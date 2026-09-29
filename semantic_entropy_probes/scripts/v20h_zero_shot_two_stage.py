"""v20h: zero-shot Qwen two-round baseline, mirroring v20g's two-stage funnel
but with BOTH stages done by Qwen3-8B zero-shot self-rank (no probe, no
hidden-state ranking) -- for direct comparison against the probe-based
funnel result (78.7% final; data/v20_branches_full/two_stage_funnel_results.json,
probe_1 alone 65.6%).

Uses the IDENTICAL 244-group test split as v20g (same groups_with_branches
set, same SEED=0 permutation over sorted ids). Stage 1 hits/picks are read
from the already-computed data/v19_concat_activations/qwen_selfrank_results.json
(zero-shot Qwen self-rank over all 1229 groups, verbatim v12 prompt) instead
of re-running the model -- that file's 1229-group universe is a strict
superset of this eval's 244-group test set, so no new stage-1 inference is
needed.

For every stage-1 MISS, this script:
  1. Reconstructs the round-1 candidate list the SAME way v19_control_qwen_
     selfrank.py did (`build_groups`), to map Qwen's own `picked_idx` back to
     the actual wrong question text it chose (not the probe's pick -- Qwen's
     own zero-shot mistake may differ from the trained probe's).
  2. Finds the round-2 branch in branches_round2.json whose `removed_question`
     matches that wrong question text EXACTLY (same matching approach as
     v20g), using elicited_round2.jsonl (keyed by its own `branch_idx` field,
     NOT list position -- the two files are not co-ordered) for the round-2
     candidate question text, cross-checked against the branch's .npz `tiers`
     array (hard skip on mismatch, never mis-score).
  3. Runs the SAME zero-shot selection prompt against the round-2 branch
     prompt + round-2 candidates, and scores hit2.

Final hit rate = stage-1 hits + stage-2 recovered misses, over the same
244-group denominator as v20g.

Usage: python v20h_zero_shot_two_stage.py
Resumable: writes data/v20_branches_full/zero_shot_two_stage_results.json
incrementally (stage-2 generations are the only new model calls -- there
are at most ~n_miss of them, a small fraction of the 244 group test set).
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_control_qwen_selfrank import (  # noqa: E402
    MODEL_NAME, SELECT_INSTRUCTION, build_candidates_block, build_groups, parse_answer,
)

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
ELICITED_PATH = ROOT / "data" / "v20_branches_full" / "elicited_round2.jsonl"
ACT_DIR = ROOT / "data" / "v20_branches_full" / "round2_activations"
STAGE1_CACHE = ROOT / "data" / "v19_concat_activations" / "qwen_selfrank_results.json"
OUT_PATH = ROOT / "data" / "v20_branches_full" / "zero_shot_two_stage_results.json"
SEED = 0
TEST_FRAC = 0.2
LOG_EVERY_N_SAVE = 5


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def reconstruct_test_groups():
    """Identical to v20g_two_stage_funnel_eval.py's test-group construction."""
    branches = json.loads(BRANCHES_PATH.read_text())
    branches_by_group = defaultdict(list)  # gid -> [(branch_idx, removed_question)]
    for i, b in enumerate(branches):
        if not (ACT_DIR / f"{i}.npz").exists():
            continue
        branches_by_group[b["id"]].append((i, b["removed_question"].strip()))

    groups_with_branches = sorted(branches_by_group.keys())
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_branches))
    n_test = max(1, int(len(groups_with_branches) * TEST_FRAC))
    test_groups = [g for g in np.array(groups_with_branches)[perm[:n_test]].tolist()]
    return test_groups, branches_by_group, branches


def main() -> None:
    test_groups, branches_by_group, branches = reconstruct_test_groups()
    print(f"[v20h] {len(test_groups)} test groups (identical split to v20g)", flush=True)

    groups = build_groups()  # round-1 candidate lists, cross-checked vs npz tiers
    stage1 = json.loads(STAGE1_CACHE.read_text())
    stage1_by_id = {r["id"]: r for r in stage1["per_problem"]}

    missing = [g for g in test_groups if g not in stage1_by_id or g not in groups]
    if missing:
        print(f"[v20h] WARNING: {len(missing)} test groups missing from stage-1 cache / "
              f"build_groups() -- excluded from denominator: {missing[:5]}", flush=True)
    scored_test_groups = [g for g in test_groups if g in stage1_by_id and g in groups]
    n1 = len(scored_test_groups)

    hit1_rows = {g: stage1_by_id[g] for g in scored_test_groups}
    hit1_count = sum(r["hit"] for r in hit1_rows.values())
    print(f"[v20h] stage-1 (zero-shot, cached) hit rate: {hit1_count}/{n1} = {hit1_count/n1:.4f}",
          flush=True)

    elicited_by_idx = {}
    for e in read_jsonl(ELICITED_PATH):
        elicited_by_idx[e["branch_idx"]] = e

    misses = [g for g in scored_test_groups if not hit1_rows[g]["hit"]]
    print(f"[v20h] {len(misses)} stage-1 misses to attempt chaining", flush=True)

    # Resolve, for each miss, which round-2 branch (if any) applies -- pure
    # bookkeeping, no model calls yet.
    plan = {}  # gid -> {"branch_idx": int} or {"skip_reason": str}
    for gid in misses:
        row = hit1_rows[gid]
        if row.get("note") != "ok" or row.get("picked_idx") is None:
            plan[gid] = {"skip_reason": f"stage1_unparsed(note={row.get('note')})"}
            continue
        candidates = groups[gid]["candidates"]
        picked_idx = row["picked_idx"]
        if not (1 <= picked_idx <= len(candidates)):
            plan[gid] = {"skip_reason": "picked_idx_out_of_range"}
            continue
        wrong_q = candidates[picked_idx - 1]["question"].strip()
        match = next((idx for idx, bq in branches_by_group[gid] if bq == wrong_q), None)
        if match is None:
            plan[gid] = {"skip_reason": "no_matching_branch"}
            continue
        e = elicited_by_idx.get(match)
        if e is None:
            plan[gid] = {"skip_reason": "branch_not_in_elicited"}
            continue
        d = np.load(ACT_DIR / f"{match}.npz")
        if [c["tier"] for c in e["candidates"]] != d["tiers"].tolist():
            plan[gid] = {"skip_reason": "elicited_npz_tier_mismatch"}
            continue
        plan[gid] = {"branch_idx": match}

    n_chainable = sum(1 for v in plan.values() if "branch_idx" in v)
    print(f"[v20h] {n_chainable}/{len(misses)} misses have a chainable round-2 branch "
          f"(reasons for the rest: "
          f"{ {r: sum(1 for v in plan.values() if v.get('skip_reason')==r) for r in set(v.get('skip_reason') for v in plan.values() if 'skip_reason' in v)} })",
          flush=True)

    done_rows = {}
    if OUT_PATH.exists():
        prev = json.loads(OUT_PATH.read_text())
        for row in prev.get("stage2_rows", []):
            done_rows[row["id"]] = row
        print(f"[v20h] resuming stage-2, {len(done_rows)} already scored", flush=True)

    to_run = [gid for gid, v in plan.items() if "branch_idx" in v and gid not in done_rows]

    if to_run:
        print(f"[v20h] loading {MODEL_NAME} for {len(to_run)} stage-2 generations ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        print("[v20h] model loaded", flush=True)

        for n, gid in enumerate(to_run, 1):
            branch_idx = plan[gid]["branch_idx"]
            b = branches[branch_idx]
            e = elicited_by_idx[branch_idx]
            candidates = e["candidates"]
            tiers = [c["tier"] for c in candidates]
            best_tier = max(tiers)
            valid_idxs = set(range(1, len(candidates) + 1))

            user_msg = SELECT_INSTRUCTION.format(
                prompt=b["branch_prompt"], candidates=build_candidates_block(candidates)
            )
            messages = [{"role": "user", "content": user_msg}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id,
                )
            gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            picked = parse_answer(gen_text, valid_idxs)
            if picked is None:
                hit2, note, picked_tier = 0, "unparsed", None
            else:
                picked_tier = candidates[picked - 1]["tier"]
                hit2, note = int(picked_tier == best_tier), "ok"

            done_rows[gid] = {
                "id": gid, "branch_idx": branch_idx, "hit2": hit2, "picked_idx": picked,
                "picked_tier": picked_tier, "best_tier": best_tier, "n_candidates": len(candidates),
                "note": note, "raw_response": gen_text.strip(),
            }
            mark = "HIT " if hit2 else "MISS"
            print(f"[v20h] stage-2 ({n}/{len(to_run)}) {mark} {gid} branch={branch_idx}: "
                  f"picked={picked} note={note}", flush=True)

            if n % LOG_EVERY_N_SAVE == 0:
                _save(n1, hit1_count, misses, plan, done_rows, partial=True)

        del model
        torch.cuda.empty_cache()

    _save(n1, hit1_count, misses, plan, done_rows, partial=False)


def _save(n1, hit1_count, misses, plan, done_rows, partial: bool) -> None:
    n_recovered = sum(1 for gid in misses if gid in done_rows and done_rows[gid]["hit2"])
    n_chainable = sum(1 for v in plan.values() if "branch_idx" in v)
    n_unrecoverable = len(misses) - n_chainable
    final_hits = hit1_count + n_recovered

    skip_reason_counts = {}
    for v in plan.values():
        if "skip_reason" in v:
            skip_reason_counts[v["skip_reason"]] = skip_reason_counts.get(v["skip_reason"], 0) + 1

    out = {
        "n_test_groups_scored": n1,
        "stage1_hit_rate": hit1_count / n1,
        "stage1_hits": hit1_count,
        "n_misses": len(misses),
        "n_misses_chainable": n_chainable,
        "n_misses_unrecoverable": n_unrecoverable,
        "unrecoverable_reason_counts": skip_reason_counts,
        "n_misses_recovered_by_round2": n_recovered,
        "final_two_stage_hit_rate": final_hits / n1,
        "stage2_rows": [done_rows[gid] for gid in misses if gid in done_rows],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    if not partial:
        print(f"\n[v20h] stage-1 (zero-shot) alone: {hit1_count}/{n1} = {hit1_count/n1:.4f}", flush=True)
        print(f"[v20h] {len(misses)} misses; {n_chainable} chainable, {n_unrecoverable} unrecoverable "
              f"({skip_reason_counts})", flush=True)
        print(f"[v20h] round-2 (zero-shot) recovered {n_recovered}/{n_chainable} of the chainable misses",
              flush=True)
        print(f"[v20h] FINAL zero-shot two-stage hit rate: {final_hits}/{n1} = {final_hits/n1:.4f} "
              f"(vs. stage-1-alone {hit1_count/n1:.4f})", flush=True)
        print(f"[v20h] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
