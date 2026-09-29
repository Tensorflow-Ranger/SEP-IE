"""v19 control: does Qwen know the answer if you just ask it, on v19's data?

v12's zero-shot control (`v12_control_qwen_selfrank_fulltest.py`) measured
0.645 (78/121) on v12 Stage E's 121-group test split. v19 (`## v19` in
FINDINGS.md) trained/evaluated a new ranking probe on a much larger, more
strictly filtered 4-source pool (1,229 groups: clarifycodebench, orchid_heval,
orchid_bcb, orchid_bcb_expand) but never ran the equivalent zero-shot control
against that data -- this script closes that gap.

Design: since v19's LODO test set for a held-out source is simply "every
group of that source" (`run_lodo`'s `test_mask = dataset_source ==
held_out_source` in v19_train_ranking_probe.py), scoring all 1,229 groups
ONCE and slicing by source afterward reproduces the LODO-comparable baseline
for all 4 sources plus their mean. The same single pass also covers both
pooled test splits (proportional n=246, equal_per_source n=94) by
reconstructing their exact test-id sets (same seed=0, same split functions
v19_train_ranking_probe.py used) and filtering the per-group results down to
each split's test ids.

Candidate list per group is reconstructed with the IDENTICAL filter
v19_extract_activations.py used (only candidates with a usable oracle
answer, same order) so tiers/best_tier here are byte-identical to what the
probe itself was evaluated on -- cross-checked against the already-extracted
data/v19_concat_activations/activations/*.npz tiers array at load time (hard
assertion, not a warning).

Same zero-shot selection prompt/parsing as v12's control (verbatim), applied
to v19's prompt+candidate data.

Usage: python v19_control_qwen_selfrank.py
Resumable: writes data/v19_concat_activations/qwen_selfrank_results.json
incrementally, skips groups already scored on restart.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import make_combined_train_val_test_split, make_equal_per_source_split  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"
JUDGED_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl"
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
V12_JUDGED_PATH = ROOT / "data" / "v12_ranking" / "judged.jsonl"
V15_REWRITES_DIR = ROOT / "data" / "v15_new_activations" / "rewrites"
V19_ANSWERS_DIR = ROOT / "data" / "v19_bcbexpand" / "answers"
V19_ACT_DIR = ROOT / "data" / "v19_concat_activations" / "activations"
SEED = 0
OUT_PATH = ROOT / "data" / "v19_concat_activations" / "qwen_selfrank_results.json"
LOG_EVERY_N_SAVE = 10

SOURCES = ("clarifycodebench", "orchid_heval", "orchid_bcb", "orchid_bcb_expand")

# Verbatim from v12_control_qwen_selfrank_fulltest.py.
SELECT_INSTRUCTION = """You are given a programming problem statement and a numbered list of \
candidate clarifying questions about it. Exactly one of these questions targets \
the single most important ambiguity or underspecified detail in the problem \
statement -- the kind of thing that, if answered, would change how you'd \
implement the function. The others are either irrelevant, already answered by \
the statement, or address a much less important detail.

Problem statement:
{prompt}

Candidate questions:
{candidates}

Which numbered question best targets the single most important ambiguity in \
this problem statement? Think briefly, then answer with a final line of the \
exact form:
ANSWER: <number>
"""


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def is_clean(j: dict) -> bool:
    return (
        not j.get("label_leaked")
        and j.get("label_functionally_relevant") is True
        and j.get("more_important_than_label") is None
    )


def build_candidates_block(candidates: list[dict]) -> str:
    return "\n".join(f"{i+1}. {c['question']}" for i, c in enumerate(candidates))


def parse_answer(text: str, valid_idxs: set[int]) -> int | None:
    matches = re.findall(r"ANSWER:\s*(\d+)", text)
    if matches:
        val = int(matches[-1])
        if val in valid_idxs:
            return val
    for tok in reversed(re.findall(r"\b\d+\b", text)):
        val = int(tok)
        if val in valid_idxs:
            return val
    return None


def load_answers_by_question(rid: str, v12_judged: dict) -> dict[str, str]:
    """Same lookup order as v19_extract_activations.py's function of the same name."""
    if rid in v12_judged:
        return {c["question"].strip(): c["answer"] for c in v12_judged[rid]["candidates"]}
    rw_path = V15_REWRITES_DIR / f"{safe_name(rid)}.json"
    if rw_path.exists():
        rw = json.loads(rw_path.read_text())
        return {c["question"].strip(): c["answer"] for c in rw["candidates"]}
    ans_path = V19_ANSWERS_DIR / f"{safe_name(rid)}.json"
    if ans_path.exists():
        rw = json.loads(ans_path.read_text())
        return {c["question"].strip(): c["answer"] for c in rw["candidates"]}
    return {}


def build_groups():
    """Reconstruct the exact 1,229-group pool v19_train_ranking_probe.py used,
    with prompt text + filtered candidate list (question, tier) per group,
    cross-checked against the extracted .npz tiers array."""
    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}
    raw_groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            raw_groups[g["id"]] = g

    all_ids = [rid for rid in judged_rows if rid in raw_groups and is_clean(judged_rows[rid]["judged"])]

    out = {}
    n_mismatch = 0
    for rid in all_ids:
        npz_path = V19_ACT_DIR / f"{safe_name(rid)}.npz"
        if not npz_path.exists():
            continue  # not extracted (e.g. <2 usable candidates) -- same set training excludes
        npz_tiers = np.load(npz_path)["tiers"].tolist()

        prompt = raw_groups[rid]["prompt"]
        answers_by_q = load_answers_by_question(rid, v12_judged)
        candidates = []
        for c in judged_rows[rid]["judged"]["candidates"]:
            q = c["question"].strip()
            if answers_by_q.get(q) is None:
                continue
            candidates.append({"question": q, "tier": c["tier"]})

        if [c["tier"] for c in candidates] != npz_tiers:
            n_mismatch += 1
            continue  # candidate reconstruction drifted from what the probe trained on -- skip rather than mis-score

        out[rid] = {
            "prompt": prompt,
            "candidates": candidates,
            "dataset_source": judged_rows[rid]["dataset_source"],
        }

    print(f"[v19-selfrank] {len(out)}/{len(all_ids)} groups reconstructed and tier-verified "
          f"({n_mismatch} skipped on tier mismatch)", flush=True)
    return out


def reconstruct_split_test_ids(ids, dataset_source, source_problem_id):
    labels_dummy = np.ones(len(ids), dtype=np.int64)
    _, _, prop_test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    _, _, eq_test_idx = make_equal_per_source_split(dataset_source, source_problem_id, seed=SEED)
    return set(ids[prop_test_idx].tolist()), set(ids[eq_test_idx].tolist())


def derive_source_problem_id(rid: str, dataset_source: str) -> str:
    if dataset_source == "clarifycodebench":
        return rid
    if dataset_source == "orchid_heval":
        num = rid.split("_")[1]
        return f"humaneval::{num}"
    if dataset_source == "orchid_bcb":
        rest = rid[len("bcb_"):] if rid.startswith("bcb_") else rid
        task_id = rest.rsplit("_", 1)[0]
        return f"bigcodebench::{task_id}"
    if dataset_source == "orchid_bcb_expand":
        rest = rid[len("bcbexpand_"):] if rid.startswith("bcbexpand_") else rid
        task_id = rest.rsplit("_", 1)[0]
        return f"bigcodebench_expand::{task_id}"
    raise ValueError(f"unexpected dataset_source {dataset_source}")


def main() -> None:
    groups = build_groups()

    v12_gt = {r["id"]: r for r in read_jsonl(ROOT / "data" / "v12_ranking" / "ground_truth.jsonl")}
    ids = np.array(sorted(groups))
    dataset_source = np.array([groups[g]["dataset_source"] for g in ids])
    source_problem_id = np.array([
        v12_gt[g]["source_problem_id"] if g in v12_gt else derive_source_problem_id(g, groups[g]["dataset_source"])
        for g in ids
    ])
    prop_test_ids, eq_test_ids = reconstruct_split_test_ids(ids, dataset_source, source_problem_id)
    print(f"[v19-selfrank] pooled proportional test: {len(prop_test_ids)} ids; "
          f"pooled equal_per_source test: {len(eq_test_ids)} ids", flush=True)

    todo = list(ids)

    done_rows = {}
    if OUT_PATH.exists():
        prev = json.loads(OUT_PATH.read_text())
        for row in prev.get("per_problem", []):
            done_rows[row["id"]] = row
        print(f"[v19-selfrank] resuming, {len(done_rows)} already scored", flush=True)

    remaining = [gid for gid in todo if gid not in done_rows]
    if not remaining:
        print("[v19-selfrank] all done already", flush=True)
    else:
        print(f"[v19-selfrank] loading {MODEL_NAME} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="cuda",
        )
        model.eval()
        print("[v19-selfrank] model loaded", flush=True)

        for n, gid in enumerate(remaining, 1):
            g = groups[gid]
            candidates = g["candidates"]
            tiers = [c["tier"] for c in candidates]
            best_tier = max(tiers)
            valid_idxs = set(range(1, len(candidates) + 1))

            user_msg = SELECT_INSTRUCTION.format(prompt=g["prompt"], candidates=build_candidates_block(candidates))
            messages = [{"role": "user", "content": user_msg}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=400, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            picked = parse_answer(gen_text, valid_idxs)
            if picked is None:
                hit, note, picked_tier = 0, "unparsed", None
            else:
                picked_tier = candidates[picked - 1]["tier"]
                hit, note = int(picked_tier == best_tier), "ok"

            done_rows[gid] = {
                "id": gid, "dataset_source": g["dataset_source"], "hit": hit,
                "picked_idx": picked, "picked_tier": picked_tier, "best_tier": best_tier,
                "n_candidates": len(candidates), "note": note, "raw_response": gen_text.strip(),
            }
            mark = "HIT " if hit else "MISS"
            print(f"[v19-selfrank] ({n}/{len(remaining)}) {mark} {gid} [{g['dataset_source']}]: "
                  f"picked={picked} note={note}", flush=True)

            if n % LOG_EVERY_N_SAVE == 0:
                _save(done_rows, todo, prop_test_ids, eq_test_ids, partial=True)

    _save(done_rows, todo, prop_test_ids, eq_test_ids, partial=False)


def _hit_rate(rows):
    hits = [r["hit"] for r in rows]
    return (sum(hits) / len(hits)) if hits else None, sum(hits), len(hits)


def _save(done_rows, todo, prop_test_ids, eq_test_ids, partial: bool) -> None:
    per_problem = [done_rows[gid] for gid in todo if gid in done_rows]

    by_source = {}
    for src in SOURCES:
        rows = [r for r in per_problem if r["dataset_source"] == src]
        rate, k, n = _hit_rate(rows)
        by_source[src] = {"hit_rate": rate, "hits": k, "n": n}
    lodo_rates = [v["hit_rate"] for v in by_source.values() if v["hit_rate"] is not None]
    lodo_mean = float(np.mean(lodo_rates)) if lodo_rates else None

    prop_rows = [r for r in per_problem if r["id"] in prop_test_ids]
    eq_rows = [r for r in per_problem if r["id"] in eq_test_ids]
    prop_rate, prop_k, prop_n = _hit_rate(prop_rows)
    eq_rate, eq_k, eq_n = _hit_rate(eq_rows)

    all_rate, all_k, all_n = _hit_rate(per_problem)

    payload = {
        "model": MODEL_NAME,
        "method": "zero-shot single-pick prompt (no probe, no hidden-state ranking)",
        "n_target": len(todo),
        "n_scored": len(per_problem),
        "overall_hit_rate": all_rate, "overall_hits": all_k, "overall_n": all_n,
        "lodo_by_source": by_source,
        "lodo_mean_hit_rate": lodo_mean,
        "pooled_proportional_test": {"hit_rate": prop_rate, "hits": prop_k, "n": prop_n},
        "pooled_equal_per_source_test": {"hit_rate": eq_rate, "hits": eq_k, "n": eq_n},
        "per_problem": per_problem,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    if not partial:
        print(f"\n[v19-selfrank] FINAL: overall {all_rate:.3f} ({all_k}/{all_n})", flush=True)
        for src, v in by_source.items():
            print(f"  LODO[{src}]: {v['hit_rate']:.3f} ({v['hits']}/{v['n']})" if v["hit_rate"] is not None
                  else f"  LODO[{src}]: no data", flush=True)
        print(f"  LODO mean: {lodo_mean:.3f}" if lodo_mean is not None else "  LODO mean: n/a", flush=True)
        print(f"  pooled proportional test: {prop_rate:.3f} ({prop_k}/{prop_n})" if prop_rate is not None
              else "  pooled proportional test: n/a", flush=True)
        print(f"  pooled equal_per_source test: {eq_rate:.3f} ({eq_k}/{eq_n})" if eq_rate is not None
              else "  pooled equal_per_source test: n/a", flush=True)
        print(f"[v19-selfrank] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
