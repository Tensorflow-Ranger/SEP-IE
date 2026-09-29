"""Full-test-set version of v12_control_qwen_selfrank.py: instead of just the
26-problem heval32 static eval, this runs the same "just ask Qwen zero-shot"
control on the actual 121-group TEST split from v12 Stage E
(v12_train_ranking_probe.py) -- the same held-out set the deployed
rank_probe_v12_pairwise.* probe reports 0.636 test hit-rate on (layer 26,
selected by val), with layer 22 hitting 0.777 as the best-of-6-layers upper
bound (see FINDINGS.md's Stage E results table).

Reproduces the EXACT same group-disjoint split (same seed=0, same
`make_combined_train_val_test_split` call, same 602-group usable pool) by
loading data/v12_ranking/activations/*.npz just to recover which ids belong
to the test split -- no activations/hidden states are used for scoring here,
only the split membership. Candidates/tiers/prompts come from
judged.jsonl / ground_truth.jsonl exactly as the probe's own eval used them.

Usage: python v12_control_qwen_selfrank_fulltest.py
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

from src.combined_splits import make_combined_train_val_test_split  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"
GROUND_TRUTH_PATH = ROOT / "data" / "v12_ranking" / "ground_truth.jsonl"
JUDGED_PATH = ROOT / "data" / "v12_ranking" / "judged.jsonl"
ACT_DIR = ROOT / "data" / "v12_ranking" / "activations"
SEED = 0
OUT_PATH = ROOT / "data" / "v12_ranking" / "qwen_selfrank_fulltest_results.json"
LOG_EVERY_N_SAVE = 10  # incremental checkpointing in case of long-running interruption

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


def build_candidates_block(candidates: list[dict]) -> str:
    return "\n".join(f"{c['idx']}. {c['question']}" for c in candidates)


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


def reconstruct_test_ids() -> list[str]:
    """Same group-selection + split logic as v12_train_ranking_probe.py's
    load_groups()+main(), but we only need ids/dataset_source/source_problem_id,
    not the activation vectors themselves."""
    ground_truth = {}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                ground_truth[row["id"]] = row

    ids, dataset_source, source_problem_id = [], [], []
    for npz_path in sorted(ACT_DIR.glob("*.npz")):
        gid = npz_path.stem
        if gid not in ground_truth:
            continue
        data = np.load(npz_path)
        tiers = data["tiers"]
        if len(tiers) < 2 or len(set(tiers.tolist())) < 2:
            continue
        ids.append(gid)
        dataset_source.append(ground_truth[gid]["dataset_source"])
        source_problem_id.append(ground_truth[gid]["source_problem_id"])

    ids = np.array(ids)
    dataset_source = np.array(dataset_source)
    source_problem_id = np.array(source_problem_id)
    labels_dummy = np.ones(len(ids), dtype=np.int64)

    train_idx, val_idx, test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    print(f"[qwen-selfrank-full] reproduced split: {len(train_idx)} train / "
          f"{len(val_idx)} val / {len(test_idx)} test (of {len(ids)} usable groups)", flush=True)
    return ids[test_idx].tolist()


def main() -> None:
    test_ids = reconstruct_test_ids()

    ground_truth = {}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                ground_truth[row["id"]] = row

    judged = {}
    with JUDGED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                judged[row["id"]] = row

    todo = [gid for gid in test_ids if gid in judged and gid in ground_truth]
    print(f"[qwen-selfrank-full] {len(todo)}/{len(test_ids)} test ids have judged+ground_truth rows", flush=True)

    # Resume support: skip ids already scored in a partial results file.
    done_rows = {}
    if OUT_PATH.exists():
        prev = json.loads(OUT_PATH.read_text())
        for row in prev.get("per_problem", []):
            done_rows[row["id"]] = row
        print(f"[qwen-selfrank-full] resuming, {len(done_rows)} already scored", flush=True)

    remaining = [gid for gid in todo if gid not in done_rows]
    if not remaining:
        print("[qwen-selfrank-full] all done already", flush=True)
    else:
        print(f"[qwen-selfrank-full] loading {MODEL_NAME} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: "13GiB", "cpu": "40GiB"},
        )
        model.eval()
        print("[qwen-selfrank-full] model loaded", flush=True)

        for n, gid in enumerate(remaining, 1):
            prompt = ground_truth[gid]["prompt"]
            candidates = judged[gid]["candidates"]
            tiers = [c["tier"] for c in candidates]
            best_tier = max(tiers)
            valid_idxs = {c["idx"] for c in candidates}

            user_msg = SELECT_INSTRUCTION.format(prompt=prompt, candidates=build_candidates_block(candidates))
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
                hit, note = 0, "unparsed"
                picked_tier = None
            else:
                picked_tier = next(c["tier"] for c in candidates if c["idx"] == picked)
                hit, note = int(picked_tier == best_tier), "ok"

            done_rows[gid] = {
                "id": gid, "hit": hit, "picked_idx": picked, "picked_tier": picked_tier,
                "best_tier": best_tier, "n_candidates": len(candidates), "note": note,
                "raw_response": gen_text.strip(),
            }
            mark = "HIT " if hit else "MISS"
            print(f"[qwen-selfrank-full] ({n}/{len(remaining)}) {mark} {gid}: picked={picked} note={note}", flush=True)

            if n % LOG_EVERY_N_SAVE == 0:
                _save(done_rows, todo, partial=True)

    _save(done_rows, todo, partial=False)


def _save(done_rows: dict, todo: list[str], partial: bool) -> None:
    per_problem = [done_rows[gid] for gid in todo if gid in done_rows]
    hits = [r["hit"] for r in per_problem]
    n_parsed = sum(1 for r in per_problem if r["note"] == "ok")
    payload = {
        "model": MODEL_NAME,
        "method": "zero-shot single-pick prompt (no probe, no hidden-state ranking)",
        "split": "v12 Stage E test split (121 groups), reproduced via same seed",
        "n_target": len(todo),
        "n_scored": len(per_problem),
        "n_parsed": n_parsed,
        "hit_rate": (sum(hits) / len(hits)) if hits else None,
        "per_problem": per_problem,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    if not partial:
        print(f"[qwen-selfrank-full] FINAL rank-1 hit rate: {payload['hit_rate']:.3f} "
              f"({sum(hits)}/{len(hits)}) over {len(per_problem)}/{len(todo)} scored", flush=True)
        print(f"[qwen-selfrank-full] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
