"""Stage C (Claude-as-judge, done inline by reading each candidate against
the known ambiguity_explanation -- tiers hand-assigned to
/tmp .../bcb_tiers.json, see conversation) + the same zero-shot
candidate-SELECTION control used on heval32/ccb
(v12_control_qwen_selfrank.py), applied to the small orchid_bcb sample
elicited by v12_bcb_elicit.py.

Builds data/v12_ranking/bcb_judged.jsonl (same schema as heval32_judged.jsonl:
{id, candidates: [{idx, question, tier}]}) from bcb_candidates.jsonl +
the tier assignments, skips any group with no tier-2 candidate at all
(elicitation never surfaced the true ambiguity -- no ranking signal, same
skip rule v12_eval_heval32.py uses for all-same-tier groups), then runs the
same "pick the numbered candidate that targets the real ambiguity" prompt
against Qwen3-8B zero-shot on the remaining groups.

Usage: python v12_bcb_judge_and_selfrank.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen3-8B"
CANDIDATES_PATH = ROOT / "data" / "v12_ranking" / "bcb_candidates.jsonl"
TIERS_PATH = Path("/tmp/claude-1000/-home-ubuntu-SEP-IE/68609f4a-7d53-4b01-bf70-fc84ec45b5a7/scratchpad/bcb_tiers.json")
JUDGED_OUT_PATH = ROOT / "data" / "v12_ranking" / "bcb_judged.jsonl"
RESULTS_OUT_PATH = ROOT / "data" / "v12_ranking" / "qwen_bcb_selfrank_results.json"

SELECT_INSTRUCTION = """You are given a Python function specification and a numbered list of \
candidate clarifying questions about it. Exactly one of these questions targets \
the single most important ambiguity or underspecified detail in the specification \
-- the kind of thing that, if answered, would change how you'd implement the \
function. The others are either irrelevant, already answered by the statement, \
or address a much less important detail.

Function specification:
{prompt}

Candidate questions:
{candidates}

Which numbered question best targets the single most important ambiguity in \
this specification? Think briefly, then answer with a final line of the exact \
form:
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


def main() -> None:
    tiers_map = json.loads(TIERS_PATH.read_text())

    rows = []
    with CANDIDATES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    judged_rows = []
    skipped = []
    for r in rows:
        t = tiers_map[r["id"]]
        tier2 = set(t["tier2_idx"])
        tier1 = set(t["tier1_idx"])
        if not tier2:
            skipped.append(r["id"])
            continue
        candidates = []
        for i, q in enumerate(r["questions"]):
            tier = 2 if i in tier2 else (1 if i in tier1 else 0)
            candidates.append({"idx": i, "question": q, "tier": tier})
        judged_rows.append({"id": r["id"], "prompt": r["prompt"], "candidates": candidates,
                             "true_explanation": r["true_explanation"]})

    with JUDGED_OUT_PATH.open("w", encoding="utf-8") as f:
        for row in judged_rows:
            f.write(json.dumps({"id": row["id"], "candidates": row["candidates"]}, ensure_ascii=False) + "\n")
    print(f"[bcb-judge] {len(judged_rows)} groups have a tier-2 candidate (usable), "
          f"{len(skipped)} skipped (no tier-2 candidate elicited at all): {skipped}", flush=True)

    if not judged_rows:
        print("[bcb-judge] nothing usable, exiting", flush=True)
        return

    print(f"[bcb-judge] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "13GiB", "cpu": "40GiB"},
    )
    model.eval()
    print("[bcb-judge] model loaded", flush=True)

    hits, per_problem = [], []
    for n, row in enumerate(judged_rows, 1):
        candidates = row["candidates"]
        best_tier = max(c["tier"] for c in candidates)
        valid_idxs = {c["idx"] for c in candidates}

        user_msg = SELECT_INSTRUCTION.format(prompt=row["prompt"], candidates=build_candidates_block(candidates))
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
            picked_tier = next(c["tier"] for c in candidates if c["idx"] == picked)
            hit, note = int(picked_tier == best_tier), "ok"

        hits.append(hit)
        per_problem.append({"id": row["id"], "hit": hit, "picked_idx": picked, "picked_tier": picked_tier,
                             "best_tier": best_tier, "n_candidates": len(candidates), "note": note,
                             "raw_response": gen_text.strip()})
        mark = "HIT " if hit else "MISS"
        print(f"[bcb-judge] ({n}/{len(judged_rows)}) {mark} {row['id']}: picked={picked} note={note}", flush=True)

    hit_rate = sum(hits) / len(hits)
    print(f"[bcb-judge] rank-1 hit rate: {hit_rate:.3f} ({sum(hits)}/{len(hits)}) "
          f"over {len(judged_rows)}/{len(rows)} usable groups", flush=True)

    RESULTS_OUT_PATH.write_text(json.dumps({
        "model": MODEL_NAME,
        "method": "zero-shot candidate-selection (same prompt style as heval32/ccb controls)",
        "n_target": len(rows),
        "n_usable": len(judged_rows),
        "n_skipped_no_tier2": len(skipped),
        "skipped_ids": skipped,
        "hit_rate": hit_rate,
        "per_problem": per_problem,
    }, indent=2))
    print(f"[bcb-judge] saved {RESULTS_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
