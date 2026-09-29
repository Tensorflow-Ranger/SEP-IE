"""Control experiment for v12: can Qwen3-8B, just by being ASKED which
candidate clarifying question best targets the true ambiguity, pick it --
without any probe, just zero-shot generation over the same candidate pools
used in the v12/old-probe static heval32 eval (data/v12_ranking/heval32_judged.jsonl,
problem text from data/v10_code_quality/held_out_orchid32.json)?

This isolates "does Qwen's forward pass even encode which candidate is
correct, if you just ask it in plain language" from "can a linear probe on
its hidden states extract that." If Qwen's own zero-shot judgment does much
better than the v12 probe's 0.538 (14/26), that says the information is in
there but the probe isn't reading it well. If Qwen does about as poorly (or
worse), that says the base ambiguity-identification task is just hard for
this model, and the probe isn't leaving much on the table.

Same 26-problem set as v12_eval_heval32.py / v12_eval_heval32_old_probes.py
(6 all-same-tier problems skipped, no ranking signal to hit).

Usage: python v12_control_qwen_selfrank.py
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
HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json"
JUDGED_PATH = ROOT / "data" / "v12_ranking" / "heval32_judged.jsonl"
V12_RESULTS_PATH = ROOT / "data" / "v12_ranking" / "heval32_eval_results.json"
OUT_PATH = ROOT / "data" / "v12_ranking" / "qwen_selfrank_heval32_results.json"

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
    lines = []
    for c in candidates:
        lines.append(f"{c['idx']}. {c['question']}")
    return "\n".join(lines)


def parse_answer(text: str, valid_idxs: set[int]) -> int | None:
    matches = re.findall(r"ANSWER:\s*(\d+)", text)
    if matches:
        val = int(matches[-1])
        if val in valid_idxs:
            return val
    # fallback: last standalone integer in the text that's a valid idx
    for tok in reversed(re.findall(r"\b\d+\b", text)):
        val = int(tok)
        if val in valid_idxs:
            return val
    return None


def main() -> None:
    held_out = json.loads(HELD_OUT_PATH.read_text())
    v12_results = json.loads(V12_RESULTS_PATH.read_text())
    skipped_ids = set(v12_results["skipped_ids"])

    judged_rows = []
    with JUDGED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                judged_rows.append(json.loads(line))

    todo = [r for r in judged_rows if r["id"] not in skipped_ids]
    print(f"[qwen-selfrank] {len(todo)} problems to evaluate (matching v12 static eval's 26)", flush=True)

    print(f"[qwen-selfrank] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "13GiB", "cpu": "40GiB"},
    )
    model.eval()
    print("[qwen-selfrank] model loaded", flush=True)

    hits, ranks, per_problem = [], [], []

    for n, jrow in enumerate(todo, 1):
        gid = jrow["id"]
        prompt = held_out[gid]["ambiguous_prompt"]
        candidates = jrow["candidates"]
        tiers = [c["tier"] for c in candidates]
        best_tier = max(tiers)
        valid_idxs = {c["idx"] for c in candidates}

        user_msg = SELECT_INSTRUCTION.format(
            prompt=prompt,
            candidates=build_candidates_block(candidates),
        )
        messages = [{"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=400,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        picked = parse_answer(gen_text, valid_idxs)
        if picked is None:
            hit = 0
            rank_of_best = len(candidates)  # worst case: treat unparsed as miss, max rank
            note = "unparsed"
        else:
            picked_tier = next(c["tier"] for c in candidates if c["idx"] == picked)
            hit = int(picked_tier == best_tier)
            rank_of_best = 1 if hit else 2  # single-pick task: no full ranking, just hit/no-hit
            note = "ok"

        hits.append(hit)
        ranks.append(rank_of_best)
        per_problem.append({
            "id": gid,
            "hit": hit,
            "picked_idx": picked,
            "picked_tier": (next((c["tier"] for c in candidates if c["idx"] == picked), None)
                            if picked is not None else None),
            "best_tier": best_tier,
            "n_candidates": len(candidates),
            "note": note,
            "raw_response": gen_text.strip(),
        })
        mark = "HIT " if hit else "MISS"
        print(f"[qwen-selfrank] ({n}/{len(todo)}) {mark} {gid}: picked={picked} note={note}", flush=True)

    n_parsed = sum(1 for p in per_problem if p["note"] == "ok")
    print(f"[qwen-selfrank] parsed {n_parsed}/{len(todo)} responses", flush=True)
    print(f"[qwen-selfrank] rank-1 hit rate: {sum(hits)/len(hits):.3f} ({sum(hits)}/{len(hits)})", flush=True)

    OUT_PATH.write_text(json.dumps({
        "model": MODEL_NAME,
        "method": "zero-shot single-pick prompt (no probe, no hidden-state ranking)",
        "n_evaluated": len(todo),
        "n_parsed": n_parsed,
        "hit_rate": sum(hits) / len(hits) if hits else None,
        "per_problem": per_problem,
    }, indent=2))
    print(f"[qwen-selfrank] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
