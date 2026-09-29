"""v20 pilot stage 3: extract Qwen3-8B hidden states for every candidate in
data/v20_pilot_multihop/pilot_results.jsonl, joined against
oracle_answers.jsonl for per-candidate answer text -- same STYLE_INSTRUCTION
inline-rewrite + last-token hidden-state extraction as v12_extract_activations.py
(imported directly), just applied per-round: each row's `prompt` field
(round 0 = raw ambiguous, round 1 = dataset's clear original, round 2 =
clear + round-1's applied answer) is the REWRITE BASE for that round's
candidates, not always the round-0 prompt -- this mirrors what the live
loop actually does (each round rewrites off the PREVIOUS round's chosen
prompt, not the original).

Output (resumable, one file pair per (id, round_depth)):
  data/v20_pilot_multihop/activations/{id}__r{round_depth}.npz
    baseline: (n_layers, hidden_dim) float32   -- this round's own base prompt, unmodified
    candidates: (n_candidates, n_layers, hidden_dim) float32
    tiers: (n_candidates,) int32
    layers: (n_layers,) int32
  data/v20_pilot_multihop/activations/{id}__r{round_depth}.json
    {"id":..., "round_depth":..., "questions":[...], "answers":[...], "inline_prompts":[...]}

Usage: python -m scripts.v20_pilot_extract_activations [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v12_extract_activations import (  # noqa: E402
    MODEL_NAME,
    STYLE_INSTRUCTION,
    LAYERS,
    generate_sane_batch,
    hidden_states_at_layers,
    hidden_states_at_layers_batch,
    is_sane_rewrite,
)

PILOT_PATH = ROOT / "data" / "v20_pilot_multihop" / "pilot_results.jsonl"
ANSWERS_PATH = ROOT / "data" / "v20_pilot_multihop" / "oracle_answers.jsonl"
OUT_DIR = ROOT / "data" / "v20_pilot_multihop" / "activations"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    pilot_rows = [r for r in read_jsonl(PILOT_PATH) if "error" not in r and r["judged"].get("candidates")]
    answers_by_key = {}
    for a in read_jsonl(ANSWERS_PATH):
        if "error" in a:
            continue
        answers_by_key[(a["id"], a["round_depth"])] = {x["question"]: x["answer"] for x in a["answers"]}

    rows = []
    for r in pilot_rows:
        key = (r["id"], r["round_depth"])
        if key not in answers_by_key:
            continue
        rows.append(r)
    if args.limit:
        rows = rows[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (OUT_DIR / f"{r['id']}__r{r['round_depth']}.npz").exists()]
    print(f"[v20-pilot-extract] {len(rows) - len(todo)} already done, {len(todo)}/{len(rows)} remaining", flush=True)
    if not todo:
        return

    print(f"[v20-pilot-extract] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v20-pilot-extract] model loaded", flush=True)

    for n, row in enumerate(todo, 1):
        t0 = time.time()
        rid, rd = row["id"], row["round_depth"]
        prompt = row["prompt"]
        log_prefix = f"[v20-pilot-extract][{rid}][r{rd}] "

        baseline_vec = hidden_states_at_layers(tokenizer, model, prompt, LAYERS)

        cands = row["judged"]["candidates"]
        ans_map = answers_by_key[(rid, rd)]
        questions = [c["question"] for c in cands]
        tiers = [c.get("tier", 0) for c in cands]
        answers = [ans_map.get(q, "") for q in questions]

        rewrite_users = [
            STYLE_INSTRUCTION.format(
                original=prompt,
                detail=f"Add this clarification, integrated naturally into the relevant sentence: {a}",
            )
            for a in answers
        ]
        is_sane_list = [(lambda raw: is_sane_rewrite(raw, prompt))] * len(rewrite_users)
        inline_prompts = generate_sane_batch(
            tokenizer, model, rewrite_users, max_new_tokens=500, is_sane_list=is_sane_list,
            log_prefix=log_prefix,
        )
        cand_vecs = hidden_states_at_layers_batch(tokenizer, model, inline_prompts, LAYERS)

        np.savez(
            OUT_DIR / f"{rid}__r{rd}.npz",
            baseline=baseline_vec,
            candidates=cand_vecs,
            tiers=np.array(tiers, dtype=np.int32),
            layers=np.array(LAYERS, dtype=np.int32),
        )
        (OUT_DIR / f"{rid}__r{rd}.json").write_text(json.dumps({
            "id": rid, "round_depth": rd, "questions": questions, "answers": answers,
            "inline_prompts": inline_prompts,
        }, indent=2))

        print(f"{log_prefix}[{n}/{len(todo)}] {len(cands)} candidates, {time.time()-t0:.1f}s", flush=True)

    print("[v20-pilot-extract] DONE", flush=True)


if __name__ == "__main__":
    main()
