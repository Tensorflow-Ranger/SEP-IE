"""v3 retrain, extraction leg: same pipeline as v12_extract_activations.py
(STYLE_INSTRUCTION inline-rewrite + last-token hidden states over the same
LAYERS sweep), applied to the 173 new v13_rejudge groups that (a) survived
the label_leaked/label_test_relevant exclusion filter and (b) have >=2
distinct candidate tiers (the rest have no ranking signal to train on, so
extracting them would waste GPU time for nothing).

Input: data/v13_rejudge/extraction_input.json (built from merged.json +
data/v13_rejudge/chunks/*.json's prompt field), each row:
  {id, dataset_source, ambiguity_type, source_problem_id, prompt,
   candidates: [{question, tier, answer}, ...]}

Output layout (resumable, same schema as v12's activations dir so both can
be merged for training):
  data/v13_rejudge/activations_new/{id}.npz
    baseline, candidates, tiers, layers -- same fields as v12.
  data/v13_rejudge/activations_new/{id}.json
    {id, questions, answers, inline_prompts, tiers} -- for spot-checking.

Usage: python v13_extract_activations.py [--limit N]
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
    LAYERS,
    STYLE_INSTRUCTION,
    generate_sane_batch,
    hidden_states_at_layers,
    hidden_states_at_layers_batch,
    is_sane_rewrite,
)

MODEL_NAME = "Qwen/Qwen3-8B"
INPUT_PATH = ROOT / "data" / "v13_rejudge" / "extraction_input.json"
OUT_DIR = ROOT / "data" / "v13_rejudge" / "activations_new"


def safe_name(row_id: str) -> str:
    """Some ids (orchid_bcb) embed a '/' from the BigCodeBench task_id -- not
    valid in a filename."""
    return row_id.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = json.loads(INPUT_PATH.read_text())
    if args.limit:
        rows = rows[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in rows if not (OUT_DIR / f"{safe_name(r['id'])}.npz").exists()]
    print(f"[v13-extract] {len(rows) - len(todo)} already done, {len(todo)}/{len(rows)} remaining", flush=True)
    if not todo:
        return

    print(f"[v13-extract] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "13GiB", "cpu": "48GiB"},
    )
    model.eval()
    print("[v13-extract] model loaded", flush=True)

    for n, row in enumerate(todo, 1):
        t0 = time.time()
        prompt = row["prompt"]
        log_prefix = f"[v13-extract][{row['id']}] "

        baseline_vec = hidden_states_at_layers(tokenizer, model, prompt, LAYERS)

        tiers = [c["tier"] for c in row["candidates"]]
        answers = [c["answer"] for c in row["candidates"]]
        questions = [c["question"] for c in row["candidates"]]
        rewrite_users = [
            STYLE_INSTRUCTION.format(
                original=prompt,
                detail=f"Add this clarification, integrated naturally into the relevant sentence: {a}",
            )
            for a in answers
        ]
        is_sane_list = [(lambda raw: is_sane_rewrite(raw, prompt))] * len(rewrite_users)
        inline_prompts = generate_sane_batch(
            tokenizer, model, rewrite_users, max_new_tokens=600, is_sane_list=is_sane_list,
            log_prefix=log_prefix,
        )
        cand_vecs = hidden_states_at_layers_batch(tokenizer, model, inline_prompts, LAYERS)

        np.savez(
            OUT_DIR / f"{safe_name(row['id'])}.npz",
            baseline=baseline_vec.astype(np.float32),
            candidates=cand_vecs.astype(np.float32),
            tiers=np.array(tiers, dtype=np.int32),
            layers=np.array(LAYERS, dtype=np.int32),
        )
        (OUT_DIR / f"{safe_name(row['id'])}.json").write_text(json.dumps({
            "id": row["id"], "questions": questions, "answers": answers,
            "inline_prompts": inline_prompts, "tiers": tiers,
        }, ensure_ascii=False, indent=2))

        print(f"[v13-extract] ({n}/{len(todo)}) {row['id']}: {len(cand_vecs)} candidates "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("[v13-extract] done", flush=True)


if __name__ == "__main__":
    main()
