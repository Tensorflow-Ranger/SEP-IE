"""v12 heval32 eval: same extraction logic as v12_extract_activations.py
(Stage D), but pointed at the 32-problem heval32 held-out benchmark instead
of the 635-row training set, using data/v12_ranking/heval32_judged.jsonl
(tier-judged by a fork agent, see v12_ranking_recipe_pipeline memory) for
candidates/answers/tiers and data/v10_code_quality/held_out_orchid32.json
for each problem's ambiguous_prompt.

heval32 problems were explicitly EXCLUDED from training (see
v12_build_ranking_ground_truth.py's held-out filtering), so this is a clean
held-out evaluation of the trained probe -- not just a re-run of the test
split.

Output: data/v12_ranking/heval32_activations/{id}.npz (same schema as
Stage D's activations/{id}.npz).

Usage: python v12_extract_heval32_activations.py
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
    MODEL_NAME,
    generate_sane_batch,
    hidden_states_at_layers,
    hidden_states_at_layers_batch,
    is_sane_rewrite,
    STYLE_INSTRUCTION,
)

VARIANT_PATHS = {
    "vagueness": {
        "held_out": ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json",
        "judged": ROOT / "data" / "v12_ranking" / "heval32_judged.jsonl",
        "out_dir": ROOT / "data" / "v12_ranking" / "heval32_activations",
    },
    "lexical": {
        "held_out": ROOT / "data" / "v10_code_quality" / "held_out_orchid32_lexical.json",
        "judged": ROOT / "data" / "v12_ranking" / "heval32_lexical_judged.jsonl",
        "out_dir": ROOT / "data" / "v12_ranking" / "heval32_lexical_activations",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["vagueness", "lexical"], default="vagueness")
    args = ap.parse_args()
    paths = VARIANT_PATHS[args.variant]
    HELD_OUT_PATH, JUDGED_PATH, OUT_DIR = paths["held_out"], paths["judged"], paths["out_dir"]

    held_out = json.loads(HELD_OUT_PATH.read_text())

    judged_rows = []
    with JUDGED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                judged_rows.append(json.loads(line))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in judged_rows if not (OUT_DIR / f"{r['id']}.npz").exists()]
    print(f"[v12-heval32-D] {len(judged_rows) - len(todo)} already done, {len(todo)}/{len(judged_rows)} remaining", flush=True)
    if not todo:
        return

    print(f"[v12-heval32-D] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 16:
        # T4-class card (~15GB usable): Qwen3-8B bf16 doesn't comfortably fit standalone
        # (see sepie_g4ad_t4_gpu_quirks memory) -- offload the last couple of layers to CPU.
        print(f"[v12-heval32-D] {total_mem_gb:.1f}GB GPU detected, using CPU-offload device_map", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
            max_memory={0: "13GiB", "cpu": "40GiB"},
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v12-heval32-D] model loaded", flush=True)

    for n, jrow in enumerate(todo, 1):
        t0 = time.time()
        prompt = held_out[jrow["id"]]["ambiguous_prompt"]
        log_prefix = f"[v12-heval32-D][{jrow['id']}] "

        baseline_vec = hidden_states_at_layers(tokenizer, model, prompt, LAYERS)

        tiers = [c["tier"] for c in jrow["candidates"]]
        answers = [c["answer"] for c in jrow["candidates"]]
        questions = [c["question"] for c in jrow["candidates"]]
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
            OUT_DIR / f"{jrow['id']}.npz",
            baseline=baseline_vec.astype(np.float32),
            candidates=cand_vecs.astype(np.float32),
            tiers=np.array(tiers, dtype=np.int32),
            layers=np.array(LAYERS, dtype=np.int32),
        )
        (OUT_DIR / f"{jrow['id']}.json").write_text(json.dumps({
            "id": jrow["id"], "questions": questions, "answers": answers,
            "inline_prompts": inline_prompts, "tiers": tiers,
        }, ensure_ascii=False, indent=2))

        print(f"[v12-heval32-D] ({n}/{len(todo)}) {jrow['id']}: {len(cand_vecs)} candidates "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("[v12-heval32-D] done", flush=True)


if __name__ == "__main__":
    main()
