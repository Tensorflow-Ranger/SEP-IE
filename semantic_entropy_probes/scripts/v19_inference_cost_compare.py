"""Inference-time cost: zero-shot generation vs. probe forward-pass, on THIS
GPU (A10G) -- v12's control found no measurable speedup either way on the old
T4 (both ~9-10s/problem, CPU-offload-bound, see FINDINGS.md's "## Control
experiments" section), but that was flagged as untested on the A10G in both
the "## v19 control" and "## v19 layer sweep" caveats. This script measures
both directly, same model load, same groups, back to back.

Zero-shot: v19_control_qwen_selfrank.py's exact generation call (up to 400
new tokens, greedy).
Probe: v19_extract_activations.py's exact hidden-state extraction (single
forward pass, output_hidden_states=True, no generation) -- the classifier
scoring on top (StandardScaler + PCA + LogisticRegression) is µs-scale and
not separately timed.

Usage: python v19_inference_cost_compare.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from v19_control_qwen_selfrank import build_groups, build_candidates_block, SELECT_INSTRUCTION  # noqa: E402

MODEL_NAME = "Qwen/Qwen3-8B"
N_SAMPLE = 20
LAYERS = [16, 18, 20, 22, 24, 26]


def main():
    groups = build_groups()
    sample_ids = sorted(groups)[:N_SAMPLE]
    print(f"[cost-compare] timing on {len(sample_ids)} groups", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[cost-compare] model loaded", flush=True)

    # warm-up (first call pays CUDA kernel compilation/allocator warmup)
    warm_g = groups[sample_ids[0]]
    msg = [{"role": "user", "content": warm_g["prompt"]}]
    text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        model(**inputs, output_hidden_states=True, use_cache=False)
    torch.cuda.synchronize()

    # --- zero-shot generation timing ---
    gen_times = []
    for gid in sample_ids:
        g = groups[gid]
        user_msg = SELECT_INSTRUCTION.format(prompt=g["prompt"], candidates=build_candidates_block(g["candidates"]))
        messages = [{"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        dt = time.time() - t0
        n_new = out.shape[1] - inputs["input_ids"].shape[1]
        gen_times.append(dt)
        print(f"[cost-compare] gen {gid}: {dt:.2f}s ({n_new} new tokens)", flush=True)

    # --- probe forward-pass (hidden states only, no generation) timing ---
    fwd_times = []
    for gid in sample_ids:
        g = groups[gid]
        messages = [{"role": "user", "content": g["prompt"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model(**inputs, output_hidden_states=True, use_cache=False)
        torch.cuda.synchronize()
        dt = time.time() - t0
        fwd_times.append(dt)
        print(f"[cost-compare] fwd {gid}: {dt:.3f}s", flush=True)

    import numpy as np
    gen_mean, gen_med = np.mean(gen_times), np.median(gen_times)
    fwd_mean, fwd_med = np.mean(fwd_times), np.median(fwd_times)
    print(f"\n[cost-compare] zero-shot generation: mean={gen_mean:.2f}s median={gen_med:.2f}s (n={len(gen_times)})")
    print(f"[cost-compare] probe forward pass:    mean={fwd_mean:.3f}s median={fwd_med:.3f}s (n={len(fwd_times)})")
    print(f"[cost-compare] speedup (probe vs zero-shot): {gen_mean/fwd_mean:.1f}x faster")


if __name__ == "__main__":
    main()
