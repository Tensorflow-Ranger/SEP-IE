"""Benchmark v12's *actual* per-round scoring cost with the confound-control
rewrite-generation step excluded: given candidate prompts that already exist
(reusing the real inline_prompts already saved to disk in
data/v12_ranking/activations/{id}.json from Stage D), how long does JUST the
batched hidden-state forward pass (hidden_states_at_layers_batch) + a probe
dot product take, per problem?

This isolates step 2+3 of v12's live-loop mode_score
(scripts/v12_heval32_live_loop.py:226-239) from step 1 (generate_sane_batch,
the inline-rewrite generation) -- step 1 exists purely to normalize surface
style against the baseline the probe was trained on (confound control), not
because ranking-via-hidden-states inherently requires autoregressive
generation. Compares against the ~9.4s/problem measured for the zero-shot
self-rank control (scripts/v12_control_qwen_selfrank_fulltest.py), which IS
pure autoregressive generation (up to 400 new tokens).

Usage: python v12_bench_forward_pass_only.py [--n 20]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v12_extract_activations import MODEL_NAME, build_prompt_text  # noqa: E402


@torch.no_grad()
def hidden_states_only_no_lm_head(tokenizer, model, prompt_texts, layers):
    """Same as v12_extract_activations.hidden_states_at_layers_batch, but calls
    the base transformer (model.model) directly instead of the full
    CausalLM wrapper -- skips materializing lm_head logits entirely (we only
    need hidden states, and lm_head's CPU->GPU offload hop was OOM-ing this
    T4 when combined with a large batch's stacked hidden_states tuple)."""
    texts = [build_prompt_text(tokenizer, p) for p in prompt_texts]
    enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=4096, padding=True).to(model.device)
    attention_mask = enc["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    out = model.model(
        input_ids=enc["input_ids"], attention_mask=attention_mask,
        position_ids=position_ids, output_hidden_states=True, use_cache=False,
    )
    hs = out.hidden_states
    return np.stack(
        [np.stack([hs[l][i, -1, :].float().cpu().numpy() for l in layers], axis=0) for i in range(len(texts))],
        axis=0,
    )

ACT_DIR = ROOT / "data" / "v12_ranking" / "activations"
MODEL_JSON_PATH = ROOT / "models" / "rank_probe_v12_pairwise.json"
MODEL_PATH = ROOT / "models" / "rank_probe_v12_pairwise.joblib"
OUT_PATH = ROOT / "data" / "v12_ranking" / "forward_pass_only_bench_results.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of problems to benchmark")
    args = ap.parse_args()

    meta = json.loads(MODEL_JSON_PATH.read_text())
    layer = meta["layer"]
    bundle = joblib.load(MODEL_PATH)
    pipe, w = bundle["pipe"], bundle["w"]

    json_paths = sorted(ACT_DIR.glob("*.json"))[: args.n]
    print(f"[fwd-bench] benchmarking {len(json_paths)} problems, layer={layer}", flush=True)

    print(f"[fwd-bench] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "13GiB", "cpu": "40GiB"},
    )
    model.eval()
    print("[fwd-bench] model loaded", flush=True)

    # warm-up call (excluded from timing) -- first CUDA call pays init/compile cost
    warm = json.loads(json_paths[0].read_text())
    _ = hidden_states_only_no_lm_head(tokenizer, model, warm["inline_prompts"], [layer])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print("[fwd-bench] warm-up done, starting timed runs", flush=True)

    per_problem = []
    for n, jpath in enumerate(json_paths, 1):
        jrow = json.loads(jpath.read_text())
        prompts = jrow["inline_prompts"]

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        cand_vecs = hidden_states_only_no_lm_head(tokenizer, model, prompts, [layer])[:, 0, :]
        scores = pipe.transform(cand_vecs) @ w
        _ = int(np.argmax(scores))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - t0

        per_problem.append({"id": jrow["id"], "n_candidates": len(prompts), "seconds": dt})
        print(f"[fwd-bench] ({n}/{len(json_paths)}) {jrow['id']}: {len(prompts)} candidates, {dt:.3f}s", flush=True)

    secs = [r["seconds"] for r in per_problem]
    print(f"[fwd-bench] mean={np.mean(secs):.3f}s median={np.median(secs):.3f}s "
          f"min={min(secs):.3f}s max={max(secs):.3f}s over {len(secs)} problems", flush=True)

    OUT_PATH.write_text(json.dumps({
        "layer": layer,
        "n_problems": len(per_problem),
        "mean_seconds": float(np.mean(secs)),
        "median_seconds": float(np.median(secs)),
        "min_seconds": float(min(secs)),
        "max_seconds": float(max(secs)),
        "per_problem": per_problem,
    }, indent=2))
    print(f"[fwd-bench] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
