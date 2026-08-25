"""Extract per-layer residual-stream hidden states at the final prompt token.

For every prompt in the canonical dataset, we:
  1. Wrap it as a single user turn and apply Qwen3's chat template with
     enable_thinking=False (non-thinking mode). This appends the standard
     assistant-turn opener plus an *empty* thinking block
     (`<think>\n\n</think>\n\n`) -- that is what "non-thinking mode" means for
     Qwen3: the thinking block is present but empty, not omitted.
  2. Run one forward pass with output_hidden_states=True.
  3. Take the residual stream at the LAST token of that templated string (the
     token the model would condition its first generated token on), for every
     layer. `hidden_states` from HF is a tuple of length n_layers+1: index 0
     is the embedding output, indices 1..n_layers are post-block residual
     streams -- we keep all of them and let the probing stage pick a layer.

Batch size is fixed at 1. With right-padding, index -1 on a padded batch would
select a PAD token's hidden state for the shorter sequences; batching would
need an attention-mask gather (`attention_mask.sum(-1) - 1`) to stay correct.
At ~500 prompts a single 8B forward pass each is cheap enough (RTX 4090) that
the extra complexity isn't worth it here.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-8B"

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "ambiguity_dataset.jsonl"
DEFAULT_OUT_DIR = ROOT / "activations"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt_text(tokenizer, problem_statement: str) -> str:
    messages = [{"role": "user", "content": problem_statement}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


@torch.no_grad()
def extract(
    rows: list[dict],
    model_name: str = MODEL_NAME,
    out_dir: Path = DEFAULT_OUT_DIR,
    max_length: int = 4096,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map="cuda",
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    n_states = n_layers + 1  # + embedding output
    print(f"[extract] {n_layers} layers, hidden_size={hidden_size}, {n_states} hidden-state tensors/prompt")

    N = len(rows)
    acts = np.zeros((N, n_states, hidden_size), dtype=np.float16)
    meta_out: list[dict] = []
    n_truncated = 0

    t0 = time.time()
    for i, row in enumerate(rows):
        text = build_prompt_text(tokenizer, row["prompt"])
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        n_tokens = enc["input_ids"].shape[1]
        if n_tokens >= max_length:
            n_truncated += 1
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple(len=n_states) of (1, seq_len, hidden_size)
        for layer_idx, layer_hs in enumerate(hs):
            acts[i, layer_idx] = layer_hs[0, -1, :].float().cpu().numpy().astype(np.float16)

        meta_out.append(
            {
                "row_index": i,
                "id": row["id"],
                "source_problem_id": row["source_problem_id"],
                "label": row["label"],  # ambiguity count K in {0,1,2,3}
                "ambiguity_types": row["ambiguity_types"],
                "prompt_token_len": n_tokens,
            }
        )

        if (i + 1) % 25 == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[extract] {i + 1}/{N} ({rate:.2f} prompts/s, elapsed {elapsed:.0f}s)")

    if n_truncated:
        print(f"[extract] WARNING: {n_truncated} prompts hit max_length={max_length} and were truncated")

    acts_path = out_dir / "activations.npy"
    meta_path = out_dir / "metadata.jsonl"
    config_path = out_dir / "extract_config.json"

    np.save(acts_path, acts)
    with meta_path.open("w", encoding="utf-8") as f:
        for m in meta_out:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "n_layers": n_layers,
                "hidden_size": hidden_size,
                "n_hidden_states": n_states,
                "n_prompts": N,
                "max_length": max_length,
                "dtype": str(dtype),
                "enable_thinking": False,
                "extraction_point": "last_token_of_templated_prompt",
            },
            f,
            indent=2,
        )

    print(f"[extract] saved activations {acts.shape} -> {acts_path}")
    print(f"[extract] saved metadata -> {meta_path}")
    print(f"[extract] saved config -> {config_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model-name", type=str, default=MODEL_NAME)
    ap.add_argument("--max-length", type=int, default=4096)
    args = ap.parse_args()

    rows = _read_jsonl(args.data)
    print(f"[extract] loaded {len(rows)} prompts from {args.data}")
    extract(rows, model_name=args.model_name, out_dir=args.out_dir, max_length=args.max_length)


if __name__ == "__main__":
    main()
