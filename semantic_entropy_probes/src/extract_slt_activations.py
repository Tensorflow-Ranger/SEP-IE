"""Extract SLT (second-to-last token) hidden states, alongside a matched TBG
extraction from the *same* generation -- OATML's other probing position,
compared head-to-head against this project's existing bare-prompt TBG
extraction (src/extract_activations.py).

## What SLT is, and why it's different from TBG

TBG = the hidden state at the last PROMPT token, before any generation.
SLT  = the hidden state at the second-to-last token of the model's OWN
       generated response (i.e. the token right before the final generated
       token) -- see OATML's paper (arXiv:2406.15927), Section 4
       "Probing Locations": "the last token of the model response, i.e. the
       token before the end-of-sequence token."

Per the paper (Section 6): SLT consistently outperforms TBG by a modest
margin ("SEPs [TBG]... performance slightly below the SLT experiments").
The intuition: TBG reflects "how uncertain am I about this input in
general," while SLT reflects "how uncertain did I turn out to be, having
actually committed to an answer" -- a more direct signal, at the cost of
requiring an actual generation.

## Short-form generation, matching OATML's best-performing setting

Per Section 5 ("Tasks"): OATML's short-form setting -- few-shot prompting
the LLM to answer "as briefly as possible" -- gets meaningfully better SEP
results than their long-form setting (Figs. 2-3 vs Fig. 4: short-form AUROC
mostly 0.7-1.0, long-form more often 0.4-0.8 and noisier; their own
explanation is that late-layer hidden states in long generations are
"preoccupied with predicting the next token," which in long-form is "more
often unrelated to the semantic uncertainty of the overall answer"). This
project's code-generation analogue of their short-form instruction is a
system prompt directing the model to output only code, no explanation --
`SHORT_FORM_SYSTEM_PROMPT` below. (An earlier attempt with no such
constraint had Qwen3-8B write full explanation + code + worked examples,
hitting a 400-token budget without producing EOS on every one of an 8-prompt
smoke test -- confirmed empirically, not assumed.)

## Two extractions from one generation

For each prompt (same underlying ClarifyCodeBench text as the existing bare
TBG dataset, now templated with the short-form system prompt):
  1. Greedily generate a response (do_sample=False, matching OATML's own
     "we first generate a high-likelihood model response via greedy
     sampling" -- Section 4). This is a single deterministic response, NOT
     the N=10-sample generation OATML uses elsewhere to compute semantic
     entropy -- this project never computes semantic entropy (see
     FINDINGS.md); it's the same greedy generation OATML uses to *locate*
     the TBG/SLT hidden states themselves.
  2. Run ONE forward pass over (prompt_tokens + generated_tokens) with
     output_hidden_states=True, and read off TWO positions from it:
       - TBG: last token of the prompt (position `prompt_len - 1`) --
         extracted from the SAME templated (short-form) prompt used to
         generate, so this is directly comparable to SLT from the same
         input/generation, not to the bare-prompt TBG in activations/.
       - SLT: second-to-last token of the full sequence (position -2) --
         the token right before the last generated token.
     Causal attention means a single full-sequence forward pass gives
     identical hidden states to an incremental decode at each position, so
     this is simpler than threading extraction through HF's generate() loop.

Batched for throughput (unlike the original TBG extraction's batch_size=1 --
generation is far more expensive than a single forward pass, so batching
matters here; padding is handled via left-padding + explicit prompt-length
bookkeeping, not a fixed index).

Outputs two directories with identical metadata (row order, labels, groups)
so both plug into scripts/train_probes.py / train_oatml_style_probe.py
unmodified via --activations-dir:
  <out-dir>-tbg/activations.npy   -- TBG, short-form-prompt
  <out-dir>-slt/activations.npy   -- SLT, short-form-prompt
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
DEFAULT_OUT_STEM = ROOT / "activations_shortform"

# Code-generation analogue of OATML's "answer as briefly as possible" --
# see module docstring for why this matters (both for token-budget
# practicality and because it's the setting that got OATML's best results).
SHORT_FORM_SYSTEM_PROMPT = (
    "You are a professional software developer. Provide only the Python "
    "code solution to the given problem. Do not include any explanation, "
    "reasoning, comments about your approach, or worked examples -- output "
    "code only, in a single code block."
)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_prompt_text(tokenizer, problem_statement: str) -> str:
    messages = [
        {"role": "system", "content": SHORT_FORM_SYSTEM_PROMPT},
        {"role": "user", "content": problem_statement},
    ]
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
    out_stem: Path = DEFAULT_OUT_STEM,
    max_prompt_length: int = 4096,
    max_new_tokens: int = 400,
    batch_size: int = 8,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    tbg_dir = Path(f"{out_stem}-tbg")
    slt_dir = Path(f"{out_stem}-slt")
    tbg_dir.mkdir(parents=True, exist_ok=True)
    slt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract_slt] loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # last prompt token always at a known offset from the end
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="cuda")
    model.eval()

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    n_states = n_layers + 1
    N = len(rows)
    print(f"[extract_slt] {n_layers} layers, hidden_size={hidden_size}, {N} prompts, "
          f"batch_size={batch_size}, max_new_tokens={max_new_tokens}")

    tbg_acts = np.zeros((N, n_states, hidden_size), dtype=np.float16)
    slt_acts = np.zeros((N, n_states, hidden_size), dtype=np.float16)
    meta_out: list[dict] = []
    n_hit_max_new_tokens = 0

    t0 = time.time()
    for start in range(0, N, batch_size):
        batch_rows = rows[start:start + batch_size]
        texts = [build_prompt_text(tokenizer, r["prompt"]) for r in batch_rows]
        enc = tokenizer(texts, return_tensors="pt", truncation=True,
                         max_length=max_prompt_length, padding=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]  # fixed across the (left-padded) batch

        gen_out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = gen_out[:, prompt_len:]
        eos_id = tokenizer.eos_token_id

        for i, row in enumerate(batch_rows):
            gen_ids = new_tokens[i]
            eos_positions = (gen_ids == eos_id).nonzero(as_tuple=True)[0]
            hit_max = len(eos_positions) == 0
            if not hit_max:
                gen_len = int(eos_positions[0].item()) + 1  # include EOS itself
            else:
                gen_len = gen_ids.shape[0]
                n_hit_max_new_tokens += 1

            full_ids = gen_out[i, :prompt_len + gen_len].unsqueeze(0)
            out = model(input_ids=full_ids, output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # tuple(len=n_states) of (1, seq_len, hidden)

            tbg_pos = prompt_len - 1
            # If generation produced nothing before EOS (gen_len==1, immediate
            # stop), there's no "generated token before the last one" to take
            # SLT from -- fall back to TBG itself for that row.
            slt_pos = -2 if full_ids.shape[1] >= prompt_len + 2 else tbg_pos

            for layer_idx, layer_hs in enumerate(hs):
                tbg_acts[start + i, layer_idx] = layer_hs[0, tbg_pos, :].float().cpu().numpy().astype(np.float16)
                slt_acts[start + i, layer_idx] = layer_hs[0, slt_pos, :].float().cpu().numpy().astype(np.float16)

            meta_out.append({
                "row_index": start + i,
                "id": row["id"],
                "source_problem_id": row["source_problem_id"],
                "label": row["label"],
                "ambiguity_types": row["ambiguity_types"],
                "prompt_token_len": prompt_len,
                "generated_token_len": gen_len,
                "hit_max_new_tokens": hit_max,
            })

        elapsed = time.time() - t0
        done = min(start + batch_size, N)
        print(f"[extract_slt] {done}/{N} ({done/elapsed:.2f} prompts/s, elapsed {elapsed:.0f}s)")

    if n_hit_max_new_tokens:
        print(f"[extract_slt] WARNING: {n_hit_max_new_tokens}/{N} generations hit max_new_tokens={max_new_tokens} "
              f"without producing EOS -- SLT position for these falls back to the last generated token, "
              f"not necessarily 'before EOS' as OATML defines it")

    def _write(out_dir: Path, acts: np.ndarray, position_name: str) -> None:
        np.save(out_dir / "activations.npy", acts)
        with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
            for m in meta_out:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        with (out_dir / "extract_config.json").open("w", encoding="utf-8") as f:
            json.dump({
                "model_name": model_name,
                "n_layers": n_layers,
                "hidden_size": hidden_size,
                "n_hidden_states": n_states,
                "n_prompts": N,
                "max_prompt_length": max_prompt_length,
                "max_new_tokens": max_new_tokens,
                "dtype": str(dtype),
                "enable_thinking": False,
                "extraction_point": position_name,
                "generation": "greedy (do_sample=False)",
                "system_prompt": SHORT_FORM_SYSTEM_PROMPT,
                "n_hit_max_new_tokens": n_hit_max_new_tokens,
            }, f, indent=2)
        print(f"[extract_slt] saved {position_name} activations {acts.shape} -> {out_dir}/activations.npy")

    _write(tbg_dir, tbg_acts, "last_token_of_prompt (TBG, short-form-prompt)")
    _write(slt_dir, slt_acts, "second_to_last_token_of_generated_response (SLT)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-stem", type=Path, default=DEFAULT_OUT_STEM)
    ap.add_argument("--model-name", type=str, default=MODEL_NAME)
    ap.add_argument("--max-new-tokens", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-k", type=int, default=1,
                     help="Restrict to label >= min_k. Default 1 excludes K=0 (see FINDINGS.md confound).")
    args = ap.parse_args()

    rows = _read_jsonl(args.data)
    if args.min_k > 0:
        rows = [r for r in rows if r["label"] >= args.min_k]
    print(f"[extract_slt] loaded {len(rows)} prompts (min_k={args.min_k}) from {args.data}")
    extract(rows, model_name=args.model_name, out_stem=args.out_stem,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
