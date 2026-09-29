"""v12 stage D: for every judged row (data/v12_ranking/judged.jsonl, produced by
stage C's oracle-answer + tier-judging pass), build each candidate's actual
CLARIFIED prompt and extract Qwen3-8B hidden states from it, plus the
unclarified baseline.

Two things this deliberately does differently from the CURRENTLY DEPLOYED
v10/v11 scoring path, both to fix [[heval32-probe-ranking-investigation]]'s
Finding 3 (scoring/codegen prompt-format mismatch) at the source instead of
inheriting it into the new ranking probe:

  1. Candidate prompts are built via the SAME STYLE_INSTRUCTION inline-rewrite
     Qwen3-8B already uses for the CODEGEN input (verbatim from
     v10_round_worker.py), not the raw `current_prompt + "\\nClarification: {a}"`
     concatenation the deployed probes were scored (and trained) on. The
     live rescoring experiment in FINDINGS.md already showed this closes
     part of the train/score format gap (rank-1 hit rate 3/16 -> 5/16); this
     pipeline bakes that fix into training itself rather than treating it as
     a scoring-time patch.
  2. Hidden states are pulled for a small LAYER sweep (not hardcoded to the
     deployed rank1/rank2 layers 20/22), since the new grouped/ranking
     training objective is different enough from the old i.i.d. binary
     objective that the best layer isn't guaranteed to carry over -- stage E
     picks the best layer(s) empirically on held-out groups.

Output layout (resumable -- one file per row, skip if it already exists):
  data/v12_ranking/activations/{id}.npz
    baseline: (n_layers, hidden_dim) float32
    candidates: (n_candidates, n_layers, hidden_dim) float32
    tiers: (n_candidates,) int32
    layers: (n_layers,) int32
  data/v12_ranking/activations/{id}.json
    {"id":..., "questions": [...], "answers": [...], "inline_prompts": [...]}
    (text alongside the vectors, for spot-checking / debugging)

Usage: python v12_extract_activations.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen3-8B"
GROUND_TRUTH_PATH = ROOT / "data" / "v12_ranking" / "ground_truth.jsonl"
JUDGED_PATH = ROOT / "data" / "v12_ranking" / "judged.jsonl"
OUT_DIR = ROOT / "data" / "v12_ranking" / "activations"

LAYERS = [16, 18, 20, 22, 24, 26]

STYLE_INSTRUCTION = (
    "You are editing a competitive-programming problem statement (AtCoder/LiveCodeBench style: "
    "terse, formal, precise, no filler commentary, uses the same mathematical notation style as the original, "
    "and preserves the original's line-break structure -- one clause/sentence per line where the original does that).\n\n"
    "Here is the ORIGINAL problem statement:\n<original>\n{original}\n</original>\n\n"
    "Rewrite this problem statement to incorporate the following change, INLINE into the existing prose "
    "(edit an existing sentence, or insert one short sentence in the same place and style where a real "
    "problem-setter would have put it originally). Do NOT append a new bullet point or trailing note at the "
    "end. Do NOT change anything else about the text -- keep the rest word-for-word identical, including the "
    "Input/Output/Constraints sections and all line breaks. Output ONLY the full rewritten problem statement, "
    "with no preamble, no explanation, and no markdown formatting around it.\n\n"
    "Change to make: {detail}"
)


def build_prompt_text(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate(tokenizer, model, user_content, max_new_tokens, temperature=0.7, top_p=0.8):
    text = build_prompt_text(tokenizer, user_content)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def generate_sane(tokenizer, model, user_content, max_new_tokens, is_sane, max_retries=3, log_prefix=""):
    last = ""
    for attempt in range(max_retries):
        raw = strip_leaked_template_tags(generate(tokenizer, model, user_content, max_new_tokens))
        last = raw
        if is_sane(raw):
            return raw
        print(f"{log_prefix}attempt {attempt}: insane rewrite ({len(raw)} chars), retrying", flush=True)
    print(f"{log_prefix}WARNING: no sane rewrite after {max_retries} attempts, using last attempt", flush=True)
    return last


@torch.no_grad()
def generate_batch(tokenizer, model, user_contents, max_new_tokens, temperature=0.7, top_p=0.8):
    """Batched version of generate() -- requires left-padding so every sequence's
    newly-generated tokens start at the same index (enc['input_ids'].shape[1])."""
    texts = [build_prompt_text(tokenizer, uc) for uc in user_contents]
    enc = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(
        **enc, max_new_tokens=max_new_tokens, do_sample=True,
        temperature=temperature, top_p=top_p, pad_token_id=tokenizer.pad_token_id,
    )
    gen_tokens = out[:, enc["input_ids"].shape[1]:]
    return [tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_tokens]


def generate_sane_batch(tokenizer, model, user_contents, max_new_tokens, is_sane_list, max_retries=3, log_prefix=""):
    """Batched first attempt across all of a row's candidates in one GPU call, then
    falls back to sequential per-item retries (generate_sane) only for items that
    failed their sanity check -- keeps the same quality guarantee as the fully
    sequential path while batching the common (already-sane) case."""
    n = len(user_contents)
    results = [None] * n
    try:
        batch_out = generate_batch(tokenizer, model, user_contents, max_new_tokens, temperature=0.7, top_p=0.8)
    except torch.cuda.OutOfMemoryError:
        print(f"{log_prefix}batched generation OOM, falling back to fully sequential", flush=True)
        torch.cuda.empty_cache()
        batch_out = [None] * n

    retry_idxs = []
    for i in range(n):
        raw = strip_leaked_template_tags(batch_out[i]) if batch_out[i] is not None else ""
        if batch_out[i] is not None and is_sane_list[i](raw):
            results[i] = raw
        else:
            retry_idxs.append(i)

    for i in retry_idxs:
        results[i] = generate_sane(
            tokenizer, model, user_contents[i], max_new_tokens, is_sane_list[i],
            max_retries=max_retries, log_prefix=f"{log_prefix}[cand {i}] ",
        )
    return results


def strip_leaked_template_tags(text: str) -> str:
    """STYLE_INSTRUCTION wraps the original in <original>...</original> -- the model
    occasionally echoes one of these tags back into its output (observed: a trailing
    "</original>" appended after the rewritten text). Since hidden states are pulled
    from the LAST token, a leaked tag directly corrupts the exact vector the probe
    reads, not just cosmetic noise -- strip it defensively."""
    text = text.strip()
    for tag in ("<original>", "</original>"):
        if text.startswith(tag):
            text = text[len(tag):].strip()
        if text.endswith(tag):
            text = text[: -len(tag)].strip()
    return text


def is_sane_rewrite(raw_text: str, original: str) -> bool:
    text = strip_leaked_template_tags(raw_text)
    return bool(text) and len(text) >= 0.5 * len(original)


@torch.no_grad()
def hidden_states_at_layers(tokenizer, model, prompt_text, layers):
    text = build_prompt_text(tokenizer, prompt_text)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return np.stack([hs[l][0, -1, :].float().cpu().numpy() for l in layers], axis=0)  # (n_layers, dim)


@torch.no_grad()
def hidden_states_at_layers_batch(tokenizer, model, prompt_texts, layers):
    """Batched version -- with left-padding, index -1 is always the true last
    real token for every sequence regardless of how much padding it got.

    IMPORTANT: a plain forward call does NOT auto-adjust position_ids for left
    padding the way model.generate() does internally -- without this, RoPE
    positions for the real tokens are offset by the padding length, desyncing
    them from what the same text would get as an unpadded single sequence
    (verified: max abs hidden-state diff of ~1.0 vs the unbatched path before
    this fix, ~0 after)."""
    texts = [build_prompt_text(tokenizer, p) for p in prompt_texts]
    enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=4096, padding=True).to(model.device)
    attention_mask = enc["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    out = model(**enc, position_ids=position_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    # (n_items, n_layers, dim)
    return np.stack(
        [np.stack([hs[l][i, -1, :].float().cpu().numpy() for l in layers], axis=0) for i in range(len(texts))],
        axis=0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    ground_truth = {}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                ground_truth[row["id"]] = row

    judged_rows = []
    with JUDGED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                judged_rows.append(json.loads(line))
    if args.limit:
        judged_rows = judged_rows[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [r for r in judged_rows if not (OUT_DIR / f"{r['id']}.npz").exists()]
    print(f"[v12-D] {len(judged_rows) - len(todo)} already done, {len(todo)}/{len(judged_rows)} remaining", flush=True)
    if not todo:
        return

    print(f"[v12-D] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # required for correct batched generate() + last-token hidden states
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v12-D] model loaded", flush=True)

    for n, jrow in enumerate(todo, 1):
        t0 = time.time()
        gt = ground_truth[jrow["id"]]
        prompt = gt["prompt"]
        log_prefix = f"[v12-D][{jrow['id']}] "

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

        print(f"[v12-D] ({n}/{len(todo)}) {jrow['id']}: {len(cand_vecs)} candidates "
              f"({time.time()-t0:.0f}s)", flush=True)

    print("[v12-D] done", flush=True)


if __name__ == "__main__":
    main()
