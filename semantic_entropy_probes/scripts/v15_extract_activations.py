"""v15 stage D: local Qwen3-8B hidden-state extraction for the 418 new
clean rows, using the text already produced by
scripts/v15_bedrock_oracle_rewrite.py (data/v15_new_activations/rewrites/).

Identical mechanism to v12_extract_activations.py (last-token hidden
states at layers [16,18,20,22,24,26], baseline = original prompt,
candidates = each candidate's inline-rewritten prompt) -- the ONLY
difference is the text is already generated (via Bedrock in the previous
stage) rather than generated here, so this script does no text generation
at all, only forward passes. Frees the GPU from competing with generation,
and skips re-implementing the sanity-check/retry logic since Bedrock
output was already validated by JSON-parse success in the previous stage.

Output: data/v15_new_activations/activations/{safe_id}.npz
  (same layout as v12: baseline, candidates, tiers, layers)
Resumable: skips ids whose .npz already exists.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen3-8B"
JUDGED_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl"
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
REWRITES_DIR = ROOT / "data" / "v15_new_activations" / "rewrites"
OUT_DIR = ROOT / "data" / "v15_new_activations" / "activations"
NEW_IDS_PATH = Path("/tmp/new_clean_ids.json")

LAYERS = [16, 18, 20, 22, 24, 26]


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def build_prompt_text(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


@torch.no_grad()
def hidden_states_at_layers(tokenizer, model, prompt_text, layers):
    text = build_prompt_text(tokenizer, prompt_text)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    out = model(**enc, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return np.stack([hs[l][0, -1, :].float().cpu().numpy() for l in layers], axis=0)


@torch.no_grad()
def hidden_states_at_layers_batch(tokenizer, model, prompt_texts, layers):
    texts = [build_prompt_text(tokenizer, p) for p in prompt_texts]
    enc = tokenizer(texts, return_tensors="pt", truncation=True, max_length=4096, padding=True).to(model.device)
    attention_mask = enc["attention_mask"]
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    out = model(**enc, position_ids=position_ids, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    return np.stack(
        [np.stack([hs[l][i, -1, :].float().cpu().numpy() for l in layers], axis=0) for i in range(len(texts))],
        axis=0,
    )


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g

    new_ids = json.loads(NEW_IDS_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [rid for rid in new_ids if not (OUT_DIR / f"{safe_name(rid)}.npz").exists()]
    print(f"[v15-D] {len(new_ids)} total, {len(new_ids)-len(todo)} already done, {len(todo)} todo", flush=True)
    if not todo:
        return

    print(f"[v15-D] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v15-D] model loaded", flush=True)

    n_skipped_missing_rewrite = 0
    for n, rid in enumerate(todo, 1):
        t0 = time.time()
        rw_path = REWRITES_DIR / f"{safe_name(rid)}.json"
        if not rw_path.exists():
            print(f"[v15-D] MISSING rewrite for {rid}, skipping", flush=True)
            n_skipped_missing_rewrite += 1
            continue
        rw = json.loads(rw_path.read_text())
        jrow = judged_rows[rid]
        g = groups[rid]
        prompt = g["prompt"]

        # align rewrite candidates to judged tiers by question text
        tier_by_q = {c["question"].strip(): c["tier"] for c in jrow["judged"]["candidates"]}
        cands = rw["candidates"]
        tiers, inline_prompts = [], []
        for c in cands:
            q = c["question"].strip()
            if q not in tier_by_q:
                continue
            tiers.append(tier_by_q[q])
            inline_prompts.append(c["inline_prompt"])
        if len(inline_prompts) < 2 or len(set(tiers)) < 2:
            print(f"[v15-D] ({n}/{len(todo)}) {rid}: skipped, <2 usable candidates/tiers after alignment", flush=True)
            continue

        baseline_vec = hidden_states_at_layers(tokenizer, model, prompt, LAYERS)
        cand_vecs = hidden_states_at_layers_batch(tokenizer, model, inline_prompts, LAYERS)

        np.savez(
            OUT_DIR / f"{safe_name(rid)}.npz",
            baseline=baseline_vec.astype(np.float32),
            candidates=cand_vecs.astype(np.float32),
            tiers=np.array(tiers, dtype=np.int32),
            layers=np.array(LAYERS, dtype=np.int32),
        )
        print(f"[v15-D] ({n}/{len(todo)}) {rid}: {len(inline_prompts)} candidates ({time.time()-t0:.1f}s)", flush=True)

    print(f"[v15-D] done. skipped_missing_rewrite={n_skipped_missing_rewrite}", flush=True)


if __name__ == "__main__":
    main()
