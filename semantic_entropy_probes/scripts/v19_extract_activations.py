"""v19 stage 3: extract concatenation-format hidden states (v17's method:
`prompt + "\\n\\nClarification: {answer}"`, one forward pass per group) for
the union of all 4 ranking-pipeline sources -- clarifycodebench, orchid_heval,
orchid_bcb (v17's existing 3), plus orchid_bcb_expand ("bcb-extend", NEW) --
under a STRICTER 3-way clean-label filter than v17 used:

  1. not label_leaked
  2. label_functionally_relevant is True
  3. NEW: more_important_than_label is None (no candidate was judged a bigger
     functional lever than the row's own label -- v14's judging already
     computed this field for every row, but no prior script filtered on it;
     v17 only applied checks 1+2)

Applying (3) is a real change, not just an addition for the new source: it
drops ~53% of the pre-existing clarifycodebench/orchid_heval/orchid_bcb rows
too (see FINDINGS.md's v14 section, 581/1095 rows flagged), so this script
re-filters the OLD sources as well rather than reusing v17's activation
files wholesale (those were built under a looser filter and cannot be
trusted to already satisfy criterion 3).

For old-source ids (already extracted by v12/v15 for baseline, v17 for
candidates): baseline vector is reused as-is (unclarified prompt text is
unchanged by this filter), candidates are RE-extracted here (cheap, same
mechanism) rather than trying to cherry-pick from v17_concat_activations,
since re-extraction from already-elicited answer text costs one forward
pass per group regardless and keeping one code path avoids subtle mismatches.

For orchid_bcb_expand ids: nothing has ever been extracted -- both baseline
(forward pass on the ambiguous prompt itself, same convention v12/v15 used:
baseline = g["prompt"], not the unambiguous complete_prompt) and candidates
are computed fresh, using answers from data/v19_bcbexpand/answers/.

Output: data/v19_concat_activations/activations/{safe_id}.npz
  (baseline, candidates, tiers, layers -- same layout as v12/v15/v17)
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
V12_JUDGED_PATH = ROOT / "data" / "v12_ranking" / "judged.jsonl"
V15_REWRITES_DIR = ROOT / "data" / "v15_new_activations" / "rewrites"
V19_ANSWERS_DIR = ROOT / "data" / "v19_bcbexpand" / "answers"
V12_ACT_DIR = ROOT / "data" / "v12_ranking" / "activations"
V15_ACT_DIR = ROOT / "data" / "v15_new_activations" / "activations"
OUT_DIR = ROOT / "data" / "v19_concat_activations" / "activations"

LAYERS = [16, 18, 20, 22, 24, 26]
CONCAT_TEMPLATE = "{prompt}\n\nClarification: {answer}"


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def is_clean(j: dict) -> bool:
    return (
        not j.get("label_leaked")
        and j.get("label_functionally_relevant") is True
        and j.get("more_important_than_label") is None
    )


def build_prompt_text(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


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


@torch.no_grad()
def hidden_states_at_layers_single(tokenizer, model, prompt_text, layers):
    return hidden_states_at_layers_batch(tokenizer, model, [prompt_text], layers)[0]


def load_answers_by_question(rid: str, v12_judged: dict) -> dict[str, str]:
    if rid in v12_judged:
        return {c["question"].strip(): c["answer"] for c in v12_judged[rid]["candidates"]}
    rw_path = V15_REWRITES_DIR / f"{safe_name(rid)}.json"
    if rw_path.exists():
        rw = json.loads(rw_path.read_text())
        return {c["question"].strip(): c["answer"] for c in rw["candidates"]}
    ans_path = V19_ANSWERS_DIR / f"{safe_name(rid)}.json"
    if ans_path.exists():
        rw = json.loads(ans_path.read_text())
        return {c["question"].strip(): c["answer"] for c in rw["candidates"]}
    return {}


def load_old_baseline_vec(rid: str) -> np.ndarray | None:
    for act_dir in (V12_ACT_DIR, V15_ACT_DIR):
        p = act_dir / f"{safe_name(rid)}.npz"
        if p.exists():
            return np.load(p)["baseline"].astype(np.float32)
    return None


def main() -> None:
    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}

    all_ids = [rid for rid in judged_rows if rid in groups and is_clean(judged_rows[rid]["judged"])]
    import collections
    print(f"[v19-extract] {len(all_ids)} clean groups total across sources:", flush=True)
    print(collections.Counter(judged_rows[rid]["dataset_source"] for rid in all_ids), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [rid for rid in all_ids if not (OUT_DIR / f"{safe_name(rid)}.npz").exists()]
    print(f"[v19-extract] {len(all_ids)-len(todo)} already done, {len(todo)} todo", flush=True)
    if not todo:
        return

    print(f"[v19-extract] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v19-extract] model loaded", flush=True)

    n_skipped = 0
    for n, rid in enumerate(todo, 1):
        t0 = time.time()
        g = groups.get(rid)
        if g is None:
            n_skipped += 1
            continue
        prompt = g["prompt"]
        answers_by_q = load_answers_by_question(rid, v12_judged)

        tiers, concat_prompts = [], []
        for c in judged_rows[rid]["judged"]["candidates"]:
            q = c["question"].strip()
            ans = answers_by_q.get(q)
            if ans is None:
                continue
            tiers.append(c["tier"])
            concat_prompts.append(CONCAT_TEMPLATE.format(prompt=prompt, answer=ans))
        if len(concat_prompts) < 2 or len(set(tiers)) < 2:
            print(f"[v19-extract] ({n}/{len(todo)}) {rid}: skipped, <2 usable candidates/tiers", flush=True)
            n_skipped += 1
            continue

        baseline_vec = load_old_baseline_vec(rid)
        if baseline_vec is None:
            baseline_vec = hidden_states_at_layers_single(tokenizer, model, prompt, LAYERS).astype(np.float32)

        cand_vecs = hidden_states_at_layers_batch(tokenizer, model, concat_prompts, LAYERS)

        np.savez(
            OUT_DIR / f"{safe_name(rid)}.npz",
            baseline=baseline_vec,
            candidates=cand_vecs.astype(np.float32),
            tiers=np.array(tiers, dtype=np.int32),
            layers=np.array(LAYERS, dtype=np.int32),
        )
        print(f"[v19-extract] ({n}/{len(todo)}) {rid}: {len(concat_prompts)} candidates ({time.time()-t0:.1f}s)", flush=True)

    print(f"[v19-extract] done. skipped={n_skipped}", flush=True)


if __name__ == "__main__":
    main()
