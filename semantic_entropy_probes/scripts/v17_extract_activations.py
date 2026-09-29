"""v17 stage D: re-extract candidate hidden states using PLAIN CONCATENATION
(`prompt + "\\n\\nClarification: {answer}"`) instead of v12/v15's LLM
inline-rewrite step, to test whether the rewrite's compute cost is actually
buying ranking accuracy or just matching codegen prompt style.

Motivation (see FINDINGS.md's v12 section + [[heval32-probe-ranking-investigation]]
Finding 3): the inline-rewrite exists to match the CODEGEN prompt format, which
only matters if the probe's readout must resemble what the model sees during
generation. For pure ranking of candidates against each other, only train/score
FORMAT CONSISTENCY matters, not naturalness -- and Stage E already trains on
WITHIN-GROUP hidden-state differences, so a constant label-token artifact from
"Clarification: ..." should mostly cancel out across candidates in the same
group. This also removes an entire per-candidate generation call (v12: local
Qwen3-8B; v15: Bedrock DeepSeek), leaving just one forward pass -- unlike the
"forward-pass-only" cost check in FINDINGS.md (which only skipped decoding but
still relied on already-generated rewrite text), this removes the generation
call altogether.

Uses the SAME 962-usable-group universe as v15 (v12's 635 clarifycodebench +
orchid_heval rows, plus v15's 416 new orchid_heval-semantic + orchid_bcb rows),
sourced uniformly from data/v13_rejudge/ (chunks for prompt text, deepseek_judged
.jsonl for tiers + clean-label filtering) so both group sets go through
identical logic here -- no v12-vs-v15-shaped special casing needed for the
concatenation step itself.

Answer text (needed to build the concatenation) is NOT re-elicited -- reused
verbatim from whichever prior stage already has it:
  - v12 groups (635 ids present in data/v12_ranking/judged.jsonl): answers
    from there (matched to deepseek_judged.jsonl's tiers by question text,
    since v12's own tiers may differ slightly from the v13 rejudge -- the
    rejudge's tiers are what v15's training actually uses, so match to those
    for an apples-to-apples comparison against v15's numbers).
  - v15-new groups (present in data/v15_new_activations/rewrites/*.json):
    answers from there (same question-text matching).

Baseline vectors are NOT recomputed -- the baseline (unclarified original
prompt) is identical text in both the old and new pipelines, so its hidden
state is reused directly from whichever of v12_ranking/activations or
v15_new_activations/activations already has it. Only the candidate-side
forward pass changes.

Output: data/v17_concat_activations/activations/{safe_id}.npz
  (same layout as v12/v15: baseline, candidates, tiers, layers)
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
V12_ACT_DIR = ROOT / "data" / "v12_ranking" / "activations"
V15_ACT_DIR = ROOT / "data" / "v15_new_activations" / "activations"
OUT_DIR = ROOT / "data" / "v17_concat_activations" / "activations"

LAYERS = [16, 18, 20, 22, 24, 26]

CONCAT_TEMPLATE = "{prompt}\n\nClarification: {answer}"


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


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


def load_answers_by_question(rid: str, v12_judged: dict, v15_rewrites_dir: Path) -> dict[str, str]:
    """Returns {question_text.strip(): answer_text} from whichever prior stage
    already elicited answers for this id."""
    if rid in v12_judged:
        return {c["question"].strip(): c["answer"] for c in v12_judged[rid]["candidates"]}
    rw_path = v15_rewrites_dir / f"{safe_name(rid)}.json"
    if rw_path.exists():
        rw = json.loads(rw_path.read_text())
        return {c["question"].strip(): c["answer"] for c in rw["candidates"]}
    return {}


def load_baseline_vec(rid: str) -> np.ndarray | None:
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

    def clean(rid):
        j = judged_rows[rid]["judged"]
        return (not j.get("label_leaked")) and (j.get("label_functionally_relevant") is True)

    all_ids = [rid for rid in judged_rows if clean(rid)]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [rid for rid in all_ids if not (OUT_DIR / f"{safe_name(rid)}.npz").exists()]
    print(f"[v17-D] {len(all_ids)} clean total, {len(all_ids)-len(todo)} already done, {len(todo)} todo", flush=True)
    if not todo:
        return

    print(f"[v17-D] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v17-D] model loaded", flush=True)

    n_skipped = 0
    for n, rid in enumerate(todo, 1):
        t0 = time.time()
        g = groups.get(rid)
        baseline_vec = load_baseline_vec(rid)
        if g is None or baseline_vec is None:
            print(f"[v17-D] {rid}: missing prompt or baseline vec, skipping", flush=True)
            n_skipped += 1
            continue
        prompt = g["prompt"]
        answers_by_q = load_answers_by_question(rid, v12_judged, V15_REWRITES_DIR)

        tiers, concat_prompts = [], []
        for c in judged_rows[rid]["judged"]["candidates"]:
            q = c["question"].strip()
            ans = answers_by_q.get(q)
            if ans is None:
                continue
            tiers.append(c["tier"])
            concat_prompts.append(CONCAT_TEMPLATE.format(prompt=prompt, answer=ans))
        if len(concat_prompts) < 2 or len(set(tiers)) < 2:
            print(f"[v17-D] ({n}/{len(todo)}) {rid}: skipped, <2 usable candidates/tiers after alignment", flush=True)
            n_skipped += 1
            continue

        cand_vecs = hidden_states_at_layers_batch(tokenizer, model, concat_prompts, LAYERS)

        np.savez(
            OUT_DIR / f"{safe_name(rid)}.npz",
            baseline=baseline_vec,
            candidates=cand_vecs.astype(np.float32),
            tiers=np.array(tiers, dtype=np.int32),
            layers=np.array(LAYERS, dtype=np.int32),
        )
        print(f"[v17-D] ({n}/{len(todo)}) {rid}: {len(concat_prompts)} candidates ({time.time()-t0:.1f}s)", flush=True)

    print(f"[v17-D] done. skipped={n_skipped}", flush=True)


if __name__ == "__main__":
    main()
