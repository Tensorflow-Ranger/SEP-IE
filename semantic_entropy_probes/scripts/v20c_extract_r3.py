"""v20b stage 4: extract Qwen3-8B hidden states for every freshly-elicited
round-2 candidate, using v19's plain CONCAT_TEMPLATE chained twice (original
prompt + wrong-candidate's answer + new-candidate's answer) -- no generation
step needed, just a forward pass, matching v19_extract_activations.py exactly.

Output: data/v20_branches/round3_activations/{branch_idx}.npz
  candidates: (n_candidates, n_layers, dim) float32
  tiers: (n_candidates,) int32
  layers: (n_layers,) int32
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from v19_extract_activations import MODEL_NAME, LAYERS, CONCAT_TEMPLATE, hidden_states_at_layers_batch  # noqa: E402

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round3.json"
ELICITED_PATH = ROOT / "data" / "v20_branches_full" / "elicited_round3.jsonl"
ANSWERS_PATH = ROOT / "data" / "v20_branches_full" / "answers_round3.jsonl"
OUT_DIR = ROOT / "data" / "v20_branches_full" / "round3_activations"


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    branches = json.loads(BRANCHES_PATH.read_text())
    elicited = {d["branch_idx"]: d for d in read_jsonl(ELICITED_PATH) if "error" not in d}
    answers = {d["branch_idx"]: d for d in read_jsonl(ANSWERS_PATH) if "error" not in d}

    idxs = sorted(set(elicited) & set(answers))
    print(f"[v20b-extract] branches ready: {len(idxs)}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [i for i in idxs if not (OUT_DIR / f"{i}.npz").exists()]
    print(f"[v20b-extract] {len(idxs)-len(todo)} already done, {len(todo)} todo", flush=True)
    if not todo:
        return

    print(f"[v20b-extract] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v20b-extract] model loaded", flush=True)

    for n, idx in enumerate(todo, 1):
        branch = branches[idx]
        branch_prompt = branch["branch_prompt"]  # already = original + Clarification: wrong_answer
        ans_by_q = {a["question"].strip(): a["answer"] for a in answers[idx]["answers"]
                    if "question" in a and "answer" in a}

        tiers, texts, questions = [], [], []
        for c in elicited[idx]["candidates"]:
            q = c["question"].strip()
            ans = ans_by_q.get(q)
            if ans is None:
                continue
            tiers.append(c["tier"])
            questions.append(q)
            texts.append(CONCAT_TEMPLATE.format(prompt=branch_prompt, answer=ans))

        if len(texts) < 2 or len(set(tiers)) < 2:
            print(f"[v20b-extract] ({n}/{len(todo)}) branch {idx}: skipped, <2 usable/tiers", flush=True)
            continue

        cand_vecs = hidden_states_at_layers_batch(tokenizer, model, texts, LAYERS)
        np.savez(
            OUT_DIR / f"{idx}.npz",
            candidates=cand_vecs.astype(np.float32),
            tiers=np.array(tiers, dtype=np.int32),
            layers=np.array(LAYERS, dtype=np.int32),
        )
        (OUT_DIR / f"{idx}.json").write_text(json.dumps({
            "branch_idx": idx, "id": branch["id"], "questions": questions,
        }, indent=2))
        print(f"[v20b-extract] ({n}/{len(todo)}) branch {idx} ({branch['id']}): {len(texts)} candidates", flush=True)

    print("[v20b-extract] DONE", flush=True)


if __name__ == "__main__":
    main()
