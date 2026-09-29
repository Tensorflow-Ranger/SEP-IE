"""v12 stage C-answer: for every elicited candidate question
(data/v12_ranking/candidates.jsonl), get an honest answer from local
Qwen3-8B, GROUNDED IN THE TRUE REFERENCE (data/v12_ranking/ground_truth.jsonl's
`reference` field -- ccb's real qa_pairs, or orchid_heval's clean_prompt +
ambiguity_explanation). This is deliberately an extraction/paraphrase task,
not open-ended guessing: the model is handed the true spec directly and
asked to answer using only it, so its reliability doesn't depend on Qwen
"knowing" the answer, just on reading comprehension over the given text.

This is independent of and runs in parallel with stage C-judge (Claude
personally assigning correct/similar/irrelevant tiers to each candidate
QUESTION against the same reference) -- judging compares the question's
topic to the reference and needs no answer text, so the two stages don't
depend on each other and are merged afterward into judged.jsonl.

Resumable: writes to data/v12_ranking/oracle_answers.jsonl, skips ids
already present.

Usage: python v12_oracle_answer.py [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen3-8B"
GROUND_TRUTH_PATH = ROOT / "data" / "v12_ranking" / "ground_truth.jsonl"
CANDIDATES_PATH = ROOT / "data" / "v12_ranking" / "candidates.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "oracle_answers.jsonl"

ORACLE_INSTRUCTION = (
    "You are answering a question about a coding problem's specification. You have privileged "
    "knowledge of the TRUE intended specification, given below, which the person asking the question "
    "cannot see. Answer using ONLY the information in the true specification. If the question is not "
    "addressed by it, answer with the most standard/reasonable convention for a competitive-programming "
    "problem (e.g. 'not specified beyond what's given; assume standard behavior') rather than inventing "
    "a new constraint.\n\n"
    "<true_specification>\n{reference_text}\n</true_specification>\n\n"
    "<question>\n{question}\n</question>\n\n"
    "Answer in ONE concise sentence. Output ONLY the answer sentence, nothing else."
)


def build_reference_text(row: dict) -> str:
    ref = row["reference"]
    if row["dataset_source"] == "clarifycodebench":
        lines = []
        for i, qa in enumerate(ref["qa_pairs"], 1):
            lines.append(f"Clarification {i} -- Q: {qa['question']}\nClarification {i} -- A: {qa['answer']}")
        return "\n".join(lines)
    else:  # orchid_heval
        return (
            f"Original UNAMBIGUOUS version of this problem:\n{ref['clean_prompt']}\n\n"
            f"What was made ambiguous relative to that original, and why:\n{ref['explanation']}"
        )


def build_prompt_text(tokenizer, user_content):
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def generate(tokenizer, model, user_content, max_new_tokens, temperature=0.3, top_p=0.9):
    text = build_prompt_text(tokenizer, user_content)
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=top_p, pad_token_id=tokenizer.eos_token_id,
        )
    gen_tokens = out[0][enc["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def generate_sane(tokenizer, model, user_content, max_new_tokens, max_retries=3, log_prefix=""):
    last = ""
    for attempt in range(max_retries):
        raw = generate(tokenizer, model, user_content, max_new_tokens)
        last = raw
        if raw and 2 <= len(raw) <= 600:
            return raw
        print(f"{log_prefix}attempt {attempt}: insane answer ({len(raw)} chars), retrying", flush=True)
    print(f"{log_prefix}WARNING: no sane answer after {max_retries} attempts, using last attempt", flush=True)
    return last


def load_done_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


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

    candidates = []
    with CANDIDATES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    if args.limit:
        candidates = candidates[: args.limit]

    done_ids = load_done_ids()
    todo = [c for c in candidates if c["id"] not in done_ids]
    print(f"[v12-C-answer] {len(done_ids)} already done, {len(todo)}/{len(candidates)} remaining", flush=True)
    if not todo:
        return

    print(f"[v12-C-answer] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    print("[v12-C-answer] model loaded", flush=True)

    with OUT_PATH.open("a", encoding="utf-8") as out_f:
        for n, c in enumerate(todo, 1):
            t0 = time.time()
            row = ground_truth[c["id"]]
            reference_text = build_reference_text(row)
            log_prefix = f"[v12-C-answer][{c['id']}] "

            answers = []
            for q in c["questions"]:
                user_content = ORACLE_INSTRUCTION.format(reference_text=reference_text, question=q)
                ans = generate_sane(tokenizer, model, user_content, max_new_tokens=120, log_prefix=log_prefix)
                answers.append(ans)

            out_f.write(json.dumps({"id": c["id"], "questions": c["questions"], "answers": answers},
                                    ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[v12-C-answer] ({n}/{len(todo)}) {c['id']}: {len(answers)} answers "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("[v12-C-answer] done", flush=True)


if __name__ == "__main__":
    main()
