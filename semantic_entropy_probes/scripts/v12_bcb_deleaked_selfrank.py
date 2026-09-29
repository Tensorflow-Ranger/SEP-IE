"""Control: re-run the exact same structured zero-shot selection
(v12_bcb_judge_and_selfrank.py) on the 4 orchid_bcb groups previously found
to have a same-prompt leak (BigCodeBench_0_lexical, BigCodeBench_0_vagueness,
BigCodeBench_2_lexical, BigCodeBench_2_vagueness), but with that leak
manually patched out of the prompt text -- same candidates, same tiers,
only the prompt's leaking span is edited. If Qwen still misses after the
leak is removed, that's evidence of a genuine capability gap on this
ambiguity. If it now hits, that confirms the leak (not a real ambiguity in
context) was suppressing the "correct" answer from being needed at all.

Usage: python v12_bcb_deleaked_selfrank.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_NAME = "Qwen/Qwen3-8B"
JUDGED_PATH = ROOT / "data" / "v12_ranking" / "bcb_judged.jsonl"
CANDIDATES_PATH = ROOT / "data" / "v12_ranking" / "bcb_candidates.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "qwen_bcb_deleaked_selfrank_results.json"

# Original (leaked) -> de-leaked prompt text, hand-patched to remove exactly
# the leaking span while changing nothing else about the ambiguity itself.
DELEAKED_PROMPTS = {
    "BigCodeBench_0_lexical": (
        "Calculates the average of the sums of absolute differences between each pair of consecutive "
        "numbers for all permutations of a given list. Each permutation is processed before calculating "
        "the differences. Args: - numbers (list): A list of numbers. Default is numbers from 1 to 10.\n"
        "The function should output with:\n"
        "    float: The average of the sums of absolute differences for each processed permutation of the list.\n"
        "You should write self-contained code starting with:\n"
        "```\nimport itertools\ndef task_func(numbers=list(range(1, 3))):\n```"
    ),  # removed: `from random import shuffle`
    "BigCodeBench_0_vagueness": (
        "Calculates the average of the sums of absolute differences between each pair of consecutive "
        "numbers for all permutations of a given list. Each permutation is processed before calculating "
        "the differences. Args: - numbers (list): A list of numbers. Default is numbers from 1 to 10.\n"
        "The function should output with:\n"
        "    float: The average of the sums of absolute differences for each processed permutation of the list.\n"
        "You should write self-contained code starting with:\n"
        "```\nimport itertools\ndef task_func(numbers=list(range(1, 3))):\n```"
    ),  # removed: `from random import shuffle` import AND "shuffled" -> "processed" in output sentence
    "BigCodeBench_2_lexical": (
        "Create a dictionary in which keys are random letters and values are lists of random integers. "
        "The dictionary is then sorted by by the central tendency of the values in descending order, "
        "demonstrating the use of the statistics library.\n"
        "The function should output with:\n"
        "    dict: The sorted dictionary with letters as keys and lists of integers as values, sorted by their computed values.\n"
        "You should write self-contained code starting with:\n"
        "```\nimport random\nimport statistics\ndef task_func(LETTERS):\n```"
    ),  # changed: "average" -> "computed" in output sentence
    "BigCodeBench_2_vagueness": (
        "Create a dictionary in which keys are random letters and values are lists of random integers. "
        "The dictionary is then sorted by statistics of the values in descending order, demonstrating the "
        "use of the statistics library.\n"
        "The function should output with:\n"
        "    dict: The sorted dictionary with letters as keys and lists of integers as values, sorted by their computed values.\n"
        "You should write self-contained code starting with:\n"
        "```\nimport random\nimport statistics\ndef task_func(LETTERS):\n```"
    ),  # changed: "mean" -> "computed" in output sentence
}

SELECT_INSTRUCTION = """You are given a Python function specification and a numbered list of \
candidate clarifying questions about it. Exactly one of these questions targets \
the single most important ambiguity or underspecified detail in the specification \
-- the kind of thing that, if answered, would change how you'd implement the \
function. The others are either irrelevant, already answered by the statement, \
or address a much less important detail.

Function specification:
{prompt}

Candidate questions:
{candidates}

Which numbered question best targets the single most important ambiguity in \
this specification? Think briefly, then answer with a final line of the exact \
form:
ANSWER: <number>
"""


def build_candidates_block(candidates: list[dict]) -> str:
    return "\n".join(f"{c['idx']}. {c['question']}" for c in candidates)


def parse_answer(text: str, valid_idxs: set[int]) -> int | None:
    matches = re.findall(r"ANSWER:\s*(\d+)", text)
    if matches:
        val = int(matches[-1])
        if val in valid_idxs:
            return val
    for tok in reversed(re.findall(r"\b\d+\b", text)):
        val = int(tok)
        if val in valid_idxs:
            return val
    return None


def main() -> None:
    judged = {}
    with JUDGED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                judged[r["id"]] = r

    target_ids = list(DELEAKED_PROMPTS.keys())
    rows = [judged[i] for i in target_ids if i in judged]
    print(f"[bcb-deleak] {len(rows)}/{len(target_ids)} target groups found in bcb_judged.jsonl", flush=True)

    print(f"[bcb-deleak] loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
        max_memory={0: "13GiB", "cpu": "40GiB"},
    )
    model.eval()
    print("[bcb-deleak] model loaded", flush=True)

    per_problem = []
    for row in rows:
        gid = row["id"]
        candidates = row["candidates"]
        best_tier = max(c["tier"] for c in candidates)
        valid_idxs = {c["idx"] for c in candidates}
        prompt = DELEAKED_PROMPTS[gid]

        user_msg = SELECT_INSTRUCTION.format(prompt=prompt, candidates=build_candidates_block(candidates))
        messages = [{"role": "user", "content": user_msg}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=400, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        picked = parse_answer(gen_text, valid_idxs)
        if picked is None:
            hit, note, picked_tier = 0, "unparsed", None
        else:
            picked_tier = next(c["tier"] for c in candidates if c["idx"] == picked)
            hit, note = int(picked_tier == best_tier), "ok"

        per_problem.append({"id": gid, "hit": hit, "picked_idx": picked, "picked_tier": picked_tier,
                             "best_tier": best_tier, "note": note, "raw_response": gen_text.strip()})
        mark = "HIT " if hit else "MISS"
        print(f"[bcb-deleak] {mark} {gid}: picked={picked} note={note}", flush=True)

    hits = [r["hit"] for r in per_problem]
    print(f"[bcb-deleak] de-leaked rank-1 hit rate: {sum(hits)}/{len(hits)}", flush=True)

    OUT_PATH.write_text(json.dumps({"per_problem": per_problem, "hit_rate": sum(hits) / len(hits) if hits else None},
                                    indent=2))
    print(f"[bcb-deleak] saved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
