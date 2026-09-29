"""v15 stage C+D-text: for the 418 new clean rows (orchid_heval-semantic,
orchid_bcb semantic+vagueness -- never processed by v12), replicate v12's
two text-generation stages -- oracle-answer (v12_oracle_answer.py) and
inline style-preserving rewrite (v12_extract_activations.py's
STYLE_INSTRUCTION) -- using DeepSeek V3.2 on Bedrock instead of local
Qwen3-8B generation (faster, frees the GPU for the activation-extraction
forward passes that DO need to stay local).

Deliberately uses v12's EXACT instruction text/mechanism, unchanged: one
concise oracle-answer sentence grounded only in the true reference, then
inline-edit that answer into the ORIGINAL prompt while keeping everything
else word-for-word identical (no restructuring, no template -- this is
the mechanism verified NOT to hallucinate/drop signatures, unlike a
from-scratch "canonicalize into one style" rewrite).

One Bedrock call per row, covering all its candidates at once (answer +
rewrite for each), to keep call count at ~418 rather than ~3,344.

Output: data/v15_new_activations/rewrites/{id}.json
  {"id":..., "candidates": [{"question":..., "answer":..., "inline_prompt":...}, ...]}
Resumable: skips ids whose output file already exists.
"""
from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
JUDGED_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl"
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
NEW_IDS_PATH = Path("/tmp/new_clean_ids.json")
OUT_DIR = ROOT / "data" / "v15_new_activations" / "rewrites"

MODEL_ID = "deepseek.v3.2"
REGION = "ap-northeast-1"
N_WORKERS = 6
MAX_RETRIES = 8

# v12_oracle_answer.py's ORACLE_INSTRUCTION, unchanged in substance.
ORACLE_RULE = (
    "You are answering a question about a coding problem's specification. You have privileged "
    "knowledge of the TRUE intended specification, given below, which the person asking the question "
    "cannot see. Answer using ONLY the information in the true specification. If the question is not "
    "addressed by it, answer with the most standard/reasonable convention for a competitive-programming "
    "problem (e.g. 'not specified beyond what's given; assume standard behavior') rather than inventing "
    "a new constraint. Answer in ONE concise sentence."
)

# v12_extract_activations.py's STYLE_INSTRUCTION, unchanged in substance
# (same "AtCoder/LiveCodeBench style" framing v12 used for every row --
# replicating v12 exactly, not fixing the per-source description).
REWRITE_RULE = (
    "You are editing a competitive-programming problem statement (AtCoder/LiveCodeBench style: "
    "terse, formal, precise, no filler commentary, uses the same mathematical notation style as the original, "
    "and preserves the original's line-break structure -- one clause/sentence per line where the original does that). "
    "Rewrite the ORIGINAL problem statement to incorporate the given clarification, INLINE into the existing prose "
    "(edit an existing sentence, or insert one short sentence in the same place and style where a real "
    "problem-setter would have put it originally). Do NOT append a new bullet point or trailing note at the "
    "end. Do NOT change anything else about the text -- keep the rest word-for-word identical, including all "
    "code, signatures, imports, and line breaks."
)

SYSTEM_PROMPT = f"""For each candidate clarifying question about a coding problem, do TWO things:

1. ANSWER: {ORACLE_RULE}

2. REWRITE: {REWRITE_RULE} The clarification to incorporate is your own answer from step 1, phrased as "Add this clarification, integrated naturally into the relevant sentence: <answer>".

Respond with ONLY a JSON object, no markdown fences:
{{"candidates": [{{"question": "<verbatim>", "answer": "<one sentence>", "inline_prompt": "<the FULL rewritten problem statement, complete, not truncated>"}}]}}"""


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    start = text.find("{")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise ValueError("unbalanced")


def call_deepseek(rt, user_prompt, max_tokens=8000):
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
    )
    delay = 5
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = rt.invoke_model(modelId=MODEL_ID, body=body)
            payload = json.loads(resp["body"].read())
            return payload["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if "Throttling" in msg or "TooManyRequests" in msg:
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(delay, 20))
                delay = min(delay * 2, 120)
                continue
            raise
    raise last_err


def build_user_prompt(g, candidates, cand_slice):
    cand_text = "\n".join(f"{i+1}. {candidates[ci]['question']}" for i, ci in enumerate(cand_slice))
    return (
        f"--- ORIGINAL PROBLEM STATEMENT ---\n{g['prompt']}\n\n"
        f"--- TRUE SPECIFICATION (privileged, for answering only) ---\n{g['reference_text']}\n\n{g['extra_context']}\n\n"
        f"--- CANDIDATE QUESTIONS ---\n{cand_text}"
    )


def process_row(rt, g, candidates):
    # split into halves if >5 candidates to keep response size safe
    if len(candidates) <= 5:
        slices = [list(range(len(candidates)))]
    else:
        mid = (len(candidates) + 1) // 2
        slices = [list(range(0, mid)), list(range(mid, len(candidates)))]

    all_results = [None] * len(candidates)
    for sl in slices:
        prompt = build_user_prompt(g, candidates, sl)
        raw = call_deepseek(rt, prompt)
        parsed = extract_json(raw)
        verdicts = parsed["candidates"]
        if len(verdicts) != len(sl):
            # best-effort match by question text
            vmap = {v["question"].strip(): v for v in verdicts}
            for ci in sl:
                v = vmap.get(candidates[ci]["question"].strip())
                if v is None:
                    raise ValueError(f"missing verdict for candidate {ci}")
                all_results[ci] = v
        else:
            for ci, v in zip(sl, verdicts):
                all_results[ci] = v
    return all_results


def main():
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]

    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g

    new_ids = json.loads(NEW_IDS_PATH.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [rid for rid in new_ids if not (OUT_DIR / f"{safe_name(rid)}.json").exists()]
    print(f"[v15-rewrite] {len(new_ids)} total, {len(new_ids)-len(todo)} already done, {len(todo)} todo", flush=True)

    def worker(rid):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        g = groups[rid]
        candidates = judged_rows[rid]["judged"]["candidates"]
        for attempt in range(3):
            try:
                results = process_row(rt, g, candidates)
                out = {"id": rid, "candidates": results}
                (OUT_DIR / f"{safe_name(rid)}.json").write_text(json.dumps(out, indent=1))
                return rid, None
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return rid, str(e)
                time.sleep(5)

    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker, rid): rid for rid in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rid, err = fut.result()
            if err:
                n_err += 1
                print(f"[{i}/{len(todo)}] ERROR {rid} {err[:150]}", flush=True)
            else:
                n_ok += 1
                if i % 20 == 0 or i == len(todo):
                    print(f"[{i}/{len(todo)}] ok={n_ok} err={n_err} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"DONE. ok={n_ok} err={n_err}", flush=True)


if __name__ == "__main__":
    main()
