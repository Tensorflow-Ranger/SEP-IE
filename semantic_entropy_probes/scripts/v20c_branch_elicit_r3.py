"""v20b stage 2: for every branch in data/v20_branches/branches_round3.json,
elicit + tier a FRESH set of ~8 candidates against that branch's own
`branch_prompt` (original + one wrong candidate's answer concatenated) --
full per-branch fidelity, no reuse/mixing with round-1 candidates, matching
how the real live loop always re-elicits a fresh pool each round.

One Bedrock call per branch. Resumable (keyed by branch index).
Output: data/v20_branches/elicited_round3.jsonl
  {"branch_idx":..., "id":..., "candidates":[{"question":...,"tier":0|1|2,
   "functionally_relevant":...,"reason":...}, ...]}
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round3.json"
OUT_PATH = ROOT / "data" / "v20_branches_full" / "elicited_round3.jsonl"

MODEL_ID = "deepseek.v3.2"
REGION = "ap-northeast-1"
N_WORKERS = 8
MAX_RETRIES = 8
N_ELICIT = 8

SYSTEM_PROMPT = """You are auditing a Python/coding function specification for remaining \
specification ambiguity. The text you are given already has ONE clarification appended at the \
end (it may or may not have actually resolved the original ambiguity well -- judge the text as \
given, don't assume it was a good clarification).

Invent up to {n_elicit} diverse, plausible clarifying questions a careful reviewer would still \
want answered before implementing this specific prompt -- not generic boilerplate (type \
validation / negative numbers / empty input) unless those genuinely apply here.

Judge each candidate on two independent axes:

TIER (categorical):
- 0 = irrelevant or nonsensical for this specific prompt, or already unambiguously answered by \
the prompt (including the appended clarification) as written
- 1 = a genuine, unresolved ambiguity in this prompt
- 2 = the SINGLE most functionally consequential ambiguity remaining (at most one candidate may \
be tier 2; if none of your candidates rise above tier 1, omit tier 2 entirely)

FUNCTIONAL IMPACT (test-blind): would two plausible, reasonable answers to this question cause a \
CORRECT implementation to actually diverge in output for SOME valid input, reasoned from the \
docstring/signature/reference evidence as evidence of "correct" -- not limited to whether given \
tests happen to exercise it. Mark functionally_relevant: false for real-but-inert ambiguities, \
don't default to true.

Respond with ONLY a single JSON object, no markdown fences, no commentary, matching exactly:
{{
  "candidates": [
    {{"question": "<text>", "tier": 0|1|2, "functionally_relevant": true|false, "reason": "<one line>"}}
  ]
}}"""


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def call_deepseek(rt, user_prompt):
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(n_elicit=N_ELICIT)},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 3000,
        "temperature": 0.2,
    })
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
                time.sleep(delay); delay = min(delay * 2, 120); continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(delay, 20)); delay = min(delay * 2, 120); continue
            raise
    raise last_err


def build_user_prompt(branch):
    return (
        f"--- PROMPT (with one clarification already appended) ---\n{branch['branch_prompt']}\n\n"
        f"--- REFERENCE EVIDENCE ---\n{branch['extra_context']}"
    )


_write_lock = threading.Lock()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]

    branches = json.loads(BRANCHES_PATH.read_text())
    log(f"branches: {len(branches)}")

    done = set()
    if OUT_PATH.exists():
        for l in open(OUT_PATH, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["branch_idx"])

    todo = [(i, b) for i, b in enumerate(branches) if i not in done]
    log(f"todo: {len(todo)}")

    out_f = open(OUT_PATH, "a", encoding="utf-8")

    def process(item):
        idx, branch = item
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        user_prompt = build_user_prompt(branch)
        for attempt in range(3):
            try:
                raw = call_deepseek(rt, user_prompt)
                parsed = extract_json(raw)
                return {"branch_idx": idx, "id": branch["id"], "candidates": parsed["candidates"]}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"branch_idx": idx, "id": branch["id"], "error": str(e)}
                time.sleep(5)

    t0 = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process, item): item for item in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            with _write_lock:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            if "error" in result:
                n_err += 1
                log(f"[{i}/{len(todo)}] ERROR idx={result['branch_idx']} {result['error'][:150]}")
            else:
                n_ok += 1
                log(f"[{i}/{len(todo)}] ok idx={result['branch_idx']} id={result['id']} "
                    f"n_cand={len(result['candidates'])}")
    log(f"DONE ok={n_ok} err={n_err} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
