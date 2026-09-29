"""v20f: re-tier round-2's ALREADY-elicited candidates (from the pre-fix
backup, data/v20_branches_full/_prelabelfix_backup/elicited_round2.jsonl)
against the dataset's own LABELED ambiguity (reference_text), instead of
re-eliciting a fresh candidate set through v20c_branch_elicit_r2.py.

Why re-tier instead of re-elicit: the elicitation instruction (invent up to
8 diverse candidates) was never buggy -- only the TIER definition was (see
v20c_branch_elicit_r2.py's docstring / FINDINGS.md's v20e note). Re-using
the exact same candidate question texts and only fixing the judgment
isolates the labeling fix cleanly, and lets the oracle-answer stage's
existing output (data/v20_branches_full/_prelabelfix_backup/answers_round2.jsonl,
keyed by exact question text) be reused as-is -- no new Bedrock calls needed
for elicitation OR oracle-answering, only for this re-tiering pass.

One Bedrock call per branch (all its ~8 existing candidates at once).
Resumable (keyed by branch index), same output schema as
v20c_branch_elicit_r2.py so v20c_extract_r2.py needs no changes.

Output: data/v20_branches_full/elicited_round2.jsonl
  {"branch_idx":..., "id":..., "candidates":[{"question":...,"tier":0|1|2,
   "functionally_relevant":...,"reason":...}, ...]}
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from v19_extract_activations import CHUNK_DIR  # noqa: E402

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
OLD_ELICITED_PATH = ROOT / "data" / "v20_branches_full" / "_prelabelfix_backup" / "elicited_round2.jsonl"
OUT_PATH = ROOT / "data" / "v20_branches_full" / "elicited_round2.jsonl"

MODEL_ID = "deepseek.v3.2"
REGION = "ap-northeast-1"
N_WORKERS = 8
MAX_RETRIES = 8

SYSTEM_PROMPT = """You are auditing a Python/coding function specification for remaining \
specification ambiguity. The text you are given already has ONE clarification appended at the \
end (it may or may not have actually resolved the original ambiguity well -- judge the text as \
given, don't assume it was a good clarification). You are also given the dataset's own LABELED \
ambiguity -- the ONE ambiguity that was originally, deliberately injected into this problem -- \
for reference, and a fixed list of candidate clarifying questions to judge (do NOT invent new \
ones -- judge exactly the candidates given, in the same order).

Judge each candidate on two independent axes:

TIER (categorical):
- 0 = irrelevant or nonsensical for this specific prompt, or already unambiguously answered by \
the prompt (including the appended clarification) as written
- 1 = a genuine, unresolved ambiguity in this prompt, but NOT the same thing as the LABELED \
ambiguity below
- 2 = matches / duplicates the LABELED ambiguity (asks about essentially the same fork in \
behavior that was deliberately injected, even if worded differently) -- judge this as a \
relevance/equivalence match against the given label, not as your own independent opinion of \
what's "most important"

FUNCTIONAL IMPACT (test-blind): would two plausible, reasonable answers to this question cause a \
CORRECT implementation to actually diverge in output for SOME valid input, reasoned from the \
docstring/signature/reference evidence as evidence of "correct" -- not limited to whether given \
tests happen to exercise it. Mark functionally_relevant: false for real-but-inert ambiguities, \
don't default to true.

Respond with ONLY a single JSON object, no markdown fences, no commentary, matching exactly:
{
  "candidates": [
    {"question": "<verbatim, must match one of the given questions exactly>", "tier": 0|1|2, \
"functionally_relevant": true|false, "reason": "<one line>"}
  ]
}"""


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


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
            {"role": "system", "content": SYSTEM_PROMPT},
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


def load_reference_text_by_id() -> dict:
    ref = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            ref[g["id"]] = g.get("reference_text", "")
    return ref


def build_user_prompt(branch, old_questions):
    q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(old_questions))
    return (
        f"--- PROMPT (with one clarification already appended) ---\n{branch['branch_prompt']}\n\n"
        f"--- LABELED AMBIGUITY (what was originally, deliberately injected -- the appended "
        f"clarification above is a WRONG answer, not this one) ---\n{branch.get('reference_text', '')}\n\n"
        f"--- REFERENCE EVIDENCE ---\n{branch['extra_context']}\n\n"
        f"--- CANDIDATES ({len(old_questions)}) ---\n{q_text}"
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
    ref_by_id = load_reference_text_by_id()
    for b in branches:
        if not b.get("reference_text"):
            b["reference_text"] = ref_by_id.get(b["id"], "")

    old_elicited = {d["branch_idx"]: d for d in read_jsonl(OLD_ELICITED_PATH) if "error" not in d}
    log(f"branches: {len(branches)}, old elicited (candidates to re-tier): {len(old_elicited)}")

    done = set()
    if OUT_PATH.exists():
        for l in open(OUT_PATH, encoding="utf-8"):
            if l.strip():
                done.add(json.loads(l)["branch_idx"])

    todo = [idx for idx in old_elicited if idx not in done]
    log(f"todo: {len(todo)}")

    out_f = open(OUT_PATH, "a", encoding="utf-8")

    def process(idx):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        branch = branches[idx]
        old_questions = [c["question"].strip() for c in old_elicited[idx]["candidates"]]
        user_prompt = build_user_prompt(branch, old_questions)
        for attempt in range(3):
            try:
                raw = call_deepseek(rt, user_prompt)
                parsed = extract_json(raw)
                cands = parsed["candidates"]
                got_qs = {c["question"].strip() for c in cands}
                missing = [q for q in old_questions if q not in got_qs]
                if missing:
                    raise ValueError(f"{len(missing)}/{len(old_questions)} questions dropped by model")
                return {"branch_idx": idx, "id": branch["id"], "candidates": cands}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"branch_idx": idx, "id": branch["id"], "error": str(e)}
                time.sleep(5)

    t0 = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process, idx): idx for idx in todo}
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
                if i % 50 == 0 or i == len(todo):
                    log(f"[{i}/{len(todo)}] ok idx={result['branch_idx']} id={result['id']}")
    log(f"DONE ok={n_ok} err={n_err} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
