"""v20 pilot stage 2: oracle-answer every candidate in
data/v20_pilot_multihop/pilot_results.jsonl (all 3 round depths), grounded
in each seed group's real solution/test_case -- ORACLE_RULE reused verbatim
from v19_bedrock_oracle_answer.py. Needed because v20_pilot_multihop.py only
produced a `chosen_answer` for each round's tier-2 pick (just enough to
build the next round's prompt); extraction needs an answer for EVERY
candidate, at every tier, to build each candidate's rewritten prompt the
same way v12_extract_activations.py does.

One Bedrock call per (id, round_depth) row, covering all its candidates at
once (mirrors v19_bedrock_oracle_answer.py's one-call-per-row choice).

Output: data/v20_pilot_multihop/oracle_answers.jsonl
  {"id":..., "round_depth":..., "answers": [{"question":..., "answer":...}, ...]}
Resumable: skips (id, round_depth) pairs already present.
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
PILOT_PATH = ROOT / "data" / "v20_pilot_multihop" / "pilot_results.jsonl"
OUT_PATH = ROOT / "data" / "v20_pilot_multihop" / "oracle_answers.jsonl"

MODEL_ID = "deepseek.v3.2"
REGION = "ap-northeast-1"
N_WORKERS = 6
MAX_RETRIES = 8

ORACLE_RULE = (
    "You are answering a question about a coding problem's specification. You have privileged "
    "knowledge of the TRUE intended specification, given below, which the person asking the question "
    "cannot see. Answer using ONLY the information in the true specification. If the question is not "
    "addressed by it, answer with the most standard/reasonable convention for a competitive-programming "
    "problem (e.g. 'not specified beyond what's given; assume standard behavior') rather than inventing "
    "a new constraint. Answer in ONE concise sentence."
)

SYSTEM_PROMPT = f"""{ORACLE_RULE}

You will be given a coding problem's TRUE reference (canonical solution + tests) and a list of \
candidate clarifying questions about it. Answer each one independently.

Respond with ONLY a JSON object, no markdown fences:
{{"answers": [{{"question": "<verbatim, must match one of the given questions exactly>", "answer": "<one concise sentence>"}}]}}"""


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def extract_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    start = text.find("{")
    depth, in_str, esc = 0, False, False
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
                time.sleep(delay)
                delay = min(delay * 2, 120)
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(delay, 20))
                delay = min(delay * 2, 120)
                continue
            raise
    raise last_err


def build_user_prompt(row, seed_by_id):
    g = seed_by_id[row["id"]]
    qs = [c["question"] for c in row["judged"].get("candidates", [])]
    q_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
    return (
        f"--- REFERENCE (TRUE) ---\nentry_point: {g['entry_point']}\nsolution:\n{g['solution']}\n"
        f"test_case:\n{g['test_case']}\n\n--- QUESTIONS ---\n{q_text}"
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

    from scripts.v20_pilot_multihop import load_seed_groups  # noqa: E402
    seed = load_seed_groups()
    seed_by_id = {g["id"]: g for g in seed}

    pilot_rows = [r for r in read_jsonl(PILOT_PATH) if "error" not in r and r["judged"].get("candidates")]
    log(f"pilot rows with candidates: {len(pilot_rows)}")

    done = set()
    if OUT_PATH.exists():
        for l in open(OUT_PATH, encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                done.add((d["id"], d["round_depth"]))

    todo = [r for r in pilot_rows if (r["id"], r["round_depth"]) not in done]
    log(f"todo: {len(todo)}")

    out_f = open(OUT_PATH, "a", encoding="utf-8")

    def process(row):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        user_prompt = build_user_prompt(row, seed_by_id)
        for attempt in range(3):
            try:
                raw = call_deepseek(rt, user_prompt)
                parsed = extract_json(raw)
                return {"id": row["id"], "round_depth": row["round_depth"], "answers": parsed["answers"]}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"id": row["id"], "round_depth": row["round_depth"], "error": str(e)}
                time.sleep(5)

    t0 = time.time()
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(process, r): r for r in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            with _write_lock:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            if "error" in result:
                n_err += 1
                log(f"[{i}/{len(todo)}] ERROR id={result['id']} rd={result['round_depth']} {result['error'][:150]}")
            else:
                n_ok += 1
                if i % 10 == 0 or i == len(todo):
                    log(f"[{i}/{len(todo)}] ok={n_ok} err={n_err}")
    log(f"DONE ok={n_ok} err={n_err} time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
