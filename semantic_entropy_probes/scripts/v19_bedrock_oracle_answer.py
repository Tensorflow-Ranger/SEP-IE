"""v19 stage 2: elicit oracle answers (ONLY -- no inline rewrite, since v19
uses v17's plain-concatenation format) for orchid_bcb_expand groups that
survive the 3-way clean filter (not leaked, label functionally relevant,
no candidate judged more important than the label -- see
`is_clean` in v19_extract_activations.py, duplicated here so this script
has no import-time dependency on it).

Reuses v15_bedrock_oracle_rewrite.py's ORACLE_RULE verbatim (grounded-only,
one concise sentence, standard-convention fallback) but drops the
inline-rewrite half entirely -- v19's activations are built by plain
`prompt + "\\n\\nClarification: {answer}"` concatenation (v17's format), so
only the answer text is needed, not a rewritten prompt.

One Bedrock call per row, covering all its judged candidates at once.

Output: data/v19_bcbexpand/answers/{safe_id}.json
  {"id":..., "candidates": [{"question":..., "answer":...}, ...]}
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
OUT_DIR = ROOT / "data" / "v19_bcbexpand" / "answers"
LOG_PATH = ROOT / "data" / "v19_bcbexpand" / "oracle_answer_run.log"

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
{{"candidates": [{{"question": "<verbatim, must match one of the given questions exactly>", "answer": "<one concise sentence>"}}]}}"""


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
        "max_tokens": 4000,
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
            delay = min(delay * 2, 120)
            if "Throttling" in msg or "TooManyRequests" in msg:
                time.sleep(delay)
                continue
            if attempt < MAX_RETRIES - 1:
                time.sleep(min(delay, 20))
                continue
            raise
    raise last_err


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_user_prompt(g, judged):
    cand_text = "\n".join(f"{i+1}. {c['question']}" for i, c in enumerate(judged["candidates"]))
    return (
        f"--- PROBLEM PROMPT (ambiguous version) ---\n{g['prompt']}\n\n"
        f"--- TRUE REFERENCE ---\n{g['reference_text']}\n\n{g['extra_context']}\n\n"
        f"--- CANDIDATE QUESTIONS ---\n{cand_text}"
    )


def load_all_groups():
    groups = {}
    for f in sorted((ROOT / "data" / "v13_rejudge" / "chunks").glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            if g["dataset_source"] == "orchid_bcb_expand":
                groups[g["id"]] = g
    return groups


def main():
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]

    groups = load_all_groups()
    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH) if r["id"] in groups}
    clean_ids = [rid for rid, r in judged_rows.items() if is_clean(r["judged"])]
    log(f"orchid_bcb_expand: {len(groups)} groups, {len(judged_rows)} judged, {len(clean_ids)} clean")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [rid for rid in clean_ids if not (OUT_DIR / f"{safe_name(rid)}.json").exists()]
    log(f"todo={len(todo)}")

    def worker(rid):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        g = groups[rid]
        j = judged_rows[rid]["judged"]
        user_prompt = build_user_prompt(g, j)
        for attempt in range(3):
            try:
                raw = call_deepseek(rt, user_prompt)
                parsed = extract_json(raw)
                (OUT_DIR / f"{safe_name(rid)}.json").write_text(
                    json.dumps({"id": rid, "candidates": parsed["candidates"]})
                )
                return rid, None
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return rid, str(e)
                time.sleep(5)

    n_ok, n_err = 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker, rid): rid for rid in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            rid, err = fut.result()
            if err:
                n_err += 1
                log(f"[{i}/{len(todo)}] ERROR {rid}: {err[:150]}")
            else:
                n_ok += 1
                if i % 50 == 0 or i == len(todo):
                    elapsed = time.time() - t0
                    log(f"[{i}/{len(todo)}] ok={n_ok} err={n_err} rate={i/elapsed*60:.1f}/min")

    log(f"DONE. ok={n_ok} err={n_err} total_time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
