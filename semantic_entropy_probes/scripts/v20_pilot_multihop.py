"""v20 pilot: multi-hop (round-depth) tiered candidate data, 50 orchid_heval
groups, to test whether round-depth (0/1/2 clarifications already applied)
behaves like a domain-shift axis the way `dataset_source` does in
`## v19 cross-source-only SupCon` -- i.e. whether a candidate's tier meaning
drifts across round depth the way the v12 live-loop note flagged (same
correct candidate ranked 3rd in round 1, dead-last by round 3).

Round 0: reused as-is from `data/v13_rejudge/deepseek_judged.jsonl`
(already-judged candidates against the raw ambiguous prompt) -- no new
compute.

Round 1: elicit + tier a FRESH candidate set against the dataset's own
`prompt` field (the human-authored clear/unambiguous original -- i.e. the
one injected ambiguity is now resolved by construction, no rewrite needed).
Tier 2 here means "the single most functionally consequential ambiguity
still remaining" (self-judged -- there's no second external label at this
depth), not "matches an injected label" the way round 0's tier 2 does.
For the tier-2 pick only, also asks for a one-sentence `chosen_answer`.

Round 2: same call, against `prompt` + `chosen_answer` appended as one
extra sentence (a plain-concatenation pilot simplification -- NOT the real
STYLE_INSTRUCTION/Qwen3-8B inline rewrite v12's actual training pipeline
uses; flagged in FINDINGS as a caveat to fix if this pilot's round-depth
signal is real). Skipped for groups whose round 1 had zero tier-2 pick.

Resumable: one JSON line per (id, round_depth) in
data/v20_pilot_multihop/pilot_results.jsonl, skips completed (id,
round_depth) pairs on restart.
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
JUDGED_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl"
RAW_ORCHID_HEVAL = ROOT / "data" / "raw" / "orchid_heval.jsonl"
OUT_DIR = ROOT / "data" / "v20_pilot_multihop"
OUT_PATH = OUT_DIR / "pilot_results.jsonl"

MODEL_ID = "deepseek.v3.2"
REGION = "ap-northeast-1"
N_WORKERS = 6
MAX_RETRIES = 8
N_ELICIT = 8
N_GROUPS = 50

SYSTEM_PROMPT = """You are auditing a Python function specification for remaining specification \
ambiguity, AFTER one previously-known ambiguity has already been resolved by whatever text you \
are given. Real prompts typically have SEVERAL live ambiguities even after one is fixed.

Invent up to {n_elicit} diverse, plausible clarifying questions a careful reviewer would still \
want answered before implementing this specific prompt -- not generic boilerplate (type \
validation / negative numbers / empty input) unless those genuinely apply here.

Judge each candidate on two independent axes:

TIER (categorical):
- 0 = irrelevant or nonsensical for this specific prompt, or already unambiguously answered by \
the prompt as written
- 1 = a genuine, unresolved ambiguity in this prompt
- 2 = the SINGLE most functionally consequential ambiguity remaining (at most one candidate may \
be tier 2; if none of your candidates rise above tier 1, leave all as tier 1/0 and omit tier 2 \
entirely -- do not force a tier-2 pick)

FUNCTIONAL IMPACT (test-blind): would two plausible, reasonable answers to this question cause a \
CORRECT implementation to actually diverge in output for SOME valid input in the function's full \
domain, reasoned from the docstring/signature/reference solution as evidence of "correct" -- not \
limited to whether the specific given tests happen to exercise it. A real, non-tier-0 ambiguity \
can still be functionally inert; mark those functionally_relevant: false, don't default to true.

If (and only if) you assigned a tier-2 candidate, also give a concise one-sentence \
`chosen_answer` -- the single most natural, reasonable resolution to that specific question, \
phrased as a plain declarative sentence (not a question), as if a spec author were adding it \
inline to the docstring.

Respond with ONLY a single JSON object, no markdown fences, no commentary, matching exactly:
{{
  "candidates": [
    {{"question": "<text>", "tier": 0|1|2, "functionally_relevant": true|false, "reason": "<one line>"}}
  ],
  "chosen_answer": null | "<one-sentence answer to the tier-2 candidate, if any>"
}}"""


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
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT.format(n_elicit=N_ELICIT)},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 3000,
            "temperature": 0.2,
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


def build_user_prompt(prompt_text, entry_point, solution, test_case):
    return (
        f"--- PROMPT (already resolved of its one previously-known ambiguity) ---\n{prompt_text}\n\n"
        f"--- REFERENCE EVIDENCE ---\nentry_point: {entry_point}\nsolution:\n{solution}\n"
        f"test_case:\n{test_case}"
    )


def load_seed_groups():
    """50 orchid_heval ids already round-0-judged in v13/v14, joined against the raw
    Orchid-HEval source for `prompt` (clear original) / `entry_point` / `solution` /
    `test_case` -- see this file's module docstring for the id-format mapping."""
    judged = read_jsonl(JUDGED_PATH)
    round0_by_id = {d["id"]: d for d in judged if d.get("dataset_source") == "orchid_heval"}

    raw = read_jsonl(RAW_ORCHID_HEVAL)
    raw_by_idx = {}
    for r in raw:
        idx = r["name"].split("/")[-1]
        raw_by_idx[idx] = r

    seed = []
    for gid in sorted(round0_by_id):
        m = re.match(r"orchidheval_(\d+)_(lexical|vagueness|semantic)", gid)
        if not m:
            continue
        idx, _amb_type = m.groups()
        r = raw_by_idx.get(idx)
        if r is None:
            continue
        seed.append(
            {
                "id": gid,
                "prompt": r["prompt"],
                "entry_point": r["entry_point"],
                "solution": r["solution"],
                "test_case": r["test_case"],
                "round0_candidates": round0_by_id[gid]["judged"].get("candidates", []),
            }
        )
        if len(seed) >= N_GROUPS:
            break
    return seed


_write_lock = threading.Lock()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_done():
    done = set()
    if OUT_PATH.exists():
        for l in open(OUT_PATH, encoding="utf-8"):
            if l.strip():
                try:
                    d = json.loads(l)
                    done.add((d["id"], d["round_depth"]))
                except Exception:  # noqa: BLE001
                    pass
    return done


def process_round1(rt, g):
    user_prompt = build_user_prompt(g["prompt"], g["entry_point"], g["solution"], g["test_case"])
    raw = call_deepseek(rt, user_prompt)
    try:
        parsed = extract_json(raw)
    except Exception as e:  # noqa: BLE001
        return {"id": g["id"], "round_depth": 1, "error": f"parse_failed: {e}", "raw": raw[:2000]}
    return {"id": g["id"], "round_depth": 1, "prompt": g["prompt"], "judged": parsed}


def process_round2(rt, g, round1_result):
    chosen_answer = round1_result["judged"].get("chosen_answer")
    if not chosen_answer:
        return None
    round2_prompt = g["prompt"].rstrip() + f" {chosen_answer}"
    user_prompt = build_user_prompt(round2_prompt, g["entry_point"], g["solution"], g["test_case"])
    raw = call_deepseek(rt, user_prompt)
    try:
        parsed = extract_json(raw)
    except Exception as e:  # noqa: BLE001
        return {"id": g["id"], "round_depth": 2, "error": f"parse_failed: {e}", "raw": raw[:2000]}
    return {"id": g["id"], "round_depth": 2, "prompt": round2_prompt, "judged": parsed}


def main():
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = load_seed_groups()
    log(f"seed groups: {len(seed)}")

    done = load_done()
    out_f = open(OUT_PATH, "a", encoding="utf-8")

    def write(result):
        with _write_lock:
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()

    # round 0: just copy through, tagged with round_depth=0, if not already written
    for g in seed:
        if (g["id"], 0) not in done:
            write({"id": g["id"], "round_depth": 0, "prompt": g["prompt"],
                   "judged": {"candidates": g["round0_candidates"]}})
            done.add((g["id"], 0))
    log("round 0 written (copied from v13/v14 judged data)")

    def worker_round1(g):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        for attempt in range(3):
            try:
                return process_round1(rt, g)
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"id": g["id"], "round_depth": 1, "error": f"failed: {e}"}
                time.sleep(5)

    todo1 = [g for g in seed if (g["id"], 1) not in done]
    log(f"round 1: todo={len(todo1)}")
    round1_results = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker_round1, g): g for g in todo1}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            write(result)
            round1_results[result["id"]] = result
            status = "ERROR" if "error" in result else "ok"
            log(f"[round1 {i}/{len(todo1)}] {status} id={result['id']}")
    log(f"round 1 done in {time.time()-t0:.0f}s")

    # re-load all round1 results (including previously-done ones from a prior run) for round 2
    for l in open(OUT_PATH, encoding="utf-8"):
        d = json.loads(l)
        if d.get("round_depth") == 1 and "error" not in d:
            round1_results[d["id"]] = d

    seed_by_id = {g["id"]: g for g in seed}

    def worker_round2(gid):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        g = seed_by_id[gid]
        r1 = round1_results[gid]
        for attempt in range(3):
            try:
                return process_round2(rt, g, r1)
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"id": gid, "round_depth": 2, "error": f"failed: {e}"}
                time.sleep(5)

    todo2 = [gid for gid in round1_results if (gid, 2) not in done]
    log(f"round 2: todo={len(todo2)} (groups with a round-1 tier-2 pick)")
    t0 = time.time()
    n_skipped_no_tier2 = 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker_round2, gid): gid for gid in todo2}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            if result is None:
                n_skipped_no_tier2 += 1
                continue
            write(result)
            status = "ERROR" if "error" in result else "ok"
            log(f"[round2 {i}/{len(todo2)}] {status} id={result['id']}")
    log(f"round 2 done in {time.time()-t0:.0f}s, skipped (no tier-2 pick)={n_skipped_no_tier2}")
    log("DONE")


if __name__ == "__main__":
    main()
