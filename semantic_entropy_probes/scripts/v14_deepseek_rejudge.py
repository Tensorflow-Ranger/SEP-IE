"""v14: redo v13's rejudging using DeepSeek V3.2 on AWS Bedrock instead of
haiku sub-agents.

One Bedrock call per group (row), not per candidate: a group's ~8
candidates need to be judged *comparatively* (tier, and "is this more
important than the label") which requires seeing them together anyway, so
batching them into one call is both cheaper and the right unit of
judgment. For `needs_elicitation` groups (the 460 orchid_heval-Semantic /
orchid_bcb rows v12 never touched), elicitation is folded into the same
call rather than split into a separate round-trip.

Per candidate, beyond the old tier (0/1/2), this asks for a *functional*
impact judgment instead of a test-flip judgment: would two plausible
answers to this question cause a CORRECT implementation to diverge in
behavior over the function's full input domain, reasoned from the
docstring/signature/reference solution -- not just whether the specific
redistributed tests happen to exercise it. That's the "test-case blind"
check the test_relevant field in v13 didn't do (it checked against the
given tests, which is exactly the information a real user doesn't have).

Also asks directly: is there a non-label candidate that is a bigger
functional lever than the row's own labeled ambiguity.

Resumable: writes one JSON line per completed group to
data/v13_rejudge/deepseek_judged.jsonl immediately (flushed), and skips
any id already present in that file on restart. Safe to kill and rerun.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
OUT_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl"
LOG_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_run.log"

MODEL_ID = "deepseek.v3.2"
REGION = "ap-northeast-1"
N_WORKERS = 6
MAX_RETRIES = 8
N_ELICIT = 8

SYSTEM_PROMPT = """You are auditing a dataset of deliberately-ambiguated coding problems. \
Each problem had ONE ambiguity injected on purpose (the "label"), but real prompts often \
have several live ambiguities, of which the label is only one candidate.

Judge every candidate clarifying question on two independent axes:

TIER (categorical):
- 0 = irrelevant or nonsensical for this specific prompt (a generic question that doesn't \
even apply, or something already unambiguously answered by the prompt as written)
- 1 = a genuine, unresolved ambiguity in this prompt, but NOT the same thing as the labeled \
ambiguity below
- 2 = matches / duplicates the labeled ambiguity (asks about essentially the same fork in \
behavior that was deliberately injected)

FUNCTIONAL IMPACT (the important, test-blind check):
Would two plausible, reasonable answers to this question cause a CORRECT implementation of \
this problem to actually diverge in output/behavior for SOME valid input within the \
function's full input domain -- reasoning from the docstring, signature, and reference \
solution/tests you're given as evidence of what "correct" looks like, NOT limited to \
whether the specific redistributed test cases happen to exercise the divergence. A real \
user deploying this code doesn't get to see the hidden tests either, so "does it change the \
given tests" is the WRONG question -- "does it change what the program actually does" is \
the right one. It is entirely possible for a real, non-tier-0 ambiguity to have NO \
functional impact (e.g. the given tests, and plausibly all reasonable inputs anyone would \
feed it, never hit the diverging case) -- mark those functionally_relevant: false with a \
one-line reason, don't default to true just because the ambiguity is "real."

For the row's own LABEL, judge two more things:
- label_leaked: is the ambiguous detail actually already pinned down somewhere else in the \
SAME prompt (an example, a type hint, other prose) such that a careful reader wouldn't need \
to ask at all? Look for concrete textual evidence, don't guess.
- label_functionally_relevant: same functional-impact test as above, applied to the label \
itself.

Finally, across all tier-1 (non-label) candidates, decide whether any ONE of them is \
functionally MORE consequential than the label for what code actually gets written (e.g. it \
changes behavior across a much wider swath of the input domain, or the label turns out to be \
functionally inert while this other one is not). Most rows will have none -- only report one \
if it's a real, defensible claim, not a coin flip.

Respond with ONLY a single JSON object, no markdown fences, no commentary, matching exactly:
{
  "candidates": [
    {"question": "<verbatim candidate text>", "tier": 0|1|2, "functionally_relevant": true|false, "reason": "<one line>"}
  ],
  "label_leaked": true|false,
  "label_leaked_reason": "<one line>",
  "label_functionally_relevant": true|false,
  "label_functionally_relevant_reason": "<one line>",
  "more_important_than_label": null | {"question": "<verbatim candidate text, must match one in candidates>", "reason": "<one line>"}
}

If the group needs elicitation (no candidates given), you must ALSO invent up to 8 diverse, \
plausible clarifying questions a careful reviewer might ask about this specific prompt \
before writing code -- not generic boilerplate (type validation / negative numbers / empty \
input) unless those genuinely apply here -- and judge each of your own invented candidates \
the same way, filling "candidates" with them (question field = your invented text)."""


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_all_groups():
    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g
    return list(groups.values())


def build_user_prompt(g):
    parts = [
        f"Dataset source: {g['dataset_source']} / ambiguity_type: {g['ambiguity_type']}",
        f"\n--- AMBIGUOUS PROMPT ---\n{g['prompt']}",
        f"\n--- LABELED AMBIGUITY (what was deliberately injected) ---\n{g['reference_text']}",
        f"\n--- REFERENCE EVIDENCE (solution / tests / qa pairs) ---\n{g['extra_context']}",
    ]
    if g["needs_elicitation"]:
        parts.append(
            f"\n--- CANDIDATES ---\n(none given -- invent up to {N_ELICIT} yourself, per the "
            "system instructions, then judge them.)"
        )
    else:
        cand_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(g["candidates"]))
        parts.append(f"\n--- CANDIDATES ({len(g['candidates'])}) ---\n{cand_text}")
    return "\n".join(parts)


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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 4000,
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


_write_lock = threading.Lock()
_log_lock = threading.Lock()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def process_group(rt, g):
    user_prompt = build_user_prompt(g)
    raw = call_deepseek(rt, user_prompt)
    try:
        parsed = extract_json(raw)
    except Exception as e:  # noqa: BLE001
        return {"id": g["id"], "error": f"parse_failed: {e}", "raw": raw[:2000]}
    return {
        "id": g["id"],
        "dataset_source": g["dataset_source"],
        "ambiguity_type": g["ambiguity_type"],
        "needs_elicitation": g["needs_elicitation"],
        "judged": parsed,
    }


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
    done_ids = set()
    if OUT_PATH.exists():
        for l in open(OUT_PATH, encoding="utf-8"):
            if l.strip():
                try:
                    done_ids.add(json.loads(l)["id"])
                except Exception:  # noqa: BLE001
                    pass
    todo = [g for g in groups if g["id"] not in done_ids]
    log(f"total groups={len(groups)} already_done={len(done_ids)} todo={len(todo)}")

    out_f = open(OUT_PATH, "a", encoding="utf-8")

    def worker(g):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        for attempt in range(3):
            try:
                return process_group(rt, g)
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"id": g["id"], "error": f"failed: {e}"}
                time.sleep(5)

    n_ok = 0
    n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker, g): g["id"] for g in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            with _write_lock:
                out_f.write(json.dumps(result) + "\n")
                out_f.flush()
            if "error" in result:
                n_err += 1
                log(f"[{i}/{len(todo)}] ERROR id={result['id']} {result['error'][:150]}")
            else:
                n_ok += 1
                if i % 20 == 0 or i == len(todo):
                    elapsed = time.time() - t0
                    rate = i / elapsed * 60
                    log(f"[{i}/{len(todo)}] ok={n_ok} err={n_err} rate={rate:.1f}/min")

    log(f"DONE. ok={n_ok} err={n_err} total_time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
