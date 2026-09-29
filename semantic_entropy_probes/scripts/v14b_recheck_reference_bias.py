"""v14b: fix two reasoning bugs found in the v14 DeepSeek rejudge pass,
both caught by manual spot-checking after the fact.

Bug A -- reference-implementation bias (342 candidates, non-bcb sources):
tier-1 "functionally_relevant: false" verdicts justified "no divergence"
by citing what the ONE reference solution happens to do (e.g. "reference
solution uses abs(), so sign doesn't matter") rather than the problem
SPEC itself. Only valid evidence when the prompt's own starter code
forces a specific implementation approach (true for orchid_bcb, which
hands you required imports + a function stub) -- for
orchid_heval/clarifycodebench, a different, equally spec-compliant
solution could still diverge on exactly that candidate's question.

Bug B -- mis-tiering (131 candidates, all sources): some "tier 1,
functionally_relevant: false" verdicts have reasons that plainly cite the
PROMPT's own text ("prompt specifies...", "prompt explicitly states...")
as already answering the question -- per the tiering system prompt's own
rule, that makes it tier 0 (already unambiguously answered), not tier 1
(a genuine, unresolved ambiguity that happens not to matter). These
aren't "real ambiguities that turned out to be inert" -- they were never
ambiguous in the first place, and got miscounted into the tier-1 pool.

464 flagged (row, candidate) pairs total (342 + 131, 9 overlap) across
251 rows -- one Bedrock call per affected row, listing only its flagged
candidates, asking the model to reconsider BOTH tier (0 vs 1, using only
the prompt's own text) and functional relevance (using the spec, not the
one given reference implementation) together.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
JUDGED_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl"
SUSPECTS_PATH = Path("/tmp/suspect_candidates_v2.json")
OUT_PATH = ROOT / "data" / "v13_rejudge" / "recheck_results.jsonl"

MODEL_ID = "deepseek.v3.2"
REGION = "ap-northeast-1"
N_WORKERS = 6
MAX_RETRIES = 8

SYSTEM_PROMPT = """You are re-checking two specific classes of judging error from a previous \
pass. For each candidate clarifying question given, re-decide BOTH tier and \
functionally_relevant from scratch, watching for these two failure modes:

BUG A -- reference-implementation bias: previously, some candidates were marked \
functionally_relevant: false with reasoning like "the reference solution does X, so no \
divergence" -- but that only proves ONE implementation didn't diverge, not that no correct \
implementation would. A reference solution's incidental choices (which library function, how \
it happens to order operations, an unstated default it picked) are NOT proof that a \
different, equally spec-compliant implementation couldn't answer the candidate's question the \
other way and produce different output. Use the reference solution/tests only as evidence of \
what "valid input" and "intended output" look like -- never as proof by itself that only one \
interpretation is possible. If a plausible alternative-but-still-correct implementation could \
diverge on this question for some real input, functionally_relevant: true, even if the one \
given reference happens not to exhibit the divergence.

BUG B -- mis-tiering: previously, some candidates were marked tier 1 (a genuine, unresolved \
ambiguity) when the reasoning actually showed the PROMPT's own text (docstring, spec prose, \
constraints -- NOT the reference solution) already answers the question explicitly. That \
should be tier 0 (already unambiguously answered by the prompt), not tier 1 -- it was never \
really ambiguous. Only use tier 0 here if the prompt TEXT itself states the answer; if the \
only evidence is "the reference solution happens to do X," that is NOT grounds for tier 0 \
(see Bug A) -- it stays tier 1, and you separately judge functional relevance per Bug A's rule.

Decision procedure per candidate:
1. Does the prompt's own docstring/spec/constraint text explicitly settle this question? -> \
tier 0, functionally_relevant: false.
2. Otherwise it's tier 1. Could a different, still spec-compliant implementation answer it the \
other way and produce different output for some valid input (reasoning independently of what \
the one given reference happens to do)? -> functionally_relevant: true if yes, false if no.
3. (tier 2, matching the row's own labeled ambiguity, does not apply here -- these were all \
previously tier 1; only reconsider whether it should now be tier 0.)

Respond with ONLY a JSON object, no markdown fences:
{"verdicts": [{"question": "<verbatim candidate text>", "tier": 0|1, "functionally_relevant": true|false, "reason": "<one line, must cite either the PROMPT's own text (for tier 0) or a concrete alternative implementation (for functionally_relevant true/false) -- never just 'the reference does X'>"}]}"""


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


def call_deepseek(rt, user_prompt):
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": 2000,
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


def main():
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]

    judged_rows = read_jsonl(JUDGED_PATH)
    by_id = {r["id"]: r for r in judged_rows}

    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g

    suspects = json.loads(SUSPECTS_PATH.read_text())
    by_row_id = {}
    for s in suspects:
        by_row_id.setdefault(s["id"], []).append(s)

    print(f"[recheck] {len(suspects)} candidates across {len(by_row_id)} rows")

    def build_prompt(rid, cand_idxs):
        g = groups[rid]
        r = by_id[rid]
        j = r["judged"]
        cand_text = "\n".join(
            f"{n+1}. {j['candidates'][ci]['question']}" for n, ci in enumerate(cand_idxs)
        )
        return (
            f"--- PROMPT ---\n{g['prompt']}\n\n"
            f"--- LABELED AMBIGUITY (for context only, not what you're judging) ---\n{g['reference_text']}\n\n"
            f"--- REFERENCE EVIDENCE ---\n{g['extra_context']}\n\n"
            f"--- CANDIDATES TO RE-CHECK ---\n{cand_text}"
        )

    def worker(rid, s_list):
        rt = boto3.client("bedrock-runtime", region_name=REGION)
        cand_idxs = [s["cand_idx"] for s in s_list]
        prompt = build_prompt(rid, cand_idxs)
        for attempt in range(3):
            try:
                raw = call_deepseek(rt, prompt)
                parsed = extract_json(raw)
                return {"id": rid, "cand_idxs": cand_idxs, "verdicts": parsed["verdicts"]}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"id": rid, "cand_idxs": cand_idxs, "error": str(e)}
                time.sleep(5)

    out_f = open(OUT_PATH, "w", encoding="utf-8")
    n_ok = n_err = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(worker, rid, s_list): rid for rid, s_list in by_row_id.items()}
        for i, fut in enumerate(as_completed(futs), 1):
            result = fut.result()
            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            if "error" in result:
                n_err += 1
                print(f"[{i}/{len(by_row_id)}] ERROR {result['id']} {result['error'][:150]}", flush=True)
            else:
                n_ok += 1
                if i % 20 == 0 or i == len(by_row_id):
                    print(f"[{i}/{len(by_row_id)}] ok={n_ok} err={n_err} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"DONE. ok={n_ok} err={n_err}")


if __name__ == "__main__":
    main()
