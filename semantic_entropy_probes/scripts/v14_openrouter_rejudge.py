"""v3 retrain, elicitation+tiering leg for the 400 new v13 rows (orchid_heval
Semantic + orchid_bcb Semantic/Vagueness that survived the label_leaked /
label_test_relevant exclusion filter).

Why this exists: v13's candidates + tiers for these rows came from 20
parallel Haiku sub-agents (FINDINGS.md's "v13: full rejudging..." section),
which that same write-up flags as unreliable -- tier-2 base rate ranging
0%-69% across chunks given identical instructions, some chunks silently
skipping elicitation, label-verdict fields defaulted rather than reasoned.
A follow-up spot-check (this conversation) confirmed the tier field
specifically is bad: sampled "real" tier reorderings vs v12's original
tiering were 0/4 correct (systematically demoted the actual label match to
tier 0). Since these 400 rows have NO v12-original tiering to fall back on
(v12 never touched them), the fix is to regenerate both the candidate
proposals and their tiers from scratch with a single structured API call per
row (OpenRouter, no multi-agent orchestration to skip steps or drift
mid-run) instead of trusting the Haiku sub-agent output.

Grounding per row comes from data/v13_rejudge/chunks/*.json's reference_text
(what was made ambiguous + the original clean prompt) and extra_context
(reference solution + real test code, for orchid_heval/orchid_bcb -- both
sources have real tests, unlike clarifycodebench).

Output (resumable, one file per row):
  data/v13_rejudge/rejudge_v2/{safe_id}.json
    {"id":..., "candidates": [{"question":, "answer":, "tier":, "test_relevant":}, x8]}

Usage: python v14_openrouter_rejudge.py [--limit N] [--workers 8]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
MERGED_PATH = ROOT / "data" / "v13_rejudge" / "merged.json"
ACT_DIR = ROOT / "data" / "v12_ranking" / "activations"
OUT_DIR = ROOT / "data" / "v13_rejudge" / "rejudge_v2"

MODEL = "openai/gpt-4o-mini"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are grading and generating clarifying questions for an ambiguous programming problem statement.

You will be given:
- PROMPT: the ambiguous problem statement as given to a coder.
- WHAT_WAS_MADE_AMBIGUOUS: the specific detail that was deliberately obscured (the "label"), and why.
- ORIGINAL: the original, unambiguous version of the same problem (ground truth for what the label's correct answer is).
- REFERENCE (if present): a canonical solution and real test cases -- use these to judge whether a question's answer would actually change test outcomes.

Task: propose exactly 8 diverse clarifying questions a careful reviewer might ask about PROMPT before implementing it. For each:
- "question": the clarifying question text.
- "answer": a specific, concrete answer grounded in ORIGINAL/REFERENCE (not a generic placeholder like "not addressed" or "a related ambiguity" -- always give the actual best-available answer).
- "tier": 2 if this question asks about exactly the thing named in WHAT_WAS_MADE_AMBIGUOUS (the label) -- even if worded differently. 1 if it's a genuine, separate ambiguity in PROMPT not covered by the label. 0 if it's not a real ambiguity (already answered in PROMPT, pedantic, or irrelevant).
- "test_relevant": true if resolving this question differently (in a plausible alternate way) would change whether the given REFERENCE test cases pass; false if the tests never exercise it; null if no REFERENCE tests were given.

Constraints: exactly 1 candidate should normally be tier 2 (the label itself) unless none of your 8 naturally hit it, in which case include the label question directly. Do not pad with near-duplicate questions. Vary specificity and topic across the 8.

Respond with ONLY a JSON object: {"candidates": [{"question":, "answer":, "tier":, "test_relevant":}, ...8 items...]}"""


def build_user_prompt(row: dict) -> str:
    parts = [f"PROMPT:\n{row['prompt']}\n"]
    ref_text = row.get("reference_text", "")
    if ref_text:
        parts.append(ref_text)
    extra = row.get("extra_context", "")
    if extra and not extra.startswith("("):
        parts.append(f"\n{extra}")
    return "\n".join(parts)


def call_openrouter(api_key: str, user_prompt: str, max_retries: int = 4) -> dict:
    body = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.4,
    }).encode("utf-8")

    last_err = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            OPENROUTER_URL, data=body,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.load(resp)
            content = data["choices"][0]["message"]["content"]
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
            parsed = json.loads(content)
            cands = parsed["candidates"]
            if not isinstance(cands, list) or len(cands) < 4:
                raise ValueError(f"expected >=4 candidates, got {len(cands) if isinstance(cands, list) else type(cands)}")
            for c in cands:
                if c.get("tier") not in (0, 1, 2):
                    raise ValueError(f"bad tier: {c.get('tier')}")
            return {"candidates": cands}
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"OpenRouter call failed after {max_retries} attempts: {last_err}")


def safe_name(row_id: str) -> str:
    return row_id.replace("/", "__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    api_key = os.environ.get("OpenRouter") or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: no OpenRouter API key found in env (OpenRouter / OPENROUTER_API_KEY)", file=sys.stderr)
        sys.exit(1)

    chunks = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for row in json.load(f.open()):
            chunks[row["id"]] = row

    merged = {r["id"]: r for r in json.load(MERGED_PATH.open())}
    act_ids = {p.stem for p in ACT_DIR.glob("*.npz")}

    def keep(r):
        return not (r["label_leaked"] is True or r["label_test_relevant"] is False)

    todo_ids = [gid for gid, r in merged.items() if keep(r) and gid not in act_ids]
    if args.limit:
        todo_ids = todo_ids[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remaining = [gid for gid in todo_ids if not (OUT_DIR / f"{safe_name(gid)}.json").exists()]
    print(f"[v14-rejudge] {len(todo_ids) - len(remaining)} already done, {len(remaining)}/{len(todo_ids)} remaining", flush=True)

    def process(gid: str):
        row = chunks[gid]
        user_prompt = build_user_prompt(row)
        t0 = time.time()
        try:
            result = call_openrouter(api_key, user_prompt)
        except Exception as e:  # noqa: BLE001
            print(f"[v14-rejudge] FAILED {gid}: {e}", flush=True)
            return
        result["id"] = gid
        (OUT_DIR / f"{safe_name(gid)}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        n_tier2 = sum(1 for c in result["candidates"] if c["tier"] == 2)
        print(f"[v14-rejudge] {gid}: {len(result['candidates'])} candidates, "
              f"{n_tier2} tier-2 ({time.time()-t0:.1f}s)", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(process, remaining))

    print("[v14-rejudge] done", flush=True)


if __name__ == "__main__":
    main()
