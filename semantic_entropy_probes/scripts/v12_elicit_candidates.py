"""v12 stage B: for every ambiguous training row in
data/v12_ranking/ground_truth.jsonl, elicit up to `--max_candidates` (8)
candidate clarifying QUESTIONS via the Groq-primary/OpenRouter-fallback API
agent (scripts/coding_agent.py's generate_agent) -- NOT local Qwen3-8B.

This matters: v10_round_worker.py originally elicited locally with Qwen3-8B,
but v11_heval_batch_worker.py's own docstring documents why that was
abandoned -- a qualitative review of the all-local run found that on ~12/32
round-1 misses, about half the time the correct clarifying question was
NEVER proposed among Qwen3-8B's up-to-8 candidates at all (a pure
elicitation-capability gap, not a probe-scoring failure). v11 switched
elicitation to the API agent for exactly this reason and kept it there ever
since; PROBE SCORING stays on local Qwen3-8B regardless (rank1/rank2 need
Qwen's own hidden states -- that part is untouched, see
v12_extract_activations.py). This script must match that same precedent, not
the older, already-superseded v10 approach -- an earlier version of this
file mistakenly copied v10's local-Qwen elicitation; its output was
discarded (data/v12_ranking/candidates.jsonl.local_qwen_STALE.bak) and
regenerated via this corrected version.

No GPU needed for this stage at all now (pure API calls) -- it no longer
contends with Stage D's local-Qwen activation extraction.

Resumable: writes one JSON line per completed row to
data/v12_ranking/candidates.jsonl immediately after each row, and skips any
id already present in that file on restart.

Usage: python v12_elicit_candidates.py [--max_candidates 8] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.coding_agent import generate_agent  # noqa: E402, F401

# NOTE (2026-08-28): OpenRouter is out of credits (402 Payment Required) as of this
# session, while Groq's earlier daily rate-limit cap (2026-08-27) has reset -- back to
# the default Groq-primary/OpenRouter-fallback generate_agent from coding_agent.py.

GROUND_TRUTH_PATH = ROOT / "data" / "v12_ranking" / "ground_truth.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "candidates.jsonl"

ELICIT_QUESTIONS_INSTRUCTION = (
    "You are reviewing a competitive-programming problem statement for specification ambiguity -- "
    "places where the statement under-specifies behavior, leaving multiple reasonable interpretations "
    "(e.g. edge cases, tie-breaking rules, formatting details, boundary conditions, data ranges, "
    "input constraints left implicit, terminology, etc.).\n\n"
    "Here is the problem statement:\n<problem>\n{original}\n</problem>\n\n"
    "List up to {max_candidates} distinct QUESTIONS you would need answered to remove ambiguity from this "
    "problem statement before implementing it. For EACH one, output a single line in this exact format:\n"
    "N. <one-sentence question>\n\n"
    "Only output the numbered list, nothing else -- no preamble, no explanation. Each item MUST be phrased "
    "as an actual question (ending in '?'), not a statement."
)


def parse_numbered(raw_text, max_items):
    items = []
    for line in raw_text.splitlines():
        line = line.strip()
        m = re.match(r'^\d+[\.\)]\s*(.+)$', line)
        if m:
            item = m.group(1).strip().strip('"')
            if item:
                items.append(item)
    return items[:max_items]


def elicit_questions_via_api(problem: str, max_candidates: int, max_sanity_retries: int = 3) -> tuple[list[str], str]:
    """Verbatim logic from v11_heval_batch_worker.py's elicit_questions_via_api."""
    uc = ELICIT_QUESTIONS_INSTRUCTION.format(original=problem, max_candidates=max_candidates)
    last_qs, last_provider = [], None
    for _ in range(max_sanity_retries):
        raw, provider = generate_agent(uc, max_tokens=500)
        qs = parse_numbered(raw, max_candidates)
        last_qs, last_provider = qs, provider
        if qs:
            return qs, provider
    return last_qs, last_provider


def load_done_ids() -> set[str]:
    if not OUT_PATH.exists():
        return set()
    done = set()
    with OUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["id"])
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_candidates", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = []
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    done_ids = load_done_ids()
    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"[v12-B] {len(done_ids)} already done, {len(todo)}/{len(rows)} remaining", flush=True)
    if not todo:
        return

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a", encoding="utf-8") as out_f:
        for n, row in enumerate(todo, 1):
            t0 = time.time()
            questions, provider = elicit_questions_via_api(row["prompt"], args.max_candidates)
            out_f.write(json.dumps({"id": row["id"], "questions": questions, "provider": provider},
                                    ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"[v12-B] ({n}/{len(todo)}) {row['id']}: {len(questions)} questions via {provider} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print("[v12-B] done", flush=True)


if __name__ == "__main__":
    main()
