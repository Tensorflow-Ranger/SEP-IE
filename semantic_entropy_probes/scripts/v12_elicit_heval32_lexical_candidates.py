"""Stage B equivalent for held_out_orchid32_lexical.json: elicit up to
`--max_candidates` (8, matching the original heval32 static eval's scale)
candidate clarifying QUESTIONS per problem via the Groq-primary/OpenRouter-
fallback API agent, using v12_heval32_live_loop.py's HumanEval-docstring
-specific ELICIT_QUESTIONS_INSTRUCTION (domain-matched, not
v12_elicit_candidates.py's competitive-programming-worded version).

Output (questions only, no tiers/answers yet -- Stage C judging is a
separate, oracle-labor step): data/v12_ranking/heval32_lexical_candidates_raw.jsonl
  {"id": "heval_1", "questions": [...]}

Resumable: skips any id already present in the output file on restart.

Usage: python v12_elicit_heval32_lexical_candidates.py [--max_candidates 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v12_heval32_live_loop import ELICIT_QUESTIONS_INSTRUCTION  # noqa: E402
from scripts.v12_elicit_candidates import parse_numbered  # noqa: E402
from scripts.coding_agent import generate_agent  # noqa: E402

HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_orchid32_lexical.json"
OUT_PATH = ROOT / "data" / "v12_ranking" / "heval32_lexical_candidates_raw.jsonl"


def elicit_questions_via_api(problem: str, max_candidates: int, max_sanity_retries: int = 3) -> tuple[list[str], str]:
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
    args = ap.parse_args()

    held_out = json.loads(HELD_OUT_PATH.read_text())
    done_ids = load_done_ids()
    todo = [aid for aid in held_out if aid not in done_ids]
    print(f"[v12-heval32-lex-B] {len(done_ids)} already done, {len(todo)}/{len(held_out)} remaining", flush=True)

    with OUT_PATH.open("a", encoding="utf-8") as f:
        for i, aid in enumerate(todo, 1):
            prompt = held_out[aid]["ambiguous_prompt"]
            qs, provider = elicit_questions_via_api(prompt, args.max_candidates)
            f.write(json.dumps({"id": aid, "questions": qs}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[v12-heval32-lex-B] ({i}/{len(todo)}) {aid}: {len(qs)} questions via {provider}", flush=True)
            if not qs:
                print(f"[v12-heval32-lex-B] WARNING: 0 questions for {aid}", flush=True)

    print("[v12-heval32-lex-B] done", flush=True)


if __name__ == "__main__":
    main()
