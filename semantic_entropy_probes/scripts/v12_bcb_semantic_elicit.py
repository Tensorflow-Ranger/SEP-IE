"""Same as v12_bcb_elicit.py but for Semantic_prompt instead of
Lexical_prompt/Vagueness_prompt -- fills the gap noted mid-session: we
checked Semantic_prompt for leaks (read-only) but never ran it through the
actual elicit->judge->select pipeline the way Lexical/Vagueness were.

Usage: python v12_bcb_semantic_elicit.py [--n 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v12_bcb_elicit import elicit_questions_via_api  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "orchid_bcb.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "bcb_semantic_candidates.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_candidates", type=int, default=8)
    args = ap.parse_args()

    rows = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows = rows[: args.n]

    out_rows = []
    for r in rows:
        prompt = r["Semantic_prompt"]
        explanation = r["Semantic_ambiguity_explanation"]
        qs, provider = elicit_questions_via_api(prompt, args.max_candidates)
        print(f"[bcb-sem-elicit] {r['task_id']} via {provider}: {len(qs)} questions", flush=True)
        for i, q in enumerate(qs):
            print(f"    {i}. {q}", flush=True)
        out_rows.append({
            "id": f"{r['task_id'].replace('/', '_')}_semantic",
            "task_id": r["task_id"],
            "kind": "Semantic",
            "prompt": prompt,
            "true_explanation": explanation,
            "questions": qs,
            "provider": provider,
        })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[bcb-sem-elicit] saved {len(out_rows)} rows to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
