"""Scale-up: elicit candidates for the Semantic variant on 10 fresh
orchid_heval rows + 10 fresh orchid_bcb rows (none previously touched this
session), to get a less tiny (n~20 vs n~3-4) read on Semantic hit rate.

Usage: python v12_semantic10_elicit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.v12_bcb_elicit import elicit_questions_via_api  # noqa: E402

HEVAL_PATH = ROOT / "data" / "raw" / "orchid_heval.jsonl"
BCB_PATH = ROOT / "data" / "raw" / "orchid_bcb.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "semantic10_candidates.jsonl"

HEVAL_IDS = ["HumanEval/5", "HumanEval/6", "HumanEval/14", "HumanEval/16", "HumanEval/18",
             "HumanEval/19", "HumanEval/20", "HumanEval/21", "HumanEval/22", "HumanEval/23"]
BCB_IDS = ["BigCodeBench/6", "BigCodeBench/7", "BigCodeBench/8", "BigCodeBench/9", "BigCodeBench/10",
           "BigCodeBench/11", "BigCodeBench/12", "BigCodeBench/13", "BigCodeBench/14", "BigCodeBench/15"]


def main() -> None:
    heval = {r["name"]: r for r in (json.loads(l) for l in HEVAL_PATH.open())}
    bcb = {r["task_id"]: r for r in (json.loads(l) for l in BCB_PATH.open())}

    out_rows = []
    for source, ids, table, id_key in [
        ("orchid_heval", HEVAL_IDS, heval, "name"),
        ("orchid_bcb", BCB_IDS, bcb, "task_id"),
    ]:
        for pid in ids:
            r = table[pid]
            prompt = r["Semantic_prompt"]
            explanation = r["Semantic_ambiguity_explanation"]
            qs, provider = elicit_questions_via_api(prompt, 8)
            print(f"[sem10-elicit] {source} {pid} via {provider}: {len(qs)} questions", flush=True)
            for i, q in enumerate(qs):
                print(f"    {i}. {q}", flush=True)
            out_rows.append({
                "id": f"{source}_{pid.replace('/', '_')}_semantic",
                "source": source,
                "task_id": pid,
                "prompt": prompt,
                "true_explanation": explanation,
                "questions": qs,
                "provider": provider,
            })

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[sem10-elicit] saved {len(out_rows)} rows to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
