"""Build a lexical-ambiguity counterpart to held_out_orchid32.json, using the
SAME 32 HumanEval problems (same test-set indices as build_orchid32_held_out.py)
but the Lexical_prompt/Lexical_ambiguity_explanation variant instead of
Vagueness_prompt/Vagueness_ambiguity_explanation.

Why this is a clean held-out set: v12_build_ranking_ground_truth.py excludes
training rows by HumanEval index (see load_held_out_heval32_indices()), not by
ambiguity type -- so the lexical variant of these same 32 problems was ALSO
never seen during v12 training, it just never had a benchmark harness built
around it. Confirmed all 32 indices have a complete Lexical_prompt +
Lexical_ambiguity_explanation in the raw orchid_heval.jsonl.

Output: data/v10_code_quality/held_out_orchid32_lexical.json (same schema as
held_out_orchid32.json).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_PATH = ROOT / "data" / "raw" / "orchid_heval.jsonl"
VAGUENESS_HELD_OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json"
OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_orchid32_lexical.json"


def main() -> None:
    vagueness_held_out = json.loads(VAGUENESS_HELD_OUT_PATH.read_text())
    task_ids = {v["task_id"] for v in vagueness_held_out.values()}

    rows = [json.loads(l) for l in open(RAW_PATH) if l.strip()]
    by_name = {r["name"]: r for r in rows}

    missing = task_ids - set(by_name.keys())
    assert not missing, f"held-out problems missing from raw file: {missing}"

    held_out = {}
    for name in sorted(task_ids, key=lambda n: int(n.split("/")[1])):
        r = by_name[name]
        n_str = name.split("/")[1]
        assert r.get("Lexical_prompt") and r.get("Lexical_ambiguity_explanation"), \
            f"{name}: missing Lexical_prompt/Lexical_ambiguity_explanation"
        tcs = r["test_case"]
        assert all(tc.get("relation") == "==" for tc in tcs), \
            f"{name}: expected all-'==' test cases (verified earlier), found other relation"
        held_out[f"heval_{n_str}"] = {
            "task_id": name,
            "entry_point": r["entry_point"],
            "ambiguous_prompt": r["Lexical_prompt"],
            "clear_prompt": r["prompt"],
            "missing_detail": r["Lexical_ambiguity_explanation"],
            "solution": r["solution"],
            "test_case": tcs,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(held_out, open(OUT_PATH, "w"), indent=2)
    print(f"[build_orchid32_lexical] {len(held_out)} problems -> {OUT_PATH}")


if __name__ == "__main__":
    main()
