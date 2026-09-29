"""Same 16 double-held-out K=1 prompts and same redundant/missing bullet
text as scripts/score_rank1_rank2_full_testset.py, but with every literal
occurrence of the word "Constraints" removed:

  - The base prompt's own pre-existing "Constraints" header line is
    stripped (its bullet list is kept, just unlabeled).
  - The redundant/missing bullet is appended as a bare trailing "- ..."
    line, never introducing a new "Constraints" header.

Purpose: scripts/score_rank1_rank2_full_testset.py found that redundant and
missing edits move rank1/rank2's P(ambiguous) by statistically
indistinguishable amounts at n=16 -- both edits nudge the score down a
little regardless of content, suggesting a structural/stylistic tell (e.g.
"another bullet under Constraints reads as more fully-specified") rather
than a real ambiguity-content signal. Since "Constraints" appears verbatim
in baseline/redundant/missing alike (not a differential confound by itself)
this isn't a clean test of that hypothesis on its own, but removing the
word throughout (base text included) checks whether that literal token is
part of what the probe is keying on independent of the bullet-list
structure itself.

Source prompts/bullets are pulled directly from the three existing dataset
files (not re-authored) to keep the comparison apples-to-apples.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "detail_addition_probe_test_v5_noheader.jsonl"

SOURCE_FILES = [
    ROOT / "data" / "detail_addition_probe_test_v2.jsonl",
    ROOT / "data" / "detail_addition_probe_test_v3_extra.jsonl",
    ROOT / "data" / "detail_addition_probe_test_v4_extra.jsonl",
]

BASE_IDS = [
    "amb_task_4", "amb_task_26", "amb_task_49", "amb_task_87", "amb_task_118",
    "amb_task_198", "amb_task_212", "amb_task_279", "amb_task_321", "amb_task_417",
    "amb_task_142", "amb_task_282", "amb_task_360",
    "amb_task_28", "amb_task_125", "amb_task_346",
]


def load_source_rows() -> dict:
    rows = {}
    for f in SOURCE_FILES:
        for line in f.open():
            r = json.loads(line)
            rows[r["id"]] = r
    return rows


def strip_constraints_header(prompt: str) -> str:
    assert prompt.count("Constraints") == 1, prompt
    return prompt.replace("Constraints\n", "", 1)


def bare_bullet_variant(base_no_header: str, bullet: str) -> str:
    return base_no_header + "\n- " + bullet


def main() -> None:
    src = load_source_rows()
    rows = []
    for base in BASE_IDS:
        b = src[f"{base}__baseline"]
        r = src[f"{base}__redundant"]
        m = src[f"{base}__missing"]

        base_prompt_no_header = strip_constraints_header(b["prompt"])
        # Recover the exact bullet text that was appended in v2/v3/v4 by
        # diffing the redundant/missing prompt against the (header-having)
        # baseline prompt -- both were built as base + "\n- {bullet}".
        redundant_bullet = r["prompt"][len(b["prompt"]):].lstrip("\n").lstrip("- ")
        missing_bullet = m["prompt"][len(b["prompt"]):].lstrip("\n").lstrip("- ")

        common = dict(base_id=base, source_problem_id=b["source_problem_id"], ambiguity_types=b["ambiguity_types"])
        rows.append(dict(common, id=f"{base}__baseline", variant="baseline", label=1, prompt=base_prompt_no_header))
        rows.append(dict(common, id=f"{base}__redundant", variant="redundant", label=1,
                          prompt=bare_bullet_variant(base_prompt_no_header, redundant_bullet)))
        rows.append(dict(common, id=f"{base}__missing", variant="missing", label=0,
                          prompt=bare_bullet_variant(base_prompt_no_header, missing_bullet)))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_bad = sum(1 for row in rows if "Constraints" in row["prompt"])
    print(f"[build_v5] wrote {len(rows)} rows -> {OUT_PATH} ('Constraints' occurrences remaining: {n_bad})")


if __name__ == "__main__":
    main()
