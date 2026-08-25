"""Build the canonical paired ambiguity dataset.

## Target: ambiguity COUNT (K in {0,1,2,3}), not a binary flag

Each ClarifyCodeBench task has 1-3 annotated ambiguity points (K = number of
qa_pairs). The probing target is this count, not a collapsed "any ambiguity
present" flag -- K=1 and K=3 are genuinely different amounts of unresolved
specification, and treating them identically throws away information the
benchmark already gives us for free. `label` is the ordinal target throughout
this pipeline; the old binary `n_ambiguities`-as-side-info field is gone,
since it would just duplicate `label`.

## Why the K=0 class is reconstructed from ClarifyCodeBench itself

The first version of this script paired each `modified_content` (ambiguous)
against the corresponding *original, unedited* LiveCodeBench
`question_content` (K=0), matched by `question_id`. That has a fatal
confound: LiveCodeBench statements almost always include "Sample Input /
Sample Output" sections, and ClarifyCodeBench's annotators apparently work
from a rendering that drops them when producing `modified_content` --
checking the actual data, 1/419 ambiguous prompts contain "Sample Input",
vs. 609/1055 LiveCodeBench statements overall (and essentially all of the 80
relevant ones, since they're the same problems). A probe -- or even a
bag-of-words baseline -- could hit near-perfect accuracy by detecting the
"Sample Input" header rather than anything about ambiguity. It was also
lopsided across K: 419 rewrites share only 80 source problems, so pairing by
`question_id` gives 80 K=0 examples against 199/169/51 for K=1/2/3.

The data card's own description of the benchmark gives a better construction:
"[e]ach edited task is underspecified initially but becomes fully specified
once the annotated clarification answers are given." So the K=0 class here is
`modified_content` with its own annotated QA-pair answers appended -- i.e.
ClarifyCodeBench's own definition of "resolved." This:
  - gives one K=0 example per ambiguous row (419, not 80),
  - keeps every class in the same provenance/formatting family (all derive
    from `modified_content`; none have Sample I/O sections),
  - introduces a smaller, expected confound of its own: K=0 prompts are
    always at least as long as their source row (answers are appended, never
    removed) and carry a "Clarifications:" header the K>=1 rows never have.
    This is exactly what the token-length baseline in train_probes.py is for.

The LiveCodeBench-question_id-paired version (with the Sample I/O confound,
and K collapsed to a binary "has LiveCodeBench full-text pair" flag) is still
written out, clearly namespaced, as `ambiguity_dataset_lcbfull.jsonl` -- for
reference / secondary comparison only. Nothing downstream reads it by
default.

## A third variant: LiveCodeBench-paired, with Sample I/O stripped

`ambiguity_dataset_lcbclean.jsonl` fixes the Sample I/O confound directly
instead of avoiding LiveCodeBench text altogether: `clean_full_text()` cuts
each `question_content` at its first "Sample Input" line and collapses the
blank lines LiveCodeBench puts after each section header, which -- checked
against all 80 relevant AtCoder problems -- reproduces ClarifyCodeBench's own
tight formatting (no blank line between a header like "Input"/"Output" and
its content) essentially exactly; the only remaining differences are the
actual deleted content, which is the point. This still gives only 80 K=0
examples (one per unique source problem, vs. 419 for K=1-3), and K=0 prompts
here are the ORIGINAL, uncut problem statements -- necessarily at least as
long as their post-deletion K>=1 siblings, since deletion only removes text.
That's a different, milder length confound than the Sample I/O or
Clarifications-header ones; check the length-only baseline on this variant's
results before trusting its layer-wise numbers.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLARIFY_PATH = ROOT.parent / "ClarifyCodeBench" / "data" / "ClarifyCodeBench.jsonl"
LCB_PATH = ROOT / "data" / "livecodebench_v6_questions.jsonl"

OUT_PRIMARY = ROOT / "data" / "ambiguity_dataset.jsonl"
OUT_LCB_SECONDARY = ROOT / "data" / "ambiguity_dataset_lcbfull.jsonl"
OUT_LCB_CLEAN = ROOT / "data" / "ambiguity_dataset_lcbclean.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _reconstruct_full(modified_content: str, qa_pairs: list[dict]) -> str:
    clarifications = "\n".join(f"- {qa['answer']}" for qa in qa_pairs)
    return f"{modified_content}\n\nClarifications:\n{clarifications}"


def clean_full_text(text: str) -> str:
    """Strip LiveCodeBench's Sample Input/Output section and reformat to
    match ClarifyCodeBench's tight (no-blank-line) formatting. See module
    docstring; validated against all 80 relevant AtCoder problems.
    """
    m = re.search(r"\nSample Input\b", text)
    if m:
        text = text[: m.start()]
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def build_primary(clarify_rows: list[dict]) -> list[dict]:
    out_rows: list[dict] = []
    for r in clarify_rows:
        qid = r["question_id"]
        k = len(r["qa_pairs"])
        out_rows.append(
            {
                "id": f"amb_{r['task_id']}",
                "source_problem_id": qid,
                "prompt": r["modified_content"],
                "label": k,
                "ambiguity_types": r["keywords"],
            }
        )
        out_rows.append(
            {
                "id": f"full_{r['task_id']}",
                "source_problem_id": qid,
                "prompt": _reconstruct_full(r["modified_content"], r["qa_pairs"]),
                "label": 0,
                "ambiguity_types": [],
            }
        )
    return out_rows


def build_lcb_secondary(clarify_rows: list[dict], lcb_by_id: dict[str, dict]) -> tuple[list[dict], set[str]]:
    """Question_id-paired version, kept for reference. See module docstring
    for why this is NOT the primary dataset (Sample I/O confound + lopsided K).
    """
    out_rows: list[dict] = []
    missing_qids: set[str] = set()
    seen_qids_for_negatives: set[str] = set()

    for r in clarify_rows:
        qid = r["question_id"]
        k = len(r["qa_pairs"])
        out_rows.append(
            {
                "id": f"amb_{r['task_id']}",
                "source_problem_id": qid,
                "prompt": r["modified_content"],
                "label": k,
                "ambiguity_types": r["keywords"],
            }
        )
        if qid not in lcb_by_id:
            missing_qids.add(qid)
        elif qid not in seen_qids_for_negatives:
            out_rows.append(
                {
                    "id": f"full_{qid}",
                    "source_problem_id": qid,
                    "prompt": lcb_by_id[qid]["question_content"],
                    "label": 0,
                    "ambiguity_types": [],
                }
            )
            seen_qids_for_negatives.add(qid)

    return out_rows, missing_qids


def build_lcb_clean(clarify_rows: list[dict], lcb_by_id: dict[str, dict]) -> tuple[list[dict], set[str]]:
    """Question_id-paired version with Sample I/O stripped from the K=0 text.
    See module docstring ("A third variant") for the cleaning method and its
    remaining (milder) length confound.
    """
    out_rows: list[dict] = []
    missing_qids: set[str] = set()
    seen_qids_for_negatives: set[str] = set()

    for r in clarify_rows:
        qid = r["question_id"]
        k = len(r["qa_pairs"])
        out_rows.append(
            {
                "id": f"amb_{r['task_id']}",
                "source_problem_id": qid,
                "prompt": r["modified_content"],
                "label": k,
                "ambiguity_types": r["keywords"],
            }
        )
        if qid not in lcb_by_id:
            missing_qids.add(qid)
        elif qid not in seen_qids_for_negatives:
            out_rows.append(
                {
                    "id": f"full_{qid}",
                    "source_problem_id": qid,
                    "prompt": clean_full_text(lcb_by_id[qid]["question_content"]),
                    "label": 0,
                    "ambiguity_types": [],
                }
            )
            seen_qids_for_negatives.add(qid)

    return out_rows, missing_qids


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    clarify_rows = _read_jsonl(CLARIFY_PATH)

    primary_rows = build_primary(clarify_rows)
    k_dist = collections.Counter(r["label"] for r in primary_rows)
    n_groups = len({r["source_problem_id"] for r in primary_rows})
    print(f"[build_dataset] PRIMARY: {len(primary_rows)} rows, K distribution "
          f"{dict(sorted(k_dist.items()))}, {n_groups} unique source problems")
    _write_jsonl(OUT_PRIMARY, primary_rows)
    print(f"[build_dataset] wrote {OUT_PRIMARY}")

    if LCB_PATH.exists():
        lcb_rows = _read_jsonl(LCB_PATH)
        lcb_by_id = {r["question_id"]: r for r in lcb_rows}
        secondary_rows, missing_qids = build_lcb_secondary(clarify_rows, lcb_by_id)
        k_dist2 = collections.Counter(r["label"] for r in secondary_rows)
        print(f"[build_dataset] SECONDARY (LiveCodeBench-paired, reference only): "
              f"{len(secondary_rows)} rows, K distribution {dict(sorted(k_dist2.items()))}, "
              f"{len(missing_qids)} question_ids missing from LiveCodeBench dump")
        _write_jsonl(OUT_LCB_SECONDARY, secondary_rows)
        print(f"[build_dataset] wrote {OUT_LCB_SECONDARY}")

        clean_rows, missing_qids3 = build_lcb_clean(clarify_rows, lcb_by_id)
        k_dist3 = collections.Counter(r["label"] for r in clean_rows)
        print(f"[build_dataset] LCBCLEAN (LiveCodeBench-paired, Sample I/O stripped): "
              f"{len(clean_rows)} rows, K distribution {dict(sorted(k_dist3.items()))}, "
              f"{len(missing_qids3)} question_ids missing from LiveCodeBench dump")
        _write_jsonl(OUT_LCB_CLEAN, clean_rows)
        print(f"[build_dataset] wrote {OUT_LCB_CLEAN}")
    else:
        print(f"[build_dataset] {LCB_PATH} not found, skipping secondary/lcbclean LiveCodeBench-paired "
              f"datasets (run scripts/fetch_livecodebench.py first if you want them)")


if __name__ == "__main__":
    main()
