"""v12 stage A: join every ambiguous training row (label==1) in
data/ambiguity_dataset_ccb_heval.jsonl -- the exact set the deployed rank1/
rank2 probes were trained on -- to its GROUND-TRUTH reference for what the
actual injected ambiguity is, recovered from each source's original
annotations:

  - clarifycodebench: `id` is "ccb_amb_{task_id}" (see build_dataset.py's
    `f"amb_{r['task_id']}"`, prefixed with "ccb_" by
    build_ccb_heval_subset.py's parent build_combined_dataset.py). Joins 1:1
    against ../../ClarifyCodeBench/data/ClarifyCodeBench.jsonl by task_id,
    which carries the original annotated `qa_pairs` (1-3 {question, answer}
    pairs -- literally the gold clarifying Q&A that resolves this row's
    ambiguity). That repo was re-cloned this session; it was empty on disk
    (the file the combined-dataset build originally read from is gone, only
    the label/count derived from it survived into the training jsonl).

  - orchid_heval: `id` is "orchidheval_{i}_{lexical|vagueness}" (see
    build_combined_dataset.py). Joins against data/raw/orchid_heval.jsonl by
    `name == f"HumanEval/{i}"`, pulling the matching `{Type}_ambiguity_explanation`
    (a free-text description of what was made ambiguous and why) plus the
    original clean `prompt`, for oracle-answering context.

Output: data/v12_ranking/ground_truth.jsonl, one row per ambiguous training
example, ready for stage B (local Qwen3-8B candidate elicitation) and stage C
(oracle-answer + tier judging).
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CCB_HEVAL_PATH = ROOT / "data" / "ambiguity_dataset_ccb_heval.jsonl"
CCB_RAW_PATH = ROOT.parent / "ClarifyCodeBench" / "data" / "ClarifyCodeBench.jsonl"
ORCHID_HEVAL_RAW_PATH = ROOT / "data" / "raw" / "orchid_heval.jsonl"
OUT_PATH = ROOT / "data" / "v12_ranking" / "ground_truth.jsonl"

# Held-out EVAL benchmarks that must never appear in training data -- held_out_3/40.json
# are ClarifyCodeBench's held-out eval tasks (held_out_3 is a subset of held_out_40's 40
# task_ids); held_out_orchid32.json is the 32 HumanEval problems the live heval32/
# heval32_control/heval32_api_elicit pipeline runs (and this whole investigation's
# ambiguity_type_ranking.json results) are evaluated on. An earlier version of this
# script didn't filter these out -- 104/739 rows (40 ccb + 64 orchid_heval, since each
# HumanEval problem yields a lexical + a vagueness row) were found to overlap directly
# with these benchmarks and would have silently invalidated any future heval32 evaluation
# of the new ranking probe.
HELD_OUT_DIR = ROOT / "data" / "v10_code_quality"


def load_held_out_ccb_task_ids() -> set[str]:
    ids = set()
    for fname in ("held_out_3.json", "held_out_40.json"):
        d = json.loads((HELD_OUT_DIR / fname).read_text())
        for row in d.values():
            ids.add(row["task_id"])  # e.g. "task_4"
    return ids


def load_held_out_heval32_indices() -> set[str]:
    d = json.loads((HELD_OUT_DIR / "held_out_orchid32.json").read_text())
    return {row["task_id"].split("/")[1] for row in d.values()}  # "HumanEval/1" -> "1"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_ccb_lookup() -> dict[str, dict]:
    """task_id -> raw ClarifyCodeBench row (has qa_pairs)."""
    rows = _read_jsonl(CCB_RAW_PATH)
    return {r["task_id"]: r for r in rows}


def build_orchid_heval_lookup() -> dict[str, dict]:
    """HumanEval index (str) -> raw orchid_heval row."""
    rows = _read_jsonl(ORCHID_HEVAL_RAW_PATH)
    out = {}
    for r in rows:
        i = r["name"].split("/")[1]
        out[i] = r
    return out


def main() -> None:
    train_rows = [r for r in _read_jsonl(CCB_HEVAL_PATH) if r["label"] == 1]
    ccb_lookup = build_ccb_lookup()
    oh_lookup = build_orchid_heval_lookup()
    held_out_ccb_task_ids = load_held_out_ccb_task_ids()
    held_out_heval32_indices = load_held_out_heval32_indices()

    out_rows = []
    missing = []
    excluded_eval_leak = []
    for r in train_rows:
        if r["dataset_source"] == "clarifycodebench":
            # id is "ccb_amb_{task_id}" -> recover task_id
            task_id = r["id"][len("ccb_amb_"):]
            if task_id in held_out_ccb_task_ids:
                excluded_eval_leak.append(r["id"])
                continue
            src = ccb_lookup.get(task_id)
            if src is None:
                missing.append(r["id"])
                continue
            reference = {"qa_pairs": src["qa_pairs"]}
        elif r["dataset_source"] == "orchid_heval":
            # id is "orchidheval_{i}_{lexical|vagueness}"
            body = r["id"][len("orchidheval_"):]
            i, amb_type = body.rsplit("_", 1)
            if i in held_out_heval32_indices:
                excluded_eval_leak.append(r["id"])
                continue
            src = oh_lookup.get(i)
            if src is None:
                missing.append(r["id"])
                continue
            type_key = amb_type.capitalize()  # "lexical" -> "Lexical"
            explanation = src.get(f"{type_key}_ambiguity_explanation")
            if explanation is None:
                missing.append(r["id"])
                continue
            reference = {"explanation": explanation, "clean_prompt": src["prompt"]}
        else:
            missing.append(r["id"])
            continue

        out_rows.append(
            {
                "id": r["id"],
                "dataset_source": r["dataset_source"],
                "source_problem_id": r["source_problem_id"],
                "ambiguity_type": r["ambiguity_type"],
                "prompt": r["prompt"],
                "reference": reference,
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    by_source = {}
    for row in out_rows:
        by_source[row["dataset_source"]] = by_source.get(row["dataset_source"], 0) + 1
    print(f"[v12-A] {len(out_rows)}/{len(train_rows)} rows joined to ground truth -> {OUT_PATH}")
    print(f"[v12-A] by source: {by_source}")
    print(f"[v12-A] excluded {len(excluded_eval_leak)} rows that overlap held_out_3/40.json "
          f"(ccb) or held_out_orchid32.json (heval32) eval benchmarks")
    if missing:
        print(f"[v12-A] WARNING: {len(missing)} rows had no ground truth match: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}")


if __name__ == "__main__":
    main()
