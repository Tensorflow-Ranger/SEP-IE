"""Build the combined BINARY ambiguity dataset from three sources.

## Target: binary ambiguous/clear, not the K-count ordinal target

This is a deliberately different target from build_dataset.py's ClarifyCodeBench
K in {0,1,2,3}. Orchid and human-eval-comm don't carry an ambiguity *count* --
each is a clear/ambiguous pair (or clear + two ambiguous variants, for Orchid).
To combine all three into one dataset we collapse ClarifyCodeBench's K>=1 into
a single `ambiguous` class, matching the other two sources: `label` is binary,
0 = clear/unambiguous, 1 = ambiguous.

## The three sources

1. **ClarifyCodeBench** (via data/ambiguity_dataset_lcbclean.jsonl, already
   built by build_dataset.py -- run scripts/fetch_livecodebench.py first if
   it doesn't exist). Deliberately NOT data/ambiguity_dataset.jsonl (the
   default/primary file for the ordinal K-count task): that file's K=0 class
   is `modified_content` with an appended "Clarifications:\n- ..." block,
   which is a trivial lexical tell -- the literal word "Clarifications"
   appears in 100% of clear rows and 0% of ambiguous rows, and was confirmed
   (via this script's own confound tests) to produce accuracy=1.00/AUROC=1.00
   for this source alone, i.e. the classifier was reading the header, not
   judging ambiguity. ambiguity_dataset_lcbclean.jsonl's K=0 class is instead
   the genuine original LiveCodeBench problem statement (Sample I/O section
   stripped, reformatted to match ClarifyCodeBench's tight formatting) --
   real content, not a fabricated marker -- at the cost of only 80 K=0 rows
   (one per unique source problem) against 418 K>=1 rows, and a milder,
   expected length confound (the uncut original is always >= the length of
   its deletion-edited K>=1 siblings; the repo's own length-only baseline on
   this variant is close to chance, rho~=0.02 -- see README.md's baselines
   table). K=0 rows -> label 0; K in {1,2,3} rows -> label 1, collapsed.
   `source_problem_id` is namespaced `livecodebench::{question_id}`.

2. **Orchid** (SII-YDD/Orchid on HuggingFace), configs Orchid-HEval,
   Orchid-BCB, and Orchid-BCB-Expand. Only the lexical-ambiguity prompt, the
   vagueness-ambiguity prompt, and the original clean prompt are used from
   each config -- Semantic_prompt/Syntactic_prompt are deliberately NOT used.
     - Orchid-HEval: clean = `prompt` (HumanEval docstring-style). Ambiguous =
       `Lexical_prompt`, `Vagueness_prompt`. `source_problem_id` is
       `humaneval::{i}` -- the SAME namespace human-eval-comm uses below,
       since both are literally the 164 HumanEval problems; sharing the id
       keeps every variant of a given underlying problem on one side of a
       split (see src/splits.py).
     - Orchid-BCB: clean = `instruct_prompt`, NOT `complete_prompt`.
       `complete_prompt` is a code-stub-with-docstring format that neither
       Lexical_prompt nor Vagueness_prompt resemble (checked directly: both
       are rewrites of `instruct_prompt`'s natural-language instruction
       text). Pairing ambiguous rows against `complete_prompt` would
       reintroduce exactly the kind of format confound this pipeline's
       README already flags for the ClarifyCodeBench/LiveCodeBench Sample-I/O
       case: two classes drawn from differently-formatted text, trivially
       separable on format alone rather than on ambiguity content.
       `source_problem_id` is `bigcodebench::{task_id}`.
     - Orchid-BCB-Expand: a DIFFERENT, larger set of 976 BigCodeBench
       problems (task_ids 164-1139) from Orchid-BCB's 164 (task_ids 0-163) --
       zero task_id overlap, verified directly. Same `instruct_prompt` clean
       column; ambiguous columns are named `Lexical-Expand_prompt`/
       `Vagueness-Expand_prompt` (note the hyphen) rather than
       `Lexical_prompt`/`Vagueness_prompt`. Kept as its OWN dataset_source
       (`orchid_bcb_expand`) and its own `source_problem_id` namespace
       (`bigcodebench_expand::{task_id}`) rather than merged into
       `orchid_bcb` -- even though both are "BigCodeBench", treating them as
       separate sources lets the group-safe leave-one-dataset-out evaluation
       (src/combined_splits.py, scripts/confound_tests.py) test transfer
       to/from this source independently.

3. **human-eval-comm** (jie-jw-wu/human-eval-comm on GitHub) -- ONLY the
   incompleteness-modified problems (`heval_modified_incomplete.jsonl`'s
   `output` field), per the task spec ("use only the incompleteness
   prompts"). This source contributes ambiguous (label=1) rows only; no
   clear counterpart is built from it, because the clean counterpart for the
   same 164 HumanEval problems is already provided by Orchid-HEval's `prompt`
   column and shares the same `humaneval::{i}` source_problem_id -- adding a
   second, textually-identical clear copy here would just double-count that
   text without adding information. Index alignment against
   Benchmark/HumanEval.jsonl was verified byte-for-byte (see
   fetch_combined_sources.py / the exploration that preceded this script):
   all 164 rows' embedded "Original Problem Description:" matches
   HumanEval.jsonl row i's `prompt` exactly.

## Deduplication

Orchid's Lexical_prompt and Vagueness_prompt are independent single-edit
rewrites of the same clean prompt; empirically they are sometimes textually
identical for a given row (when the two ambiguity injections happened to land
on the same span/word). Exact-duplicate prompts (same text) are dropped,
keeping the first occurrence, so no single string is double-counted in the
label distribution -- this is a within-source_problem_id dedup, it never
merges rows across different source problems.

## Split-stratification key

`dataset_source` (one of clarifycodebench/orchid_heval/orchid_bcb/
humanevalcomm) is written per row specifically so a downstream split can
stratify by (label, dataset_source) jointly, not just label -- "a good split
from all three" means every source appears in both train and test, not just
that the pooled label ratio is preserved. See src/combined_splits.py.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

CCB_PATH = ROOT / "data" / "ambiguity_dataset_lcbclean.jsonl"
ORCHID_HEVAL_PATH = RAW_DIR / "orchid_heval.jsonl"
ORCHID_BCB_PATH = RAW_DIR / "orchid_bcb.jsonl"
ORCHID_BCB_EXPAND_PATH = RAW_DIR / "orchid_bcb_expand.jsonl"
HEC_INCOMPLETE_PATH = RAW_DIR / "heval_modified_incomplete.jsonl"
HUMANEVAL_PATH = RAW_DIR / "HumanEval.jsonl"

OUT_PATH = ROOT / "data" / "ambiguity_dataset_combined.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize(text: str) -> str:
    """Strip leading/trailing whitespace uniformly across every row of every
    source. Discovered via the tfidf_bow/per-source-breakdown confound tests:
    Orchid-HEval's clean 'prompt' field ends in a trailing newline right
    after the closing docstring quotes (163/164 rows) while its
    Lexical_prompt/Vagueness_prompt columns -- and human-eval-comm's
    LLM-generated incomplete rewrites -- consistently do NOT (163/164 and
    141/164 respectively), evidently because whatever pipeline generated the
    ambiguous/incomplete variants ran a trailing-whitespace strip that the
    original clean text never went through. That difference sits at the
    last token of the prompt, which is exactly
    src/extract_activations.py's extraction point -- a probe (or even a
    length-1 feature) can trivially separate the classes on that alone,
    with zero regard for content. Orchid-BCB has no such artifact
    (instruct_prompt/Lexical_prompt/Vagueness_prompt all end identically,
    with a code-fence opener, for ~164/164 rows) and is not affected either
    way by this normalization. Stripping every prompt the same way here
    removes the artifact at its root rather than leaving it for a
    downstream baseline to (partially) explain away.
    """
    return text.strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_clarifycodebench_rows() -> list[dict]:
    rows = _read_jsonl(CCB_PATH)
    out = []
    for r in rows:
        label = 0 if r["label"] == 0 else 1
        ambiguity_type = "clear" if label == 0 else f"ccb_k{r['label']}"
        out.append(
            {
                "id": f"ccb_{r['id']}",
                "dataset_source": "clarifycodebench",
                "source_problem_id": f"livecodebench::{r['source_problem_id']}",
                "prompt": _normalize(r["prompt"]),
                "label": label,
                "ambiguity_type": ambiguity_type,
            }
        )
    return out


def build_orchid_heval_rows() -> list[dict]:
    rows = _read_jsonl(ORCHID_HEVAL_PATH)
    out = []
    for r in rows:
        i = r["name"].split("/")[1]  # "HumanEval/0" -> "0"
        spid = f"humaneval::{i}"
        out.append({"id": f"orchidheval_{i}_clear", "dataset_source": "orchid_heval",
                     "source_problem_id": spid, "prompt": _normalize(r["prompt"]), "label": 0,
                     "ambiguity_type": "clear"})
        out.append({"id": f"orchidheval_{i}_lexical", "dataset_source": "orchid_heval",
                     "source_problem_id": spid, "prompt": _normalize(r["Lexical_prompt"]), "label": 1,
                     "ambiguity_type": "lexical"})
        out.append({"id": f"orchidheval_{i}_vagueness", "dataset_source": "orchid_heval",
                     "source_problem_id": spid, "prompt": _normalize(r["Vagueness_prompt"]), "label": 1,
                     "ambiguity_type": "vagueness"})
    return out


def build_orchid_bcb_rows() -> list[dict]:
    rows = _read_jsonl(ORCHID_BCB_PATH)
    out = []
    for r in rows:
        tid = r["task_id"]  # "BigCodeBench/0"
        spid = f"bigcodebench::{tid}"
        out.append({"id": f"orchidbcb_{tid.replace('/', '_')}_clear", "dataset_source": "orchid_bcb",
                     "source_problem_id": spid, "prompt": _normalize(r["instruct_prompt"]), "label": 0,
                     "ambiguity_type": "clear"})
        out.append({"id": f"orchidbcb_{tid.replace('/', '_')}_lexical", "dataset_source": "orchid_bcb",
                     "source_problem_id": spid, "prompt": _normalize(r["Lexical_prompt"]), "label": 1,
                     "ambiguity_type": "lexical"})
        out.append({"id": f"orchidbcb_{tid.replace('/', '_')}_vagueness", "dataset_source": "orchid_bcb",
                     "source_problem_id": spid, "prompt": _normalize(r["Vagueness_prompt"]), "label": 1,
                     "ambiguity_type": "vagueness"})
    return out


def build_orchid_bcb_expand_rows() -> list[dict]:
    rows = _read_jsonl(ORCHID_BCB_EXPAND_PATH)
    out = []
    for r in rows:
        tid = r["task_id"]  # "BigCodeBench/164".."BigCodeBench/1139"
        spid = f"bigcodebench_expand::{tid}"
        out.append({"id": f"orchidbcbexpand_{tid.replace('/', '_')}_clear", "dataset_source": "orchid_bcb_expand",
                     "source_problem_id": spid, "prompt": _normalize(r["instruct_prompt"]), "label": 0,
                     "ambiguity_type": "clear"})
        out.append({"id": f"orchidbcbexpand_{tid.replace('/', '_')}_lexical", "dataset_source": "orchid_bcb_expand",
                     "source_problem_id": spid, "prompt": _normalize(r["Lexical-Expand_prompt"]), "label": 1,
                     "ambiguity_type": "lexical"})
        out.append({"id": f"orchidbcbexpand_{tid.replace('/', '_')}_vagueness", "dataset_source": "orchid_bcb_expand",
                     "source_problem_id": spid, "prompt": _normalize(r["Vagueness-Expand_prompt"]), "label": 1,
                     "ambiguity_type": "vagueness"})
    return out


def build_human_eval_comm_rows() -> list[dict]:
    rows = _read_jsonl(HEC_INCOMPLETE_PATH)
    out = []
    for i, r in enumerate(rows):
        spid = f"humaneval::{i}"
        out.append({"id": f"humanevalcomm_{i}_incomplete", "dataset_source": "humanevalcomm",
                     "source_problem_id": spid, "prompt": _normalize(r["output"]), "label": 1,
                     "ambiguity_type": "incomplete"})
    return out


def dedup_within_source_problem(rows: list[dict]) -> list[dict]:
    seen: dict[tuple[str, str], None] = {}
    out = []
    n_dropped = 0
    for r in rows:
        key = (r["source_problem_id"], r["prompt"].strip())
        if key in seen:
            n_dropped += 1
            continue
        seen[key] = None
        out.append(r)
    if n_dropped:
        print(f"[build_combined_dataset] dropped {n_dropped} exact-duplicate prompts "
              f"(same source_problem_id + text)")
    return out


def main() -> None:
    if not CCB_PATH.exists():
        raise SystemExit(f"{CCB_PATH} not found -- run scripts/fetch_livecodebench.py "
                         f"then scripts/build_dataset.py first")
    for p in (ORCHID_HEVAL_PATH, ORCHID_BCB_PATH, ORCHID_BCB_EXPAND_PATH, HEC_INCOMPLETE_PATH, HUMANEVAL_PATH):
        if not p.exists():
            raise SystemExit(f"{p} not found -- run scripts/fetch_combined_sources.py first")

    all_rows = (
        build_clarifycodebench_rows()
        + build_orchid_heval_rows()
        + build_orchid_bcb_rows()
        + build_orchid_bcb_expand_rows()
        + build_human_eval_comm_rows()
    )
    all_rows = dedup_within_source_problem(all_rows)

    by_source = collections.Counter(r["dataset_source"] for r in all_rows)
    by_label = collections.Counter(r["label"] for r in all_rows)
    by_source_label = collections.Counter((r["dataset_source"], r["label"]) for r in all_rows)
    n_groups = len({r["source_problem_id"] for r in all_rows})

    print(f"[build_combined_dataset] {len(all_rows)} rows total, {n_groups} unique source_problem_id groups")
    print(f"[build_combined_dataset] by dataset_source: {dict(by_source)}")
    print(f"[build_combined_dataset] by label (0=clear,1=ambiguous): {dict(by_label)}")
    print("[build_combined_dataset] by (dataset_source, label):")
    for src in sorted(by_source):
        print(f"  {src}: clear={by_source_label[(src, 0)]} ambiguous={by_source_label[(src, 1)]}")

    _write_jsonl(OUT_PATH, all_rows)
    print(f"[build_combined_dataset] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
