"""v27: quantify (not just hand-inspect) whether orchid_bcb_expand's labeled
ambiguities are (a) functionally important and (b) the MOST important
candidate present, using fields DeepSeek's judging pass already computed for
every row of `data/v13_rejudge/deepseek_judged.jsonl`
(`label_functionally_relevant`, `more_important_than_label`,
per-candidate `functionally_relevant`) -- no new LLM calls. Also quantifies
the "long free-form prompt -> small-fraction word edit" hidden-state-dilution
hypothesis (`## A fourth source` / `## v19 control` above) directly from
`data/raw/{orchid_heval,orchid_bcb,orchid_bcb_expand}.jsonl`, which had only
ever been asserted qualitatively.

This is the "natural next step" flagged at the end of the `## v21/v22/v23`
section: "diffing candidate answers against canonical-solution behavior...
would let this be quantified per-problem rather than demonstrated
case-by-case."

Output: prints all tables used in the FINDINGS.md follow-up write-up.
"""
from __future__ import annotations

import difflib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def label_importance_table():
    judged = read_jsonl(ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl")
    per_src = defaultdict(list)
    for row in judged:
        src = row.get("dataset_source")
        if not src:
            continue
        per_src[src].append(row["judged"])

    print("\n=== Per-source label-importance stats (from deepseek_judged.jsonl) ===")
    header = (
        f"{'source':22s} {'n':>5s} {'leaked':>7s} {'label_rel':>10s} "
        f"{'has_mit':>8s} {'label_only_rel':>15s} {'other_rel_tier1':>16s} {'clean_strict':>13s}"
    )
    print(header)
    for src, rows in per_src.items():
        n = len(rows)
        leaked = sum(bool(j.get("label_leaked")) for j in rows)
        label_rel = sum(j.get("label_functionally_relevant") is True for j in rows)
        has_mit = sum(j.get("more_important_than_label") is not None for j in rows)
        other_rel_counts = []
        label_only_rel = 0
        clean_strict = 0
        for j in rows:
            cands = j.get("candidates", [])
            n_tier1_rel = sum(1 for c in cands if c.get("tier") == 1 and c.get("functionally_relevant"))
            other_rel_counts.append(n_tier1_rel)
            if j.get("label_functionally_relevant") and n_tier1_rel == 0:
                label_only_rel += 1
            if (
                not j.get("label_leaked")
                and j.get("label_functionally_relevant") is True
                and j.get("more_important_than_label") is None
            ):
                clean_strict += 1
        print(
            f"{src:22s} {n:5d} {leaked/n:7.3f} {label_rel/n:10.3f} "
            f"{has_mit/n:8.3f} {label_only_rel/n:15.3f} {statistics.mean(other_rel_counts):16.2f} "
            f"{clean_strict/n:13.3f}"
        )
    print(
        "\nColumns: leaked=label_leaked rate; label_rel=label_functionally_relevant==True "
        "rate; has_mit=some OTHER candidate judged more important than the label; "
        "label_only_rel=fraction of rows where the label is the ONLY functionally-relevant "
        "tier-1/2 candidate (no competing relevant detail at all); other_rel_tier1=mean count "
        "of OTHER functionally-relevant tier-1 candidates per row; clean_strict=fraction "
        "passing v19's (not leaked & label_rel & no mit) filter."
    )


def edit_fraction_table():
    print("\n=== Prompt-length / edit-fraction by source (word-level SequenceMatcher) ===")

    def edit_stats(rows, clean_key, amb_keys):
        lens, fracs = [], []
        for r in rows:
            clean = r.get(clean_key) or ""
            if not clean:
                continue
            for k in amb_keys:
                amb = r.get(k)
                if not amb:
                    continue
                ratio = difflib.SequenceMatcher(None, clean.split(), amb.split()).ratio()
                lens.append(len(clean.split()))
                fracs.append(1 - ratio)
        return lens, fracs

    heval = read_jsonl(ROOT / "data" / "raw" / "orchid_heval.jsonl")
    bcb = read_jsonl(ROOT / "data" / "raw" / "orchid_bcb.jsonl")
    bcbx = read_jsonl(ROOT / "data" / "raw" / "orchid_bcb_expand.jsonl")

    configs = [
        ("orchid_heval", heval, "prompt", ["Semantic_prompt", "Vagueness_prompt"]),
        ("orchid_bcb", bcb, "instruct_prompt", ["Semantic_prompt", "Vagueness_prompt"]),
        (
            "orchid_bcb_expand",
            bcbx,
            "instruct_prompt",
            ["Semantic-Expand_prompt", "Vagueness-Expand_prompt"],
        ),
    ]
    print(f"{'source':22s} {'n':>5s} {'mean_words':>11s} {'mean_edit_frac':>15s} {'median_edit_frac':>17s}")
    for name, rows, clean_key, keys in configs:
        lens, fracs = edit_stats(rows, clean_key, keys)
        print(
            f"{name:22s} {len(lens):5d} {statistics.mean(lens):11.1f} "
            f"{statistics.mean(fracs):15.4f} {statistics.median(fracs):17.4f}"
        )
    print(
        "\nNote: orchid_bcb/orchid_bcb_expand must be diffed against `instruct_prompt`, "
        "NOT `complete_prompt` (BigCodeBench's code-format docstring) -- the ambiguous "
        "`{Semantic,Vagueness}[-Expand]_prompt` fields are written in instruct-prompt "
        "free-text style. Diffing against complete_prompt (the wrong pairing) inflates "
        "the apparent edit fraction to ~50% by comparing two different templates, not "
        "the actual injected edit."
    )


if __name__ == "__main__":
    label_importance_table()
    edit_fraction_table()
