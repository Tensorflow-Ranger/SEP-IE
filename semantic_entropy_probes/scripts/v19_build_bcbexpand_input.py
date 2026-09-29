"""v19 stage 1: build rejudge-input chunks for orchid_bcb_expand (the
"bcb-extend" source, 976 BigCodeBench/164-1139 problems, zero group overlap
with orchid_bcb), mirroring v13_build_rejudge_input.py's treatment of
orchid_bcb exactly -- same two ambiguity types (semantic, vagueness; lexical
and syntactic were never brought into the ranking pipeline for orchid_bcb
either, so this keeps the new source consistent with its sibling rather than
introducing new scope), needs_elicitation=True for all rows (never touched by
any prior stage), own `bigcodebench_expand::{task_id}` source_problem_id
namespace so it can never leak into another source's CV group.

Output: appends new chunk_20..chunk_39.json files into the SAME
data/v13_rejudge/chunks/ directory used by v13/v14, so v14_deepseek_rejudge.py
picks these up unmodified on its next (resumable) run -- no changes needed to
that script.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
N_NEW_CHUNKS = 20
CHUNK_OFFSET = 20  # v13 already wrote chunk_0..chunk_19


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main():
    groups = []
    for r in read_jsonl(ROOT / "data" / "raw" / "orchid_bcb_expand.jsonl"):
        for amb_type, prompt_key, expl_key in [
            ("semantic", "Semantic-Expand_prompt", "Semantic-Expand_ambiguity_explanation"),
            ("vagueness", "Vagueness-Expand_prompt", "Vagueness-Expand_ambiguity_explanation"),
        ]:
            expl = r.get(expl_key)
            amb_prompt = r.get(prompt_key)
            if not expl or not amb_prompt:
                continue
            ref_text = (
                f"What was made ambiguous, and why:\n{expl}\n\n"
                f"Original unambiguous version (complete_prompt):\n{r.get('complete_prompt', '')[:2000]}"
            )
            extra = (
                f"Reference canonical solution:\n{r.get('canonical_solution', '')[:2000]}\n\n"
                f"Test code (unittest-style):\n{r.get('test', '')[:2000]}"
            )
            groups.append({
                "id": f"bcbexpand_{r['task_id']}_{amb_type}",
                "dataset_source": "orchid_bcb_expand",
                "ambiguity_type": amb_type,
                "needs_elicitation": True,
                "prompt": amb_prompt,
                "reference_text": ref_text,
                "extra_context": extra,
                "candidates": [],
            })

    print(f"[v19-build] total orchid_bcb_expand groups: {len(groups)}")
    import collections
    print(collections.Counter((g["dataset_source"], g["ambiguity_type"]) for g in groups))

    for i in range(N_NEW_CHUNKS):
        chunk = groups[i::N_NEW_CHUNKS]
        out_path = CHUNK_DIR / f"chunk_{CHUNK_OFFSET + i}.json"
        out_path.write_text(json.dumps(chunk, indent=1))
        print(f"  {out_path.name}: {len(chunk)} groups")


if __name__ == "__main__":
    main()
