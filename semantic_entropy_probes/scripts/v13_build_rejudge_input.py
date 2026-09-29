"""v13: build input chunks for haiku-based rejudging.

Two scopes:
  1. EXISTING v12 train/test groups (635: 378 ccb + 257 orchid_heval
     lexical/vagueness), reusing already-elicited candidates from
     data/v12_ranking/candidates.jsonl -- these just need re-judging,
     grounded in the real solution+tests where available (orchid_heval),
     or the full qa_pairs list (ccb, already the ground truth).
  2. NEW rows never in v12: orchid_heval Semantic type, and orchid_bcb
     Semantic + Vagueness types -- these need BOTH candidate elicitation
     AND judging, since v12 never touched bcb and never used Semantic.

Held-out exclusions reused from v12 (held_out_3/40.json for ccb,
held_out_orchid32.json for heval) so this stays consistent with what
v12 actually trained/tested on. orchid_bcb has no existing holdout list
in this repo, so all 164 rows are included for the new bcb types.

Output: data/v13_rejudge/chunks/chunk_{i}.json, each a list of group dicts:
  {id, dataset_source, ambiguity_type, needs_elicitation, prompt,
   reference_text, candidates (list of question strings, or [] if
   needs_elicitation)}
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "v13_rejudge"
CHUNK_DIR = OUT_DIR / "chunks"

N_CHUNKS = 20


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_held_out_ccb():
    ids = set()
    for f in ("held_out_3.json", "held_out_40.json"):
        d = json.loads((ROOT / "data" / "v10_code_quality" / f).read_text())
        for row in d.values():
            ids.add(row["task_id"])
    return ids


def load_held_out_heval():
    d = json.loads((ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json").read_text())
    return {row["task_id"].split("/")[1] for row in d.values()}


def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    groups = []

    # ---- Scope 1: existing v12 groups, reusing elicited candidates ----
    gt = {r["id"]: r for r in read_jsonl(ROOT / "data" / "v12_ranking" / "ground_truth.jsonl")}
    cand_by_id = {r["id"]: r for r in read_jsonl(ROOT / "data" / "v12_ranking" / "candidates.jsonl")}
    heval_raw = {r["name"]: r for r in read_jsonl(ROOT / "data" / "raw" / "orchid_heval.jsonl")}

    n_missing_cand = 0
    for rid, row in gt.items():
        cand_row = cand_by_id.get(rid)
        if cand_row is None:
            n_missing_cand += 1
            continue
        if row["dataset_source"] == "clarifycodebench":
            ref = row["reference"]
            ref_text = "\n".join(
                f"Annotated key Q{i}: {qa['question']}\nAnnotated key A{i}: {qa['answer']}"
                for i, qa in enumerate(ref["qa_pairs"], 1)
            )
            extra = "(No reference solution/tests available for this source -- ClarifyCodeBench hidden tests are not redistributed. Judge tier only against whether the candidate matches one of the annotated key Q/A pairs above.)"
        else:
            pid = row["source_problem_id"].split("::")[1]
            raw = heval_raw.get(f"HumanEval/{pid}")
            ref = row["reference"]
            ref_text = f"What was made ambiguous, and why:\n{ref['explanation']}\n\nOriginal unambiguous version:\n{ref['clean_prompt']}"
            if raw:
                extra = f"Reference solution:\n{raw['solution']}\n\nTest cases (input/expected output pairs):\n{json.dumps(raw['test_case'])[:2000]}"
            else:
                extra = "(No reference solution/tests found -- judge against the explanation only.)"
        groups.append({
            "id": rid,
            "dataset_source": row["dataset_source"],
            "ambiguity_type": row.get("ambiguity_type", ""),
            "needs_elicitation": False,
            "prompt": row["prompt"],
            "reference_text": ref_text,
            "extra_context": extra,
            "candidates": cand_row["questions"],
        })

    # ---- Scope 2: NEW rows -- orchid_heval Semantic ----
    held_out_heval = load_held_out_heval()
    for r in read_jsonl(ROOT / "data" / "raw" / "orchid_heval.jsonl"):
        i = r["name"].split("/")[1]
        if i in held_out_heval:
            continue
        expl = r.get("Semantic_ambiguity_explanation")
        sem_prompt = r.get("Semantic_prompt")
        if not expl or not sem_prompt:
            continue
        ref_text = f"What was made ambiguous, and why:\n{expl}\n\nOriginal unambiguous version:\n{r['prompt']}"
        extra = f"Reference solution:\n{r['solution']}\n\nTest cases (input/expected output pairs):\n{json.dumps(r['test_case'])[:2000]}"
        groups.append({
            "id": f"orchidheval_{i}_semantic",
            "dataset_source": "orchid_heval",
            "ambiguity_type": "semantic",
            "needs_elicitation": True,
            "prompt": sem_prompt,
            "reference_text": ref_text,
            "extra_context": extra,
            "candidates": [],
        })

    # ---- Scope 2: NEW rows -- orchid_bcb Semantic + Vagueness ----
    for r in read_jsonl(ROOT / "data" / "raw" / "orchid_bcb.jsonl"):
        for amb_type, prompt_key, expl_key in [
            ("semantic", "Semantic_prompt", "Semantic_ambiguity_explanation"),
            ("vagueness", "Vagueness_prompt", "Vagueness_ambiguity_explanation"),
        ]:
            expl = r.get(expl_key)
            amb_prompt = r.get(prompt_key)
            if not expl or not amb_prompt:
                continue
            ref_text = f"What was made ambiguous, and why:\n{expl}\n\nOriginal unambiguous version (complete_prompt):\n{r.get('complete_prompt', '')[:2000]}"
            extra = f"Reference canonical solution:\n{r.get('canonical_solution', '')[:2000]}\n\nTest code (unittest-style):\n{r.get('test', '')[:2000]}"
            groups.append({
                "id": f"bcb_{r['task_id']}_{amb_type}",
                "dataset_source": "orchid_bcb",
                "ambiguity_type": amb_type,
                "needs_elicitation": True,
                "prompt": amb_prompt,
                "reference_text": ref_text,
                "extra_context": extra,
                "candidates": [],
            })

    print(f"[v13-build] total groups: {len(groups)} ({n_missing_cand} v12 rows had no elicited candidates, skipped)")
    import collections
    print(collections.Counter((g["dataset_source"], g["ambiguity_type"]) for g in groups))

    # split into N_CHUNKS roughly equal chunks, existing-vs-new mixed evenly
    # so no single agent gets all-elicitation or all-judging-only work
    for i in range(N_CHUNKS):
        chunk = groups[i::N_CHUNKS]
        (CHUNK_DIR / f"chunk_{i}.json").write_text(json.dumps(chunk, indent=1))
        n_elicit = sum(g["needs_elicitation"] for g in chunk)
        print(f"  chunk_{i}: {len(chunk)} groups ({n_elicit} need elicitation)")


if __name__ == "__main__":
    main()
