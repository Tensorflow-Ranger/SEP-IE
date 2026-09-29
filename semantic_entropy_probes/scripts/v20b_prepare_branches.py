"""v20b stage 1 (small test): pick a handful of v19-pool groups, train a
round-1 probe on everything else, and for each test group build its round-2
BRANCH PROMPTS -- one per top-3 wrong candidate, using v19's own plain
CONCAT_TEMPLATE (`{prompt}\\n\\nClarification: {answer}`), matching how the
real live loop actually works: round 2 does NOT keep/reuse round-1's
rejected candidates, it re-elicits an ENTIRELY FRESH pool against whichever
answer got applied. So no candidate depletion/mixing here -- round 2's pool
will be 100% new candidates, elicited fresh per branch in the next stage.

Output: data/v20_branches/branches_round2.json
  [{"id":..., "dataset_source":..., "removed_question":..., "branch_prompt":...,
    "extra_context":...}, ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v19_train_ranking_probe as v19  # noqa: E402
from v19_extract_activations import (  # noqa: E402
    CONCAT_TEMPLATE, JUDGED_PATH, CHUNK_DIR, V12_JUDGED_PATH, load_answers_by_question, safe_name,
)

OUT_DIR = ROOT / "data" / "v20_branches"
OUT_PATH = OUT_DIR / "branches_round2.json"
LAYER = 22
N_TEST_GROUPS_PER_SOURCE = 13  # ~52 groups total across 4 sources (was 2 / 8 total for the small test)
N_BRANCH = 3


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fit_pairwise_lr(pools):
    all_vecs = np.concatenate([v for v, _ in pools], axis=0)
    pipe = make_pipeline(StandardScaler()).fit(all_vecs)
    X, y = [], []
    for vecs, tiers in pools:
        z = pipe.transform(vecs)
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    X.append(z[i] - z[j]); y.append(1)
                    X.append(z[j] - z[i]); y.append(0)
    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    return pipe, clf.coef_[0]


def main():
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    li = layers.tolist().index(LAYER)
    print(f"[v20b-prepare] loaded {len(ids)} groups", flush=True)

    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}

    # pick 2 test groups per source (8 total), require >=2 tiers and >=4 candidates
    # (so there's a real "wrong" pool of >=3 after excluding the best tier)
    by_source = {}
    for gid, ds in zip(ids, dataset_source):
        tiers = per_group_tiers[gid]
        if len(set(tiers.tolist())) < 2 or len(tiers) < 4:
            continue
        by_source.setdefault(ds, []).append(gid)

    test_ids = []
    for src in v19.SOURCES:
        cand = by_source.get(src, [])
        test_ids.extend(cand[:N_TEST_GROUPS_PER_SOURCE])
    test_ids = set(test_ids)
    print(f"[v20b-prepare] test groups ({len(test_ids)}): {sorted(test_ids)}", flush=True)

    train_ids = [g for g in ids if g not in test_ids and len(set(per_group_tiers[g].tolist())) >= 2]
    train_pools = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_ids]
    pipe1, w1 = fit_pairwise_lr(train_pools)
    print(f"[v20b-prepare] round1 probe trained on {len(train_ids)} groups", flush=True)

    branches = []
    for gid in sorted(test_ids):
        g = groups.get(gid)
        if g is None:
            print(f"[v20b-prepare] {gid}: missing from chunks, skipping", flush=True)
            continue
        prompt = g["prompt"]
        answers_by_q = load_answers_by_question(gid, v12_judged)

        # rebuild the SAME filtered/ordered candidate list v19_extract_activations.py used,
        # so index i here matches row i of per_group_vecs[gid]
        filtered = []
        for c in judged_rows[gid]["judged"]["candidates"]:
            q = c["question"].strip()
            ans = answers_by_q.get(q)
            if ans is None:
                continue
            filtered.append({"question": q, "answer": ans, "tier": c["tier"]})
        vecs = per_group_vecs[gid][li]
        tiers = per_group_tiers[gid]
        if len(filtered) != len(tiers):
            print(f"[v20b-prepare] {gid}: alignment mismatch ({len(filtered)} vs {len(tiers)}), skipping", flush=True)
            continue

        z = pipe1.transform(vecs)
        scores = z @ w1
        best_tier = tiers.max()
        wrong_idx = [i for i in range(len(tiers)) if tiers[i] != best_tier]
        wrong_idx_sorted = sorted(wrong_idx, key=lambda i: -scores[i])[:N_BRANCH]

        for i in wrong_idx_sorted:
            wrong_q = filtered[i]["question"]
            wrong_ans = filtered[i]["answer"]
            branch_prompt = CONCAT_TEMPLATE.format(prompt=prompt, answer=wrong_ans)
            branches.append({
                "id": gid,
                "dataset_source": judged_rows[gid]["dataset_source"],
                "removed_question": wrong_q,
                "removed_answer": wrong_ans,
                "removed_tier": int(tiers[i]),
                "branch_prompt": branch_prompt,
                "extra_context": g.get("extra_context", ""),
            })
        print(f"[v20b-prepare] {gid}: {len(wrong_idx_sorted)} branches built", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(branches, indent=2))
    print(f"\n[v20b-prepare] wrote {len(branches)} round-2 branches to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
