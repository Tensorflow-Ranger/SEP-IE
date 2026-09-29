"""v20c stage 1 (full scale): build round-2 branches for ALL usable groups in
the v19 pool (not just a held-out test subset like v20b_prepare_branches.py),
using 5-fold cross-validation so every group's top-3-wrong-candidate
selection is made by a probe that never saw that group during training (no
leakage) -- fold_k's ~1/5 of groups get their branches built using a probe
trained on the OTHER 4/5.

Output: data/v20_branches_full/branches_round2.json
  [{"id":..., "fold":..., "dataset_source":..., "removed_question":...,
    "removed_answer":..., "removed_tier":..., "branch_prompt":...,
    "extra_context":..., "reference_text":...}, ...]

`reference_text` is the dataset's own "what was deliberately injected" label
(v13_build_rejudge_input.py's field) -- carried forward so round-2 tiering
(v20c_branch_elicit_r2.py) can judge tier 2 as "matches the original labeled
ambiguity" the same way v14_deepseek_rejudge.py did for round 1, instead of
an ungrounded "which of my own candidates seems most important" judgment
call (see FINDINGS.md's "v20e" note for why this was a real bug: this field
was previously dropped entirely when building branches).
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
    CONCAT_TEMPLATE, JUDGED_PATH, CHUNK_DIR, V12_JUDGED_PATH, load_answers_by_question,
)

OUT_DIR = ROOT / "data" / "v20_branches_full"
OUT_PATH = OUT_DIR / "branches_round2.json"
LAYER = 22
SEED = 0
N_FOLDS = 5
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
    print(f"[v20c-prepare] loaded {len(ids)} groups", flush=True)

    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    groups = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            groups[g["id"]] = g
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}

    usable_ids = np.array([g for g in ids if len(set(per_group_tiers[g].tolist())) >= 2 and len(per_group_tiers[g]) >= 4])
    print(f"[v20c-prepare] usable groups (>=2 tiers, >=4 candidates): {len(usable_ids)}", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(usable_ids))
    folds = np.array_split(perm, N_FOLDS)

    branches = []
    n_skipped = 0
    for fold_i, test_perm_idx in enumerate(folds):
        test_fold_ids = set(usable_ids[test_perm_idx].tolist())
        train_fold_ids = [g for g in ids if g not in test_fold_ids and len(set(per_group_tiers[g].tolist())) >= 2]
        train_pools = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_fold_ids]
        pipe_k, w_k = fit_pairwise_lr(train_pools)
        print(f"[v20c-prepare] fold {fold_i}: trained probe on {len(train_fold_ids)} groups, "
              f"building branches for {len(test_fold_ids)} held-out groups", flush=True)

        for gid in sorted(test_fold_ids):
            g = groups.get(gid)
            if g is None:
                n_skipped += 1
                continue
            prompt = g["prompt"]
            answers_by_q = load_answers_by_question(gid, v12_judged)

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
                n_skipped += 1
                continue

            z = pipe_k.transform(vecs)
            scores = z @ w_k
            best_tier = tiers.max()
            wrong_idx = [i for i in range(len(tiers)) if tiers[i] != best_tier]
            wrong_idx_sorted = sorted(wrong_idx, key=lambda i: -scores[i])[:N_BRANCH]

            for i in wrong_idx_sorted:
                wrong_q = filtered[i]["question"]
                wrong_ans = filtered[i]["answer"]
                branch_prompt = CONCAT_TEMPLATE.format(prompt=prompt, answer=wrong_ans)
                branches.append({
                    "id": gid, "fold": fold_i,
                    "dataset_source": judged_rows[gid]["dataset_source"],
                    "removed_question": wrong_q, "removed_answer": wrong_ans,
                    "removed_tier": int(tiers[i]),
                    "branch_prompt": branch_prompt,
                    "extra_context": g.get("extra_context", ""),
                    "reference_text": g.get("reference_text", ""),
                })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(branches, indent=2))
    print(f"\n[v20c-prepare] wrote {len(branches)} round-2 branches ({n_skipped} groups skipped) to {OUT_PATH}",
          flush=True)


if __name__ == "__main__":
    main()
