"""v20b stage 5: train a round-2-specific probe on the 50-problem branch set
(data/v20_branches/round2_activations) and compare it against probe_1
(trained on the rest of the v19 pool, same as v20b_prepare_branches.py)
applied naively to the same held-out branches -- the actual test of whether
round-specific training helps, at a scale big enough to trust (unlike the
n=20 small test).

Group-disjoint split: branches are split into train/test by their ORIGINAL
problem id (not by branch), so all of a group's 1-3 branches stay on the
same side -- no leakage of near-duplicate pools across the split.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v19_train_ranking_probe as v19  # noqa: E402

BRANCHES_PATH = ROOT / "data" / "v20_branches" / "branches_round2.json"
ACT_DIR = ROOT / "data" / "v20_branches" / "round2_activations"
LAYER = 22
SEED = 0
TEST_FRAC = 0.25


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


def hit_rate(pools, pipe, w):
    hits = []
    for vecs, tiers in pools:
        z = pipe.transform(vecs)
        scores = z @ w
        order = np.argsort(-scores)
        hits.append(int(tiers[order[0]] == tiers.max()))
    return float(np.mean(hits)), len(pools)


def main():
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    li = layers.tolist().index(LAYER)

    branches = json.loads(BRANCHES_PATH.read_text())
    test_group_ids = sorted(set(b["id"] for b in branches))
    print(f"[v20b-train-eval] 50-problem set: {len(test_group_ids)} groups", flush=True)

    # probe_1: trained on everything else in the v19 pool (unchanged from v20b_prepare_branches.py)
    train_ids_probe1 = [g for g in ids if g not in set(test_group_ids)
                        and len(set(per_group_tiers[g].tolist())) >= 2]
    pools_probe1 = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_ids_probe1]
    pipe1, w1 = fit_pairwise_lr(pools_probe1)
    print(f"[v20b-train-eval] probe_1 trained on {len(train_ids_probe1)} groups (rest of v19 pool)", flush=True)

    # load round-2 branch pools, grouped by original problem id
    branches_by_group = defaultdict(list)
    for i, b in enumerate(branches):
        npz_path = ACT_DIR / f"{i}.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
        tiers = d["tiers"]
        branches_by_group[b["id"]].append((vecs, tiers))

    groups_with_branches = sorted(branches_by_group.keys())
    print(f"[v20b-train-eval] {len(groups_with_branches)} groups have >=1 usable round-2 branch "
          f"({sum(len(v) for v in branches_by_group.values())} branches total)", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_branches))
    n_test = max(1, int(len(groups_with_branches) * TEST_FRAC))
    test_groups = set(np.array(groups_with_branches)[perm[:n_test]].tolist())
    train_groups = [g for g in groups_with_branches if g not in test_groups]
    print(f"[v20b-train-eval] group-disjoint split: train_groups={len(train_groups)} "
          f"test_groups={len(test_groups)}", flush=True)

    train_pools_r2 = [p for g in train_groups for p in branches_by_group[g]]
    test_pools_r2 = [p for g in test_groups for p in branches_by_group[g]]
    print(f"[v20b-train-eval] train_branches={len(train_pools_r2)} test_branches={len(test_pools_r2)}", flush=True)

    pipe2, w2 = fit_pairwise_lr(train_pools_r2)

    r2_probe2_hit, n2 = hit_rate(test_pools_r2, pipe2, w2)
    r2_probe1_hit, n1 = hit_rate(test_pools_r2, pipe1, w1)
    print(f"\n[v20b-train-eval] === held-out round-2 branches (n={n2}) ===", flush=True)
    print(f"[v20b-train-eval] probe_2 (round-2-trained): hit_rate={r2_probe2_hit:.3f}", flush=True)
    print(f"[v20b-train-eval] probe_1 (naive transfer):  hit_rate={r2_probe1_hit:.3f}", flush=True)

    out = {
        "n_groups_50problem_set": len(test_group_ids),
        "n_groups_with_usable_branches": len(groups_with_branches),
        "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
        "n_train_branches": len(train_pools_r2), "n_test_branches": n2,
        "probe_2_test_hit_rate": r2_probe2_hit,
        "probe_1_naive_test_hit_rate": r2_probe1_hit,
    }
    out_path = ROOT / "data" / "v20_branches" / "round2_50problem_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[v20b-train-eval] saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
