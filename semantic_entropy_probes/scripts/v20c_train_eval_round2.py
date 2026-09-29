"""v20c stage 5: train the full-scale round-2 probe and evaluate it against
naive round-1 transfer, on a single group-disjoint 80/20 split across all
groups that have >=1 usable round-2 branch.

Output: data/v20_branches_full/round2_results.json
Also saves the trained round-2 probe pieces are kept in-memory only (this
stage is eval-focused, not for deployment) -- re-run to reproduce.
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

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
ACT_DIR = ROOT / "data" / "v20_branches_full" / "round2_activations"
OUT_PATH = ROOT / "data" / "v20_branches_full" / "round2_results.json"
LAYER = 22
SEED = 0
TEST_FRAC = 0.2


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
    n_total_branches = sum(len(v) for v in branches_by_group.values())
    print(f"[v20c-r2-eval] {len(groups_with_branches)} groups have usable round-2 branches "
          f"({n_total_branches} branches total)", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_branches))
    n_test = max(1, int(len(groups_with_branches) * TEST_FRAC))
    test_groups = set(np.array(groups_with_branches)[perm[:n_test]].tolist())
    train_groups_r2 = [g for g in groups_with_branches if g not in test_groups]

    # probe_1: trained on round-1 pools of the SAME train-side groups (fresh, consistent baseline)
    train_pools_r1 = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_groups_r2
                      if g in per_group_vecs and len(set(per_group_tiers[g].tolist())) >= 2]
    pipe1, w1 = fit_pairwise_lr(train_pools_r1)
    print(f"[v20c-r2-eval] probe_1 trained on {len(train_pools_r1)} round-1 pools", flush=True)

    train_pools_r2 = [p for g in train_groups_r2 for p in branches_by_group[g]]
    test_pools_r2 = [p for g in test_groups for p in branches_by_group[g]]
    print(f"[v20c-r2-eval] round-2 train_branches={len(train_pools_r2)} test_branches={len(test_pools_r2)} "
          f"(train_groups={len(train_groups_r2)} test_groups={len(test_groups)})", flush=True)

    pipe2, w2 = fit_pairwise_lr(train_pools_r2)

    r2_probe2_hit, n2 = hit_rate(test_pools_r2, pipe2, w2)
    r2_probe1_hit, n1 = hit_rate(test_pools_r2, pipe1, w1)
    print(f"[v20c-r2-eval] probe_2 (round-2-trained): hit_rate={r2_probe2_hit:.3f} (n={n2})", flush=True)
    print(f"[v20c-r2-eval] probe_1 (naive transfer):  hit_rate={r2_probe1_hit:.3f} (n={n1})", flush=True)

    out = {
        "n_groups_with_usable_branches": len(groups_with_branches),
        "n_total_branches": n_total_branches,
        "n_train_groups": len(train_groups_r2), "n_test_groups": len(test_groups),
        "n_train_branches": len(train_pools_r2), "n_test_branches": n2,
        "probe_2_test_hit_rate": r2_probe2_hit,
        "probe_1_naive_test_hit_rate": r2_probe1_hit,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[v20c-r2-eval] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
