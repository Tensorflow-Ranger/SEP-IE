"""v20c stage 10: train the full-scale round-3 probe and evaluate it against
naive round-1 AND naive round-2 transfer, on a single group-disjoint 80/20
split by ORIGINAL problem id (so no branch derived from a test-side problem
appears in any of the three probes' training data).

Output: data/v20_branches_full/round3_results.json
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

OUT_DIR = ROOT / "data" / "v20_branches_full"
BRANCHES_R2_PATH = OUT_DIR / "branches_round2.json"
ACT_R2_DIR = OUT_DIR / "round2_activations"
BRANCHES_R3_PATH = OUT_DIR / "branches_round3.json"
ACT_R3_DIR = OUT_DIR / "round3_activations"
OUT_PATH = OUT_DIR / "round3_results.json"
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


def load_pool_dir(branches_path, act_dir, id_field="id"):
    branches = json.loads(branches_path.read_text())
    by_group = defaultdict(list)
    for i, b in enumerate(branches):
        p = act_dir / f"{i}.npz"
        if not p.exists():
            continue
        d = np.load(p)
        li = list(d["layers"]).index(LAYER)
        vecs = d["candidates"][:, li, :]
        tiers = d["tiers"]
        by_group[b[id_field]].append((vecs, tiers))
    return by_group


def main():
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    li = layers.tolist().index(LAYER)

    r2_by_group = load_pool_dir(BRANCHES_R2_PATH, ACT_R2_DIR)
    r3_by_group = load_pool_dir(BRANCHES_R3_PATH, ACT_R3_DIR)

    groups_with_r3 = sorted(r3_by_group.keys())
    n_r3_total = sum(len(v) for v in r3_by_group.values())
    print(f"[v20c-r3-eval] {len(groups_with_r3)} groups have usable round-3 branches "
          f"({n_r3_total} branches total)", flush=True)

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_r3))
    n_test = max(1, int(len(groups_with_r3) * TEST_FRAC))
    test_groups = set(np.array(groups_with_r3)[perm[:n_test]].tolist())
    train_groups = [g for g in groups_with_r3 if g not in test_groups]

    # probe_1: round-1 pools, train-side groups only
    train_pools_r1 = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_groups
                      if g in per_group_vecs and len(set(per_group_tiers[g].tolist())) >= 2]
    pipe1, w1 = fit_pairwise_lr(train_pools_r1)

    # probe_2: round-2 pools, train-side groups only
    train_pools_r2 = [p for g in train_groups for p in r2_by_group.get(g, [])]
    pipe2, w2 = fit_pairwise_lr(train_pools_r2) if train_pools_r2 else (None, None)

    # probe_3: round-3 pools, train-side groups only
    train_pools_r3 = [p for g in train_groups for p in r3_by_group[g]]
    test_pools_r3 = [p for g in test_groups for p in r3_by_group[g]]
    print(f"[v20c-r3-eval] round-3 train_branches={len(train_pools_r3)} test_branches={len(test_pools_r3)} "
          f"(train_groups={len(train_groups)} test_groups={len(test_groups)})", flush=True)
    pipe3, w3 = fit_pairwise_lr(train_pools_r3)

    r3_probe3_hit, n3 = hit_rate(test_pools_r3, pipe3, w3)
    r3_probe1_hit, _ = hit_rate(test_pools_r3, pipe1, w1)
    out = {
        "n_groups_with_usable_r3_branches": len(groups_with_r3),
        "n_total_r3_branches": n_r3_total,
        "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
        "n_train_branches": len(train_pools_r3), "n_test_branches": n3,
        "probe_3_test_hit_rate": r3_probe3_hit,
        "probe_1_naive_test_hit_rate": r3_probe1_hit,
    }
    if pipe2 is not None:
        r3_probe2_hit, _ = hit_rate(test_pools_r3, pipe2, w2)
        out["probe_2_naive_test_hit_rate"] = r3_probe2_hit
        print(f"[v20c-r3-eval] probe_2 (naive, round-2-trained): hit_rate={r3_probe2_hit:.3f}", flush=True)

    print(f"[v20c-r3-eval] probe_3 (round-3-trained): hit_rate={r3_probe3_hit:.3f} (n={n3})", flush=True)
    print(f"[v20c-r3-eval] probe_1 (naive, round-1-trained): hit_rate={r3_probe1_hit:.3f}", flush=True)

    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[v20c-r3-eval] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
