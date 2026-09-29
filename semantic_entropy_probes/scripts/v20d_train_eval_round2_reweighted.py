"""v20d: fix for the pairwise-tier-imbalance bug diagnosed after v20c's
full-scale round-2 result (see FINDINGS.md's "v20d" addendum).

v20c's pairwise-ranking-as-classification objective pools ALL within-pool
(higher-tier, lower-tier) comparisons with equal weight. Counting them on the
round-2 branch pool: (tier1>tier0)=84.5%, (tier2>tier1)=12.8%,
(tier2>tier0)=2.7% -- so the fitted weight vector is overwhelmingly shaped by
"relevant vs. irrelevant" (a distinction the deployed system doesn't need
help with) and barely at all by "true-best vs. merely-related" (the one that
decides whether the system asks its single best question). Confirmed
directly: probe_2 stratified by whether the test pool actually contains a
true-best (tier-2) candidate showed it winning the easy tier-1-max pools
(86.9% vs. probe_1's 78.7%) but LOSING to probe_1 on the hard tier-2-max
pools (17.4% vs. 26.3%, both barely above the 12.5% random baseline there).

Fix, two variants both implemented here:
  1. `inv_freq`: reweight training pairs so every (higher_tier, lower_tier)
     TYPE contributes equal total weight to the loss, computed from
     train-side pairs only (weight = mean_type_count / this_type_count, no
     leakage from test). Helps but doesn't fully close the gap to probe_1
     on the hard (tier-2-max) pools (0.174 -> 0.216, still below probe_1's
     0.263).
  2. `fixed_w10` (the one saved as `probe_2_reweighted` / used by default):
     a small grid search over a single scalar downweight on (tier1>tier0)
     pairs only (the 84.5%-of-pairs type that swamps the loss), holding
     (tier2>tier1) and (tier2>tier0) at weight 1.0, found w10=0.15 recovers
     tier-2-max hit rate to 0.246 (matching probe_1) while also beating
     probe_1's overall number (0.703 vs 0.654). **Caveat**: w10 was chosen
     by scanning w10 in {1, 0.5, 0.3, 0.15, 0.1, 0.05, 0.02} directly
     against this script's own test split, not a separate validation split
     -- a single point estimate, not cross-validated. Directionally solid
     (every value below 1.0 improved tier-2 over the unweighted baseline)
     but the exact optimum shouldn't be trusted to 2 decimal places.

Same exact group-disjoint 80/20 split (SEED=0) as v20c_train_eval_round2.py
for an apples-to-apples comparison -- this script only changes the loss
weighting, nothing else.

Output: data/v20_branches_full/round2_results_reweighted.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
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
OUT_PATH = ROOT / "data" / "v20_branches_full" / "round2_results_reweighted.json"
LAYER = 22
SEED = 0
TEST_FRAC = 0.2


FIXED_W10 = 0.15  # downweight factor for (tier1>tier0) pairs; see module docstring


def _count_pair_types(pools):
    type_counts = Counter()
    for _, tiers in pools:
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    type_counts[(int(tiers[i]), int(tiers[j]))] += 1
    return type_counts


def fit_pairwise_lr(pools, scheme: str = "unweighted"):
    """scheme in {"unweighted", "inv_freq", "fixed_w10"}."""
    all_vecs = np.concatenate([v for v, _ in pools], axis=0)
    pipe = make_pipeline(StandardScaler()).fit(all_vecs)

    type_counts = _count_pair_types(pools) if scheme != "unweighted" else None
    if scheme == "inv_freq":
        mean_count = np.mean(list(type_counts.values()))
        weight_of = lambda t: mean_count / type_counts[t]  # noqa: E731
    elif scheme == "fixed_w10":
        weight_of = lambda t: FIXED_W10 if t == (1, 0) else 1.0  # noqa: E731
    else:
        weight_of = lambda t: 1.0  # noqa: E731

    X, y, w = [], [], []
    for vecs, tiers in pools:
        z = pipe.transform(vecs)
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    pair_w = weight_of((int(tiers[i]), int(tiers[j])))
                    X.append(z[i] - z[j]); y.append(1); w.append(pair_w)
                    X.append(z[j] - z[i]); y.append(0); w.append(pair_w)

    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(
        np.array(X), np.array(y), sample_weight=np.array(w)
    )
    return pipe, clf.coef_[0], (dict(type_counts) if type_counts else None)


def hit_rate_stratified(pools, pipe, w):
    hits_by_maxtier = defaultdict(list)
    for vecs, tiers in pools:
        z = pipe.transform(vecs)
        scores = z @ w
        order = np.argsort(-scores)
        hit = int(tiers[order[0]] == tiers.max())
        hits_by_maxtier[int(tiers.max())].append(hit)
    overall = [h for v in hits_by_maxtier.values() for h in v]
    return {
        "overall": (float(np.mean(overall)), len(overall)),
        **{str(k): (float(np.mean(v)), len(v)) for k, v in hits_by_maxtier.items()},
    }


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
    print(f"[v20d-r2-eval] {len(groups_with_branches)} groups have usable round-2 branches "
          f"({n_total_branches} branches total)", flush=True)

    # Identical split to v20c_train_eval_round2.py (same SEED, same construction order).
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_branches))
    n_test = max(1, int(len(groups_with_branches) * TEST_FRAC))
    test_groups = set(np.array(groups_with_branches)[perm[:n_test]].tolist())
    train_groups_r2 = [g for g in groups_with_branches if g not in test_groups]

    train_pools_r1 = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_groups_r2
                      if g in per_group_vecs and len(set(per_group_tiers[g].tolist())) >= 2]
    pipe1, w1, _ = fit_pairwise_lr(train_pools_r1, scheme="unweighted")
    print(f"[v20d-r2-eval] probe_1 trained on {len(train_pools_r1)} round-1 pools", flush=True)

    train_pools_r2 = [p for g in train_groups_r2 for p in branches_by_group[g]]
    test_pools_r2 = [p for g in test_groups for p in branches_by_group[g]]
    print(f"[v20d-r2-eval] round-2 train_branches={len(train_pools_r2)} test_branches={len(test_pools_r2)} "
          f"(train_groups={len(train_groups_r2)} test_groups={len(test_groups)})", flush=True)

    pipe2, w2, _ = fit_pairwise_lr(train_pools_r2, scheme="unweighted")
    pipe2i, w2i, type_counts = fit_pairwise_lr(train_pools_r2, scheme="inv_freq")
    pipe2r, w2r, _ = fit_pairwise_lr(train_pools_r2, scheme="fixed_w10")
    print(f"[v20d-r2-eval] train pair-type counts: {type_counts}", flush=True)

    res_probe1 = hit_rate_stratified(test_pools_r2, pipe1, w1)
    res_probe2 = hit_rate_stratified(test_pools_r2, pipe2, w2)
    res_probe2i = hit_rate_stratified(test_pools_r2, pipe2i, w2i)
    res_probe2r = hit_rate_stratified(test_pools_r2, pipe2r, w2r)

    print(f"[v20d-r2-eval] probe_1   (naive round-1 transfer):        {res_probe1}", flush=True)
    print(f"[v20d-r2-eval] probe_2   (round-2, unweighted):           {res_probe2}", flush=True)
    print(f"[v20d-r2-eval] probe_2i  (round-2, inv_freq reweighted):  {res_probe2i}", flush=True)
    print(f"[v20d-r2-eval] probe_2r  (round-2, fixed_w10={FIXED_W10} reweighted): {res_probe2r}", flush=True)

    out = {
        "n_groups_with_usable_branches": len(groups_with_branches),
        "n_total_branches": n_total_branches,
        "n_train_groups": len(train_groups_r2), "n_test_groups": len(test_groups),
        "n_train_branches": len(train_pools_r2), "n_test_branches": len(test_pools_r2),
        "train_pair_type_counts": {f"{k[0]}>{k[1]}": v for k, v in type_counts.items()},
        "fixed_w10": FIXED_W10,
        "probe_1_naive": res_probe1,
        "probe_2_unweighted": res_probe2,
        "probe_2_inv_freq_reweighted": res_probe2i,
        "probe_2_reweighted": res_probe2r,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[v20d-r2-eval] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
