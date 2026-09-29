"""v20: curriculum-combinatorial round-specific probes, built ENTIRELY offline
from the existing v19 activation pool (data/v19_concat_activations/activations,
1229 groups, no new Bedrock/GPU calls) -- an alternative to the real multi-hop
re-elicitation pilot (v20_pilot_multihop.py) and the shared-probe cross-round
SupCon attempt (v20_train_round_supcon.py), both of which hit data-scarcity
(tier-2 candidates dry up fast under real re-elicitation) or underperformed a
plain linear probe (too little data for the SupCon encoder).

Idea (from the user): round depth = POOL DEPLETION, not new elicitation.
Train a separate probe per round:
  - probe_1: plain pairwise-ranking probe on each group's full candidate pool
    (identical to the existing v15/v19 baseline).
  - round-2 branches: for every group (train AND test), score its full pool
    with probe_1, take the top-3 WRONG (non-best-tier) candidates by score
    (probe_1's most-confident wrong guesses -- "curriculum"), and for each of
    the 3, build a branch = pool minus that one candidate (pretend it was
    already tried and rejected in round 1). This is done for EVERY group
    regardless of whether probe_1's actual top-1 pick was already correct --
    pure data augmentation, not a literal deployment trace ("combinatory").
    3 branches/group.
  - probe_2: trained on TRAIN groups' round-2 branches.
  - round-3 branches: for every round-2 branch, score its (already-depleted)
    pool with probe_2, take ITS top-3 wrong picks, remove each -> 3
    sub-branches per branch = 9 per original group.
  - probe_3: trained on TRAIN groups' round-3 branches.

Evaluation: one group-disjoint 80/20 split by original problem id, used for
ALL rounds' branches (a branch's parent id determines its split side, so no
branch derived from a test-split problem ever appears in any round's training
pool). For each round k, report probe_k's test hit rate on round-k TEST
branches AGAINST probe_1's hit rate on those same round-k TEST branches --
the direct test of the user's hypothesis (does round-1's probe get worse as
the pool gets depleted, and does a dedicated probe_k recover it).

Usage: python -m scripts.v20_curriculum_round_probes
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

OUT_PATH = ROOT / "data" / "v20_pilot_multihop" / "curriculum_round_results.json"
LAYER = 22
SEED = 0
TEST_FRAC = 0.2
N_BRANCH = 3


def group_disjoint_split(ids: np.ndarray, seed: int, test_frac: float):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ids))
    n_test = int(len(ids) * test_frac)
    test_idx = set(perm[:n_test].tolist())
    train_mask = np.array([i not in test_idx for i in range(len(ids))])
    return ids[train_mask], ids[~train_mask]


def fit_pairwise_lr(pools):
    """pools: list of (vecs (n_cand, dim), tiers (n_cand,))."""
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


def score_pool(pipe, w, vecs):
    return pipe.transform(vecs) @ w


def hit(tiers, scores):
    order = np.argsort(-scores)
    return int(tiers[order[0]] == tiers.max())


def hit_rate(pools, pipe, w):
    return float(np.mean([hit(tiers, score_pool(pipe, w, vecs)) for vecs, tiers in pools]))


def build_next_round_branches(parent_pools, pipe, w):
    """parent_pools: list of (parent_id, vecs, tiers). Returns list of
    (parent_id, vecs_subset, tiers_subset) -- up to N_BRANCH branches per
    parent, each with one additional "already-tried" wrong candidate removed."""
    branches = []
    for parent_id, vecs, tiers in parent_pools:
        if len(tiers) < 2 or len(set(tiers.tolist())) < 2:
            continue
        best_tier = tiers.max()
        wrong_idx = [i for i in range(len(tiers)) if tiers[i] != best_tier]
        if not wrong_idx:
            continue
        scores = score_pool(pipe, w, vecs)
        wrong_idx_sorted = sorted(wrong_idx, key=lambda i: -scores[i])
        for remove_i in wrong_idx_sorted[:N_BRANCH]:
            keep = [i for i in range(len(tiers)) if i != remove_i]
            if len(set(tiers[keep].tolist())) < 2:
                continue
            branches.append((parent_id, vecs[keep], tiers[keep]))
    return branches


def main():
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    li = layers.tolist().index(LAYER)
    print(f"[v20-curriculum] loaded {len(ids)} groups, layer={LAYER}", flush=True)

    train_ids, test_ids = group_disjoint_split(ids, SEED, TEST_FRAC)
    print(f"[v20-curriculum] split: train={len(train_ids)} test={len(test_ids)}", flush=True)

    def pool_of(gid):
        return per_group_vecs[gid][li], per_group_tiers[gid]

    train_pools_r1 = [pool_of(g) for g in train_ids
                      if len(set(per_group_tiers[g].tolist())) >= 2]
    test_pools_r1 = [pool_of(g) for g in test_ids
                     if len(set(per_group_tiers[g].tolist())) >= 2]
    print(f"[v20-curriculum] round1 usable: train={len(train_pools_r1)} test={len(test_pools_r1)}", flush=True)

    # --- round 1 ---
    pipe1, w1 = fit_pairwise_lr(train_pools_r1)
    r1_test_hit = hit_rate(test_pools_r1, pipe1, w1)
    print(f"[v20-curriculum] probe_1 on round1 test: hit_rate={r1_test_hit:.3f} (n={len(test_pools_r1)})", flush=True)

    results = {"round1": {"probe_1_on_own_round": {"hit_rate": r1_test_hit, "n": len(test_pools_r1)}}}

    # --- round 2 branches (train + test, both scored by probe_1) ---
    train_parents_r1 = [(g, *pool_of(g)) for g in train_ids if len(set(per_group_tiers[g].tolist())) >= 2]
    test_parents_r1 = [(g, *pool_of(g)) for g in test_ids if len(set(per_group_tiers[g].tolist())) >= 2]

    train_branches_r2 = build_next_round_branches(train_parents_r1, pipe1, w1)
    test_branches_r2 = build_next_round_branches(test_parents_r1, pipe1, w1)
    print(f"[v20-curriculum] round2 branches: train={len(train_branches_r2)} test={len(test_branches_r2)}", flush=True)

    train_pools_r2 = [(v, t) for _, v, t in train_branches_r2]
    test_pools_r2 = [(v, t) for _, v, t in test_branches_r2]

    pipe2, w2 = fit_pairwise_lr(train_pools_r2)
    r2_probe2_hit = hit_rate(test_pools_r2, pipe2, w2)
    r2_probe1_hit = hit_rate(test_pools_r2, pipe1, w1)
    print(f"[v20-curriculum] round2 test: probe_2={r2_probe2_hit:.3f} probe_1(naive)={r2_probe1_hit:.3f} "
          f"(n={len(test_pools_r2)})", flush=True)
    results["round2"] = {
        "probe_2_on_own_round": {"hit_rate": r2_probe2_hit, "n": len(test_pools_r2)},
        "probe_1_naive_transfer": {"hit_rate": r2_probe1_hit, "n": len(test_pools_r2)},
    }

    # --- round 3 branches (from round2 branches, scored by probe_2) ---
    train_branches_r3 = build_next_round_branches(train_branches_r2, pipe2, w2)
    test_branches_r3 = build_next_round_branches(test_branches_r2, pipe2, w2)
    print(f"[v20-curriculum] round3 branches: train={len(train_branches_r3)} test={len(test_branches_r3)}", flush=True)

    train_pools_r3 = [(v, t) for _, v, t in train_branches_r3]
    test_pools_r3 = [(v, t) for _, v, t in test_branches_r3]

    pipe3, w3 = fit_pairwise_lr(train_pools_r3)
    r3_probe3_hit = hit_rate(test_pools_r3, pipe3, w3)
    r3_probe1_hit = hit_rate(test_pools_r3, pipe1, w1)
    r3_probe2_hit = hit_rate(test_pools_r3, pipe2, w2)
    print(f"[v20-curriculum] round3 test: probe_3={r3_probe3_hit:.3f} probe_2(naive)={r3_probe2_hit:.3f} "
          f"probe_1(naive)={r3_probe1_hit:.3f} (n={len(test_pools_r3)})", flush=True)
    results["round3"] = {
        "probe_3_on_own_round": {"hit_rate": r3_probe3_hit, "n": len(test_pools_r3)},
        "probe_2_naive_transfer": {"hit_rate": r3_probe2_hit, "n": len(test_pools_r3)},
        "probe_1_naive_transfer": {"hit_rate": r3_probe1_hit, "n": len(test_pools_r3)},
    }

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[v20-curriculum] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
