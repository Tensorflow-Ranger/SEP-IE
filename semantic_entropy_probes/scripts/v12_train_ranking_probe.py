"""v12 stage E: train a probe on the GROUPED ranking objective instead of the
deployed rank1/rank2 probes' i.i.d. binary (ambiguous/clear) objective.

Why: the deployed probes are scored, at inference time, by ranking up to 8
same-problem candidate clarifications against each other and picking the
top-scoring one (v10_round_worker.py / v11_heval_batch_worker.py) -- but they
were never trained on that relative, within-group task, only on an absolute
clear-vs-ambiguous decision boundary. See [[heval32-probe-ranking-investigation]].

This trains via the standard pairwise-ranking-as-classification reduction
(RankSVM/RankNet-style): for every within-group pair of candidates with
different tiers (2=correct/matches the true injected ambiguity,
1=similar/related-but-not-the-actual-gap, 0=irrelevant/generic -- assigned by
Claude acting as oracle+judge in stage C, grounded only in each row's true
qa_pairs/ambiguity_explanation), build a training example from the feature
DIFFERENCE of the higher- and lower-tier candidate's hidden state, label 1;
and the reverse difference, label 0. A `fit_intercept=False` logistic
regression on these differences learns a single linear scoring direction w
such that w.x_i ranks candidates the same way the pairwise labels do (w.(xi
- xj) > 0 iff tier_i > tier_j) -- so at inference the SAME w just needs one
dot product per candidate, not a pairwise comparison.

Splits are group-disjoint by source_problem_id (reusing
src/combined_splits.py, the same machinery the deployed probes' train/test
split used) so no candidate set for the same underlying coding problem
leaks across train/val/test. Layers are swept (see v12_extract_activations.py
for why the deployed layers 20/22 aren't assumed to still be best) and picked
by validation rank-1 hit rate.

Usage: python v12_train_ranking_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.combined_splits import (  # noqa: E402
    make_combined_grouped_cv_folds,
    make_combined_train_val_test_split,
    report_split_composition,
)

GROUND_TRUTH_PATH = ROOT / "data" / "v12_ranking" / "ground_truth.jsonl"
ACT_DIR = ROOT / "data" / "v12_ranking" / "activations"
OUT_DIR = ROOT / "models"
PCA_DIM = 100
SEED = 0

# Layer selection: a single val split (~120 groups) is too small to reliably
# rank 6 layers -- resampling the split seed alone swings the "best" layer's
# held-out hit rate by >10 points. Pick the layer by grouped, repeated k-fold
# CV within the train+val pool only (test stays untouched), then do one final
# fit at that layer for the deployed probe + reporting.
CV_N_SPLITS = 4
CV_N_REPEATS = 3


def load_groups():
    ground_truth = {}
    with GROUND_TRUTH_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                ground_truth[row["id"]] = row

    ids, dataset_source, source_problem_id = [], [], []
    layers = None
    per_group_vecs = {}  # id -> (n_layers, n_candidates, dim)
    per_group_tiers = {}  # id -> (n_candidates,)

    for npz_path in sorted(ACT_DIR.glob("*.npz")):
        gid = npz_path.stem
        if gid not in ground_truth:
            continue
        data = np.load(npz_path)
        if layers is None:
            layers = data["layers"]
        cand = data["candidates"]  # (n_candidates, n_layers, dim)
        tiers = data["tiers"]
        if len(tiers) < 2 or len(set(tiers.tolist())) < 2:
            continue  # need at least 2 distinct tiers to form a pair in this group
        per_group_vecs[gid] = np.transpose(cand, (1, 0, 2))  # (n_layers, n_candidates, dim)
        per_group_tiers[gid] = tiers
        ids.append(gid)
        dataset_source.append(ground_truth[gid]["dataset_source"])
        source_problem_id.append(ground_truth[gid]["source_problem_id"])

    return (
        np.array(ids), np.array(dataset_source), np.array(source_problem_id),
        layers, per_group_vecs, per_group_tiers,
    )


def build_pairwise(group_ids, per_group_vecs, per_group_tiers, layer_idx):
    """All within-group ordered pairs (higher tier, lower tier) -> (diff, label=1)
    plus the mirrored (lower - higher, label=0), for one layer."""
    X, y = [], []
    for gid in group_ids:
        vecs = per_group_vecs[gid][layer_idx]  # (n_candidates, dim)
        tiers = per_group_tiers[gid]
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    X.append(vecs[i] - vecs[j])
                    y.append(1)
                    X.append(vecs[j] - vecs[i])
                    y.append(0)
    return np.array(X), np.array(y)


def eval_ranking(group_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, w):
    """rank-1 hit rate (does the top-scored candidate have the max tier in its
    group?) and mean rank (1-indexed) of a max-tier candidate, per group."""
    hits, ranks = [], []
    for gid in group_ids:
        vecs = per_group_vecs[gid][layer_idx]
        tiers = per_group_tiers[gid]
        z = pipe.transform(vecs)
        scores = z @ w
        order = np.argsort(-scores)  # descending
        best_tier = tiers.max()
        top_idx = order[0]
        hits.append(int(tiers[top_idx] == best_tier))
        rank_of_best = 1 + int(np.where(tiers[order] == best_tier)[0][0])
        ranks.append(rank_of_best)
    return float(np.mean(hits)), float(np.mean(ranks)), len(group_ids)


def fit_layer_probe(train_ids, per_group_vecs, per_group_tiers, layer_idx):
    """Fit the StandardScaler+PCA feature pipeline and the pairwise-difference
    logistic regression scoring direction w for one layer, on one set of
    groups."""
    train_vecs_pool = np.concatenate([per_group_vecs[g][layer_idx] for g in train_ids], axis=0)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=min(PCA_DIM, train_vecs_pool.shape[0] - 1), random_state=SEED))
    pipe.fit(train_vecs_pool)

    X, y = [], []
    for g in train_ids:
        z = pipe.transform(per_group_vecs[g][layer_idx])
        tiers = per_group_tiers[g]
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    X.append(z[i] - z[j]); y.append(1)
                    X.append(z[j] - z[i]); y.append(0)
    X, y = np.array(X), np.array(y)

    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
    clf.fit(X, y)
    return pipe, clf.coef_[0]


def cv_select_best_layer(layers, per_group_vecs, per_group_tiers, trainval_idx, ids,
                          dataset_source, source_problem_id, labels_dummy):
    """Grouped, repeated k-fold CV over the train+val pool only (never touches
    test) to pick the layer whose mean fold-held-out hit rate is highest.
    Returns {layer: mean_cv_hit_rate}."""
    cv_hits = {layer: [] for layer in layers}
    for repeat in range(CV_N_REPEATS):
        fold_seed = SEED * 1000 + repeat
        for fold_train_sub, fold_val_sub in make_combined_grouped_cv_folds(
            labels_dummy[trainval_idx], dataset_source[trainval_idx], source_problem_id[trainval_idx],
            n_splits=CV_N_SPLITS, seed=fold_seed,
        ):
            fold_train_ids = ids[trainval_idx][fold_train_sub]
            fold_val_ids = ids[trainval_idx][fold_val_sub]
            for li, layer in enumerate(layers):
                pipe, w = fit_layer_probe(fold_train_ids, per_group_vecs, per_group_tiers, li)
                hit, _, _ = eval_ranking(fold_val_ids, per_group_vecs, per_group_tiers, li, pipe, w)
                cv_hits[layer].append(hit)
    cv_mean = {layer: float(np.mean(hits)) for layer, hits in cv_hits.items()}
    for layer in layers:
        print(f"[v12-E] CV layer={layer}: mean hit_rate={cv_mean[layer]:.3f} "
              f"over {len(cv_hits[layer])} folds ({CV_N_REPEATS} repeats x {CV_N_SPLITS} splits)", flush=True)
    return cv_mean


def main() -> None:
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = load_groups()
    print(f"[v12-E] {len(ids)} usable groups (>=2 distinct tiers) out of activations available", flush=True)
    if len(ids) < 20:
        print("[v12-E] too few usable groups to train/eval meaningfully yet -- exiting", flush=True)
        return

    labels_dummy = np.ones(len(ids), dtype=np.int64)  # all rows are "ambiguous" here; stratify by source only
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    report_split_composition("train", train_idx, labels_dummy, dataset_source)
    report_split_composition("val", val_idx, labels_dummy, dataset_source)
    report_split_composition("test", test_idx, labels_dummy, dataset_source)

    train_ids, val_ids, test_ids = ids[train_idx], ids[val_idx], ids[test_idx]
    trainval_idx = np.concatenate([train_idx, val_idx])

    print(f"[v12-E] running {CV_N_REPEATS}x{CV_N_SPLITS}-fold grouped CV over the "
          f"train+val pool ({len(trainval_idx)} groups) to pick the layer...", flush=True)
    cv_mean = cv_select_best_layer(layers.tolist(), per_group_vecs, per_group_tiers, trainval_idx,
                                    ids, dataset_source, source_problem_id, labels_dummy)
    cv_best_layer = max(cv_mean, key=cv_mean.get)
    print(f"[v12-E] CV-selected layer={cv_best_layer} (mean CV hit_rate={cv_mean[cv_best_layer]:.3f})", flush=True)

    results = []
    for li, layer in enumerate(layers.tolist()):
        pipe, w = fit_layer_probe(train_ids, per_group_vecs, per_group_tiers, li)

        val_hit, val_rank, n_val = eval_ranking(val_ids, per_group_vecs, per_group_tiers, li, pipe, w)
        test_hit, test_rank, n_test = eval_ranking(test_ids, per_group_vecs, per_group_tiers, li, pipe, w)
        print(f"[v12-E] layer={layer}: cv hit_rate={cv_mean[layer]:.3f} | "
              f"val hit_rate={val_hit:.3f} mean_rank={val_rank:.2f} (n={n_val}) | "
              f"test hit_rate={test_hit:.3f} mean_rank={test_rank:.2f} (n={n_test})", flush=True)
        results.append({
            "layer": layer, "cv_hit_rate": cv_mean[layer],
            "val_hit_rate": val_hit, "val_mean_rank": val_rank, "n_val": n_val,
            "test_hit_rate": test_hit, "test_mean_rank": test_rank, "n_test": n_test,
            "pipe": pipe, "w": w,
        })

    best = next(r for r in results if r["layer"] == cv_best_layer)
    print(f"[v12-E] BEST layer={best['layer']} (selected by {CV_N_REPEATS}x{CV_N_SPLITS}-fold CV "
          f"over train+val, test set untouched during selection)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump({"pipe": best["pipe"], "w": best["w"]}, OUT_DIR / "rank_probe_v12_pairwise.joblib")
    for r in results:
        joblib.dump({"pipe": r["pipe"], "w": r["w"]},
                     OUT_DIR / f"rank_probe_v12_pairwise_layer{r['layer']}.joblib")
    with (OUT_DIR / "rank_probe_v12_pairwise.json").open("w") as f:
        json.dump({
            "layer": best["layer"],
            "selection_method": f"{CV_N_REPEATS}x{CV_N_SPLITS}-fold grouped CV over train+val",
            "cv_hit_rate": best["cv_hit_rate"],
            "val_hit_rate": best["val_hit_rate"], "val_mean_rank": best["val_mean_rank"],
            "test_hit_rate": best["test_hit_rate"], "test_mean_rank": best["test_mean_rank"],
            "all_layers": [{k: v for k, v in r.items() if k not in ("pipe", "w")} for r in results],
        }, f, indent=2)
    print(f"[v12-E] saved models/rank_probe_v12_pairwise.joblib + .json", flush=True)


if __name__ == "__main__":
    main()
