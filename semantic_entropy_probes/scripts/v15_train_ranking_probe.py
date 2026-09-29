"""v15 stage E: extend v12's pairwise-ranking probe training to the full
clean 986-row set (570 reused v12 activations + 416 newly extracted, both
clean-label-only, both using v12's exact extraction mechanism -- see
scripts/v15_bedrock_oracle_rewrite.py / v15_extract_activations.py), and
to a wider training-method grid:

  feature variant ∈ {pca (v12's original: StandardScaler+PCA-100),
                      raw (StandardScaler only, no PCA)}
  head ∈ {lr  (v12's original: fit_intercept=False LogisticRegression on
               pairwise diffs -- reduces to one linear scoring direction w,
               ranking = argsort(z @ w)),
          mlp (MLPClassifier on the same pairwise-diff formulation -- since
               this can't reduce to a single dot product, ranking is done
               by pairwise win-counting: score_i = sum_j P(i beats j) over
               all ordered pairs in the group, at inference time)}

  split ∈ {pooled  (v12's original: StratifiedGroupKFold on the
                     (label, dataset_source) composite key -- every source
                     present at its proportion in train/val/test)),
           lodo    (NEW: leave-one-dataset-out -- train on 2 sources,
                     test on the 3rd, for each of the 3 sources; answers
                     whether a probe generalizes to a genuinely unseen
                     problem source, which the pooled split cannot measure
                     since sources co-occur in both train and test there)}

SupCon is NOT included here (deferred, per plan, to a later pass).

Usage: python v15_train_ranking_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.combined_splits import (  # noqa: E402
    composite_key,
    make_combined_grouped_cv_folds,
    make_combined_train_val_test_split,
    report_split_composition,
)

JUDGED_PATH = ROOT / "data" / "v13_rejudge" / "deepseek_judged.jsonl"
CHUNK_DIR = ROOT / "data" / "v13_rejudge" / "chunks"
V12_ACT_DIR = ROOT / "data" / "v12_ranking" / "activations"
V15_ACT_DIR = ROOT / "data" / "v15_new_activations" / "activations"
OUT_DIR = ROOT / "models"
RESULTS_PATH = ROOT / "data" / "v15_new_activations" / "training_results.json"

PCA_DIM = 100
SEED = 0
CV_N_SPLITS = 4
CV_N_REPEATS = 3
MLP_HIDDEN = (64,)
MLP_MAX_ITER = 500


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def safe_name(rid: str) -> str:
    return rid.replace("/", "__")


def derive_source_problem_id(rid: str, dataset_source: str) -> str:
    if dataset_source == "orchid_heval":
        num = rid.split("_")[1]
        return f"humaneval::{num}"
    if dataset_source == "orchid_bcb":
        # id format: bcb_BigCodeBench/45_semantic -> bigcodebench::BigCodeBench/45
        rest = rid[len("bcb_") :]
        task_id = rest.rsplit("_", 1)[0]
        return f"bigcodebench::{task_id}"
    raise ValueError(f"unexpected new-row dataset_source {dataset_source}")


def load_groups():
    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    ground_truth_v12 = {}
    v12_gt_path = ROOT / "data" / "v12_ranking" / "ground_truth.jsonl"
    for r in read_jsonl(v12_gt_path):
        ground_truth_v12[r["id"]] = r

    def clean(rid):
        j = judged_rows[rid]["judged"]
        return (not j.get("label_leaked")) and (j.get("label_functionally_relevant") is True)

    ids, dataset_source, source_problem_id = [], [], []
    layers = None
    per_group_vecs, per_group_tiers = {}, {}

    n_v12, n_v15 = 0, 0
    for npz_path in sorted(V12_ACT_DIR.glob("*.npz")):
        gid = npz_path.stem
        if gid not in judged_rows or not clean(gid):
            continue
        data = np.load(npz_path)
        if layers is None:
            layers = data["layers"]
        cand = data["candidates"]
        tiers = data["tiers"]
        if len(tiers) < 2 or len(set(tiers.tolist())) < 2:
            continue
        per_group_vecs[gid] = np.transpose(cand, (1, 0, 2))
        per_group_tiers[gid] = tiers
        ids.append(gid)
        gt = ground_truth_v12[gid]
        dataset_source.append(gt["dataset_source"])
        source_problem_id.append(gt["source_problem_id"])
        n_v12 += 1

    for npz_path in sorted(V15_ACT_DIR.glob("*.npz")):
        safe_gid = npz_path.stem
        # recover original id: try direct, else with '/' substituted back in
        gid = None
        for cand_id in judged_rows:
            if safe_name(cand_id) == safe_gid:
                gid = cand_id
                break
        if gid is None or not clean(gid):
            continue
        data = np.load(npz_path)
        cand = data["candidates"]
        tiers = data["tiers"]
        if len(tiers) < 2 or len(set(tiers.tolist())) < 2:
            continue
        per_group_vecs[gid] = np.transpose(cand, (1, 0, 2))
        per_group_tiers[gid] = tiers
        ids.append(gid)
        ds = judged_rows[gid]["dataset_source"]
        dataset_source.append(ds)
        source_problem_id.append(derive_source_problem_id(gid, ds))
        n_v15 += 1

    print(f"[v15-E] loaded {n_v12} v12-reused + {n_v15} newly-extracted = {len(ids)} usable groups", flush=True)
    return (
        np.array(ids), np.array(dataset_source), np.array(source_problem_id),
        layers, per_group_vecs, per_group_tiers,
    )


def build_pipe(feature_variant: str, train_vecs_pool: np.ndarray):
    if feature_variant == "pca":
        pipe = make_pipeline(StandardScaler(), PCA(n_components=min(PCA_DIM, train_vecs_pool.shape[0] - 1), random_state=SEED))
    elif feature_variant == "raw":
        pipe = make_pipeline(StandardScaler())
    else:
        raise ValueError(feature_variant)
    pipe.fit(train_vecs_pool)
    return pipe


def build_pairwise_xy(train_ids, per_group_vecs, per_group_tiers, layer_idx, pipe):
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
    return np.array(X), np.array(y)


def fit_layer_probe(train_ids, per_group_vecs, per_group_tiers, layer_idx, feature_variant, head):
    train_vecs_pool = np.concatenate([per_group_vecs[g][layer_idx] for g in train_ids], axis=0)
    pipe = build_pipe(feature_variant, train_vecs_pool)
    X, y = build_pairwise_xy(train_ids, per_group_vecs, per_group_tiers, layer_idx, pipe)

    if head == "lr":
        clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000)
        clf.fit(X, y)
        return pipe, {"w": clf.coef_[0]}
    elif head == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER, random_state=SEED, early_stopping=True)
        clf.fit(X, y)
        return pipe, {"clf": clf}
    raise ValueError(head)


def eval_ranking(group_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, model, head):
    hits, ranks = [], []
    for gid in group_ids:
        vecs = per_group_vecs[gid][layer_idx]
        tiers = per_group_tiers[gid]
        z = pipe.transform(vecs)
        n = len(tiers)
        if head == "lr":
            scores = z @ model["w"]
        else:  # mlp: pairwise win-counting
            diffs, pair_idx = [], []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diffs.append(z[i] - z[j])
                    pair_idx.append(i)
            proba = model["clf"].predict_proba(np.array(diffs))[:, 1]
            scores = np.zeros(n)
            for p, i in zip(proba, pair_idx):
                scores[i] += p
        order = np.argsort(-scores)
        best_tier = tiers.max()
        top_idx = order[0]
        hits.append(int(tiers[top_idx] == best_tier))
        rank_of_best = 1 + int(np.where(tiers[order] == best_tier)[0][0])
        ranks.append(rank_of_best)
    return float(np.mean(hits)), float(np.mean(ranks)), len(group_ids)


def cv_select_best_layer(layers, per_group_vecs, per_group_tiers, trainval_idx, ids,
                          dataset_source, source_problem_id, feature_variant, head):
    labels_dummy = np.ones(len(ids), dtype=np.int64)
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
                pipe, model = fit_layer_probe(fold_train_ids, per_group_vecs, per_group_tiers, li, feature_variant, head)
                hit, _, _ = eval_ranking(fold_val_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
                cv_hits[layer].append(hit)
    return {layer: float(np.mean(hits)) for layer, hits in cv_hits.items()}


def run_pooled(layers, per_group_vecs, per_group_tiers, ids, dataset_source, source_problem_id,
                feature_variant, head):
    labels_dummy = np.ones(len(ids), dtype=np.int64)
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    train_ids, val_ids, test_ids = ids[train_idx], ids[val_idx], ids[test_idx]
    trainval_idx = np.concatenate([train_idx, val_idx])

    cv_mean = cv_select_best_layer(layers.tolist(), per_group_vecs, per_group_tiers, trainval_idx,
                                    ids, dataset_source, source_problem_id, feature_variant, head)
    best_layer = max(cv_mean, key=cv_mean.get)
    li = layers.tolist().index(best_layer)
    pipe, model = fit_layer_probe(train_ids, per_group_vecs, per_group_tiers, li, feature_variant, head)
    val_hit, val_rank, n_val = eval_ranking(val_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
    test_hit, test_rank, n_test = eval_ranking(test_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
    return {
        "best_layer": best_layer, "cv_hit_rate": cv_mean[best_layer],
        "val_hit_rate": val_hit, "n_val": n_val,
        "test_hit_rate": test_hit, "n_test": n_test,
        "pipe": pipe, "model": model,
    }


def run_lodo(layers, per_group_vecs, per_group_tiers, ids, dataset_source, source_problem_id,
             feature_variant, head, held_out_source):
    train_mask = dataset_source != held_out_source
    test_mask = dataset_source == held_out_source
    train_ids_all = ids[train_mask]
    test_ids = ids[test_mask]
    if len(test_ids) < 5 or len(train_ids_all) < 20:
        return None

    # small held-out val slice from the training sources (grouped, 10%) for layer CV
    rng = np.random.RandomState(SEED)
    train_dataset_source = dataset_source[train_mask]
    train_source_problem_id = source_problem_id[train_mask]
    labels_dummy = np.ones(len(train_ids_all), dtype=np.int64)
    cv_mean = cv_select_best_layer(layers.tolist(), per_group_vecs, per_group_tiers,
                                    np.arange(len(train_ids_all)), train_ids_all,
                                    train_dataset_source, train_source_problem_id, feature_variant, head)
    best_layer = max(cv_mean, key=cv_mean.get)
    li = layers.tolist().index(best_layer)
    pipe, model = fit_layer_probe(train_ids_all, per_group_vecs, per_group_tiers, li, feature_variant, head)
    test_hit, test_rank, n_test = eval_ranking(test_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
    return {"held_out_source": held_out_source, "best_layer": best_layer,
            "cv_hit_rate": cv_mean[best_layer], "test_hit_rate": test_hit, "n_test": n_test}


def main() -> None:
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = load_groups()
    report_split_composition("ALL", np.arange(len(ids)), np.ones(len(ids), dtype=np.int64), dataset_source)

    results = {"pooled": {}, "lodo": {}}
    grid = [(fv, h) for fv in ("pca", "raw") for h in ("lr", "mlp")]

    for feature_variant, head in grid:
        t0 = time.time()
        key = f"{feature_variant}_{head}"
        print(f"\n[v15-E] === POOLED: {key} ===", flush=True)
        r = run_pooled(layers, per_group_vecs, per_group_tiers, ids, dataset_source, source_problem_id,
                        feature_variant, head)
        print(f"[v15-E] {key}: best_layer={r['best_layer']} cv={r['cv_hit_rate']:.3f} "
              f"val={r['val_hit_rate']:.3f}(n={r['n_val']}) test={r['test_hit_rate']:.3f}(n={r['n_test']}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
        import joblib
        joblib.dump({"pipe": r["pipe"], "model": r["model"], "head": head},
                     OUT_DIR / f"rank_probe_v15_{key}_pooled.joblib")
        results["pooled"][key] = {k: v for k, v in r.items() if k not in ("pipe", "model")}

        print(f"[v15-E] === LODO: {key} ===", flush=True)
        lodo_results = []
        for held_out in ("clarifycodebench", "orchid_heval", "orchid_bcb"):
            t1 = time.time()
            lr = run_lodo(layers, per_group_vecs, per_group_tiers, ids, dataset_source, source_problem_id,
                           feature_variant, head, held_out)
            if lr is None:
                print(f"[v15-E] {key} LODO held_out={held_out}: skipped (too few rows)", flush=True)
                continue
            print(f"[v15-E] {key} LODO held_out={held_out}: best_layer={lr['best_layer']} "
                  f"cv={lr['cv_hit_rate']:.3f} test={lr['test_hit_rate']:.3f}(n={lr['n_test']}) "
                  f"[{time.time()-t1:.0f}s]", flush=True)
            lodo_results.append(lr)
        results["lodo"][key] = lodo_results

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[v15-E] saved results to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
