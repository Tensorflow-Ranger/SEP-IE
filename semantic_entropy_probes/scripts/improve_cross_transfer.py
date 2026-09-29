"""Try a few feature-engineering candidates for the hidden-state probe, and
see whether any of them close (part of) the pooled-vs-LODO generalization gap
found by confound_tests.py.

All candidates are deployable: every learned parameter (source-discriminant
directions, layer band, etc.) is fit on TRAIN data only (features + the
dataset_source label used as an unsupervised grouping variable, never the
ambiguity label and never target-domain statistics), then applied identically
to train and test. This is standard supervised-domain-generalization practice,
not test-time peeking -- a real deployment prompt wouldn't need to declare
which benchmark it came from for any of these transforms to apply.

Candidates (each then goes through the existing scale -> PCA -> LogisticRegression
pipeline from src/probes.py, so the *classifier* is held fixed and only the
input features change -- an apples-to-apples comparison):

  - baseline          : single layer's raw activations (what confound_tests.py
                         already evaluates).
  - multilayer_concat : concatenate activations from 3 layers spanning early/
                         mid/late depth. Rationale: a single layer's direction
                         may be partly an artifact of that specific layer's
                         idiosyncratic geometry; a wider span gives the
                         classifier more to work with and may average out
                         layer-specific noise.
  - layer_mean_pool   : elementwise mean across a band of consecutive layers.
                         Cheaper than concat (no dimensionality increase) and
                         a smoother, lower-variance summary of "how the
                         representation evolves through this stretch of the
                         network" than any single layer.
  - domain_scrubbed   : fit LDA(n_components=2) on (X_train, dataset_source
                         label) -- i.e. the directions in activation space
                         that best predict WHICH SOURCE DATASET a row came
                         from, using only the sources present in train for
                         this LODO fold -- then orthogonally project those
                         directions out of every row (train and test) before
                         scaling/PCA/classification. This directly targets
                         the failure mode confound_tests.py's LODO results
                         point at: the classifier partly keying on
                         "which construction pipeline produced this text"
                         rather than a source-invariant ambiguity signal.

Evaluated via the SAME group-safe leave-one-dataset-out protocol as
confound_tests.py (source_problem_id groups shared across dataset_source
labels -- e.g. orchid_heval/humanevalcomm -- are excluded from train, not
just rows literally labeled with the held-out source), over the 3 valid
folds (clarifycodebench, orchid_heval, orchid_bcb; humanevalcomm is
ambiguous-only and can't be a held-out test fold on its own).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import DATASET_SOURCES
from src.data_utils import load_combined_activation_bundle
from src.probes import _build_pipeline, _fit, _safe_auroc_ovr, _safe_spearman

MACRO_LABELS = (0, 1)
C_GRID = (0.001, 0.01, 0.1, 1.0)


def feat_baseline(acts: np.ndarray, layer: int) -> np.ndarray:
    return acts[:, layer, :]


def feat_multilayer_concat(acts: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    return np.concatenate([acts[:, l, :] for l in layers], axis=1)


def feat_layer_mean_pool(acts: np.ndarray, layers: tuple[int, ...]) -> np.ndarray:
    return acts[:, list(layers), :].mean(axis=1)


def fit_domain_scrub(X_train: np.ndarray, source_train: np.ndarray, n_components: int = 2):
    """Fit LDA on (X_train, source_train); return a function that projects
    the top n_components source-discriminant directions out of any X.
    """
    n_present = len(set(source_train))
    n_components = min(n_components, n_present - 1, X_train.shape[1])
    if n_components < 1:
        return lambda X: X
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda.fit(X_train, source_train)
    directions = lda.scalings_[:, :n_components].T  # (n_components, n_features)
    # Orthonormalize via QR so the projection is a clean orthogonal projector.
    q, _ = np.linalg.qr(directions.T)  # (n_features, n_components)

    def _scrub(X: np.ndarray) -> np.ndarray:
        return X - (X @ q) @ q.T

    return _scrub


def evaluate_lodo(
    X: np.ndarray, y: np.ndarray, dataset_source: np.ndarray, groups: np.ndarray,
    seed: int = 0, backend: str = "torch",
) -> dict:
    """Group-safe LODO (same protocol as confound_tests.py's fixed version),
    hidden-state features only, LogisticRegression backend. Returns per-fold
    and mean test AUROC.
    """
    results = {}
    rng = np.random.default_rng(seed)
    sources_present = [s for s in DATASET_SOURCES if (dataset_source == s).any()]

    for held_out in sources_present:
        test_mask = dataset_source == held_out
        if len(set(y[test_mask])) < 2:
            continue
        test_groups = set(groups[test_mask])
        train_mask = (dataset_source != held_out) & (~np.isin(groups, list(test_groups)))

        train_idx_all = np.where(train_mask)[0]
        rng.shuffle(train_idx_all)
        n_val = max(1, int(0.2 * len(train_idx_all)))
        val_idx, tr_idx = train_idx_all[:n_val], train_idx_all[n_val:]
        test_idx = np.where(test_mask)[0]
        if len(set(y[tr_idx])) < 2 or len(set(y[val_idx])) < 2:
            continue

        best_c, best_val_sp = C_GRID[0], -np.inf
        for c in C_GRID:
            pipe = _build_pipeline(c, seed, n_train=len(tr_idx), n_features=X.shape[1], backend=backend)
            _fit(pipe, X[tr_idx], y[tr_idx])
            val_expected = pipe.predict_proba(X[val_idx]) @ pipe.classes_.astype(np.float64)
            sp = _safe_spearman(y[val_idx], val_expected)
            if not np.isnan(sp) and sp > best_val_sp:
                best_c, best_val_sp = c, sp

        trval_idx = np.concatenate([tr_idx, val_idx])
        pipe_final = _build_pipeline(best_c, seed, n_train=len(trval_idx), n_features=X.shape[1], backend=backend)
        _fit(pipe_final, X[trval_idx], y[trval_idx])
        probs = pipe_final.predict_proba(X[test_idx])
        y_pred = pipe_final.predict(X[test_idx])

        results[held_out] = {
            "test_accuracy": float(accuracy_score(y[test_idx], y_pred)),
            "test_auroc": _safe_auroc_ovr(y[test_idx], probs, pipe_final.classes_),
            "best_C": best_c,
        }

    aurocs = [v["test_auroc"] for v in results.values() if not np.isnan(v["test_auroc"])]
    results["_mean_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    return results


def evaluate_lodo_with_domain_scrub(
    X: np.ndarray, y: np.ndarray, dataset_source: np.ndarray, groups: np.ndarray, seed: int = 0,
    backend: str = "torch",
) -> dict:
    """Like evaluate_lodo, but fits the domain-scrub projection fresh inside
    each fold (from that fold's train rows only) before the scale/PCA/LR
    pipeline -- the scrub direction must be re-learned per fold since the set
    of "other" sources changes with which one is held out.
    """
    results = {}
    rng = np.random.default_rng(seed)
    sources_present = [s for s in DATASET_SOURCES if (dataset_source == s).any()]

    for held_out in sources_present:
        test_mask = dataset_source == held_out
        if len(set(y[test_mask])) < 2:
            continue
        test_groups = set(groups[test_mask])
        train_mask = (dataset_source != held_out) & (~np.isin(groups, list(test_groups)))

        train_idx_all = np.where(train_mask)[0]
        rng.shuffle(train_idx_all)
        n_val = max(1, int(0.2 * len(train_idx_all)))
        val_idx, tr_idx = train_idx_all[:n_val], train_idx_all[n_val:]
        test_idx = np.where(test_mask)[0]
        if len(set(y[tr_idx])) < 2 or len(set(y[val_idx])) < 2:
            continue

        scrub = fit_domain_scrub(X[tr_idx], dataset_source[tr_idx], n_components=2)
        X_scrubbed = scrub(X)

        best_c, best_val_sp = C_GRID[0], -np.inf
        for c in C_GRID:
            pipe = _build_pipeline(c, seed, n_train=len(tr_idx), n_features=X.shape[1], backend=backend)
            _fit(pipe, X_scrubbed[tr_idx], y[tr_idx])
            val_expected = pipe.predict_proba(X_scrubbed[val_idx]) @ pipe.classes_.astype(np.float64)
            sp = _safe_spearman(y[val_idx], val_expected)
            if not np.isnan(sp) and sp > best_val_sp:
                best_c, best_val_sp = c, sp

        trval_idx = np.concatenate([tr_idx, val_idx])
        pipe_final = _build_pipeline(best_c, seed, n_train=len(trval_idx), n_features=X.shape[1], backend=backend)
        _fit(pipe_final, X_scrubbed[trval_idx], y[trval_idx])
        probs = pipe_final.predict_proba(X_scrubbed[test_idx])
        y_pred = pipe_final.predict(X_scrubbed[test_idx])

        results[held_out] = {
            "test_accuracy": float(accuracy_score(y[test_idx], y_pred)),
            "test_auroc": _safe_auroc_ovr(y[test_idx], probs, pipe_final.classes_),
            "best_C": best_c,
        }

    aurocs = [v["test_auroc"] for v in results.values() if not np.isnan(v["test_auroc"])]
    results["_mean_auroc"] = float(np.mean(aurocs)) if aurocs else float("nan")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations_combined")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_combined")
    ap.add_argument("--best-layer", type=int, default=None,
                     help="Defaults to run_summary.json's best_layer_lr.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows = \
        load_combined_activation_bundle(args.activations_dir)

    if args.best_layer is None:
        with (args.results_dir / "run_summary.json").open("r", encoding="utf-8") as f:
            args.best_layer = json.load(f)["best_layer_lr"]
    L = args.best_layer
    print(f"[improve_cross_transfer] center layer = {L}")

    band = tuple(l for l in (L - 8, L, L + 8) if 0 <= l < acts.shape[1])
    mean_pool_band = tuple(l for l in range(max(0, L - 8), min(acts.shape[1], L + 9)))
    print(f"[improve_cross_transfer] multilayer_concat layers = {band}, "
          f"layer_mean_pool band = {mean_pool_band}")

    report = {}

    print("\n[improve_cross_transfer] === baseline (single layer) ===")
    X = feat_baseline(acts, L)
    report["baseline"] = evaluate_lodo(X, labels, dataset_source, groups, seed=args.seed)
    print(json.dumps(report["baseline"], indent=2))

    print("\n[improve_cross_transfer] === multilayer_concat ===")
    X = feat_multilayer_concat(acts, band)
    report["multilayer_concat"] = evaluate_lodo(X, labels, dataset_source, groups, seed=args.seed)
    print(json.dumps(report["multilayer_concat"], indent=2))

    print("\n[improve_cross_transfer] === layer_mean_pool ===")
    X = feat_layer_mean_pool(acts, mean_pool_band)
    report["layer_mean_pool"] = evaluate_lodo(X, labels, dataset_source, groups, seed=args.seed)
    print(json.dumps(report["layer_mean_pool"], indent=2))

    print("\n[improve_cross_transfer] === domain_scrubbed (single layer + LDA nullspace projection) ===")
    X = feat_baseline(acts, L)
    report["domain_scrubbed"] = evaluate_lodo_with_domain_scrub(X, labels, dataset_source, groups, seed=args.seed)
    print(json.dumps(report["domain_scrubbed"], indent=2))

    print("\n[improve_cross_transfer] === SUMMARY (mean LODO AUROC across 3 valid folds) ===")
    for name, res in report.items():
        print(f"  {name}: {res['_mean_auroc']:.4f}")

    with (args.results_dir / "cross_transfer_feature_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[improve_cross_transfer] wrote {args.results_dir / 'cross_transfer_feature_report.json'}")


if __name__ == "__main__":
    main()
