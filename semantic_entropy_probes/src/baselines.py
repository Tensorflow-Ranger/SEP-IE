"""Trivial baselines: majority class and random, for sanity-checking probes.

Same metric set as src/probes.py (accuracy, macro-F1, MAE, Spearman) so the
per-layer probe results and these baselines are directly comparable in
results/baselines.json vs results/layer_metrics.csv.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

ALL_CLASSES = (0, 1, 2, 3)


def _safe_spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        return float("nan")
    rho, _ = spearmanr(y_true, y_score)
    return float(rho)


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray, macro_labels=ALL_CLASSES) -> dict:
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = int(values[np.argmax(counts)]) if len(y_train) else 0
    y_pred = np.full_like(y_test, majority_class)
    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=list(macro_labels), zero_division=0)),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_spearman": float("nan"),  # constant predictor -- correlation undefined
        "majority_class": majority_class,
    }


def random_baseline(y_train: np.ndarray, y_test: np.ndarray, seed: int = 0, n_trials: int = 200,
                     macro_labels=ALL_CLASSES) -> dict:
    """Predict K ~ Categorical(train class distribution); average over trials."""
    rng = np.random.default_rng(seed)
    values, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum() if len(y_train) else np.full(len(ALL_CLASSES), 1 / len(ALL_CLASSES))
    classes = values if len(y_train) else np.array(ALL_CLASSES)

    accs, f1s, maes, rhos = [], [], [], []
    for _ in range(n_trials):
        y_pred = rng.choice(classes, size=len(y_test), p=probs)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro", labels=list(macro_labels), zero_division=0))
        maes.append(mean_absolute_error(y_test, y_pred))
        rhos.append(_safe_spearman(y_test, y_pred))

    rhos = [r for r in rhos if not np.isnan(r)]
    return {
        "test_accuracy": float(np.mean(accs)),
        "test_macro_f1": float(np.mean(f1s)),
        "test_mae": float(np.mean(maes)),
        "test_spearman": float(np.mean(rhos)) if rhos else float("nan"),
        "train_class_distribution": {int(v): float(p) for v, p in zip(classes, probs)},
    }
