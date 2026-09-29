"""Persist LR/MLP probes at layer 36 -- the layer with the largest
missing-vs-redundant entropy shift found by scripts/layer_sweep_confidence_jump.py
-- as a drop-in alternative to rank1 (layer 20)/rank2 (layer 22).

Same methodology as scripts/save_best_models.py: best C reused from the
original per-layer sweep (results_ccb_heval/layer_metrics_{lr,mlp}.csv),
refit on train+val. Saved to CPU so it loads without CUDA, same as rank1/rank2.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import make_combined_train_val_test_split
from src.data_utils import load_combined_activation_bundle
from src.probes import MAX_ITER, N_PCA_COMPONENTS, _expected_value, _safe_auroc_ovr, _safe_spearman
from src.torch_mlp_probe import TorchMLPProbe
from src.torch_probe import TorchLogisticProbe

SEED = 0
LAYER = 36
ACTIVATIONS_DIR = ROOT / "activations_ccb_heval"
MODELS_DIR = ROOT / "models"
LAYER_METRICS = {"lr": ROOT / "results_ccb_heval" / "layer_metrics_lr.csv", "mlp": ROOT / "results_ccb_heval" / "layer_metrics_mlp.csv"}

CONFIGS = [
    {"name": f"rank1b_lr_layer{LAYER}_pca100_ccb_heval", "backend": "sklearn", "tag": "lr"},
    {"name": f"rank2b_mlp_layer{LAYER}_pca100_ccb_heval", "backend": "torch_mlp", "tag": "mlp"},
]


def build_pipeline_cpu(C: float, seed: int, n_train: int, n_features: int, backend: str) -> Pipeline:
    steps = [("scale", StandardScaler())]
    n_components = min(N_PCA_COMPONENTS, n_train - 1, n_features)
    if n_features > n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    if backend == "torch_mlp":
        clf = TorchMLPProbe(C=C, class_weight="balanced", seed=seed, device="cpu")
    elif backend == "torch":
        clf = TorchLogisticProbe(C=C, class_weight="balanced", seed=seed, device="cpu")
    else:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
    steps.append(("clf", clf))
    return Pipeline(steps)


def load_best_c(csv_path: Path, layer: int) -> float:
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if int(row["layer"]) == layer:
                return float(row["best_C"])
    raise ValueError(f"layer {layer} not found in {csv_path}")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    acts, labels, groups, _token_lens, dataset_source, _ambiguity_type, _meta = \
        load_combined_activation_bundle(ACTIVATIONS_DIR)
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(labels, dataset_source, groups, seed=SEED)

    for cfg in CONFIGS:
        C = load_best_c(LAYER_METRICS[cfg["tag"]], LAYER)
        X = acts[:, LAYER, :]
        X_train, y_train = X[train_idx], labels[train_idx]
        X_val, y_val = X[val_idx], labels[val_idx]
        X_test, y_test = X[test_idx], labels[test_idx]
        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)

        pipe = build_pipeline_cpu(C, SEED, n_train=len(y_trainval), n_features=X.shape[1], backend=cfg["backend"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipe.fit(X_trainval, y_trainval)

        y_prob = pipe.predict_proba(X_test)
        y_pred = pipe.predict(X_test)
        test_auroc = _safe_auroc_ovr(y_test, y_prob, pipe.classes_)
        test_acc = float((y_pred == y_test).mean())
        test_spearman = _safe_spearman(y_test, _expected_value(y_prob, pipe.classes_))

        model_path = MODELS_DIR / f"{cfg['name']}.joblib"
        joblib.dump(pipe, model_path)
        meta = {
            "name": cfg["name"], "backend": cfg["backend"], "layer": LAYER, "best_C": C,
            "test_accuracy": test_acc, "test_auroc": test_auroc, "test_spearman": test_spearman,
            "n_train": int(len(y_trainval)), "n_test": int(len(y_test)),
            "note": "picked for largest missing-vs-redundant entropy shift in the inline-edit confidence-jump test, not for best in-distribution AUROC",
        }
        json.dump(meta, open(MODELS_DIR / f"{cfg['name']}.json", "w"), indent=2)
        print(f"[save_layer36] {cfg['name']}: C={C} test_auroc={test_auroc:.3f} test_acc={test_acc:.3f} -> {model_path}")


if __name__ == "__main__":
    main()
