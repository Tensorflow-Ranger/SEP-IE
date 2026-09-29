"""Persist the two best-performing configurations found in this investigation
to disk: plain LR (layer 20) and plain MLP (layer 22), PCA-100 features, on
the clarifycodebench+orchid_heval subset -- pooled AUROC 0.778/0.780, LODO
mean AUROC 0.616/0.599, the best numbers found across every probe/feature/
loss variant tried (plain baselines, feature engineering, SupCon, raw
activations).

Each saved pipeline is StandardScaler -> PCA(100) -> classifier, fit on
train+val (the same refit fit_and_eval does internally before scoring test),
at the C selected by val-Spearman during the original run. The underlying
torch model is moved to CPU before saving so the pipeline loads and predicts
identically on a machine without CUDA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import make_combined_train_val_test_split
from src.data_utils import load_combined_activation_bundle
from src.probes import DEFAULT_C_GRID, _build_pipeline, _expected_value, _fit, _safe_auroc_ovr, _safe_spearman
from src.torch_mlp_probe import TorchMLPProbe
from src.torch_probe import TorchLogisticProbe

MACRO_LABELS = (0, 1)
ACTIVATIONS_DIR = ROOT / "activations_ccb_heval"
MODELS_DIR = ROOT / "models"
SEED = 0

CONFIGS = [
    # backend must match scripts/train_binary_probes.py's actual per-tag backend
    # ("sklearn" for "lr", "torch_mlp" for "mlp") -- NOT torch/GPU for LR, even
    # though this repo uses GPU where possible elsewhere. Confirmed by rerunning
    # with backend="torch" first: it silently produced a different, worse model
    # (AUROC 0.760 vs the 0.778 that's actually on record for "rank 1").
    {"name": "rank1_lr_layer20_pca100_ccb_heval", "backend": "sklearn", "layer": 20},
    {"name": "rank2_mlp_layer22_pca100_ccb_heval", "backend": "torch_mlp", "layer": 22},
]


def _to_cpu(clf) -> None:
    if isinstance(clf, (TorchLogisticProbe, TorchMLPProbe)):
        clf.device = "cpu"
        clf._model.to("cpu")


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows = \
        load_combined_activation_bundle(ACTIVATIONS_DIR)
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(labels, dataset_source, groups, seed=SEED)

    for cfg in CONFIGS:
        layer, backend = cfg["layer"], cfg["backend"]
        X = acts[:, layer, :]
        X_train, y_train = X[train_idx], labels[train_idx]
        X_val, y_val = X[val_idx], labels[val_idx]
        X_test, y_test = X[test_idx], labels[test_idx]

        best_c, best_val_sp = DEFAULT_C_GRID[0], -np.inf
        for c in DEFAULT_C_GRID:
            pipe = _build_pipeline(c, SEED, n_train=len(y_train), n_features=X.shape[1], backend=backend)
            _fit(pipe, X_train, y_train)
            val_expected = _expected_value(pipe.predict_proba(X_val), pipe.classes_)
            sp = _safe_spearman(y_val, val_expected)
            if not np.isnan(sp) and sp > best_val_sp:
                best_c, best_val_sp = c, sp

        X_trainval = np.concatenate([X_train, X_val], axis=0)
        y_trainval = np.concatenate([y_train, y_val], axis=0)
        pipe_final = _build_pipeline(best_c, SEED, n_train=len(y_trainval), n_features=X.shape[1], backend=backend)
        _fit(pipe_final, X_trainval, y_trainval)

        y_prob = pipe_final.predict_proba(X_test)
        y_pred = pipe_final.predict(X_test)
        test_auroc = _safe_auroc_ovr(y_test, y_prob, pipe_final.classes_)
        test_acc = float((y_pred == y_test).mean())

        _to_cpu(pipe_final.named_steps["clf"])

        model_path = MODELS_DIR / f"{cfg['name']}.joblib"
        joblib.dump(pipe_final, model_path)

        meta = {
            "name": cfg["name"],
            "backend": backend,
            "layer": layer,
            "best_C": best_c,
            "val_spearman_at_best_C": float(best_val_sp),
            "test_accuracy": test_acc,
            "test_auroc": test_auroc,
            "n_train": int(len(y_trainval)),
            "n_test": int(len(y_test)),
            "n_features_in": int(X.shape[1]),
            "activations_dir": str(ACTIVATIONS_DIR.relative_to(ROOT)),
            "dataset_sources": sorted(set(dataset_source.tolist())),
            "seed": SEED,
            "device_saved_on": "cpu",
            "pipeline_steps": [s[0] for s in pipe_final.steps],
        }
        meta_path = MODELS_DIR / f"{cfg['name']}.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[save_best_models] {cfg['name']}: best_C={best_c} test_acc={test_acc:.4f} "
              f"test_auroc={test_auroc:.4f}")
        print(f"[save_best_models]   -> {model_path}")
        print(f"[save_best_models]   -> {meta_path}")


if __name__ == "__main__":
    main()
