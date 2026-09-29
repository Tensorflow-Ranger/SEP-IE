"""Is the missing-vs-redundant confidence jump bigger at a layer other than
rank1's 20 (LR) / rank2's 22 (MLP)?

For every layer 0..36, train a fresh LR and MLP probe (StandardScaler ->
PCA(100) -> classifier, same methodology as scripts/save_best_models.py:
best C reused from the original per-layer sweep in
results_ccb_heval/layer_metrics_{lr,mlp}.csv, refit on train+val), then
evaluate it on the SAME 40 held-out inline-edit prompts used throughout this
investigation (activations_detail_test_v6_inline + _v7_expand, which already
have all 37 hidden-state layers cached -- no new extraction needed).

Runs entirely on CPU (torch probes are forced to device="cpu") so it doesn't
contend with the GPU-bound Qwen generation jobs running concurrently.
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import make_combined_train_val_test_split
from src.data_utils import load_activation_bundle, load_combined_activation_bundle
from src.probes import MAX_ITER, N_PCA_COMPONENTS, _expected_value, _safe_auroc_ovr, _safe_spearman
from src.torch_mlp_probe import TorchMLPProbe
from src.torch_probe import TorchLogisticProbe

SEED = 0
TRAIN_ACTS_DIR = ROOT / "activations_ccb_heval"
HELDOUT_DIRS = [ROOT / "activations_detail_test_v6_inline", ROOT / "activations_detail_test_v7_expand"]
LAYER_METRICS = {"lr": ROOT / "results_ccb_heval" / "layer_metrics_lr.csv", "mlp": ROOT / "results_ccb_heval" / "layer_metrics_mlp.csv"}
RESULTS_DIR = ROOT / "results_layer_sweep_confidence_jump"


def H(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def build_pipeline_cpu(C: float, seed: int, n_train: int, n_features: int, backend: str) -> Pipeline:
    steps = [("scale", StandardScaler())]
    n_components = min(N_PCA_COMPONENTS, n_train - 1, n_features)
    if n_features > n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    if backend == "torch":
        clf = TorchLogisticProbe(C=C, class_weight="balanced", seed=seed, device="cpu")
    elif backend == "torch_mlp":
        clf = TorchMLPProbe(C=C, class_weight="balanced", seed=seed, device="cpu")
    else:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
    steps.append(("clf", clf))
    return Pipeline(steps)


def load_best_c_per_layer(csv_path: Path) -> dict[int, float]:
    out = {}
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            out[int(row["layer"])] = float(row["best_C"])
    return out


def load_heldout_pool():
    """Return {base_id: {variant: acts_row (n_states, hidden)}}, matching the
    40-prompt pool used throughout (v6 16 + v7 24)."""
    pool = {}
    for d in HELDOUT_DIRS:
        acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(d)
        id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}
        base_ids = sorted({m["id"].rsplit("__", 1)[0] for m in meta_rows})
        for base in base_ids:
            pool[base] = {v: acts[id_to_row[f"{base}__{v}"]] for v in ("baseline", "redundant", "missing")}
    return pool


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("[sweep] loading training activation bundle + combined split ...", flush=True)
    acts, labels, groups, _token_lens, dataset_source, _ambiguity_type, _meta = \
        load_combined_activation_bundle(TRAIN_ACTS_DIR)
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(labels, dataset_source, groups, seed=SEED)

    print("[sweep] loading 40 held-out inline-edit prompts (all layers already cached) ...", flush=True)
    heldout = load_heldout_pool()
    base_ids = sorted(heldout.keys(), key=lambda x: int(x.split("_")[-1]))
    print(f"[sweep] {len(base_ids)} held-out base prompts", flush=True)

    n_layers = acts.shape[1]
    rows_out = []

    for backend_tag, backend in [("lr", "sklearn"), ("mlp", "torch_mlp")]:
        best_c_per_layer = load_best_c_per_layer(LAYER_METRICS[backend_tag])
        for layer in range(n_layers):
            C = best_c_per_layer.get(layer, 0.01)
            X = acts[:, layer, :]
            X_train, y_train = X[train_idx], labels[train_idx]
            X_val, y_val = X[val_idx], labels[val_idx]
            X_test, y_test = X[test_idx], labels[test_idx]
            X_trainval = np.concatenate([X_train, X_val], axis=0)
            y_trainval = np.concatenate([y_train, y_val], axis=0)

            pipe = build_pipeline_cpu(C, SEED, n_train=len(y_trainval), n_features=X.shape[1], backend=backend)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                pipe.fit(X_trainval, y_trainval)

            y_prob_test = pipe.predict_proba(X_test)
            test_expected = _expected_value(y_prob_test, pipe.classes_)
            test_auroc = _safe_auroc_ovr(y_test, y_prob_test, pipe.classes_)
            test_spearman = _safe_spearman(y_test, test_expected)
            pos_idx = list(pipe.classes_).index(1)

            d_red, d_mis, dH_red, dH_mis = [], [], [], []
            for base in base_ids:
                pb = float(pipe.predict_proba(heldout[base]["baseline"][layer].reshape(1, -1))[0, pos_idx])
                pr = float(pipe.predict_proba(heldout[base]["redundant"][layer].reshape(1, -1))[0, pos_idx])
                pm = float(pipe.predict_proba(heldout[base]["missing"][layer].reshape(1, -1))[0, pos_idx])
                d_red.append(pr - pb)
                d_mis.append(pm - pb)
                dH_red.append(float(H(np.array(pr)) - H(np.array(pb))))
                dH_mis.append(float(H(np.array(pm)) - H(np.array(pb))))
            d_red, d_mis = np.array(d_red), np.array(d_mis)
            dH_red, dH_mis = np.array(dH_red), np.array(dH_mis)
            diff = d_mis - d_red
            t_p = float(stats.ttest_rel(d_mis, d_red).pvalue)
            w_p = float(stats.wilcoxon(d_mis, d_red).pvalue) if np.any(diff != 0) else 1.0

            row = {
                "backend": backend_tag, "layer": layer, "C": C,
                "test_auroc": test_auroc, "test_spearman": test_spearman,
                "mean_delta_redundant": float(d_red.mean()), "mean_delta_missing": float(d_mis.mean()),
                "mean_missing_minus_redundant": float(diff.mean()),
                "n_missing_lower_than_redundant": int((diff < 0).sum()),
                "paired_ttest_p": t_p, "wilcoxon_p": w_p,
                "mean_deltaH_redundant": float(dH_red.mean()), "mean_deltaH_missing": float(dH_mis.mean()),
            }
            rows_out.append(row)
            print(f"[sweep] {backend_tag:4s} layer={layer:2d} C={C:<7g} test_auroc={test_auroc:.3f} "
                  f"mean(missing-redundant)={row['mean_missing_minus_redundant']:+.4f} "
                  f"t_p={t_p:.4f} w_p={w_p:.4f} dH_missing={row['mean_deltaH_missing']:+.4f}", flush=True)

    with (RESULTS_DIR / "layer_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print("\n=== Top 5 layers by most-negative mean(missing - redundant), per backend ===")
    for backend_tag in ("lr", "mlp"):
        sub = sorted([r for r in rows_out if r["backend"] == backend_tag], key=lambda r: r["mean_missing_minus_redundant"])[:5]
        print(f"\n  -- {backend_tag} --")
        for r in sub:
            print(f"  layer={r['layer']:2d}  mean(missing-redundant)={r['mean_missing_minus_redundant']:+.4f}  "
                  f"t_p={r['paired_ttest_p']:.4f} w_p={r['wilcoxon_p']:.4f}  "
                  f"test_auroc={r['test_auroc']:.3f}  dH_missing={r['mean_deltaH_missing']:+.4f}")

    print("\n=== Top 5 layers by most-negative entropy shift on missing (dH_missing), per backend ===")
    for backend_tag in ("lr", "mlp"):
        sub = sorted([r for r in rows_out if r["backend"] == backend_tag], key=lambda r: r["mean_deltaH_missing"])[:5]
        print(f"\n  -- {backend_tag} --")
        for r in sub:
            print(f"  layer={r['layer']:2d}  dH_missing={r['mean_deltaH_missing']:+.4f}  "
                  f"mean(missing-redundant)={r['mean_missing_minus_redundant']:+.4f}  test_auroc={r['test_auroc']:.3f}")

    # highlight rank1 (layer 20, lr) / rank2 (layer 22, mlp) for reference
    ref = {(r["backend"], r["layer"]): r for r in rows_out}
    print("\n=== reference: current rank1 (lr, layer 20) / rank2 (mlp, layer 22) ===")
    for tag, layer in [("lr", 20), ("mlp", 22)]:
        r = ref.get((tag, layer))
        if r:
            print(f"  {tag} layer {layer}: mean(missing-redundant)={r['mean_missing_minus_redundant']:+.4f} "
                  f"t_p={r['paired_ttest_p']:.4f} w_p={r['wilcoxon_p']:.4f} dH_missing={r['mean_deltaH_missing']:+.4f} "
                  f"test_auroc={r['test_auroc']:.3f}")

    print(f"\n[sweep] wrote {RESULTS_DIR / 'layer_sweep.csv'}")


if __name__ == "__main__":
    main()
