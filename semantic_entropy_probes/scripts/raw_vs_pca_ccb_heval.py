"""Rerun the two best configurations found so far -- plain LR (layer 20) and
plain MLP (layer 22) on the clarifycodebench+orchid_heval subset -- with raw
standardized activations instead of PCA-100, both pooled and LODO. Same
splits/seed/backends as everywhere else in this repo; only the PCA step is
removed (src/probes.py's fit_and_eval(use_pca=False)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import make_combined_train_val_test_split
from src.data_utils import load_combined_activation_bundle
from src.probes import fit_and_eval

MACRO_LABELS = (0, 1)
ACTIVATIONS_DIR = ROOT / "activations_ccb_heval"
CONFIGS = [("lr", "torch", 20), ("mlp", "torch_mlp", 22)]
SEED = 0


def pooled(acts, labels, groups, dataset_source, layer, backend):
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(labels, dataset_source, groups, seed=SEED)
    X = acts[:, layer, :]
    m = fit_and_eval(
        X[train_idx], labels[train_idx], X[val_idx], labels[val_idx], X[test_idx], labels[test_idx],
        seed=SEED, macro_labels=MACRO_LABELS, backend=backend, use_pca=False,
    )
    return m["test_accuracy"], m["test_auroc_ovr_macro"]


def lodo(acts, labels, groups, dataset_source, layer, backend):
    X = acts[:, layer, :]
    rng = np.random.default_rng(SEED)
    sources_present = sorted(set(dataset_source))
    fold_aurocs = {}
    for held_out in sources_present:
        test_mask = dataset_source == held_out
        test_groups = set(groups[test_mask])
        train_mask = (dataset_source != held_out) & (~np.isin(groups, list(test_groups)))
        train_idx_all = np.where(train_mask)[0]
        rng.shuffle(train_idx_all)
        n_val = max(1, int(0.2 * len(train_idx_all)))
        val_idx, tr_idx = train_idx_all[:n_val], train_idx_all[n_val:]
        test_idx = np.where(test_mask)[0]
        m = fit_and_eval(
            X[tr_idx], labels[tr_idx], X[val_idx], labels[val_idx], X[test_idx], labels[test_idx],
            seed=SEED, macro_labels=MACRO_LABELS, backend=backend, use_pca=False,
        )
        fold_aurocs[held_out] = m["test_auroc_ovr_macro"]
    mean_auroc = float(np.mean(list(fold_aurocs.values())))
    return fold_aurocs, mean_auroc


def main() -> None:
    acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows = \
        load_combined_activation_bundle(ACTIVATIONS_DIR)

    for tag, backend, layer in CONFIGS:
        acc, auroc = pooled(acts, labels, groups, dataset_source, layer, backend)
        fold_aurocs, mean_auroc = lodo(acts, labels, groups, dataset_source, layer, backend)
        print(f"[raw_vs_pca] {tag} (layer {layer}, raw activations, D=4096):")
        print(f"  pooled: acc={acc:.4f} auroc={auroc:.4f}")
        print(f"  lodo:   {fold_aurocs}  mean={mean_auroc:.4f}")


if __name__ == "__main__":
    main()
