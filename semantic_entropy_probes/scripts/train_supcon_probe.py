"""Evaluate SupCon-projected embeddings as a probe, both pooled (in-distribution)
and via the group-safe leave-one-dataset-out protocol, so results sit directly
next to every number already in FINDINGS.md's "Combined binary dataset"
section (LR/MLP pooled 0.752/0.768, LODO mean ~0.57-0.59).

Grid: {pca, raw} input features x {vanilla, cross_source_only} loss variant x
temperature in {0.05, 0.1, 0.2}. Stage 2 (the actual classifier on top of the
frozen SupCon embedding) reuses src/probes.py's fit_and_eval unchanged
(sklearn LogisticRegression, C swept on val by Spearman) -- the SupCon
embedding is the only thing that changes; the classifier and its selection
procedure are identical to every other result in this repo, for a fair
comparison.

pca input: StandardScaler + PCA(100) fit on that fold's train only (same as
src/probes.py's default pipeline).
raw input: StandardScaler only, fit on that fold's train only, D=4096.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import DATASET_SOURCES, make_combined_train_val_test_split
from src.data_utils import load_combined_activation_bundle
from src.probes import fit_and_eval
from src.supcon import fit_supcon_projector

MACRO_LABELS = (0, 1)


def make_features(X_train, X_val, X_test, input_kind: str, seed: int, n_components: int = 100):
    scaler = StandardScaler().fit(X_train)
    Xtr, Xva, Xte = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    if input_kind == "raw":
        return Xtr, Xva, Xte
    n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed).fit(Xtr)
    return pca.transform(Xtr), pca.transform(Xva), pca.transform(Xte)


def supcon_stage(
    X_train, y_train, X_val, y_val, X_test, y_test,
    source_train, input_kind: str, loss_kind: str, temperature: float, seed: int,
    embed_dim: int = 32,
) -> dict:
    Xtr, Xva, Xte = make_features(X_train, X_val, X_test, input_kind, seed)
    encode, _ = fit_supcon_projector(
        Xtr, y_train, embed_dim=embed_dim, temperature=temperature, seed=seed,
        sources_train=source_train if loss_kind == "cross_source_only" else None,
        cross_source_only=(loss_kind == "cross_source_only"),
    )
    Ztr, Zva, Zte = encode(Xtr), encode(Xva), encode(Xte)
    return fit_and_eval(Ztr, y_train, Zva, y_val, Zte, y_test, seed=seed, macro_labels=MACRO_LABELS, backend="sklearn")


def evaluate_pooled(acts, labels, groups, dataset_source, layer: int, seed: int) -> dict:
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(labels, dataset_source, groups, seed=seed)
    X = acts[:, layer, :]
    results = {}
    for input_kind in ["pca", "raw"]:
        for loss_kind in ["vanilla", "cross_source_only"]:
            for temperature in [0.05, 0.1, 0.2]:
                key = f"{input_kind}_{loss_kind}_t{temperature}"
                m = supcon_stage(
                    X[train_idx], labels[train_idx], X[val_idx], labels[val_idx], X[test_idx], labels[test_idx],
                    dataset_source[train_idx], input_kind, loss_kind, temperature, seed,
                )
                results[key] = {"test_accuracy": m["test_accuracy"], "test_auroc": m["test_auroc_ovr_macro"]}
                print(f"[train_supcon_probe][pooled] {key}: acc={m['test_accuracy']:.3f} auroc={m['test_auroc_ovr_macro']:.3f}")
    return results


def evaluate_lodo(acts, labels, groups, dataset_source, layer: int, seed: int, held_out_sources=None) -> dict:
    X = acts[:, layer, :]
    rng = np.random.default_rng(seed)
    sources_present = [s for s in DATASET_SOURCES if (dataset_source == s).any()]
    if held_out_sources is not None:
        sources_present = [s for s in sources_present if s in held_out_sources]

    results = {}
    for input_kind in ["pca", "raw"]:
        for loss_kind in ["vanilla", "cross_source_only"]:
            for temperature in [0.05, 0.1, 0.2]:
                key = f"{input_kind}_{loss_kind}_t{temperature}"
                fold_aurocs = []
                for held_out in sources_present:
                    test_mask = dataset_source == held_out
                    if len(set(labels[test_mask])) < 2:
                        continue
                    test_groups = set(groups[test_mask])
                    train_mask = (dataset_source != held_out) & (~np.isin(groups, list(test_groups)))
                    train_idx_all = np.where(train_mask)[0]
                    rng.shuffle(train_idx_all)
                    n_val = max(1, int(0.2 * len(train_idx_all)))
                    val_idx, tr_idx = train_idx_all[:n_val], train_idx_all[n_val:]
                    test_idx = np.where(test_mask)[0]
                    if len(set(labels[tr_idx])) < 2 or len(set(labels[val_idx])) < 2:
                        continue
                    if loss_kind == "cross_source_only" and len(set(dataset_source[tr_idx])) < 2:
                        # Only one source left in train once held_out is excluded (can
                        # happen with few total sources) -- cross_source_only has no
                        # valid positive pairs in that case (see src/supcon.py), so
                        # skip rather than silently training on zero gradient signal.
                        print(f"[train_supcon_probe][lodo] {key} held_out={held_out}: "
                              f"skipped, only 1 source in train")
                        continue
                    m = supcon_stage(
                        X[tr_idx], labels[tr_idx], X[val_idx], labels[val_idx], X[test_idx], labels[test_idx],
                        dataset_source[tr_idx], input_kind, loss_kind, temperature, seed,
                    )
                    fold_aurocs.append(m["test_auroc_ovr_macro"])
                mean_auroc = float(np.nanmean(fold_aurocs)) if fold_aurocs else float("nan")
                results[key] = {"fold_aurocs": fold_aurocs, "mean_auroc": mean_auroc}
                print(f"[train_supcon_probe][lodo] {key}: fold_aurocs={['%.3f' % a for a in fold_aurocs]} mean={mean_auroc:.4f}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations_combined")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_combined")
    ap.add_argument("--layer", type=int, default=None, help="Defaults to run_summary.json's best_layer_lr.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows = \
        load_combined_activation_bundle(args.activations_dir)

    if args.layer is None:
        with (args.results_dir / "run_summary.json").open("r", encoding="utf-8") as f:
            args.layer = json.load(f)["best_layer_lr"]
    print(f"[train_supcon_probe] layer={args.layer}")

    print("\n[train_supcon_probe] === POOLED ===")
    pooled = evaluate_pooled(acts, labels, groups, dataset_source, args.layer, args.seed)

    print("\n[train_supcon_probe] === LODO (group-safe) ===")
    lodo = evaluate_lodo(acts, labels, groups, dataset_source, args.layer, args.seed)

    best_pooled = max(pooled.items(), key=lambda kv: kv[1]["test_auroc"])
    best_lodo = max(lodo.items(), key=lambda kv: kv[1]["mean_auroc"] if not np.isnan(kv[1]["mean_auroc"]) else -1)
    print(f"\n[train_supcon_probe] best pooled config: {best_pooled[0]} (auroc={best_pooled[1]['test_auroc']:.4f})")
    print(f"[train_supcon_probe] best LODO config: {best_lodo[0]} (mean_auroc={best_lodo[1]['mean_auroc']:.4f})")

    out = {"layer": args.layer, "pooled": pooled, "lodo": lodo,
           "best_pooled_config": best_pooled[0], "best_lodo_config": best_lodo[0]}
    with (args.results_dir / "supcon_report.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[train_supcon_probe] wrote {args.results_dir / 'supcon_report.json'}")


if __name__ == "__main__":
    main()
