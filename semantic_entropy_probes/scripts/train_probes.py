"""Train and evaluate one logistic probe per layer on frozen Qwen3-8B activations.

Target: ambiguity COUNT K in {0,1,2,3} (ordinal), not a binary flag.

Pipeline:
  1. Load activations.npy + metadata.jsonl (from src/extract_activations.py).
  2. Build one grouped, stratified train/val/test split (grouped by
     source_problem_id, so a problem's ambiguous rewrites and its
     reconstructed-full siblings never straddle a split boundary).
  3. For each layer: standardize -> sweep C on val (by Spearman correlation)
     -> refit on train+val -> score on test (accuracy, macro-F1, OVR-macro
     AUROC, Spearman, MAE). Also report grouped 5-fold CV Spearman per layer
     at the selected C, since a single held-out test split is noisy across
     37 layers.
  4. Baselines: majority class, random (draws from train's K distribution),
     and a token-length-only probe -- K=0 rows are always at least as long as
     their source row (answers appended), so length alone may carry some
     signal; this baseline tells us how much of any layer's result could be
     explained by length rather than content.

--min-k / the K=0 confound
---------------------------
Every K=0 row carries an appended "Clarifications:" block that no K>=1 row
has (see build_dataset.py) -- a trivially separable surface feature that can
dominate Spearman on the full K=0..3 range even if a probe has learned
nothing about *degree* among K=1/2/3 (a probe that only detects the header
and predicts a constant K>=1 value would already show strong positive
correlation). `--min-k 1` restricts to the K in {1,2,3} rows -- all
`modified_content` verbatim, same provenance, no header -- to isolate the
actual "does the residual stream track degree" signal from that artifact.
Run both: `--min-k 0` (default, full range, upper-bound/reference) and
`--min-k 1` (load-bearing for the degree claim), into separate --results-dir
runs, and report both in the README rather than only the flattering one.

Outputs (results/):
  - layer_metrics.csv     : per-layer test accuracy/macro-F1/AUROC/Spearman/MAE + chosen C
  - layer_cv_spearman.csv : per-layer grouped-CV Spearman mean/std
  - baselines.json        : majority / random / length-only baseline metrics
  - run_summary.json      : split sizes, class balance, config
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import majority_baseline, random_baseline
from src.data_utils import load_activation_bundle
from src.probes import cv_spearman, fit_and_eval
from src.splits import make_grouped_cv_folds, make_train_val_test_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--min-k", type=int, default=0,
                     help="Restrict to rows with label >= min_k. min_k=1 excludes the "
                          "K=0 (Clarifications-header) class entirely -- see module docstring.")
    ap.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn",
                     help="sklearn=CPU lbfgs (default). torch=GPU Adam (src/torch_probe.py), "
                          "much faster with the GPU otherwise idle for this step.")
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    acts, labels, groups, token_lens, meta_rows = load_activation_bundle(args.activations_dir)

    if args.min_k > 0:
        keep = labels >= args.min_k
        acts, labels, groups, token_lens = acts[keep], labels[keep], groups[keep], token_lens[keep]
        print(f"[train_probes] --min-k {args.min_k}: restricted to {keep.sum()} rows "
              f"(K=0 Clarifications-header class excluded)")

    macro_labels = tuple(k for k in (0, 1, 2, 3) if k >= args.min_k)

    N, n_states, hidden_size = acts.shape
    values, counts = np.unique(labels, return_counts=True)
    k_dist = {int(v): int(c) for v, c in zip(values, counts)}
    print(f"[train_probes] loaded activations {acts.shape}, K distribution {k_dist}, "
          f"{len(set(groups))} unique source problems")

    train_idx, val_idx, test_idx = make_train_val_test_split(labels, groups, seed=args.seed)
    print(f"[train_probes] split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        v, c = np.unique(labels[idx], return_counts=True)
        print(f"[train_probes]   {name}: {len(idx)} rows, K dist {dict(zip(v.tolist(), c.tolist()))}")

    cv_folds = list(make_grouped_cv_folds(labels, groups, n_splits=args.cv_folds, seed=args.seed))

    # ---- Baselines -----------------------------------------------------
    baselines = {
        "majority": majority_baseline(labels[train_idx], labels[test_idx], macro_labels=macro_labels),
        "random": random_baseline(labels[train_idx], labels[test_idx], seed=args.seed, macro_labels=macro_labels),
        "length_only": fit_and_eval(
            token_lens[train_idx], labels[train_idx],
            token_lens[val_idx], labels[val_idx],
            token_lens[test_idx], labels[test_idx],
            seed=args.seed, macro_labels=macro_labels, backend=args.backend,
        ),
    }
    print("[train_probes] baselines:")
    for name, m in baselines.items():
        print(f"[train_probes]   {name}: acc={m['test_accuracy']:.3f} macro_f1={m['test_macro_f1']:.3f} "
              f"mae={m['test_mae']:.3f} spearman={m['test_spearman']}")

    with (args.results_dir / "baselines.json").open("w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    # ---- Per-layer probes ------------------------------------------------
    layer_metrics_rows = []
    layer_cv_rows = []
    n_unconverged_layers = 0

    for layer in range(n_states):
        X = acts[:, layer, :]
        metrics = fit_and_eval(
            X[train_idx], labels[train_idx],
            X[val_idx], labels[val_idx],
            X[test_idx], labels[test_idx],
            seed=args.seed, macro_labels=macro_labels, backend=args.backend,
        )
        if metrics["any_unconverged"]:
            n_unconverged_layers += 1
        layer_metrics_rows.append({"layer": layer, **metrics})

        cv_mean, cv_std = cv_spearman(X, labels, cv_folds, C=metrics["best_C"], seed=args.seed, backend=args.backend)
        layer_cv_rows.append({"layer": layer, "cv_spearman_mean": cv_mean, "cv_spearman_std": cv_std, "C": metrics["best_C"]})

        print(f"[train_probes] layer {layer:2d}/{n_states - 1}: "
              f"test_spearman={metrics['test_spearman']:.3f} test_acc={metrics['test_accuracy']:.3f} "
              f"test_macro_f1={metrics['test_macro_f1']:.3f} cv_spearman={cv_mean:.3f}+/-{cv_std:.3f} "
              f"(C={metrics['best_C']}{', UNCONVERGED' if metrics['any_unconverged'] else ''})")

    if n_unconverged_layers:
        print(f"[train_probes] WARNING: {n_unconverged_layers}/{n_states} layers had at least one "
              f"non-converged lbfgs fit (see 'any_unconverged' column in layer_metrics.csv)")

    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(args.results_dir / "layer_metrics.csv", layer_metrics_rows)
    _write_csv(args.results_dir / "layer_cv_spearman.csv", layer_cv_rows)

    def _rank_key(r):
        return r["test_spearman"] if not np.isnan(r["test_spearman"]) else -1
    best_layer = max(layer_metrics_rows, key=_rank_key)
    summary = {
        "min_k": args.min_k,
        "backend": args.backend,
        "macro_labels": list(macro_labels),
        "n_prompts": int(N),
        "n_hidden_states": int(n_states),
        "hidden_size": int(hidden_size),
        "k_distribution": k_dist,
        "n_unique_source_problems": int(len(set(groups))),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_layer_by_test_spearman": best_layer["layer"],
        "best_layer_test_spearman": best_layer["test_spearman"],
        "best_layer_test_accuracy": best_layer["test_accuracy"],
        "best_layer_test_macro_f1": best_layer["test_macro_f1"],
        "n_unconverged_layers": n_unconverged_layers,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
    }
    with (args.results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[train_probes] done. Best layer by test Spearman: {best_layer['layer']} "
          f"(rho={best_layer['test_spearman']:.3f})")
    print(f"[train_probes] results written to {args.results_dir}")


if __name__ == "__main__":
    main()
