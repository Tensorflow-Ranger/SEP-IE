"""Stage 1: predict WHICH ambiguity type is present, for K=1 rows only.

K=1 rows have exactly one ambiguity_types tag -- ordinary nominal multiclass
classification (which single type), not the ordinal K-count problem in
train_probes.py. See src/type_probes.py module docstring for the full
methodology (rare-type dropping, splitting, metrics).

Outputs (results-dir):
  - layer_metrics.csv   : per-layer test accuracy/macro-F1/macro-AUROC + chosen C
  - layer_cv_macro_f1.csv : per-layer grouped-CV macro-F1 mean/std
  - baselines.json      : majority-class + random baselines
  - run_summary.json    : vocab, split sizes, class balance, config
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

from src.data_utils import load_activation_bundle
from src.type_probes import (
    MIN_SUPPORT,
    cv_multiclass_macro_auroc,
    cv_multiclass_macro_f1,
    fit_and_eval_multiclass_type,
    majority_baseline_multiclass,
    make_type_split,
    random_baseline_multiclass,
    select_type_vocab,
    singlelabel_subset,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_types_k1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    acts, labels, groups, token_lens, meta_rows = load_activation_bundle(args.activations_dir)

    vocab = select_type_vocab(meta_rows, k=1, min_support=args.min_support)
    keep, y = singlelabel_subset(meta_rows, vocab, k=1)
    print(f"[k1] vocab ({len(vocab)} types, min_support={args.min_support}): {vocab}")
    print(f"[k1] {len(keep)} K=1 rows kept (of 199 total K=1 rows)")

    acts_k = acts[keep]
    groups_k = groups[keep]
    token_lens_k = token_lens[keep]

    train_idx, val_idx, test_idx = make_type_split(y, groups_k, seed=args.seed)
    print(f"[k1] split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        v, c = np.unique(y[idx], return_counts=True)
        dist = {vocab[vi]: int(ci) for vi, ci in zip(v, c)}
        print(f"[k1]   {name}: {len(idx)} rows, {dist}")

    length_only = fit_and_eval_multiclass_type(
        token_lens_k[train_idx], y[train_idx], token_lens_k[val_idx], y[val_idx], token_lens_k[test_idx], y[test_idx],
        vocab, seed=args.seed,
    )
    length_only.pop("confusion_matrix")
    baselines = {
        "majority": majority_baseline_multiclass(y[train_idx], y[test_idx], vocab),
        "random": random_baseline_multiclass(y[train_idx], y[test_idx], vocab, seed=args.seed),
        "length_only": length_only,
    }
    print(f"[k1] baselines: majority acc={baselines['majority']['test_accuracy']:.3f} "
          f"macro_f1={baselines['majority']['test_macro_f1']:.3f} | "
          f"random acc={baselines['random']['test_accuracy']:.3f} macro_f1={baselines['random']['test_macro_f1']:.3f} | "
          f"length_only acc={length_only['test_accuracy']:.3f} macro_f1={length_only['test_macro_f1']:.3f} "
          f"macro_auroc={length_only['test_auroc_ovr_macro']:.3f}")
    with (args.results_dir / "baselines.json").open("w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    n_states = acts.shape[1]
    layer_rows, cv_rows = [], []
    n_unconverged = 0

    for layer in range(n_states):
        X = acts_k[:, layer, :]
        m = fit_and_eval_multiclass_type(
            X[train_idx], y[train_idx], X[val_idx], y[val_idx], X[test_idx], y[test_idx],
            vocab, seed=args.seed,
        )
        cm = m.pop("confusion_matrix")
        if m["any_unconverged"]:
            n_unconverged += 1
        layer_rows.append({"layer": layer, **m})

        cv_mean, cv_std = cv_multiclass_macro_f1(X, y, groups_k, vocab, C=m["best_C"], seed=args.seed, n_splits=args.cv_folds)
        cv_auroc_mean, cv_auroc_std = cv_multiclass_macro_auroc(X, y, groups_k, vocab, C=m["best_C"], seed=args.seed, n_splits=args.cv_folds)
        cv_rows.append({"layer": layer, "cv_macro_f1_mean": cv_mean, "cv_macro_f1_std": cv_std,
                         "cv_macro_auroc_mean": cv_auroc_mean, "cv_macro_auroc_std": cv_auroc_std, "C": m["best_C"]})

        print(f"[k1] layer {layer:2d}/{n_states - 1}: acc={m['test_accuracy']:.3f} "
              f"macro_f1={m['test_macro_f1']:.3f} macro_auroc={m['test_auroc_ovr_macro']:.3f} "
              f"cv_macro_f1={cv_mean:.3f}+/-{cv_std:.3f} cv_macro_auroc={cv_auroc_mean:.3f}+/-{cv_auroc_std:.3f} "
              f"(C={m['best_C']}{', UNCONVERGED' if m['any_unconverged'] else ''})")

        if layer == n_states - 1:
            np.save(args.results_dir / "last_layer_confusion_matrix.npy", np.array(cm))

    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(args.results_dir / "layer_metrics.csv", layer_rows)
    _write_csv(args.results_dir / "layer_cv_macro_f1.csv", cv_rows)

    def _rank(r):
        return r["test_auroc_ovr_macro"] if not np.isnan(r["test_auroc_ovr_macro"]) else -1
    best = max(layer_rows, key=_rank)
    best_cv = cv_rows[best["layer"]]
    assert best_cv["layer"] == best["layer"]

    values, counts = np.unique(y, return_counts=True)
    summary = {
        "stage": "k1_singlelabel_multiclass",
        "vocab": vocab,
        "min_support": args.min_support,
        "n_rows_kept": int(len(keep)),
        "n_unique_source_problems": int(len(set(groups_k))),
        "type_distribution": {vocab[v]: int(c) for v, c in zip(values, counts)},
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_layer_by_test_auroc": best["layer"],
        "best_layer_test_auroc": best["test_auroc_ovr_macro"],
        "best_layer_n_classes_in_macro_auroc": best.get("n_classes_in_macro_auroc"),
        "best_layer_cv_macro_auroc_mean": best_cv["cv_macro_auroc_mean"],
        "best_layer_cv_macro_auroc_std": best_cv["cv_macro_auroc_std"],
        "best_layer_test_macro_f1": best["test_macro_f1"],
        "best_layer_test_accuracy": best["test_accuracy"],
        "n_unconverged_layers": n_unconverged,
        "seed": args.seed,
        "cv_folds": args.cv_folds,
    }
    with (args.results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[k1] done. Best layer by test macro-AUROC: {best['layer']} (AUROC={best['test_auroc_ovr_macro']:.3f})")
    print(f"[k1] results written to {args.results_dir}")


if __name__ == "__main__":
    main()
