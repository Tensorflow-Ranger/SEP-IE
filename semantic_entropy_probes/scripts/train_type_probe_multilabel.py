"""Stage 2: predict WHICH ambiguity type(s) are present, for K=2 or K=3 rows.

K=2/K=3 rows carry 2 or 3 simultaneous ambiguity_types tags -> multi-label
(binary relevance) classification: one independent present/absent logistic
head per type in the Stage-1 vocab, sharing a PCA projection per layer.
See src/type_probes.py module docstring for full methodology.

Run separately per --k (2 or 3) per user's "progressive" scope decision;
same fixed type vocabulary (computed from K=1 support) is reused across
both so results are comparable across K values.

Outputs (results-dir):
  - layer_metrics.csv    : per-layer macro_auroc / exact_match / jaccard / hamming
  - layer_per_type.json  : per-layer, per-type AUROC/F1 (list of dicts)
  - baselines.json       : per-type majority + prevalence-matched-random baselines
  - run_summary.json     : vocab, split sizes, class balance, config
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
    cv_macro_auroc_multilabel,
    fit_and_eval_multilabel_type,
    majority_and_random_baseline_multilabel,
    make_type_split,
    multilabel_subset,
    select_type_vocab,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, choices=[2, 3])
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    args = ap.parse_args()
    results_dir = args.results_dir or (ROOT / f"results_types_k{args.k}")
    results_dir.mkdir(parents=True, exist_ok=True)

    acts, labels, groups, token_lens, meta_rows = load_activation_bundle(args.activations_dir)

    # Vocab is always computed from K=1 support (fixed across stages per user decision).
    vocab = select_type_vocab(meta_rows, k=1, min_support=args.min_support)
    keep, Y = multilabel_subset(meta_rows, vocab, k=args.k)
    n_total_k = int((labels == args.k).sum())
    print(f"[k{args.k}] vocab ({len(vocab)} types, from K=1 support): {vocab}")
    print(f"[k{args.k}] {len(keep)} rows kept (of {n_total_k} total K={args.k} rows; "
          f"dropped = rows with >=1 out-of-vocab type)")

    acts_k = acts[keep]
    groups_k = groups[keep]
    token_lens_k = token_lens[keep]
    prevalence = {vocab[i]: int(Y[:, i].sum()) for i in range(len(vocab))}
    print(f"[k{args.k}] per-type prevalence among kept rows: {prevalence}")

    if len(set(groups_k)) < 5:
        print(f"[k{args.k}] WARNING: only {len(set(groups_k))} unique source problems -- "
              f"split/CV results at this K will be very noisy.")

    # GroupShuffleSplit needs a dummy 1-D y; use row-sum (a coarse proxy) just to satisfy the API.
    dummy_y = Y.sum(axis=1)
    train_idx, val_idx, test_idx = make_type_split(dummy_y, groups_k, seed=args.seed)
    print(f"[k{args.k}] split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        dist = {vocab[i]: int(Y[idx, i].sum()) for i in range(len(vocab))}
        print(f"[k{args.k}]   {name}: {len(idx)} rows, per-type positives {dist}")

    baselines = majority_and_random_baseline_multilabel(Y[train_idx], Y[test_idx], vocab, seed=args.seed)
    length_only = fit_and_eval_multilabel_type(
        token_lens_k[train_idx], Y[train_idx], token_lens_k[val_idx], Y[val_idx], token_lens_k[test_idx], Y[test_idx],
        vocab, seed=args.seed,
    )
    length_only.pop("per_type")
    baselines["length_only"] = length_only
    print(f"[k{args.k}] baselines: majority macro_f1={baselines['majority']['macro_f1']:.3f} "
          f"exact_match={baselines['majority']['exact_match_accuracy']:.3f} | "
          f"random macro_f1={baselines['random_prevalence_matched']['macro_f1']:.3f} | "
          f"length_only macro_auroc={length_only['macro_auroc']:.3f} exact_match={length_only['exact_match_accuracy']:.3f}")
    with (results_dir / "baselines.json").open("w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    n_states = acts.shape[1]
    layer_rows, per_type_by_layer = [], []
    n_unconverged = 0

    for layer in range(n_states):
        X = acts_k[:, layer, :]
        m = fit_and_eval_multilabel_type(
            X[train_idx], Y[train_idx], X[val_idx], Y[val_idx], X[test_idx], Y[test_idx],
            vocab, seed=args.seed,
        )
        per_type = m.pop("per_type")
        if m["any_unconverged"]:
            n_unconverged += 1
        layer_rows.append({"layer": layer, "macro_auroc": m["macro_auroc"],
                            "n_types_in_macro_auroc": m["n_types_in_macro_auroc"],
                            "exact_match_accuracy": m["exact_match_accuracy"],
                            "jaccard_macro": m["jaccard_macro"], "hamming_loss": m["hamming_loss"],
                            "any_unconverged": m["any_unconverged"],
                            "n_train": m["n_train"], "n_val": m["n_val"], "n_test": m["n_test"]})
        per_type_by_layer.append({"layer": layer, "per_type": per_type})

        cv_mean, cv_std = cv_macro_auroc_multilabel(X, Y, groups_k, vocab, seed=args.seed)
        layer_rows[-1]["cv_macro_auroc_mean"] = cv_mean
        layer_rows[-1]["cv_macro_auroc_std"] = cv_std

        print(f"[k{args.k}] layer {layer:2d}/{n_states - 1}: macro_auroc={m['macro_auroc']:.3f} "
              f"exact_match={m['exact_match_accuracy']:.3f} jaccard={m['jaccard_macro']:.3f} "
              f"hamming={m['hamming_loss']:.3f} cv_macro_auroc={cv_mean:.3f}+/-{cv_std:.3f}"
              f"{', UNCONVERGED' if m['any_unconverged'] else ''}")

    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    _write_csv(results_dir / "layer_metrics.csv", layer_rows)
    with (results_dir / "layer_per_type.json").open("w", encoding="utf-8") as f:
        json.dump(per_type_by_layer, f, indent=2)

    def _rank(r):
        return r["macro_auroc"] if not np.isnan(r["macro_auroc"]) else -1
    best = max(layer_rows, key=_rank)

    summary = {
        "stage": f"k{args.k}_multilabel",
        "vocab": vocab,
        "min_support": args.min_support,
        "n_rows_kept": int(len(keep)),
        "n_rows_total_at_k": n_total_k,
        "n_unique_source_problems": int(len(set(groups_k))),
        "per_type_prevalence": prevalence,
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_layer_by_macro_auroc": best["layer"],
        "best_layer_macro_auroc": best["macro_auroc"],
        "best_layer_n_types_in_macro_auroc": best["n_types_in_macro_auroc"],
        "best_layer_cv_macro_auroc_mean": best["cv_macro_auroc_mean"],
        "best_layer_cv_macro_auroc_std": best["cv_macro_auroc_std"],
        "best_layer_exact_match_accuracy": best["exact_match_accuracy"],
        "best_layer_jaccard_macro": best["jaccard_macro"],
        "n_unconverged_layers": n_unconverged,
        "seed": args.seed,
    }
    with (results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[k{args.k}] done. Best layer by macro-AUROC: {best['layer']} (AUROC={best['macro_auroc']:.3f})")
    print(f"[k{args.k}] results written to {results_dir}")


if __name__ == "__main__":
    main()
