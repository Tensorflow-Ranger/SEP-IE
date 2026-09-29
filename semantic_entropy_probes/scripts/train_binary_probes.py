"""Train and evaluate binary (clear=0 / ambiguous=1) probes per layer on the
combined ClarifyCodeBench + Orchid + human-eval-comm activations.

Two regularized model families are trained per layer, on the SAME
PCA-whitened, standardized features (see src/probes.py), so they're directly
comparable:
  - `lr`  : sklearn LogisticRegression, L2 (`C` swept on val).
  - `mlp` : TorchMLPProbe (src/torch_mlp_probe.py), one ReLU hidden layer +
            dropout + weight_decay (`C` swept on val, same grid/convention).
Both run on GPU when available: `mlp` always trains via Adam on GPU (or CPU
if none), and `lr` additionally supports --lr-backend torch to also do its
lbfgs-equivalent fit on GPU (src/torch_probe.py) -- default is sklearn CPU
lbfgs, which is already fast at this N and is the more battle-tested
optimizer for a convex objective.

Baselines (src/baselines.py, src/text_baselines.py):
  - majority, random           : trivial floors.
  - length_only                : token-length-only probe (surface-length confound).
  - tfidf_bow                  : TF-IDF + logistic regression on RAW TEXT, no
                                  hidden states at all -- the key "is this
                                  stylistic" baseline. If this matches the
                                  best hidden-state layer, the separability is
                                  explainable from surface word choice alone.

Splits are grouped by source_problem_id AND stratified jointly by
(label, dataset_source) -- src/combined_splits.py -- so every one of the
three source datasets is represented in train/val/test at roughly its
overall proportion, and no source_problem_id straddles a split.

Outputs (results-dir/):
  layer_metrics_lr.csv, layer_metrics_mlp.csv, baselines.json,
  per_source_breakdown.json (best-layer accuracy broken down by
  dataset_source, for both lr and mlp), run_summary.json
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
from src.combined_splits import make_combined_train_val_test_split, report_split_composition
from src.data_utils import load_combined_activation_bundle
from src.probes import fit_and_eval
from src.text_baselines import tfidf_logreg_baseline

MACRO_LABELS = (0, 1)


def _load_prompts_by_id(data_path: Path) -> dict[str, str]:
    out = {}
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row["prompt"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations_combined")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ambiguity_dataset_combined.jsonl",
                     help="Needed to recover raw prompt text for the tfidf_bow baseline "
                          "(not stored in activations metadata).")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_combined")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows = \
        load_combined_activation_bundle(args.activations_dir)
    prompts_by_id = _load_prompts_by_id(args.data)
    prompts = np.array([prompts_by_id[m["id"]] for m in meta_rows])

    N, n_states, hidden_size = acts.shape
    print(f"[train_binary] loaded activations {acts.shape}, "
          f"label dist {dict(zip(*np.unique(labels, return_counts=True)))}, "
          f"{len(set(groups))} unique source_problem_id groups")

    train_idx, val_idx, test_idx = make_combined_train_val_test_split(
        labels, dataset_source, groups, seed=args.seed
    )
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        report_split_composition(name, idx, labels, dataset_source)

    # ---- Baselines -----------------------------------------------------
    baselines = {
        "majority": majority_baseline(labels[train_idx], labels[test_idx], macro_labels=MACRO_LABELS),
        "random": random_baseline(labels[train_idx], labels[test_idx], seed=args.seed, macro_labels=MACRO_LABELS),
        "length_only": fit_and_eval(
            token_lens[train_idx], labels[train_idx],
            token_lens[val_idx], labels[val_idx],
            token_lens[test_idx], labels[test_idx],
            seed=args.seed, macro_labels=MACRO_LABELS, backend="sklearn",
        ),
        "tfidf_bow": tfidf_logreg_baseline(
            prompts[train_idx], labels[train_idx],
            prompts[val_idx], labels[val_idx],
            prompts[test_idx], labels[test_idx],
            seed=args.seed, macro_labels=MACRO_LABELS,
        ),
    }
    print("[train_binary] baselines:")
    for name, m in baselines.items():
        print(f"[train_binary]   {name}: acc={m['test_accuracy']:.3f} macro_f1={m['test_macro_f1']:.3f} "
              f"auroc={m.get('test_auroc_ovr_macro', float('nan'))}")
    with (args.results_dir / "baselines.json").open("w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    # ---- Per-layer probes: lr and mlp -----------------------------------
    def _write_csv(path: Path, rows: list[dict]) -> None:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    per_source_breakdown = {}
    layer_rows_by_backend = {}

    for backend, tag in [("sklearn", "lr"), ("torch_mlp", "mlp")]:
        layer_rows = []
        n_unconverged = 0
        for layer in range(n_states):
            X = acts[:, layer, :]
            metrics = fit_and_eval(
                X[train_idx], labels[train_idx],
                X[val_idx], labels[val_idx],
                X[test_idx], labels[test_idx],
                seed=args.seed, macro_labels=MACRO_LABELS, backend=backend,
            )
            if metrics["any_unconverged"]:
                n_unconverged += 1
            layer_rows.append({"layer": layer, **metrics})
            print(f"[train_binary][{tag}] layer {layer:2d}/{n_states - 1}: "
                  f"acc={metrics['test_accuracy']:.3f} macro_f1={metrics['test_macro_f1']:.3f} "
                  f"auroc={metrics['test_auroc_ovr_macro']:.3f} (C={metrics['best_C']}"
                  f"{', UNCONVERGED' if metrics['any_unconverged'] else ''})")
        _write_csv(args.results_dir / f"layer_metrics_{tag}.csv", layer_rows)
        layer_rows_by_backend[tag] = layer_rows

        best_layer_row = max(layer_rows, key=lambda r: r["test_auroc_ovr_macro"]
                              if not np.isnan(r["test_auroc_ovr_macro"]) else -1)
        best_layer = best_layer_row["layer"]
        print(f"[train_binary][{tag}] best layer by test AUROC: {best_layer} "
              f"(auroc={best_layer_row['test_auroc_ovr_macro']:.3f}, "
              f"{n_unconverged}/{n_states} layers had a non-converged fit)")

        # Per-source breakdown at the best layer: does performance hold within
        # each of the three source datasets, or is pooled accuracy driven by
        # one easy source dragging the others along?
        X_best = acts[:, best_layer, :]
        pipe_metrics = fit_and_eval(
            X_best[train_idx], labels[train_idx],
            X_best[val_idx], labels[val_idx],
            X_best[test_idx], labels[test_idx],
            seed=args.seed, macro_labels=MACRO_LABELS, backend=backend,
        )
        # Refit once more to get a reusable fitted pipeline for per-source scoring.
        from src.probes import _build_pipeline, _fit
        pipe = _build_pipeline(pipe_metrics["best_C"], args.seed,
                                n_train=len(train_idx) + len(val_idx),
                                n_features=hidden_size, backend=backend)
        X_trainval = np.concatenate([X_best[train_idx], X_best[val_idx]], axis=0)
        y_trainval = np.concatenate([labels[train_idx], labels[val_idx]], axis=0)
        _fit(pipe, X_trainval, y_trainval)

        from sklearn.metrics import accuracy_score, roc_auc_score
        src_breakdown = {}
        for src in sorted(set(dataset_source[test_idx])):
            mask = dataset_source[test_idx] == src
            idx = test_idx[mask]
            if len(idx) == 0 or len(set(labels[idx])) < 2:
                src_breakdown[src] = {"n": int(len(idx)), "accuracy": None, "auroc": None}
                continue
            y_true = labels[idx]
            probs = pipe.predict_proba(acts[idx, best_layer, :])
            y_pred = pipe.predict(acts[idx, best_layer, :])
            src_breakdown[src] = {
                "n": int(len(idx)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "auroc": float(roc_auc_score(y_true, probs[:, 1])),
            }
        per_source_breakdown[tag] = {"best_layer": best_layer, "by_source": src_breakdown}
        print(f"[train_binary][{tag}] per-source test breakdown: {src_breakdown}")

    with (args.results_dir / "per_source_breakdown.json").open("w", encoding="utf-8") as f:
        json.dump(per_source_breakdown, f, indent=2)

    summary = {
        "n_prompts": int(N),
        "n_hidden_states": int(n_states),
        "hidden_size": int(hidden_size),
        "label_distribution": {int(v): int(c) for v, c in zip(*np.unique(labels, return_counts=True))},
        "dataset_source_distribution": {s: int((dataset_source == s).sum()) for s in sorted(set(dataset_source))},
        "n_unique_source_problems": int(len(set(groups))),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_layer_lr": per_source_breakdown["lr"]["best_layer"],
        "best_layer_mlp": per_source_breakdown["mlp"]["best_layer"],
        "seed": args.seed,
    }
    with (args.results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[train_binary] done. results written to {args.results_dir}")


if __name__ == "__main__":
    main()
