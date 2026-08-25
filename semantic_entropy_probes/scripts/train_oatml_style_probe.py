"""Train a probe using OATML's actual Semantic Entropy Probes method,
adapted to this project's ambiguity-count target.

Ported functions: ../semantic-entropy-probes (OATML, Kossen et al.
arXiv:2406.15927), semantic_entropy_probes/train-latent-probe.ipynb.
See src/oatml_sep.py's module docstring for exactly what's verbatim vs.
adapted -- including the important caveat that K (this project's target) is
NOT semantic entropy: K is annotator-counted missing specification detail
(an input property), not measured output disagreement under sampling (which
this project never computes). Only OATML's binarization/bootstrap *methods*
are reused here, not their target.

## Scope: K in {1,2,3} only -- no K=0

This deliberately excludes K=0. Every K=0 construction tried in this project
(appended-answers header, LiveCodeBench-paired even with Sample I/O
stripped) carries a residual confound unrelated to ambiguity (see
FINDINGS.md) -- including it here would reopen that problem. Restricting to
K>=1 means every row went through the same ClarifyCodeBench deletion-editing
process; only *how much* was deleted varies, which is the point.

## Two modes

--mode binary (default): OATML's actual approach -- `best_split` picks the
  variance-minimizing binary cut of K (threshold~1.0: K=1 "low" vs K in
  {2,3} "high"), one `LogisticRegression()` per layer, binary AUROC.

--mode multiclass: NOT part of OATML's notebook (they never classify more
  than 2 classes -- their whole framework is built around collapsing a
  continuous target to a binary decision). This trains the same
  `LogisticRegression()` directly on the three raw values K in {1,2,3}
  (sklearn handles multinomial fitting natively), evaluated with one-vs-rest
  macro AUROC instead of binary AUROC (src/oatml_sep.ovr_macro_auroc).

Both modes report Spearman correlation between the true label and the
model's probability-weighted expected value (src/oatml_sep.spearman_vs_
expected) -- also not in OATML's notebook, added because K is ordinal and
"does the probe track degree" is a meaningful question their accuracy/AUROC
metrics alone don't answer.

The best layer's fitted model is pickled to `<results-dir>/probe.pkl`,
mirroring OATML's own "save trained probes for inference" cell (their
notebook cell 44).

## --backend {sklearn, torch}

sklearn (default) is OATML's exact `LogisticRegression()` call, CPU-only
(scikit-learn has no GPU backend). torch is src/torch_probe.py's
TorchLogisticProbe -- the same model (one linear layer + softmax, L2
regularization) fit with Adam on GPU instead of lbfgs on CPU; empirically
~15-80x faster at this project's D=4096 scale (see src/torch_probe.py's
docstring for the validation against sklearn that motivated the specific
hyperparameters used). Both backends implement the same `.fit` /
`.predict_proba` / `.classes_` interface, so every evaluation function in
src/oatml_sep.py works unchanged with either.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_activation_bundle
from src.oatml_sep import (
    best_split,
    binarize_entropy,
    make_logistic_regression,
    save_probe,
    sklearn_evaluate_on_test,
    sklearn_train_and_evaluate,
    spearman_vs_expected,
)
from src.splits import make_grouped_cv_folds, make_train_val_test_split
from src.torch_probe import TorchLogisticProbe


def make_model(backend: str, max_iter: int, seed: int = 0):
    if backend == "sklearn":
        return make_logistic_regression(max_iter=max_iter)
    return TorchLogisticProbe(C=1.0, seed=seed)  # uses tuned defaults, see module docstring


def _fit_quiet(model, X, y):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return converged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations-dir", type=Path, default=ROOT / "activations")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_oatml")
    ap.add_argument("--mode", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--backend", choices=["sklearn", "torch"], default="sklearn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=1000)
    ap.add_argument("--cv-folds", type=int, default=5)
    args = ap.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)  # matches OATML's notebook rng seed

    acts, labels, groups, token_lens, meta_rows = load_activation_bundle(args.activations_dir)
    mask = labels >= 1
    acts, labels, groups, token_lens = acts[mask], labels[mask], groups[mask], token_lens[mask]
    print(f"[oatml_probe] mode={args.mode} backend={args.backend}, restricted to K>=1: {mask.sum()} rows, "
          f"K distribution {dict(zip(*(lambda v, c: (v.tolist(), c.tolist()))(*np.unique(labels, return_counts=True))))}")

    if args.mode == "binary":
        # OATML's actual method: find the variance-minimizing binary split of K.
        split_threshold = best_split(labels.astype(np.float64))
        target = binarize_entropy(labels.astype(np.float64), split_threshold).astype(int)
        print(f"[oatml_probe] best_split threshold = {split_threshold:.4f} "
              f"(K < thresh -> 0 'low', K > thresh -> 1 'high')")
    else:
        # Not in OATML's notebook -- raw 3-class target, see module docstring.
        split_threshold = None
        target = labels.astype(int)

    v, c = np.unique(target, return_counts=True)
    print(f"[oatml_probe] target distribution: {dict(zip(v.tolist(), c.tolist()))}")

    N, n_states, hidden_size = acts.shape
    train_idx, val_idx, test_idx = make_train_val_test_split(target, groups, seed=args.seed)
    print(f"[oatml_probe] split sizes: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        vv, cc = np.unique(target[idx], return_counts=True)
        print(f"[oatml_probe]   {name}: {len(idx)} rows, dist {dict(zip(vv.tolist(), cc.tolist()))}")

    cv_folds = list(make_grouped_cv_folds(target, groups, n_splits=args.cv_folds, seed=args.seed))

    layer_rows = []
    layer_models = []
    n_unconverged = 0

    for layer in range(n_states):
        X = acts[:, layer, :]
        model = make_model(args.backend, args.max_iter, seed=args.seed)

        if args.backend == "sklearn":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                train_metrics = sklearn_train_and_evaluate(
                    model, X[train_idx], target[train_idx], X[val_idx], target[val_idx]
                )
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        else:
            train_metrics = sklearn_train_and_evaluate(
                model, X[train_idx], target[train_idx], X[val_idx], target[val_idx]
            )
            # torch backend: no ConvergenceWarning mechanism; hitting max_iter
            # (rather than early-stopping via patience) is the analogous signal
            # that optimization may not have fully settled.
            converged = model.n_iter_ < model.max_iter
        if not converged:
            n_unconverged += 1

        test_metrics = sklearn_evaluate_on_test(model, X[test_idx], target[test_idx], rng)
        layer_models.append(model)

        # grouped 5-fold CV AUROC + Spearman (robustness check, mirrors train_probes.py)
        cv_aucs, cv_rhos = [], []
        for tr_idx, te_idx in cv_folds:
            y_te = target[te_idx]
            if len(np.unique(y_te)) < 2:
                continue
            m = make_model(args.backend, args.max_iter, seed=args.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                m.fit(X[tr_idx], target[tr_idx])
            probs = m.predict_proba(X[te_idx])
            cv_rhos.append(spearman_vs_expected(y_te, probs, m.classes_))
            if args.mode == "binary":
                from sklearn.metrics import roc_auc_score
                cv_aucs.append(roc_auc_score(y_te, probs[:, 1]))
            else:
                from src.oatml_sep import ovr_macro_auroc
                cv_aucs.append(ovr_macro_auroc(y_te, probs, labels=m.classes_))
        cv_auc_mean = float(np.mean(cv_aucs)) if cv_aucs else float("nan")
        cv_auc_std = float(np.std(cv_aucs)) if cv_aucs else float("nan")
        cv_rhos = [r for r in cv_rhos if not np.isnan(r)]
        cv_rho_mean = float(np.mean(cv_rhos)) if cv_rhos else float("nan")
        cv_rho_std = float(np.std(cv_rhos)) if cv_rhos else float("nan")

        row = {
            "layer": layer,
            "converged": converged,
            "train_loss": train_metrics["train_loss"],
            "valid_loss": train_metrics["valid_loss"],
            "valid_accuracy": train_metrics["valid_accuracy"],
            "valid_auroc": train_metrics["valid_auroc"],
            "valid_spearman": train_metrics["valid_spearman"],
            "test_loss": test_metrics["test_loss"],
            "test_accuracy": test_metrics["test_accuracy"],
            "test_auroc": test_metrics["test_auroc"],
            "test_auroc_ci_low": test_metrics["test_auroc_ci"]["low"] if test_metrics["test_auroc_ci"] else float("nan"),
            "test_auroc_ci_high": test_metrics["test_auroc_ci"]["high"] if test_metrics["test_auroc_ci"] else float("nan"),
            "test_spearman": test_metrics["test_spearman"],
            "cv_auroc_mean": cv_auc_mean,
            "cv_auroc_std": cv_auc_std,
            "cv_spearman_mean": cv_rho_mean,
            "cv_spearman_std": cv_rho_std,
        }
        layer_rows.append(row)
        print(f"[oatml_probe] layer {layer:2d}/{n_states-1}: test_auroc={row['test_auroc']:.3f} "
              f"[{row['test_auroc_ci_low']:.3f},{row['test_auroc_ci_high']:.3f}] "
              f"test_spearman={row['test_spearman']:.3f} test_acc={row['test_accuracy']:.3f} "
              f"cv_auroc={cv_auc_mean:.3f}+/-{cv_auc_std:.3f} cv_spearman={cv_rho_mean:.3f}+/-{cv_rho_std:.3f}"
              f"{' UNCONVERGED' if not converged else ''}")

    if n_unconverged:
        print(f"[oatml_probe] WARNING: {n_unconverged}/{n_states} layers did not converge "
              f"(max_iter={args.max_iter})")

    with (args.results_dir / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    def _rank_key(r):
        return r["test_spearman"] if not np.isnan(r["test_spearman"]) else -1
    best_layer = max(layer_rows, key=_rank_key)
    best_layer_idx = best_layer["layer"]
    best_model = layer_models[best_layer_idx]

    probe_path = args.results_dir / "probe.pkl"
    save_probe(best_model, probe_path, metadata={
        "mode": args.mode,
        "layer": best_layer_idx,
        "split_threshold": split_threshold,
        "classes": best_model.classes_.tolist(),
        "test_auroc": best_layer["test_auroc"],
        "test_spearman": best_layer["test_spearman"],
        "activations_dir": str(args.activations_dir),
        "extraction_point": "last_token_of_templated_prompt (TBG)",
        "seed": args.seed,
    })
    print(f"[oatml_probe] saved best-layer ({best_layer_idx}) probe -> {probe_path}")

    summary = {
        "method": f"OATML semantic-entropy-probes, adapted, mode={args.mode}, backend={args.backend}",
        "backend": args.backend,
        "split_threshold": split_threshold,
        "target_distribution": {int(vv): int(cc) for vv, cc in zip(v, c)},
        "n_prompts": int(N),
        "n_hidden_states": int(n_states),
        "hidden_size": int(hidden_size),
        "split_sizes": {"train": len(train_idx), "val": len(val_idx), "test": len(test_idx)},
        "best_layer_by_test_spearman": best_layer_idx,
        "best_layer_test_auroc": best_layer["test_auroc"],
        "best_layer_test_spearman": best_layer["test_spearman"],
        "best_layer_test_accuracy": best_layer["test_accuracy"],
        "n_unconverged_layers": n_unconverged,
        "max_iter": args.max_iter,
        "seed": args.seed,
        "probe_path": str(probe_path),
    }
    with (args.results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[oatml_probe] done. Best layer by test Spearman: {best_layer_idx} "
          f"(rho={best_layer['test_spearman']:.3f}, AUROC={best_layer['test_auroc']:.3f})")
    print(f"[oatml_probe] results written to {args.results_dir}")


if __name__ == "__main__":
    main()
