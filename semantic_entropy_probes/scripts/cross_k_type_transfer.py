"""Attempted robustness check: does the K=3 type signal survive training on K=1 only?
(Spoiler, see RESULT below: inconclusive -- pair with cross_k_positive_control.py.)

Motivation (advisor review of the staged K=1/K=2/K=3 type-probing results):
K=3's 43 kept rows come from 43 *distinct* source problems (one K=3 row per
problem) -- grouping is vacuous there, so a probe trained AND tested within
K=3 could partly be recognizing "what kind of problem is this" rather than
"which ambiguity types are present," since which types an annotator
introduces plausibly correlates with the problem itself. K=1/K=2 don't have
this issue (~2.3-2.4 rows per group, so the type label varies within a
group, forcing genuine within-problem contrast). This confound could
entirely explain the K=1 -> K=2 -> K=3 signal-strength trend without type
identity being any more decodable at higher K.

The test: train the per-type binary heads on K=1 rows only, restricted to
source problems that share NO group with the eval set (i.e. genuinely
unseen problems), then evaluate on ALL rows at the target K.

RESULT (--eval-k 3, and see cross_k_positive_control.py): this comes back
near chance (best layer 18, macro-AUROC=0.566), which on its own looks like
it falsifies transfer -- but it does NOT, by itself, distinguish "no
transferable type signal" from "the 43-row/35-problem training set is too
small to learn anything at all, in-distribution or not." That ambiguity is
exactly why cross_k_positive_control.py exists: it evaluates the SAME
trained classifiers on held-out K=1 rows (same task, same distribution,
still group-disjoint from training) instead of K=3 rows. That positive
control also lands near chance (0.544 best layer, vs. 0.571 for a
properly-sized within-K1 split) -- confirming the training set genuinely
can't learn the task reliably at this size. Conclusion: the K=1->K3
transfer null is INCONCLUSIVE, not evidence against a real K=3 signal. Do
not read a near-chance result from this script alone as confirming the
problem-topic confound -- always check the positive control first. Full
three-way discussion (confound / K=3-context-specific signal / PCA
distribution shift) is in FINDINGS.md's "Cross-K transfer check" section.

All 43 K=3 groups turn out to overlap with K=1's 78 groups (the same ~80
LiveCodeBench problems are reused across K), leaving only 59 of 188 K=1
rows (35 groups) usable as a clean, K=3-disjoint training pool -- smaller
than the full K=1 stage's 112-row train split, which is exactly why the
positive control above was necessary rather than assumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_activation_bundle
from src.type_probes import (
    MIN_SUPPORT,
    fit_and_eval_multilabel_type,
    multilabel_subset,
    select_type_vocab,
    singlelabel_subset,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-k", type=int, required=True, choices=[2, 3])
    args = ap.parse_args()

    results_dir = ROOT / f"results_types_cross_k1_to_k{args.eval_k}"
    results_dir.mkdir(parents=True, exist_ok=True)
    seed = 0

    acts, labels, groups, token_lens, meta_rows = load_activation_bundle(ROOT / "activations")
    vocab = select_type_vocab(meta_rows, k=1, min_support=MIN_SUPPORT)

    keep1, y1 = singlelabel_subset(meta_rows, vocab, k=1)
    keep_eval, Y_eval = multilabel_subset(meta_rows, vocab, k=args.eval_k)
    groups1, groups_eval = groups[keep1], groups[keep_eval]

    eval_groups = set(groups_eval)
    disjoint_mask = np.array([g not in eval_groups for g in groups1])
    train_pool_idx = keep1[disjoint_mask]
    train_pool_groups = groups1[disjoint_mask]
    train_pool_y = y1[disjoint_mask]

    # Convert Stage-1 single-label ints to one-hot multi-hot targets so
    # fit_and_eval_multilabel_type (Stage-2's function) can be reused as-is.
    Y_train_pool = np.zeros((len(train_pool_y), len(vocab)), dtype=np.int64)
    Y_train_pool[np.arange(len(train_pool_y)), train_pool_y] = 1

    print(f"[cross-k] eval_k={args.eval_k}, vocab: {vocab}")
    print(f"[cross-k] K=1 rows disjoint from all K={args.eval_k} groups: {len(train_pool_idx)} "
          f"(of {len(keep1)} total kept K=1 rows, {len(set(train_pool_groups))} unique problems)")
    print(f"[cross-k] K={args.eval_k} eval rows (untouched, held out entirely): {len(keep_eval)}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    sub_idx = np.arange(len(train_pool_idx))
    train_sub, val_sub = next(gss.split(sub_idx, train_pool_y, train_pool_groups))
    train_idx, val_idx = train_pool_idx[train_sub], train_pool_idx[val_sub]
    Y_train, Y_val = Y_train_pool[train_sub], Y_train_pool[val_sub]
    assert set(groups[train_idx]).isdisjoint(groups[val_idx])
    assert set(groups[train_idx]).isdisjoint(eval_groups)
    assert set(groups[val_idx]).isdisjoint(eval_groups)
    print(f"[cross-k] train={len(train_idx)} val={len(val_idx)} (both group-disjoint from K={args.eval_k} eval set)")

    n_states = acts.shape[1]
    layer_rows = []
    for layer in range(n_states):
        X = acts[:, layer, :]
        m = fit_and_eval_multilabel_type(
            X[train_idx], Y_train, X[val_idx], Y_val, X[keep_eval], Y_eval,
            vocab, seed=seed,
        )
        per_type = m.pop("per_type")
        layer_rows.append({"layer": layer, **{k: v for k, v in m.items() if k != "per_type"}})
        print(f"[cross-k] layer {layer:2d}/{n_states-1}: macro_auroc={m['macro_auroc']:.3f} "
              f"exact_match={m['exact_match_accuracy']:.3f} jaccard={m['jaccard_macro']:.3f}"
              f"{', UNCONVERGED' if m['any_unconverged'] else ''}")
        if layer in (10, 11, 12, 16):
            print(f"[cross-k]   per-type @ layer {layer}: " +
                  ", ".join(f"{p['type']}={p['test_auroc']:.3f}" if not np.isnan(p['test_auroc']) else f"{p['type']}=NaN"
                             for p in per_type))

    with (results_dir / "layer_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader()
        writer.writerows(layer_rows)

    best = max(layer_rows, key=lambda r: r["macro_auroc"] if not np.isnan(r["macro_auroc"]) else -1)
    summary = {
        "description": f"Trained on K=1 rows only (group-disjoint from all K={args.eval_k} problems), "
                        f"evaluated on all {len(keep_eval)} K={args.eval_k} rows",
        "vocab": vocab,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_eval": len(keep_eval),
        "best_layer": best["layer"],
        "best_layer_macro_auroc": best["macro_auroc"],
    }
    with (results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[cross-k] done. Best layer {best['layer']}: macro_auroc={best['macro_auroc']:.3f}")
    print(f"[cross-k] results written to {results_dir}")


if __name__ == "__main__":
    main()
