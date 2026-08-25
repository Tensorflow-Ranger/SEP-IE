"""Positive control for scripts/cross_k_type_transfer.py --eval-k 3.

The K=1->K3 transfer check trains on only 59 K=1 rows (train+val: 43 train
after the val split), across 35 source problems -- small enough that a
near-chance result could mean either "no transferable type signal" or "this
training set is simply too small to learn 7 binary heads at all," and those
have very different implications for how to read the K=3 within-stage CV
number. This script disambiguates: evaluate the EXACT SAME K=1-trained
classifiers (same train/val split, same seed, same code path) on the 129
K=1 rows that were excluded from that training pool (because their source
problem overlaps a K=3 problem) instead of on K=3 rows. Those 129 rows are
still group-disjoint from the 43-row training pool by construction, are the
SAME task (single-type K=1 classification, same vocab) and the SAME
feature distribution (K=1 prompts, not K=3 prompts) the classifier was
fit on -- the only thing that changes vs. the actual training data is which
held-out problems are being scored.

RESULT: near chance -- best layer 18, macro-AUROC=0.544, vs. 0.571+/-0.052
for a properly-sized within-K1 split (results_types_k1/). The 43-row/35-
problem training pool cannot reliably relearn even the in-distribution K=1
task, so cross_k_type_transfer.py --eval-k 3's near-chance result on K=3
is INCONCLUSIVE -- it would look identical whether or not a transferable
type signature exists, since the classifiers underlying it likely never
learned real signal even from K=1 in the first place. See FINDINGS.md's
"Cross-K transfer check" section for the full discussion.
"""

from __future__ import annotations

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
    results_dir = ROOT / "results_types_positive_control_k1"
    results_dir.mkdir(parents=True, exist_ok=True)
    seed = 0

    acts, labels, groups, token_lens, meta_rows = load_activation_bundle(ROOT / "activations")
    vocab = select_type_vocab(meta_rows, k=1, min_support=MIN_SUPPORT)

    keep1, y1 = singlelabel_subset(meta_rows, vocab, k=1)
    keep3, _Y3 = multilabel_subset(meta_rows, vocab, k=3)
    groups1 = groups[keep1]
    k3_groups = set(groups[keep3])

    disjoint_mask = np.array([g not in k3_groups for g in groups1])
    train_pool_idx, train_pool_groups, train_pool_y = keep1[disjoint_mask], groups1[disjoint_mask], y1[disjoint_mask]
    excluded_idx, excluded_y = keep1[~disjoint_mask], y1[~disjoint_mask]

    print(f"[pos-control] train pool: {len(train_pool_idx)} rows / {len(set(train_pool_groups))} problems "
          f"(same as cross_k_type_transfer.py --eval-k 3)")
    print(f"[pos-control] held-out-but-same-distribution eval pool: {len(excluded_idx)} K=1 rows "
          f"(the ones excluded from training because their problem overlaps K=3)")

    Y_train_pool = np.zeros((len(train_pool_y), len(vocab)), dtype=np.int64)
    Y_train_pool[np.arange(len(train_pool_y)), train_pool_y] = 1
    Y_eval = np.zeros((len(excluded_y), len(vocab)), dtype=np.int64)
    Y_eval[np.arange(len(excluded_y)), excluded_y] = 1

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    sub_idx = np.arange(len(train_pool_idx))
    train_sub, val_sub = next(gss.split(sub_idx, train_pool_y, train_pool_groups))
    train_idx, val_idx = train_pool_idx[train_sub], train_pool_idx[val_sub]
    Y_train, Y_val = Y_train_pool[train_sub], Y_train_pool[val_sub]
    assert set(groups[train_idx]).isdisjoint(groups[val_idx])
    assert set(groups[train_idx]).isdisjoint(groups[excluded_idx])
    print(f"[pos-control] train={len(train_idx)} val={len(val_idx)} (identical split to the K1->K3 transfer run)")

    n_states = acts.shape[1]
    layer_rows = []
    for layer in range(n_states):
        X = acts[:, layer, :]
        m = fit_and_eval_multilabel_type(
            X[train_idx], Y_train, X[val_idx], Y_val, X[excluded_idx], Y_eval,
            vocab, seed=seed,
        )
        m.pop("per_type")
        layer_rows.append({"layer": layer, **m})
        print(f"[pos-control] layer {layer:2d}/{n_states-1}: macro_auroc={m['macro_auroc']:.3f}")

    best = max(layer_rows, key=lambda r: r["macro_auroc"] if not np.isnan(r["macro_auroc"]) else -1)
    summary = {
        "description": "Same K=1-trained classifiers as cross_k_type_transfer.py --eval-k 3, "
                        "evaluated on the 129 held-out (but same-distribution) K=1 rows instead of K=3 rows",
        "n_train": len(train_idx), "n_val": len(val_idx), "n_eval": len(excluded_idx),
        "best_layer": best["layer"], "best_layer_macro_auroc": best["macro_auroc"],
    }
    with (results_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[pos-control] done. Best layer {best['layer']}: macro_auroc={best['macro_auroc']:.3f}")
    print(f"[pos-control] compare against within-K1 CV macro-AUROC ~0.571 (results_types_k1/run_summary.json)")


if __name__ == "__main__":
    main()
