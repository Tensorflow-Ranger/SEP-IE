"""Group-aware, label-stratified train/val/test splitting.

ClarifyCodeBench's ambiguous rewrites are grouped: ~5 different underspecified
edits share one `source_problem_id` (the LiveCodeBench question_id), and that
problem's single full-requirement negative shares the same id. If two rows
with the same source_problem_id land in different splits, a probe can partly
solve the task by recognizing the *problem* rather than its ambiguity status
-- especially since a positive and its paired negative differ by only one
deleted sentence. All splits here are therefore grouped by source_problem_id
via StratifiedGroupKFold, which also keeps the label ratio close to balanced
across splits.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_train_val_test_split(
    labels: np.ndarray,
    groups: np.ndarray,
    test_frac_folds: int = 5,
    val_frac_folds: int = 4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx), grouped by `groups`, stratified by `labels`.

    Two nested StratifiedGroupKFold splits: first peel off ~1/test_frac_folds
    of the groups as test, then split the remainder into train/val with the
    same mechanism. With test_frac_folds=5, val_frac_folds=4 this yields an
    approx 60/20/20 split while keeping every source_problem_id's rows
    entirely on one side of each split.
    """
    idx = np.arange(len(labels))

    sgkf_test = StratifiedGroupKFold(n_splits=test_frac_folds, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(sgkf_test.split(idx, labels, groups))

    sgkf_val = StratifiedGroupKFold(n_splits=val_frac_folds, shuffle=True, random_state=seed)
    trainval_labels = labels[trainval_idx]
    trainval_groups = groups[trainval_idx]
    train_sub_idx, val_sub_idx = next(sgkf_val.split(trainval_idx, trainval_labels, trainval_groups))
    train_idx = trainval_idx[train_sub_idx]
    val_idx = trainval_idx[val_sub_idx]

    assert set(groups[train_idx]).isdisjoint(groups[val_idx])
    assert set(groups[train_idx]).isdisjoint(groups[test_idx])
    assert set(groups[val_idx]).isdisjoint(groups[test_idx])

    return train_idx, val_idx, test_idx


def make_grouped_cv_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
):
    """Yield (train_idx, test_idx) for grouped, stratified k-fold CV.

    Used as a robustness check alongside the single train/val/test split: with
    only ~80 source problems, one held-out test split gives a noisy per-layer
    estimate. Cross-validated AUROC (mean +/- std across folds) makes it
    possible to tell a genuine layer-wise signal from split luck.
    """
    idx = np.arange(len(labels))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_sub, test_sub in sgkf.split(idx, labels, groups):
        yield idx[train_sub], idx[test_sub]
