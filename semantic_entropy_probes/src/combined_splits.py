"""Split helpers for the combined multi-source binary dataset.

Reuses src/splits.py's grouped StratifiedGroupKFold machinery unchanged --
the only addition here is the stratification *key*. A plain label-only
stratify key balances clear/ambiguous ratio across splits but says nothing
about which of the source datasets ends up where; with clarifycodebench
having 80 groups, humaneval (orchid_heval + humanevalcomm) 164,
bigcodebench (orchid_bcb) 164, and bigcodebench_expand (orchid_bcb_expand)
976, an unlucky single StratifiedGroupKFold draw could leave one source thin
(or even entirely on one side) in a given split.

`composite_key(labels, dataset_source)` folds dataset_source into the
stratify key (key = label * n_sources + source_index) so
StratifiedGroupKFold balances (label, source) jointly -- "a good split from
all sources" means every source is represented at roughly its overall
proportion in both train and test, not just that the pooled label ratio
matches.
"""

from __future__ import annotations

import numpy as np

from src.splits import make_grouped_cv_folds, make_train_val_test_split

DATASET_SOURCES = ("clarifycodebench", "orchid_heval", "orchid_bcb", "orchid_bcb_expand", "humanevalcomm")


def composite_key(labels: np.ndarray, dataset_source: np.ndarray) -> np.ndarray:
    src_to_idx = {s: i for i, s in enumerate(DATASET_SOURCES)}
    src_idx = np.array([src_to_idx[s] for s in dataset_source], dtype=np.int64)
    return labels.astype(np.int64) * len(DATASET_SOURCES) + src_idx


def make_combined_train_val_test_split(
    labels: np.ndarray,
    dataset_source: np.ndarray,
    groups: np.ndarray,
    test_frac_folds: int = 5,
    val_frac_folds: int = 4,
    seed: int = 0,
):
    key = composite_key(labels, dataset_source)
    return make_train_val_test_split(key, groups, test_frac_folds=test_frac_folds,
                                      val_frac_folds=val_frac_folds, seed=seed)


def make_combined_grouped_cv_folds(
    labels: np.ndarray,
    dataset_source: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
):
    key = composite_key(labels, dataset_source)
    yield from make_grouped_cv_folds(key, groups, n_splits=n_splits, seed=seed)


def make_equal_per_source_split(
    dataset_source: np.ndarray,
    groups: np.ndarray,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    seed: int = 0,
):
    """Balanced-by-source alternative to make_combined_train_val_test_split:
    every source contributes the SAME number of ROWS to train/val/test,
    rather than its natural proportion of the pool -- so a 6x-larger source
    (e.g. orchid_bcb_expand) can't dominate either split just by having more
    rows. Equalizing by row count (not by unique-group count) matters here
    because `groups` (source_problem_id) collapses multiple ambiguity-type
    rows of the SAME underlying problem into one group for sources other
    than clarifycodebench (which is 1 row per group) -- equalizing group
    counts across sources with different rows-per-group ratios would have
    left row counts unequal (the bug this replaced).

    All of a group's rows are still kept together in one split (no leakage
    of a problem's other ambiguity-type variant across train/val/test) --
    groups are walked in shuffled order and greedily assigned to whichever
    split still needs rows, so the achieved row counts are AS CLOSE AS
    POSSIBLE to equal (not exact, since group sizes vary), capped to the
    smallest source's total row count.

    Returns (train_idx, val_idx, test_idx) as boolean masks over the
    original (ungrouped, i.e. per-row) arrays, matching
    make_combined_train_val_test_split's return shape.
    """
    rng = np.random.default_rng(seed)
    sources = sorted(set(dataset_source.tolist()))
    n_rows_by_source = {s: int((dataset_source == s).sum()) for s in sources}
    n_min_rows = min(n_rows_by_source.values())
    n_train_target = int(round(n_min_rows * train_frac))
    n_val_target = int(round(n_min_rows * val_frac))
    n_test_target = n_min_rows - n_train_target - n_val_target

    train_idx_mask = np.zeros(len(dataset_source), dtype=bool)
    val_idx_mask = np.zeros(len(dataset_source), dtype=bool)
    test_idx_mask = np.zeros(len(dataset_source), dtype=bool)

    for s in sources:
        src_row_idx = np.where(dataset_source == s)[0]
        rows_by_group: dict = {}
        for i in src_row_idx:
            rows_by_group.setdefault(groups[i], []).append(i)
        group_ids = list(rows_by_group.keys())
        rng.shuffle(group_ids)

        n_train, n_val, n_test = 0, 0, 0
        for gid in group_ids:
            row_ids = rows_by_group[gid]
            k = len(row_ids)
            # assign to whichever split is furthest below its target (relative deficit)
            deficits = [
                (n_train_target - n_train, train_idx_mask),
                (n_val_target - n_val, val_idx_mask),
                (n_test_target - n_test, test_idx_mask),
            ]
            deficits.sort(key=lambda t: -t[0])
            target_deficit, target_mask = deficits[0]
            if target_deficit <= 0:
                continue  # every split already at/over its target for this source
            for i in row_ids:
                target_mask[i] = True
            if target_mask is train_idx_mask:
                n_train += k
            elif target_mask is val_idx_mask:
                n_val += k
            else:
                n_test += k

    return np.where(train_idx_mask)[0], np.where(val_idx_mask)[0], np.where(test_idx_mask)[0]


def report_split_composition(name: str, idx: np.ndarray, labels: np.ndarray, dataset_source: np.ndarray) -> None:
    print(f"[split] {name}: {len(idx)} rows")
    for src in DATASET_SOURCES:
        mask = dataset_source[idx] == src
        n = int(mask.sum())
        if n == 0:
            print(f"[split]   {src}: 0 rows")
            continue
        n_amb = int(labels[idx][mask].sum())
        print(f"[split]   {src}: {n} rows (clear={n - n_amb} ambiguous={n_amb})")
