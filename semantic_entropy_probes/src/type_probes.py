"""Probing WHICH ambiguity type(s) are present, not how many (K).

K (src/probes.py) asks "how much ambiguity." This module asks "what kind":
`ambiguity_types` in the dataset is a list drawn from 11 category tags
(Output Format, Terminology, Behavior, Edge Cases, Indices & Ranges,
Collection Semantics, Comparison Rules, Ordering & Atomicity, Units,
Numerical Precision, String & Localization) -- e.g. a K=2 row has exactly 2
of these tags, a K=1 row exactly 1 (verified: no K=1 row carries >1 tag in
this dataset).

Two distinct problems, staged by K per user request:

  Stage 1 (K=1 only): each row has exactly one type -> ordinary nominal
  multiclass classification (which single type is it). No ordinal structure
  here (unlike K itself) -- "Output Format" isn't "more" than "Terminology"
  -- so this reuses PCA+LogisticRegression but drops Spearman/expected-value
  scoring in favor of accuracy/macro-F1/macro-AUROC + a confusion matrix.

  Stage 2 (K=2, K=3): each row has 2 or 3 types simultaneously -> multi-label
  classification (binary relevance): one independent present/absent logistic
  head per type, sharing a single PCA projection per layer/split (fit once,
  reused across all type-heads) for efficiency and to keep the feature space
  identical across heads. Evaluated per-type AUROC/F1 plus macro-averages and
  subset-level metrics (exact-match accuracy, Jaccard/IoU, Hamming loss).

## Type vocabulary + rare-type handling

5 of 11 types are extremely rare at K=1 (Ordering & Atomicity=5, Units=3,
Numerical Precision=2, String & Localization=1, out of 199 K=1 rows) --
too few to appear reliably across grouped train/val/test/CV splits. Per
explicit user decision: rows whose (sole, at K=1) type falls below
MIN_SUPPORT are DROPPED (not collapsed into "Other"), and the resulting
vocabulary (types with >=MIN_SUPPORT examples at K=1) is reused as-is for
the K=2/K=3 multi-label stages, so all three stages classify over the same
fixed set of types. K=2/K=3 rows are kept as long as at least one of their
types survives in-vocab; out-of-vocab types simply aren't tracked as a
label dimension for that row (a K=2 row with one in-vocab and one dropped
type becomes a single positive label, not two).

## Splitting

Grouped by source_problem_id (same leakage rationale as src/splits.py) via
plain GroupShuffleSplit / GroupKFold -- unstratified, because within a
single-K subset there is no longer a scalar label to stratify by (Stage 1
already conditions on K=1; Stage 2's target is a multi-hot vector, and
sklearn has no built-in multi-label group stratification). At K=3 there are
only 51 rows and 51 unique source problems (every K=3 row is its own
group), so a single held-out test split is noisier there and grouped CV
(mean +/- std) is reported alongside it -- but note GroupKFold is
deterministic given n_splits (no shuffle/random_state), so that +/- is
fold-to-fold variance within ONE fixed partition, not a robustness check
across partitions; don't read a tight CV std as more reassuring than a
single split would be.

IMPORTANT, K=3 specifically: because every K=3 row is its own group, its
within-stage CV number (~0.72) cannot rule out a "recognizes the problem's
topic" confound rather than genuine per-deletion signal (K=1/K=2 don't have
this issue -- verified, their multi-row groups have varying type labels
within group). scripts/cross_k_type_transfer.py attempted to check this via
K=1-to-K=3 transfer and came back near chance, but
scripts/cross_k_positive_control.py showed that same training set can't
even reliably relearn the in-distribution K=1 task (0.544 vs. 0.571 for a
properly-sized split) -- so the transfer null is inconclusive, not evidence
either way. See FINDINGS.md's "Cross-K transfer check" section for the full
three-way enumeration of what could explain 0.72; this dataset doesn't have
enough K=1 rows disjoint from all 43 K=3 problems to resolve it further.
"""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    jaccard_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

MIN_SUPPORT = 8
DEFAULT_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MAX_ITER = 1000
N_PCA_COMPONENTS = 100


def select_type_vocab(meta_rows: list[dict], k: int = 1, min_support: int = MIN_SUPPORT) -> list[str]:
    """Types with >= min_support examples among K==k rows, most-supported first."""
    counts = Counter()
    for m in meta_rows:
        if m["label"] == k:
            for t in m["ambiguity_types"]:
                counts[t] += 1
    vocab = [t for t, n in counts.most_common() if n >= min_support]
    return vocab


def singlelabel_subset(meta_rows: list[dict], vocab: list[str], k: int = 1):
    """Row indices (into meta_rows) with label==k and their sole type in vocab, plus int targets."""
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    keep, y = [], []
    for i, m in enumerate(meta_rows):
        if m["label"] != k:
            continue
        types = m["ambiguity_types"]
        assert len(types) == k, f"row {m['id']}: label={k} but {len(types)} types {types}"
        if types[0] in vocab_idx:
            keep.append(i)
            y.append(vocab_idx[types[0]])
    return np.array(keep, dtype=np.int64), np.array(y, dtype=np.int64)


def multilabel_subset(meta_rows: list[dict], vocab: list[str], k: int):
    """Row indices with label==k and ALL of the row's types in vocab, plus multi-hot targets.

    Consistent with the Stage-1 "drop rare rows" decision: a K=2/K=3 row is
    kept only if every one of its annotated types survived the min-support
    filter, not merely at-least-one. Keeping a row with a mix of in-vocab and
    dropped types would silently mislabel it (a true K=2 row would present as
    a single positive label with no signal that a second, unmodeled ambiguity
    is also present) -- dropping it entirely keeps every kept row's label
    vector a complete description of its ambiguity types.
    """
    vocab_idx = {t: i for i, t in enumerate(vocab)}
    keep, Y = [], []
    for i, m in enumerate(meta_rows):
        if m["label"] != k:
            continue
        types = m["ambiguity_types"]
        if not types or not all(t in vocab_idx for t in types):
            continue
        row_y = np.zeros(len(vocab), dtype=np.int64)
        for t in types:
            row_y[vocab_idx[t]] = 1
        keep.append(i)
        Y.append(row_y)
    return np.array(keep, dtype=np.int64), (np.stack(Y) if Y else np.zeros((0, len(vocab)), dtype=np.int64))


# ---------------------------------------------------------------------------
# Stage 1: single-label nominal multiclass (K=1)
# ---------------------------------------------------------------------------

def _pca_fit_transform(X_train, X_val, X_test, seed, n_train):
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
    n_components = min(N_PCA_COMPONENTS, n_train - 1, X_train.shape[1])
    if X_train.shape[1] > n_components:
        pca = PCA(n_components=n_components, random_state=seed).fit(X_train)
        X_train, X_val, X_test = pca.transform(X_train), pca.transform(X_val), pca.transform(X_test)
    return X_train, X_val, X_test


def fit_and_eval_multiclass_type(
    X_train, y_train, X_val, y_val, X_test, y_test,
    vocab: list[str], c_grid=DEFAULT_C_GRID, seed: int = 0,
) -> dict:
    """Standardize+PCA on train, select C by val macro-F1, refit train+val, score test.

    Nominal target (which type) -- no Spearman/expected-value here, unlike
    src/probes.py's ordinal K target. Model selection uses macro-F1 (not
    accuracy) so the majority type (Output Format) can't dominate C choice.
    """
    n_train = len(y_train)
    Xtr, Xv, Xte = _pca_fit_transform(X_train, X_val, X_test, seed, n_train)

    best_c, best_val_f1 = c_grid[0], -np.inf
    any_unconverged = False
    for c in c_grid:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(Xtr, y_train)
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        any_unconverged = any_unconverged or not converged
        val_pred = clf.predict(Xv)
        val_f1 = f1_score(y_val, val_pred, average="macro", labels=list(range(len(vocab))), zero_division=0)
        if val_f1 > best_val_f1:
            best_c, best_val_f1 = c, val_f1

    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)
    Xtv, _, Xte = _pca_fit_transform(X_trainval, X_test, X_test, seed, len(y_trainval))

    clf_final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf_final.fit(Xtv, y_trainval)
        converged_final = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    any_unconverged = any_unconverged or not converged_final

    y_pred = clf_final.predict(Xte)
    y_prob = clf_final.predict_proba(Xte)

    # Per-class one-vs-rest AUC, averaged only over classes that actually have
    # both a positive and a negative example in this test split. With only
    # ~40 test rows spread over 7 classes, a rare class (e.g. Comparison
    # Rules, n=8 total) can land with zero test examples in a given split --
    # sklearn's built-in multi_class="ovr" macro average either warns+NaNs
    # that class or raises for the whole call in that case; computing
    # per-class AUC by hand and nanmean-ing lets the other 6 classes still
    # contribute a defined score instead of the whole layer going NaN.
    per_class_auroc = []
    for ci, c in enumerate(clf_final.classes_):
        y_bin = (y_test == c).astype(int)
        if len(np.unique(y_bin)) < 2:
            continue
        per_class_auroc.append(roc_auc_score(y_bin, y_prob[:, ci]))
    macro_auroc = float(np.mean(per_class_auroc)) if per_class_auroc else float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(vocab))))

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=list(range(len(vocab))), zero_division=0)),
        "test_auroc_ovr_macro": macro_auroc,
        "n_classes_in_macro_auroc": len(per_class_auroc),  # out of len(vocab) -- rare classes can be absent from a small test split
        "best_C": best_c,
        "val_macro_f1_at_best_C": float(best_val_f1),
        "any_unconverged": any_unconverged,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "confusion_matrix": cm.tolist(),
    }


def majority_baseline_multiclass(y_train: np.ndarray, y_test: np.ndarray, vocab: list[str]) -> dict:
    """Predict the train-majority type for every test row.

    With Output Format at ~30% prevalence, a majority baseline can post
    non-trivial accuracy on this task by construction -- macro-AUROC is the
    metric that actually can't be gamed this way (a constant predictor has
    no ranking information, so it's undefined/chance, reported as NaN here
    rather than a misleading 0.5 pretending to be "average of 7 coin flips").
    """
    values, counts = np.unique(y_train, return_counts=True)
    majority = int(values[np.argmax(counts)]) if len(y_train) else 0
    y_pred = np.full_like(y_test, majority)
    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=list(range(len(vocab))), zero_division=0)),
        "test_auroc_ovr_macro": float("nan"),
        "majority_type": vocab[majority] if vocab else None,
    }


def random_baseline_multiclass(y_train: np.ndarray, y_test: np.ndarray, vocab: list[str], seed: int = 0, n_trials: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    values, counts = np.unique(y_train, return_counts=True)
    probs = counts / counts.sum() if len(y_train) else np.full(len(vocab), 1 / len(vocab))
    accs, f1s = [], []
    for _ in range(n_trials):
        y_pred = rng.choice(values, size=len(y_test), p=probs)
        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average="macro", labels=list(range(len(vocab))), zero_division=0))
    return {
        "test_accuracy": float(np.mean(accs)),
        "test_macro_f1": float(np.mean(f1s)),
        "test_auroc_ovr_macro": 0.5,  # random scores carry no ranking info -> chance AUROC by construction
    }


def majority_and_random_baseline_multilabel(Y_train: np.ndarray, Y_test: np.ndarray, vocab: list[str], seed: int = 0, n_trials: int = 200) -> dict:
    """Per-type majority-class and random-guess baselines, same metric set as fit_and_eval_multilabel_type.

    Reported so a low per-type F1 from the probe can be read against what a
    prevalence-matched guesser gets for that SAME type, instead of an
    across-types average that a rare type could distort. macro_auroc for
    majority is NaN (constant predictor, no ranking signal); for random it's
    0.5 by construction for every type with both classes present in test.
    """
    n_types = len(vocab)
    maj_f1, rand_f1, maj_acc, rand_acc = [], [], [], []
    rng = np.random.default_rng(seed)
    for t in range(n_types):
        y_tr, y_te = Y_train[:, t], Y_test[:, t]
        values, counts = np.unique(y_tr, return_counts=True)
        majority = int(values[np.argmax(counts)]) if len(y_tr) else 0
        maj_pred = np.full_like(y_te, majority)
        maj_f1.append(f1_score(y_te, maj_pred, zero_division=0))
        maj_acc.append(accuracy_score(y_te, maj_pred))

        p1 = counts[values == 1][0] / counts.sum() if 1 in values and len(y_tr) else 0.0
        trial_f1, trial_acc = [], []
        for _ in range(n_trials):
            rand_pred = (rng.random(len(y_te)) < p1).astype(int)
            trial_f1.append(f1_score(y_te, rand_pred, zero_division=0))
            trial_acc.append(accuracy_score(y_te, rand_pred))
        rand_f1.append(float(np.mean(trial_f1)))
        rand_acc.append(float(np.mean(trial_acc)))

    maj_pred_matrix = np.tile(
        np.array([int(np.unique(Y_train[:, t], return_counts=True)[0][np.argmax(np.unique(Y_train[:, t], return_counts=True)[1])]) for t in range(n_types)]),
        (len(Y_test), 1),
    )
    return {
        "majority": {
            "per_type_f1": dict(zip(vocab, maj_f1)),
            "per_type_accuracy": dict(zip(vocab, maj_acc)),
            "macro_f1": float(np.mean(maj_f1)),
            "macro_auroc": float("nan"),
            "exact_match_accuracy": float(np.mean(np.all(maj_pred_matrix == Y_test, axis=1))),
        },
        "random_prevalence_matched": {
            "per_type_f1": dict(zip(vocab, rand_f1)),
            "per_type_accuracy": dict(zip(vocab, rand_acc)),
            "macro_f1": float(np.mean(rand_f1)),
            "macro_auroc": 0.5,
        },
    }


def cv_macro_auroc_multilabel(acts_layer, Y, groups, vocab, C=1.0, seed=0, n_splits=5):
    """Grouped-CV macro-AUROC robustness check, at a fixed C (no inner val split).

    Unlike fit_and_eval_multilabel_type's per-type C-sweep (which needs a
    held-out val fold), this uses one fixed C across all folds/types -- with
    K=3's ~43 rows there isn't enough data left for a nested train/val/test
    inside each CV fold. C=1.0 (sklearn's default) is a reasonable fixed
    choice since it was the C fit_and_eval_multilabel_type actually selected
    for most types/layers in practice. This is a single-split-noise check,
    not a replacement for the tuned single-split headline number.
    """
    n_splits = min(n_splits, len(set(groups)))
    scores = []
    for train_idx, test_idx in make_type_cv_folds(Y.sum(axis=1), groups, n_splits=n_splits, seed=seed):
        Xtr, Xte = acts_layer[train_idx], acts_layer[test_idx]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
        n_components = min(N_PCA_COMPONENTS, len(train_idx) - 1, Xtr_s.shape[1])
        if Xtr_s.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=seed).fit(Xtr_s)
            Xtr_s, Xte_s = pca.transform(Xtr_s), pca.transform(Xte_s)

        per_type_auc = []
        for t_idx in range(len(vocab)):
            y_tr_t, y_te_t = Y[train_idx, t_idx], Y[test_idx, t_idx]
            if len(np.unique(y_tr_t)) < 2 or len(np.unique(y_te_t)) < 2:
                continue
            clf = LogisticRegression(C=C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
            clf.fit(Xtr_s, y_tr_t)
            pos_col = list(clf.classes_).index(1) if 1 in clf.classes_ else None
            if pos_col is None:
                continue
            prob_t = clf.predict_proba(Xte_s)[:, pos_col]
            per_type_auc.append(roc_auc_score(y_te_t, prob_t))
        if per_type_auc:
            scores.append(float(np.mean(per_type_auc)))
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def make_type_split(y: np.ndarray, groups: np.ndarray, seed: int = 0):
    """Grouped (unstratified) 60/20/20 train/val/test split by source_problem_id."""
    idx = np.arange(len(y))
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    trainval_idx, test_idx = next(gss_test.split(idx, y, groups))
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)  # 0.25 of 80% = 20% overall
    tv_idx2 = np.arange(len(trainval_idx))
    train_sub, val_sub = next(gss_val.split(tv_idx2, y[trainval_idx], groups[trainval_idx]))
    train_idx = trainval_idx[train_sub]
    val_idx = trainval_idx[val_sub]

    assert set(groups[train_idx]).isdisjoint(groups[val_idx])
    assert set(groups[train_idx]).isdisjoint(groups[test_idx])
    assert set(groups[val_idx]).isdisjoint(groups[test_idx])
    return train_idx, val_idx, test_idx


def make_type_cv_folds(y: np.ndarray, groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    idx = np.arange(len(y))
    n_groups = len(set(groups))
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    for train_sub, test_sub in gkf.split(idx, y, groups):
        yield idx[train_sub], idx[test_sub]


def cv_multiclass_macro_f1(acts_layer, y, groups, vocab, C, seed=0, n_splits=5):
    scores = []
    for train_idx, test_idx in make_type_cv_folds(y, groups, n_splits=n_splits, seed=seed):
        if len(np.unique(y[test_idx])) < 2:
            continue
        Xtr = StandardScaler().fit(acts_layer[train_idx])
        X_train_s = Xtr.transform(acts_layer[train_idx])
        X_test_s = Xtr.transform(acts_layer[test_idx])
        n_components = min(N_PCA_COMPONENTS, len(train_idx) - 1, X_train_s.shape[1])
        if X_train_s.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=seed).fit(X_train_s)
            X_train_s, X_test_s = pca.transform(X_train_s), pca.transform(X_test_s)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
        clf.fit(X_train_s, y[train_idx])
        pred = clf.predict(X_test_s)
        scores.append(f1_score(y[test_idx], pred, average="macro", labels=list(range(len(vocab))), zero_division=0))
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def cv_multiclass_macro_auroc(acts_layer, y, groups, vocab, C, seed=0, n_splits=5):
    """Grouped-CV macro-AUROC companion to cv_multiclass_macro_f1, for a fair
    same-metric comparison against Stage 2's cv_macro_auroc_multilabel.
    Per-class OVR AUC, averaged only over classes present (both labels) in
    that fold's test set -- same rationale as fit_and_eval_multiclass_type.
    """
    scores = []
    for train_idx, test_idx in make_type_cv_folds(y, groups, n_splits=n_splits, seed=seed):
        if len(np.unique(y[test_idx])) < 2:
            continue
        scaler = StandardScaler().fit(acts_layer[train_idx])
        X_train_s = scaler.transform(acts_layer[train_idx])
        X_test_s = scaler.transform(acts_layer[test_idx])
        n_components = min(N_PCA_COMPONENTS, len(train_idx) - 1, X_train_s.shape[1])
        if X_train_s.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=seed).fit(X_train_s)
            X_train_s, X_test_s = pca.transform(X_train_s), pca.transform(X_test_s)
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
        clf.fit(X_train_s, y[train_idx])
        prob = clf.predict_proba(X_test_s)
        per_class = []
        for ci, c in enumerate(clf.classes_):
            y_bin = (y[test_idx] == c).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            per_class.append(roc_auc_score(y_bin, prob[:, ci]))
        if per_class:
            scores.append(float(np.mean(per_class)))
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
# Stage 2: multi-label binary relevance (K=2, K=3)
# ---------------------------------------------------------------------------

def fit_and_eval_multilabel_type(
    X_train, Y_train, X_val, Y_val, X_test, Y_test,
    vocab: list[str], c_grid=DEFAULT_C_GRID, seed: int = 0,
) -> dict:
    """Shared PCA per layer/split, one independent binary logistic head per type.

    Each type gets its own C (swept on val AUROC) since prevalence differs
    widely across types even within a fixed K. Predictions are thresholded
    at 0.5 per-type for the subset-level metrics (exact-match / Jaccard /
    Hamming); AUROC itself is threshold-free.
    """
    n_train = len(Y_train)
    Xtr, Xv, Xte = _pca_fit_transform(X_train, X_val, X_test, seed, n_train)
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    Y_trainval = np.concatenate([Y_train, Y_val], axis=0)
    Xtv, _, Xte_final = _pca_fit_transform(X_trainval, X_test, X_test, seed, len(Y_trainval))

    per_type = []
    test_pred = np.zeros_like(Y_test)
    test_prob = np.zeros(Y_test.shape, dtype=np.float64)
    any_unconverged = False

    for t_idx, type_name in enumerate(vocab):
        y_train_t, y_val_t, y_test_t = Y_train[:, t_idx], Y_val[:, t_idx], Y_test[:, t_idx]

        if len(np.unique(y_train_t)) < 2:
            # Type is constant (all-present or all-absent) in this training
            # split -- can't fit a discriminative classifier; predict the
            # constant and report AUROC as undefined (NaN) rather than fake.
            const = int(y_train_t[0])
            test_pred[:, t_idx] = const
            test_prob[:, t_idx] = float(const)
            per_type.append({"type": type_name, "test_auroc": float("nan"), "test_f1": float("nan"),
                              "n_test_pos": int(y_test_t.sum()), "best_C": None, "degenerate": True})
            continue

        best_c, best_val_auroc = c_grid[0], -np.inf
        for c in c_grid:
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(Xtr, y_train_t)
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            any_unconverged = any_unconverged or not converged
            if len(np.unique(y_val_t)) < 2:
                continue
            val_prob = clf.predict_proba(Xv)[:, list(clf.classes_).index(1)] if 1 in clf.classes_ else np.zeros(len(y_val_t))
            val_auroc = roc_auc_score(y_val_t, val_prob)
            if val_auroc > best_val_auroc:
                best_c, best_val_auroc = c, val_auroc

        clf_final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf_final.fit(Xtv, Y_trainval[:, t_idx])
            converged_final = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        any_unconverged = any_unconverged or not converged_final

        pos_col = list(clf_final.classes_).index(1) if 1 in clf_final.classes_ else None
        prob_t = clf_final.predict_proba(Xte_final)[:, pos_col] if pos_col is not None else np.zeros(len(y_test_t))
        pred_t = (prob_t >= 0.5).astype(int)
        test_pred[:, t_idx] = pred_t
        test_prob[:, t_idx] = prob_t

        auroc_t = float("nan")
        if len(np.unique(y_test_t)) >= 2:
            auroc_t = float(roc_auc_score(y_test_t, prob_t))
        per_type.append({
            "type": type_name, "test_auroc": auroc_t,
            "test_f1": float(f1_score(y_test_t, pred_t, zero_division=0)),
            "n_test_pos": int(y_test_t.sum()), "best_C": best_c, "degenerate": False,
        })

    valid_aurocs = [p["test_auroc"] for p in per_type if not np.isnan(p["test_auroc"])]
    macro_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")

    return {
        "per_type": per_type,
        "macro_auroc": macro_auroc,
        "n_types_in_macro_auroc": len(valid_aurocs),  # out of len(vocab) -- a type absent/constant in this test split is excluded
        "exact_match_accuracy": float(np.mean(np.all(test_pred == Y_test, axis=1))),
        "jaccard_macro": float(jaccard_score(Y_test, test_pred, average="samples", zero_division=0)) if len(Y_test) else float("nan"),
        "hamming_loss": float(hamming_loss(Y_test, test_pred)),
        "any_unconverged": any_unconverged,
        "n_train": int(len(Y_train)),
        "n_val": int(len(Y_val)),
        "n_test": int(len(Y_test)),
    }
