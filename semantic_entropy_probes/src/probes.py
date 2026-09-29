"""Multinomial-logistic probing on frozen features, for an ORDINAL target.

The target is the ambiguity count K in {0,1,2,3}, not a binary flag -- K=1 and
K=3 are genuinely different amounts of unresolved specification. We fit a
standard multinomial LogisticRegression (sklearn's default multi_class
handling for >2 classes) rather than a dedicated ordinal/proportional-odds
model, but report ordinal-aware metrics alongside the classification ones so
"how far off" is visible, not just "right or wrong":

  - test_accuracy, test_macro_f1, test_auroc_ovr_macro: standard multiclass
    metrics (macro-averaged over `classes` so no class dominates just by
    being larger).
  - test_spearman: Spearman correlation between the true K and the model's
    probability-weighted expected value E[K] = sum_k P(k) * k. This is the
    metric most directly answering "does this layer's probe track the
    *degree* of ambiguity" rather than just classifying it.
  - test_mae: mean absolute error between the true K and the argmax predicted
    class -- "off by how many ambiguity points, on average."

IMPORTANT CAVEAT ABOUT test_spearman ON THE FULL K=0..3 RANGE: every K=0 row
in the canonical dataset carries an appended "Clarifications:" block that no
K>=1 row has (see build_dataset.py). That makes K=0-vs-K>=1 trivially
separable on a surface feature, which can dominate Spearman even if a probe
has learned nothing about *degree* among K=1/2/3 -- a probe that only detects
"has a Clarifications header" and predicts a constant for every K>=1 row
would still show strong positive correlation. `train_probes.py --min-k 1`
restricts to the K in {1,2,3} rows only (all of them `modified_content`
verbatim, no appended text, no header) to isolate the real degree signal from
this artifact; read that run as the load-bearing one for the "does it encode
degree" question, and the full-range run as upper-bound/reference only.

## Why PCA before the classifier

Hidden-size is 4096 but train splits are only ~300-700 rows: D >> N. Fitting
multinomial lbfgs directly in 4096-D is both slow to converge (an earlier
version of this pipeline took ~3 CPU-hours for a 37-layer sweep) and poorly
conditioned. We whiten with PCA (fit on train only, same n_components across
the whole C-sweep for a given call) before the classifier. This keeps the
overall map linear in the original activation space (PCA is a linear
projection; logistic regression on top of it is still linear-in-activations
end to end) while cutting the effective feature count ~40x, which is what
actually fixes both the runtime and the D>>N conditioning problem -- reducing
`max_iter` alone does not, since fits were converging, just slowly.

Shared by both the per-layer activation probes and the length-confound
baseline, since both are "standardize a feature matrix, sweep C on a held-out
val set, evaluate on test" -- the only difference is what X is. The 1-D
length baseline has n_features=1, so PCA is skipped for it automatically.

## backend={"sklearn","torch","torch_mlp"}

`sklearn` (default) is the pipeline described above. `torch` swaps in
src/torch_probe.py's TorchLogisticProbe (Adam on GPU, same L2-regularized
softmax objective) for the final classifier step -- it exposes the same
.fit/.predict_proba/.classes_ interface sklearn's LogisticRegression does,
so it drops directly into the same sklearn Pipeline as the PCA step. Useful
when running many layers back-to-back and the GPU is otherwise idle; see
src/torch_probe.py's docstring for the empirical sklearn-vs-torch validation
that motivated its specific hyperparameters (an under-tuned patience setting
initially converged to chance-level results on some layers before that was
caught and fixed). `torch_mlp` swaps in src/torch_mlp_probe.py's
TorchMLPProbe -- one ReLU hidden layer + dropout, GPU/Adam, same C=inverse-
weight-decay convention -- as the nonlinear "ML" counterpart to plain
logistic regression, regularized via both weight_decay and dropout.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.torch_mlp_probe import TorchMLPProbe
from src.torch_probe import TorchLogisticProbe

DEFAULT_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
MAX_ITER = 1000
N_PCA_COMPONENTS = 100
ALL_CLASSES = (0, 1, 2, 3)


def _build_pipeline(C: float, seed: int, n_train: int, n_features: int, backend: str = "sklearn",
                     use_pca: bool = True) -> Pipeline:
    steps = [("scale", StandardScaler())]
    n_components = min(N_PCA_COMPONENTS, n_train - 1, n_features)
    if use_pca and n_features > n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    if backend == "torch":
        clf = TorchLogisticProbe(C=C, class_weight="balanced", seed=seed)
    elif backend == "torch_mlp":
        clf = TorchMLPProbe(C=C, class_weight="balanced", seed=seed)
    else:
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
    steps.append(("clf", clf))
    return Pipeline(steps)


def _fit(pipeline: Pipeline, X, y) -> bool:
    """Fit in place; return whether the classifier actually converged (don't
    trust a fit that didn't). sklearn's LogisticRegression signals this via
    ConvergenceWarning; TorchLogisticProbe never raises that (it's not
    sklearn), so for it we instead check whether it stopped via early-stop
    (converged) or hit max_iter (did not) -- see src/torch_probe.py.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(X, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    clf = pipeline.named_steps["clf"]
    if isinstance(clf, (TorchLogisticProbe, TorchMLPProbe)):
        converged = clf.n_iter_ < clf.max_iter
    return converged


def _safe_spearman(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        return float("nan")
    rho, _ = spearmanr(y_true, y_score)
    return float(rho)


def _safe_auroc_ovr(y_true: np.ndarray, probs: np.ndarray, classes) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        if len(classes) == 2:
            # sklearn's roc_auc_score rejects multi_class="ovr" with a 2-column
            # probs matrix when y_true is genuinely binary (raises ValueError,
            # silently swallowed below into NaN) -- standard binary AUROC needs
            # the positive-class column (index 1, since `classes` -- from
            # clf.classes_ -- is always sorted ascending) and no multi_class kwarg.
            return float(roc_auc_score(y_true, probs[:, 1]))
        return float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro", labels=classes))
    except ValueError:
        return float("nan")


def _expected_value(probs: np.ndarray, classes) -> np.ndarray:
    classes = np.asarray(classes, dtype=np.float64)
    return probs @ classes


def _score_predictions(y_true, y_pred, probs, clf_classes, macro_labels) -> dict:
    expected = _expected_value(probs, clf_classes)
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(macro_labels), zero_division=0)),
        "test_auroc_ovr_macro": _safe_auroc_ovr(y_true, probs, clf_classes),
        "test_spearman": _safe_spearman(y_true, expected),
        "test_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def fit_and_eval(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c_grid=DEFAULT_C_GRID,
    seed: int = 0,
    macro_labels=ALL_CLASSES,
    backend: str = "sklearn",
    use_pca: bool = True,
) -> dict:
    """Standardize+PCA on train, select C by val Spearman, refit on train+val, score on test.

    X_* may be 1-D (single feature, e.g. token length) or 2-D (n_samples, n_features).
    Model selection uses Spearman correlation (val) rather than accuracy/F1,
    since the question this probe exists to answer is ordinal ("does this
    layer track ambiguity *degree*"), not just classification quality.
    `macro_labels` should be narrowed (e.g. to (1,2,3)) when a class is
    structurally absent from the data being scored -- otherwise macro-F1 is
    dragged down by a guaranteed-zero column for a class that can't appear.
    `backend`: "sklearn" (default, CPU lbfgs) or "torch" (GPU Adam) -- see
    module docstring.
    """
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)

    n_features = X_train.shape[1]

    best_c, best_val_spearman = c_grid[0], -np.inf
    any_unconverged = False
    for c in c_grid:
        pipe = _build_pipeline(c, seed, n_train=len(y_train), n_features=n_features, backend=backend, use_pca=use_pca)
        converged = _fit(pipe, X_train, y_train)
        any_unconverged = any_unconverged or not converged
        val_expected = _expected_value(pipe.predict_proba(X_val), pipe.classes_)
        val_spearman = _safe_spearman(y_val, val_expected)
        if not np.isnan(val_spearman) and val_spearman > best_val_spearman:
            best_c, best_val_spearman = c, val_spearman

    # Refit on train+val with the selected C, then score on the held-out test split.
    X_trainval = np.concatenate([X_train, X_val], axis=0)
    y_trainval = np.concatenate([y_train, y_val], axis=0)

    pipe_final = _build_pipeline(best_c, seed, n_train=len(y_trainval), n_features=n_features, backend=backend,
                                  use_pca=use_pca)
    converged_final = _fit(pipe_final, X_trainval, y_trainval)
    any_unconverged = any_unconverged or not converged_final

    y_prob = pipe_final.predict_proba(X_test)
    y_pred = pipe_final.predict(X_test)

    metrics = _score_predictions(y_test, y_pred, y_prob, pipe_final.classes_, macro_labels)
    metrics.update({
        "best_C": best_c,
        "val_spearman_at_best_C": float(best_val_spearman),
        "any_unconverged": any_unconverged,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    })
    return metrics


def cv_spearman(
    acts_layer: np.ndarray,
    labels: np.ndarray,
    fold_indices,
    C: float,
    seed: int = 0,
    backend: str = "sklearn",
) -> tuple[float, float]:
    """Mean/std Spearman correlation over pre-computed grouped CV folds, at a fixed C.

    Robustness check: with ~80 source-problem groups, a single test split
    gives a noisy per-layer estimate, so we also report cross-validated
    Spearman correlation (mean +/- std across folds).
    """
    scores = []
    for train_idx, test_idx in fold_indices:
        y_tr, y_te = labels[train_idx], labels[test_idx]
        if len(np.unique(y_te)) < 2:
            continue
        pipe = _build_pipeline(C, seed, n_train=len(y_tr), n_features=acts_layer.shape[1], backend=backend)
        _fit(pipe, acts_layer[train_idx], y_tr)
        expected = _expected_value(pipe.predict_proba(acts_layer[test_idx]), pipe.classes_)
        scores.append(_safe_spearman(y_te, expected))
    scores = [s for s in scores if not np.isnan(s)]
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))
