"""Functions ported directly from OATML's Semantic Entropy Probes codebase.

Source: https://github.com/OATML/semantic-entropy-probes
        semantic_entropy_probes/train-latent-probe.ipynb, cells 13-19

Kossen, Han, Razzak, Schut, Malik, Gal. "Semantic Entropy Probes: Robust and
Cheap Hallucination Detection in LLMs." arXiv:2406.15927

`best_split`, `binarize_entropy`, `bootstrap_func`, `bootstrap`, `auroc`, and
`compatible_bootstrap` are copied close to verbatim from that notebook, so
this project's use of their binarization/bootstrap method is the actual
method, not a reimplementation from the paper description. Only cosmetic
changes: no wandb plotting side-effect in best_split, type hints/docstrings
added.

`sklearn_train_and_evaluate` and `sklearn_evaluate_on_test` started as
verbatim ports of OATML's binary-only versions (cell 14) and were then
generalized to also support >2 classes (see `ovr_macro_auroc`), since
OATML's notebook only ever exercises the binary path (they always binarize
their target first) even though sklearn's `LogisticRegression` itself
handles multi-class natively. `expected_value` and `spearman_vs_expected`
are new additions, not in OATML's notebook at all -- see their docstrings
for why they were added.

## What's different in how we use them -- and what is NOT being claimed

OATML's target is *semantic entropy*: sample many generations for a prompt,
cluster them by semantic equivalence, measure how spread out that cluster
distribution is. It is a property of the model's OUTPUT under repeated
sampling -- "if asked again, how much would the answers disagree."

Here the target is ClarifyCodeBench's annotated ambiguity count K, a
property of the INPUT -- how many pieces of information a human annotator
judged missing from the problem statement, found by diffing against the
original fully-specified version. **K is not semantic entropy, and this
project does not compute or verify semantic entropy anywhere** (no sampling,
no generation, no clustering). Whether K predicts actual output disagreement
under sampling is a real, open, and explicitly deferred question -- a prompt
could have K=3 missing details the model resolves the same way every time
(low true entropy despite high K), or a single K=1 ambiguity on a
consequential axis could cause wide disagreement (high true entropy despite
low K).

What IS being reused from OATML, faithfully, is their *method* for turning
an ordinal/continuous scalar target into a well-chosen binary split
(`best_split`) rather than an arbitrary threshold or an unordered multi-class
setup -- applied here to K restricted to {1,2,3} (K=0 is excluded; see
project FINDINGS.md for why every K=0 construction tried carries a residual
confound unrelated to ambiguity). `best_split` cuts at whichever integer
boundary (1|2 or 2|3) minimizes within-group variance -- the direct analogue
of their "best-universal-split" binarization, just applied to a different
target than theirs.

OATML's probe is trained on TBG (token before generation) hidden states,
which is exactly what this project already extracts (the last prompt token,
before any generation) -- no re-extraction needed. Their SLT (second-to-last
generated token) has no analogue here since this project never generates.

OATML's `create_Xs_and_ys` uses a plain (non-grouped) `train_test_split`;
this project's `src/splits.py` group-aware split is used instead, since here
-- unlike OATML's QA datasets -- multiple rows share a `source_problem_id`
and an ungrouped split would leak.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import scipy.stats
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

# ---------------------------------------------------------------------------
# Binarization (notebook cell 16) -- verbatim except no plotting side effect.
# ---------------------------------------------------------------------------


def best_split(scores: np.ndarray) -> float:
    """Identify the split minimizing reconstruction error via low/high mean
    estimates -- OATML's "best-universal-split" binarization (Section 4 of
    arXiv:2406.15927), applied to any scalar target. Here `scores` is K
    (ClarifyCodeBench's annotated ambiguity-point count) restricted to
    {1,2,3} -- NOT semantic entropy. K measures specification incompleteness
    (how many details an annotator judged missing from the prompt); semantic
    entropy measures output disagreement (how much sampled generations
    diverge). Whether the two correlate is an open question this project
    does not test (no generation/sampling is performed anywhere in this
    pipeline). We reuse OATML's binarization *method* because it's the right
    tool for turning an ordinal scalar into a well-chosen binary split -- not
    because K has been shown to approximate their target.
    """
    ents = np.asarray(scores, dtype=np.float64)
    splits = np.linspace(1e-10, ents.max(), 100)
    split_mses = []
    for split in splits:
        low_idxs, high_idxs = ents < split, ents >= split
        low_mean = np.mean(ents[low_idxs]) if low_idxs.any() else 0.0
        high_mean = np.mean(ents[high_idxs]) if high_idxs.any() else 0.0
        mse = np.sum((ents[low_idxs] - low_mean) ** 2) + np.sum((ents[high_idxs] - high_mean) ** 2)
        split_mses.append(mse)
    split_mses = np.array(split_mses)
    return float(splits[np.argmin(split_mses)])


def binarize_entropy(entropy: np.ndarray, thres: float) -> np.ndarray:
    """Binarize entropy (here: K) scores into 0s and 1s at `thres`."""
    entropy = np.asarray(entropy, dtype=np.float64)
    binary = np.full_like(entropy, -1, dtype=np.float64)
    binary[entropy < thres] = 0
    binary[entropy > thres] = 1
    return binary


# ---------------------------------------------------------------------------
# Bootstrap + AUROC (notebook cell 13) -- verbatim.
# ---------------------------------------------------------------------------


def auroc(y_true, y_score) -> float:
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)


def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data,), function, n_resamples=n_resamples, confidence_level=0.9,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }
    return inner


def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        return func(y_true, y_score)

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        return bootstrap(helper, rng=rng)(wrap_inputs(y_true, y_score))
    return converted_func


def bootstrap_func(y_true, y_score, func, rng) -> dict:
    metric_dict = {'mean': func(y_true, y_score)}
    metric_dict['bootstrap'] = compatible_bootstrap(func, rng)(y_true, y_score)
    return metric_dict


# ---------------------------------------------------------------------------
# Train/eval (notebook cell 14) -- generalized from binary-only to
# binary-or-multiclass. OATML's exact code hardcodes `probs[:, 1]` and
# `labels=[0, 1]`, which only works for 2 classes; sklearn's LogisticRegression
# itself is not binary-only (it fits one-vs-rest / multinomial automatically
# for >2 classes), OATML's notebook just never exercises that path since they
# always binarize first. Here both branches use the *same* LogisticRegression
# call and the *same* evaluation shape (accuracy, AUROC with bootstrap CI,
# log-loss); only the AUROC computation and label set adapt to n_classes.
# ---------------------------------------------------------------------------


def ovr_macro_auroc(y_true, y_score_matrix, labels=None) -> float:
    """One-vs-rest macro-averaged AUROC for >2 classes -- the natural
    multiclass generalization of OATML's binary `auroc()` above. Not in
    their notebook (they never classify >2 classes); this is the standard
    sklearn approach (same one used elsewhere in this project's own
    src/probes.py) for extending ROC-AUC past binary.
    """
    return float(roc_auc_score(y_true, y_score_matrix, multi_class="ovr", average="macro", labels=labels))


def expected_value(probs: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """probs @ classes -- the probability-weighted expected class value,
    E[K] = sum_k P(k) * k. For binary {0,1} classes this is just P(class=1).
    """
    return probs @ np.asarray(classes, dtype=np.float64)


def spearman_vs_expected(y_true, probs, classes) -> float:
    """Spearman correlation between true labels and the model's expected
    value. Not part of OATML's notebook (their evaluation is accuracy/AUROC/
    log-loss only) -- added here because it's the metric that directly
    answers "does this probe's output track the *degree* of K," which
    matters when y_true has an ordinal structure (as K does) rather than
    being genuinely unordered classes. Degenerates gracefully to NaN if
    either side is constant (undefined correlation).
    """
    y_true = np.asarray(y_true)
    exp = expected_value(probs, classes)
    if len(np.unique(y_true)) < 2 or len(np.unique(exp)) < 2:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(y_true, exp)
    return float(rho)


def sklearn_train_and_evaluate(model, X_train, y_train, X_valid, y_valid) -> dict:
    model.fit(X_train, y_train)
    classes = model.classes_
    n_classes = len(classes)

    train_probs = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_probs, labels=classes)

    valid_preds = model.predict(X_valid)
    valid_probs = model.predict_proba(X_valid)
    valid_loss = log_loss(y_valid, valid_probs, labels=classes)
    val_accuracy = float(np.mean((valid_preds == y_valid).astype(int)))

    if len(np.unique(y_valid)) > 1:
        valid_auroc = (float(roc_auc_score(y_valid, valid_probs[:, 1])) if n_classes == 2
                        else ovr_macro_auroc(y_valid, valid_probs, labels=classes))
    else:
        valid_auroc = float('nan')
    valid_spearman = spearman_vs_expected(y_valid, valid_probs, classes)

    return {
        'train_loss': float(train_loss),
        'valid_loss': float(valid_loss),
        'valid_accuracy': val_accuracy,
        'valid_auroc': valid_auroc,
        'valid_spearman': valid_spearman,
    }


def sklearn_evaluate_on_test(model, X_test, y_test, rng, bootstrap_ci=True) -> dict:
    classes = model.classes_
    n_classes = len(classes)

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    test_loss = float(log_loss(y_test, test_probs, labels=classes))
    test_accuracy = float(np.mean((test_preds == y_test).astype(int)))
    test_spearman = spearman_vs_expected(y_test, test_probs, classes)

    has_two_classes_present = len(np.unique(y_test)) > 1
    if n_classes == 2:
        auroc_func = auroc
        y_score = test_probs[:, 1]
    else:
        def auroc_func(y_true, y_score_matrix):
            return ovr_macro_auroc(y_true, y_score_matrix, labels=classes)
        y_score = test_probs

    if bootstrap_ci and has_two_classes_present:
        auroc_boot = bootstrap_func(y_test, y_score, auroc_func, rng)
        test_auroc = auroc_boot['mean']
        auroc_ci = auroc_boot['bootstrap']
    elif has_two_classes_present:
        test_auroc = auroc_func(y_test, y_score)
        auroc_ci = None
    else:
        test_auroc = float('nan')
        auroc_ci = None

    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'test_auroc': test_auroc,
        'test_auroc_ci': auroc_ci,
        'test_spearman': test_spearman,
    }


def make_logistic_regression(max_iter: int = 5000) -> LogisticRegression:
    """OATML's notebook calls `LogisticRegression()` with library defaults
    (C=1.0, max_iter=100). At this project's D=4096 >> N~few-hundred scale,
    sklearn's default max_iter=100 does not converge -- confirmed empirically
    (see scripts/train_oatml_style_probe.py output). We raise max_iter so the
    same C=1.0 default actually reaches its optimum, and report per-layer
    convergence explicitly rather than silently accepting an under-fit model;
    this is the one deliberate deviation from their exact call. For >2
    classes, sklearn's LogisticRegression handles multinomial fitting
    automatically -- no separate code path needed.
    """
    return LogisticRegression(max_iter=max_iter)


# ---------------------------------------------------------------------------
# Model persistence -- mirrors OATML's own "Save trained probes for
# inference" cell (notebook cell 44: pickle a dict of fitted sklearn models
# + metadata to models/<name>.pkl). We save one layer's fitted model here
# rather than a dict of many datasets' models, since this project has one
# dataset, not several.
# ---------------------------------------------------------------------------


def save_probe(model: LogisticRegression, path: Path, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump({'model': model, 'metadata': metadata}, f)


def load_probe(path: Path) -> dict:
    with Path(path).open('rb') as f:
        return pickle.load(f)
