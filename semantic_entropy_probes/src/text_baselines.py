"""Text-only baselines: is the signal in the hidden states, or already on the surface?

These operate on raw prompt text, never touching the Qwen3-8B activations, and
exist specifically to answer the question this task asks for: is a probe's
accuracy evidence of the model *representing* ambiguity/uncertainty, or could
a much cheaper surface-level classifier get the same number just from word
choice, punctuation, and formatting -- the kind of stylistic artifact this
repo's FINDINGS.md already documented once (the ClarifyCodeBench
"Clarifications:" header, the LiveCodeBench "Sample Input" header). If a
TF-IDF bag-of-words logistic regression matches or beats the best hidden-state
layer, that's strong evidence the dataset (not the model) is the source of
the separability. If the hidden-state probe clears the BoW baseline by a
wide margin, that's evidence the residual stream carries something beyond
lexical surface cues.

`tfidf_logreg_baseline` fits TfidfVectorizer (train-only vocabulary, as with
any preprocessing step) + LogisticRegression(C swept on val, same grid/
selection-by-Spearman convention as src/probes.py) on raw prompt text.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.probes import DEFAULT_C_GRID, MAX_ITER, _safe_auroc_ovr, _safe_spearman
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error


def tfidf_logreg_baseline(
    text_train: np.ndarray,
    y_train: np.ndarray,
    text_val: np.ndarray,
    y_val: np.ndarray,
    text_test: np.ndarray,
    y_test: np.ndarray,
    c_grid=DEFAULT_C_GRID,
    seed: int = 0,
    macro_labels=(0, 1),
    ngram_range=(1, 2),
    max_features: int = 20000,
) -> dict:
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features, sublinear_tf=True)
    Xtr = vectorizer.fit_transform(text_train)
    Xva = vectorizer.transform(text_val)
    Xte = vectorizer.transform(text_test)

    best_c, best_val_spearman, best_clf = c_grid[0], -np.inf, None
    for c in c_grid:
        clf = LogisticRegression(C=c, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
        clf.fit(Xtr, y_train)
        val_expected = clf.predict_proba(Xva) @ clf.classes_.astype(np.float64)
        val_spearman = _safe_spearman(y_val, val_expected)
        if not np.isnan(val_spearman) and val_spearman > best_val_spearman:
            best_c, best_val_spearman = c, val_spearman

    from scipy.sparse import vstack
    Xtrval = vstack([Xtr, Xva])
    ytrval = np.concatenate([y_train, y_val])
    clf_final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=MAX_ITER, random_state=seed)
    clf_final.fit(Xtrval, ytrval)

    probs = clf_final.predict_proba(Xte)
    y_pred = clf_final.predict(Xte)
    expected = probs @ clf_final.classes_.astype(np.float64)

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro", labels=list(macro_labels), zero_division=0)),
        "test_auroc_ovr_macro": _safe_auroc_ovr(y_test, probs, clf_final.classes_),
        "test_spearman": _safe_spearman(y_test, expected),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "best_C": best_c,
        "val_spearman_at_best_C": float(best_val_spearman),
        "n_train": int(len(y_train)) + int(len(y_val)),
        "n_test": int(len(y_test)),
        "vocab_size": len(vectorizer.vocabulary_),
    }
