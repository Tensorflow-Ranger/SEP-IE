"""GPU-accelerated logistic regression, as a drop-in for sklearn's.

sklearn's `LogisticRegression` (what OATML's notebook calls, and what
src/oatml_sep.py otherwise uses) is CPU-only -- there is no GPU backend in
scikit-learn itself. RAPIDS/cuML has a GPU LogisticRegression, but pulling it
in means a separate CUDA-version-matched install alongside the existing
torch install, which is not "easy" in the sense asked for.

What IS easy: this project already has torch+CUDA set up (used for the
Qwen3-8B extraction). Logistic regression is exactly one linear layer
followed by a softmax -- literally `nn.Linear(D, n_classes)` -- so the same
*model* sklearn fits can be trained with Adam on GPU instead of lbfgs on
CPU. This is not a different method, just a different (much faster, given
the GPU is otherwise idle for this step) optimizer for the same convex
L2-regularized cross-entropy objective sklearn's LogisticRegression(C=...)
solves. It is NOT guaranteed to converge to numerically the same point as
lbfgs (different optimizer, different convergence criterion), but for a
convex problem both should land close to the same optimum given enough
iterations -- this is validated empirically in
scripts/train_oatml_style_probe.py's --backend comparison, not just assumed.

`TorchLogisticProbe` exposes the same interface src/oatml_sep.py's
evaluation functions expect (`.fit`, `.predict_proba`, `.predict`,
`.classes_`), so it's a drop-in for `make_logistic_regression()` -- nothing
in oatml_sep.py's bootstrap/Spearman/AUROC code needs to change.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchLogisticProbe(ClassifierMixin, BaseEstimator):
    """sklearn-compatible wrapper around a one-linear-layer softmax classifier,
    trained with Adam on GPU (falls back to CPU if CUDA unavailable).

    C mirrors sklearn's inverse-regularization-strength convention:
    weight_decay = 1 / (C * n_train), so larger C = weaker regularization,
    matching sklearn's `LogisticRegression(C=...)` semantics.

    Inherits BaseEstimator/ClassifierMixin purely so this can sit inside an
    sklearn Pipeline (which, on recent sklearn, checks `__sklearn_tags__`
    when validating `is_fitted` -- only estimators descending from
    BaseEstimator get that for free). No sklearn validation/fit logic is
    inherited or relied on beyond that; `__init__` still just stores every
    constructor arg under its own name, satisfying BaseEstimator's
    get_params/set_params contract without needing to override them.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 20000,
        lr: float = 0.05,
        tol: float = 1e-7,
        patience: int = 1000,
        min_iter: int = 500,
        class_weight: str | None = None,
        device: str | None = None,
        seed: int = 0,
    ):
        self.C = C
        self.max_iter = max_iter
        self.lr = lr
        self.tol = tol
        self.patience = patience
        self.min_iter = min_iter
        self.class_weight = class_weight
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self.classes_: np.ndarray | None = None
        self.n_iter_: int = 0
        self._model: nn.Linear | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchLogisticProbe":
        torch.manual_seed(self.seed)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.classes_, counts = np.unique(y, return_counts=True)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[v] for v in y], dtype=np.int64)

        n_train, n_features = X.shape
        n_classes = len(self.classes_)
        weight_decay = 1.0 / (self.C * n_train)

        Xt = torch.tensor(X, device=self.device)
        yt = torch.tensor(y_idx, device=self.device)

        self._model = nn.Linear(n_features, n_classes).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=weight_decay)
        if self.class_weight == "balanced":
            # Matches sklearn's class_weight="balanced": n_samples / (n_classes * count_per_class).
            class_weights = n_train / (n_classes * counts)
            weight_t = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_t)
        else:
            loss_fn = nn.CrossEntropyLoss()

        best_loss = float("inf")
        stall_count = 0
        for it in range(self.max_iter):
            optimizer.zero_grad()
            logits = self._model(Xt)
            loss = loss_fn(logits, yt)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            if best_loss - loss_val > self.tol:
                best_loss = loss_val
                stall_count = 0
            else:
                stall_count += 1
            # Adam's loss curve plateaus temporarily before continuing to
            # improve -- a naive patience check stops far too early (empirically
            # confirmed: landed at chance-level AUROC on layers where sklearn's
            # lbfgs found real signal). min_iter is a floor below which patience
            # cannot trigger a stop.
            if it >= self.min_iter and stall_count >= self.patience:
                break
        self.n_iter_ = it + 1
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.tensor(X, device=self.device)
        logits = self._model(Xt)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
