"""GPU-trained, regularized 1-hidden-layer MLP -- the "ML" counterpart to
TorchLogisticProbe (src/torch_probe.py) / sklearn's LogisticRegression.

Logistic regression is linear in the (PCA-whitened) activations; this adds
exactly one nonlinearity (a hidden ReLU layer) so it can pick up interaction
effects a linear probe structurally cannot, while staying small enough
(default hidden_dim=64, one hidden layer) not to just memorize a train split
of a few hundred to ~1500 rows. Regularization is two-pronged, both required
given how easy it is for a nonlinear model to overfit at this N:
  - weight_decay (L2 on all weights), same C-as-inverse-regularization-strength
    convention as TorchLogisticProbe, so `C` sweeps identically across both
    probe types and the results are comparable entry-for-entry.
  - dropout on the hidden activations, which weight_decay alone does not
    provide and which specifically curbs co-adapted hidden units -- the
    standard failure mode for a small MLP on a small, high-dimensional-input
    dataset.

Same sklearn-compatible .fit/.predict_proba/.predict/.classes_ interface as
TorchLogisticProbe, so it drops into the same sklearn Pipeline (PCA + scale)
used throughout src/probes.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin


class TorchMLPProbe(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        C: float = 1.0,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        max_iter: int = 20000,
        lr: float = 0.01,
        tol: float = 1e-7,
        patience: int = 1000,
        min_iter: int = 500,
        class_weight: str | None = None,
        device: str | None = None,
        seed: int = 0,
    ):
        self.C = C
        self.hidden_dim = hidden_dim
        self.dropout = dropout
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
        self._model: nn.Module | None = None

    def _build_model(self, n_features: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, n_classes),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchMLPProbe":
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

        self._model = self._build_model(n_features, n_classes).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr, weight_decay=weight_decay)
        if self.class_weight == "balanced":
            class_weights = n_train / (n_classes * counts)
            weight_t = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            loss_fn = nn.CrossEntropyLoss(weight=weight_t)
        else:
            loss_fn = nn.CrossEntropyLoss()

        best_loss = float("inf")
        stall_count = 0
        self._model.train()
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
            if it >= self.min_iter and stall_count >= self.patience:
                break
        self.n_iter_ = it + 1
        self._model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        Xt = torch.tensor(X, device=self.device)
        self._model.eval()
        logits = self._model(Xt)
        probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        idx = np.argmax(probs, axis=1)
        return self.classes_[idx]
