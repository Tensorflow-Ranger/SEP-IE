"""Supervised Contrastive Loss (Khosla et al. 2020) probe: a small projection
head trained to pull same-label embeddings together and push different-label
embeddings apart, evaluated with a plain linear classifier on the resulting
frozen embeddings (the standard two-stage SupCon recipe).

Motivation specific to this repo's cross-source-transfer problem: plain
cross-entropy (src/probes.py's LR/MLP) only needs *a* decision boundary that
works on the pooled training mix -- it has no explicit pressure to make
same-label points from DIFFERENT dataset_source values sit near each other in
representation space. SupCon does, because every training step's positive
pairs are drawn from the whole batch: with ~1000 train rows fitting in one
full batch (no minibatch sampler needed), every step already contrasts
same-label pairs across all four sources at once, which is a more direct
objective for "cluster by ambiguity, not by which benchmark this came from"
than anything cross-entropy training does implicitly.

Two input-feature variants are evaluated side by side (see
scripts/train_supcon_probe.py): the existing PCA-100 features (matches
src/probes.py's pipeline, D>>N conditioning already validated) and raw
(standardized) layer activations (D=4096, lets the projection head's own
backprop-trained weights do task-aware dimensionality reduction instead of
PCA's unsupervised, contrastive-objective-unaware one). Neither is assumed
correct going in -- both go through the identical evaluation harness.

The projection head is deliberately small (a single Linear layer by default,
optionally one hidden layer) given how little training data there is;
weight_decay and dropout (when a hidden layer is used) are the regularizers,
matching the C-as-inverse-weight-decay convention used throughout
src/torch_probe.py / src/torch_mlp_probe.py so results stay comparable.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def supcon_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1,
    sources: torch.Tensor | None = None, cross_source_only: bool = False,
) -> torch.Tensor:
    """Supervised contrastive loss, single-view (one embedding per example).

    embeddings: (N, D), L2-normalized. labels: (N,) int64.
    Standard SupCon "out" formulation (Khosla et al. 2020, eq. 2): for each
    anchor, positives are every OTHER same-label row in the batch; the
    denominator sums over every other row (positives and negatives alike).

    `cross_source_only=True` (requires `sources`, an integer-encoded
    dataset_source id per row) restricts the POSITIVE set further, to
    same-label rows from a DIFFERENT source only -- e.g. an orchid_heval
    ambiguous anchor's positives become clarifycodebench-ambiguous and
    humanevalcomm-ambiguous rows, never other orchid_heval-ambiguous rows.
    The denominator is left unchanged (still sums over every other row,
    same-source included) -- this is the minimal modification that makes
    cross-source same-label pairs the thing being explicitly pulled
    together, without also fighting the (harmless) same-source same-label
    similarity structure via the partition function. This directly targets
    what plain same-label SupCon only does implicitly and dilutes among
    same-source pairs: forcing "ambiguous" to mean the same thing in
    representation space regardless of which benchmark generated the row.
    """
    device = embeddings.device
    N = embeddings.shape[0]
    sim = embeddings @ embeddings.T / temperature  # (N, N)
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()  # numerical stability, standard log-sum-exp shift

    labels = labels.view(-1, 1)
    same_label = torch.eq(labels, labels.T).float().to(device)
    self_mask = torch.eye(N, device=device)
    pos_mask = same_label - self_mask  # exclude self as its own positive

    if cross_source_only:
        if sources is None:
            raise ValueError("cross_source_only=True requires `sources`")
        sources = sources.view(-1, 1)
        diff_source = (1.0 - torch.eq(sources, sources.T).float()).to(device)
        pos_mask = pos_mask * diff_source

    exp_sim = torch.exp(sim) * (1 - self_mask)  # exclude self from the denominator
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0  # guard: an anchor with zero (cross-source) same-label peers in-batch
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1)[valid] / pos_counts[valid]
    return -mean_log_prob_pos.mean()


class SupConProjector(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int = 32, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.net = nn.Linear(in_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def fit_supcon_projector(
    X_train: np.ndarray,
    y_train: np.ndarray,
    embed_dim: int = 32,
    hidden_dim: int | None = None,
    temperature: float = 0.1,
    weight_decay: float = 1e-3,
    lr: float = 0.01,
    epochs: int = 500,
    seed: int = 0,
    device: str | None = None,
    sources_train: np.ndarray | None = None,
    cross_source_only: bool = False,
):
    """Full-batch SupCon training (the whole train split fits in one batch at
    this N, so no sampler is needed -- every step already contrasts pairs
    across all dataset sources present in train). Returns (encode_fn, model);
    encode_fn maps a numpy array of raw/PCA features to frozen embeddings.

    `sources_train` (string array, one dataset_source per row) +
    `cross_source_only=True` switches to the cross-source-positives-only
    variant of the loss (see supcon_loss's docstring).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xt = torch.tensor(np.asarray(X_train, dtype=np.float32), device=device)
    yt = torch.tensor(np.asarray(y_train, dtype=np.int64), device=device)

    sources_t = None
    if cross_source_only:
        if sources_train is None:
            raise ValueError("cross_source_only=True requires sources_train")
        uniq = {s: i for i, s in enumerate(sorted(set(sources_train)))}
        if len(uniq) < 2:
            # With only one source present in train, every anchor's
            # cross-source positive set is empty (supcon_loss's `valid` mask
            # is all-False), so mean_log_prob_pos.mean() is NaN with a
            # gradient of exactly zero -- the loss silently contributes NO
            # signal. Training still "completes" (500 epochs, no crash), but
            # the projector head just decays toward the zero matrix under
            # weight_decay alone, producing a near-arbitrary embedding that
            # looks like a normal frozen embedding to downstream code while
            # having learned nothing. Fail loudly instead.
            raise ValueError(
                f"cross_source_only=True but only one dataset_source present in "
                f"sources_train ({next(iter(uniq))!r}) -- the loss has no valid "
                f"cross-source positive pairs and silently trains on zero "
                f"gradient (NaN loss, empty backward). This is meaningless; "
                f"the caller must skip this config for this fold."
            )
        sources_t = torch.tensor([uniq[s] for s in sources_train], dtype=torch.int64, device=device)

    model = SupConProjector(X_train.shape[1], embed_dim=embed_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        z = model(Xt)
        loss = supcon_loss(z, yt, temperature=temperature, sources=sources_t, cross_source_only=cross_source_only)
        loss.backward()
        opt.step()
    model.eval()

    @torch.no_grad()
    def encode(X: np.ndarray) -> np.ndarray:
        Xte = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        return model(Xte).cpu().numpy()

    return encode, model
