"""Does a better layer choice raise v19's probe hit rate? (GPU version)

v19_train_ranking_probe.py only ever reports the test hit rate of the layer
picked by 3x4-fold grouped CV on train+val -- it never evaluates the other 5
layers [16,18,20,22,24,26] on test, so there's no way to tell from
training_results.json alone whether CV picked the actual best-performing
layer or left points on the table (the same question v12's Stage E control
answered: CV picked layer 22, which also happened to be the best-of-6 test
performer, 0.777 -- see FINDINGS.md's "## Control experiments" section).

This script reuses v19_train_ranking_probe.py's data loading + split
reconstruction (same seed=0) and re-trains/evaluates EVERY layer x
{pca,raw}x{lr,mlp} combo on every split (pooled proportional, pooled
equal_per_source, LODOx4 sources) -- hidden states are already extracted in
the .npz files, no model/tokenizer needed.

GPU version: PCA/StandardScaler/LR/MLP are reimplemented in PyTorch and run
on CUDA (no cuML available on this box) -- the CPU version of this sweep
(144 total layer x config x split fits, several involving 4096-dim raw
features and MLP training) was too slow to finish in a reasonable time on
CPU alone. Not bit-identical to sklearn's LogisticRegression/MLPClassifier
(different optimizer/convergence details) but same model family/capacity,
run as a genuinely faster substitute rather than a from-scratch algorithm.

Usage: python v19_layer_sweep_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.combined_splits import make_combined_train_val_test_split, make_equal_per_source_split  # noqa: E402
from v19_train_ranking_probe import SOURCES, load_groups, SEED  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PCA_DIM = 100
OUT_PATH = ROOT / "data" / "v19_concat_activations" / "layer_sweep_results.json"
torch.manual_seed(SEED)


def t(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=DEVICE)


class GpuPipe:
    """StandardScaler + optional PCA, fit once, reused as a callable transform."""

    def __init__(self, feature_variant: str, train_pool: np.ndarray):
        x = t(train_pool)
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        xs = (x - self.mean) / self.std
        self.feature_variant = feature_variant
        if feature_variant == "pca":
            n_comp = min(PCA_DIM, xs.shape[0] - 1, xs.shape[1])
            # economy SVD on the standardized, mean-centered pool (mean already 0 post-scaling)
            u, s, vh = torch.linalg.svd(xs, full_matrices=False)
            self.components = vh[:n_comp].T  # (D, n_comp)
        else:
            self.components = None

    def transform(self, x_np: np.ndarray) -> torch.Tensor:
        x = t(x_np)
        xs = (x - self.mean) / self.std
        if self.components is not None:
            xs = xs @ self.components
        return xs


def fit_logreg(X: torch.Tensor, y: torch.Tensor, n_features: int, epochs: int = 300, lr: float = 0.1, wd: float = 1e-3):
    w = torch.zeros(n_features, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=lr, max_iter=epochs, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = X @ w
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y) + wd * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return {"type": "lr", "w": w.detach()}


class TinyMLP(torch.nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(X: torch.Tensor, y: torch.Tensor, n_features: int, max_epochs: int = 200, patience: int = 10):
    n_val = max(1, int(0.1 * X.shape[0]))
    perm = torch.randperm(X.shape[0], device=DEVICE)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr, Xval, yval = X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    model = TinyMLP(n_features).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train()
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vloss = torch.nn.functional.binary_cross_entropy_with_logits(model(Xval), yval).item()
        if vloss < best_val - 1e-4:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return {"type": "mlp", "model": model}


def build_pairwise_xy(train_ids, per_group_vecs, per_group_tiers, layer_idx, pipe):
    X, y = [], []
    for g in train_ids:
        z = pipe.transform(per_group_vecs[g][layer_idx])
        tiers = per_group_tiers[g]
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    X.append(z[i] - z[j]); y.append(1.0)
                    X.append(z[j] - z[i]); y.append(0.0)
    return torch.stack(X), torch.tensor(y, device=DEVICE)


def fit_layer_probe_gpu(train_ids, per_group_vecs, per_group_tiers, layer_idx, feature_variant, head):
    train_vecs_pool = np.concatenate([per_group_vecs[g][layer_idx] for g in train_ids], axis=0)
    pipe = GpuPipe(feature_variant, train_vecs_pool)
    X, y = build_pairwise_xy(train_ids, per_group_vecs, per_group_tiers, layer_idx, pipe)
    n_features = X.shape[1]
    model = fit_logreg(X, y, n_features) if head == "lr" else fit_mlp(X, y, n_features)
    return pipe, model


@torch.no_grad()
def eval_ranking_gpu(group_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, model, head):
    hits = []
    for gid in group_ids:
        z = pipe.transform(per_group_vecs[gid][layer_idx])
        tiers = per_group_tiers[gid]
        n = len(tiers)
        if head == "lr":
            scores = (z @ model["w"]).cpu().numpy()
        else:
            diffs, pair_idx = [], []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diffs.append(z[i] - z[j])
                    pair_idx.append(i)
            proba = torch.sigmoid(model["model"](torch.stack(diffs))).cpu().numpy()
            scores = np.zeros(n)
            for p, i in zip(proba, pair_idx):
                scores[i] += p
        order = np.argsort(-scores)
        best_tier = tiers.max()
        hits.append(int(tiers[order[0]] == best_tier))
    return float(np.mean(hits)), len(group_ids)


def main() -> None:
    print(f"[layer-sweep] device={DEVICE}", flush=True)
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = load_groups()
    layers = layers.tolist()
    labels_dummy = np.ones(len(ids), dtype=np.int64)

    prop_train_idx, prop_val_idx, prop_test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    eq_train_idx, eq_val_idx, eq_test_idx = make_equal_per_source_split(
        dataset_source, source_problem_id, seed=SEED,
    )

    results = {"pooled": {}, "pooled_equal_per_source": {}, "lodo": {}}
    grid = [(fv, h) for fv in ("pca", "raw") for h in ("lr", "mlp")]

    for feature_variant, head in grid:
        key = f"{feature_variant}_{head}"
        print(f"\n=== {key} ===", flush=True)

        for tag, (tr_idx, te_idx) in (
            ("proportional", (prop_train_idx, prop_test_idx)),
            ("equal_per_source", (eq_train_idx, eq_test_idx)),
        ):
            t0 = time.time()
            train_ids, test_ids = ids[tr_idx], ids[te_idx]
            per_layer = {}
            for li, layer in enumerate(layers):
                pipe, model = fit_layer_probe_gpu(train_ids, per_group_vecs, per_group_tiers, li, feature_variant, head)
                test_hit, n_test = eval_ranking_gpu(test_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
                per_layer[layer] = test_hit
            best_layer = max(per_layer, key=per_layer.get)
            print(f"  pooled[{tag}]: " + " ".join(f"L{l}={h:.3f}" for l, h in per_layer.items())
                  + f"  -> best={best_layer} ({per_layer[best_layer]:.3f}) [{time.time()-t0:.0f}s]", flush=True)
            slot = results["pooled"] if tag == "proportional" else results["pooled_equal_per_source"]
            slot[key] = {"per_layer_test_hit_rate": per_layer, "n_test": n_test,
                         "best_layer_oracle": best_layer, "best_test_hit_rate_oracle": per_layer[best_layer]}

        lodo_out = []
        for held_out in SOURCES:
            t0 = time.time()
            train_mask = dataset_source != held_out
            test_mask = dataset_source == held_out
            train_ids_all = ids[train_mask]
            test_ids = ids[test_mask]
            if len(test_ids) < 5 or len(train_ids_all) < 20:
                continue
            per_layer = {}
            for li, layer in enumerate(layers):
                pipe, model = fit_layer_probe_gpu(train_ids_all, per_group_vecs, per_group_tiers, li, feature_variant, head)
                test_hit, n_test = eval_ranking_gpu(test_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
                per_layer[layer] = test_hit
            best_layer = max(per_layer, key=per_layer.get)
            print(f"  LODO[{held_out}]: " + " ".join(f"L{l}={h:.3f}" for l, h in per_layer.items())
                  + f"  -> best={best_layer} ({per_layer[best_layer]:.3f}) [{time.time()-t0:.0f}s]", flush=True)
            lodo_out.append({"held_out_source": held_out, "per_layer_test_hit_rate": per_layer, "n_test": n_test,
                              "best_layer_oracle": best_layer, "best_test_hit_rate_oracle": per_layer[best_layer]})
        results["lodo"][key] = lodo_out

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
