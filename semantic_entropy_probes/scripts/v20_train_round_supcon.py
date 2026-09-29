"""v20 pilot stage 4: does treating round-depth (0/1/2 clarifications already
applied) as the SupCon `cross_source_only` grouping axis -- instead of
dataset_source, as in `## v19 cross-source-only SupCon` -- help a candidate's
tier meaning transfer across rounds? Direct test of the v12 live-loop note's
open question (same correct candidate ranked 3rd in round 1, dead-last by
round 3 -- is that a round-depth representation-drift problem the same
cross-source-only fix addresses?).

Leave-one-round-out (3 folds: train on the other 2 round depths, test on the
held-out one), fixed layer 22 (v19's fallback layer -- no CV sweep, this
pilot is 117 rows total, too small to CV-select a layer reliably), raw
features only (v19 found cross-source-only SupCon's gains concentrated in
`raw`, not `pca`). Three configs per fold:
  - raw_lr: plain pairwise LR on raw features, no SupCon (the v15/v19 baseline)
  - supcon_raw_mlp: plain SupCon (any other same-tier row is a positive)
  - supconxsrc_raw_mlp: cross-round-only SupCon (positives only from a
    DIFFERENT round depth, same tier) -- the one under test

Usage: python -m scripts.v20_train_round_supcon
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.supcon import fit_supcon_projector  # noqa: E402

ACT_DIR = ROOT / "data" / "v20_pilot_multihop" / "activations"
OUT_PATH = ROOT / "data" / "v20_pilot_multihop" / "round_supcon_results.json"

LAYER = 22
SEED = 0
MLP_HIDDEN = (64,)
MLP_MAX_ITER = 500
ROUNDS = ("0", "1", "2")


def load_groups():
    ids, round_depth = [], []
    layers = None
    per_group_vecs, per_group_tiers = {}, {}
    for npz_path in sorted(ACT_DIR.glob("*.npz")):
        m = re.match(r"(.+)__r(\d)$", npz_path.stem)
        if not m:
            continue
        gid, rd = m.group(1) + f"__r{m.group(2)}", m.group(2)
        data = np.load(npz_path)
        if layers is None:
            layers = data["layers"].tolist()
        cand = data["candidates"]
        tiers = data["tiers"]
        if len(tiers) < 2 or len(set(tiers.tolist())) < 2:
            continue
        per_group_vecs[gid] = np.transpose(cand, (1, 0, 2))  # (n_layers, n_cand, dim)
        per_group_tiers[gid] = tiers
        ids.append(gid)
        round_depth.append(rd)
    print(f"[v20-round-supcon] loaded {len(ids)} usable groups "
          f"(round counts: {[(r, round_depth.count(r)) for r in ROUNDS]})", flush=True)
    return np.array(ids), np.array(round_depth), layers, per_group_vecs, per_group_tiers


def build_pairwise_xy(train_ids, per_group_vecs, per_group_tiers, layer_idx, pipe):
    X, y = [], []
    for g in train_ids:
        z = pipe.transform(per_group_vecs[g][layer_idx])
        tiers = per_group_tiers[g]
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    X.append(z[i] - z[j]); y.append(1)
                    X.append(z[j] - z[i]); y.append(0)
    return np.array(X), np.array(y)


def eval_ranking(group_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, model, head):
    hits = []
    for gid in group_ids:
        vecs = per_group_vecs[gid][layer_idx]
        tiers = per_group_tiers[gid]
        z = pipe.transform(vecs)
        n = len(tiers)
        if head == "lr":
            scores = z @ model["w"]
        else:
            diffs, pair_idx = [], []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diffs.append(z[i] - z[j])
                    pair_idx.append(i)
            proba = model["clf"].predict_proba(np.array(diffs))[:, 1]
            scores = np.zeros(n)
            for p, i in zip(proba, pair_idx):
                scores[i] += p
        order = np.argsort(-scores)
        best_tier = tiers.max()
        hits.append(int(tiers[order[0]] == best_tier))
    return float(np.mean(hits)), len(group_ids)


def fit_raw_lr(train_ids, per_group_vecs, per_group_tiers, layer_idx):
    pool = np.concatenate([per_group_vecs[g][layer_idx] for g in train_ids], axis=0)
    pipe = make_pipeline(StandardScaler()).fit(pool)
    X, y = build_pairwise_xy(train_ids, per_group_vecs, per_group_tiers, layer_idx, pipe)
    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(X, y)
    return pipe, {"w": clf.coef_[0]}, "lr"


class SupConPipe:
    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device

    def transform(self, X):
        import torch
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(Xs, dtype=np.float32), device=self.device)
            return self.model(Xt).cpu().numpy()


def fit_supcon(train_ids, per_group_vecs, per_group_tiers, round_by_id, layer_idx, cross_source_only):
    X_pool, y_pool, src_pool = [], [], []
    for g in train_ids:
        vecs = per_group_vecs[g][layer_idx]
        tiers = per_group_tiers[g]
        for i in range(len(tiers)):
            X_pool.append(vecs[i]); y_pool.append(int(tiers[i])); src_pool.append(round_by_id[g])
    X_pool, y_pool, src_pool = np.array(X_pool), np.array(y_pool), np.array(src_pool)
    scaler = StandardScaler().fit(X_pool)
    X_scaled = scaler.transform(X_pool)

    encode_fn, supcon_model = fit_supcon_projector(
        X_scaled, y_pool, embed_dim=32, hidden_dim=None, temperature=0.1,
        weight_decay=1e-3, lr=0.01, epochs=500, seed=SEED,
        sources_train=src_pool, cross_source_only=cross_source_only,
    )
    device = next(supcon_model.parameters()).device
    pipe = SupConPipe(supcon_model, scaler, device)

    X, y = build_pairwise_xy(train_ids, per_group_vecs, per_group_tiers, layer_idx, pipe)
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER, random_state=SEED, early_stopping=True)
    clf.fit(X, y)
    return pipe, {"clf": clf}, "mlp"


def main():
    import torch
    print(f"[v20-round-supcon] cuda available: {torch.cuda.is_available()}", flush=True)
    ids, round_depth, layers, per_group_vecs, per_group_tiers = load_groups()
    layer_idx = layers.index(LAYER)
    round_by_id = dict(zip(ids, round_depth))

    results = {"lodo": {}}
    for held_out in ROUNDS:
        train_mask = round_depth != held_out
        test_mask = round_depth == held_out
        train_ids, test_ids = ids[train_mask], ids[test_mask]
        if len(test_ids) < 3 or len(train_ids) < 10:
            print(f"[v20-round-supcon] held_out={held_out}: skipped (too few rows, "
                  f"train={len(train_ids)} test={len(test_ids)})", flush=True)
            continue
        print(f"\n[v20-round-supcon] === held_out round={held_out} (train={len(train_ids)}, test={len(test_ids)}) ===",
              flush=True)

        fold = {}
        for name, fit_fn in (
            ("raw_lr", lambda: fit_raw_lr(train_ids, per_group_vecs, per_group_tiers, layer_idx)),
            ("supcon_raw_mlp", lambda: fit_supcon(train_ids, per_group_vecs, per_group_tiers, round_by_id,
                                                   layer_idx, cross_source_only=False)),
            ("supconxsrc_raw_mlp", lambda: fit_supcon(train_ids, per_group_vecs, per_group_tiers, round_by_id,
                                                       layer_idx, cross_source_only=True)),
        ):
            pipe, model, head = fit_fn()
            test_hit, n_test = eval_ranking(test_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, model, head)
            print(f"[v20-round-supcon] held_out={held_out} {name}: test_hit_rate={test_hit:.3f} (n={n_test})",
                  flush=True)
            fold[name] = {"test_hit_rate": test_hit, "n_test": n_test}
        results["lodo"][held_out] = fold

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[v20-round-supcon] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
