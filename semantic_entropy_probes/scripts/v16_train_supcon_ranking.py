"""v16: SupCon + MLP for the ranking objective (deferred piece of the v15
grid). Never tried before for this objective -- prior SupCon work
(src/supcon.py, scripts/train_supcon_probe.py) was all for the earlier
i.i.d. ambiguous/clear binary classification task.

Adaptation: SupCon needs a per-example scalar label to contrast on. For the
ranking objective the natural choice is TIER (0/1/2) itself -- pool every
candidate from every training group as one example (X=activation vector,
y=tier, source=that row's dataset_source), fit the projector so same-tier
candidates cluster and different-tier ones separate (optionally pulling
same-tier candidates from DIFFERENT sources together, exactly like the
original recipe's cross-source pressure). Then, frozen, encode every
candidate and train an MLP on the SupCon embeddings' pairwise differences
-- same win-counting inference as v15's mlp head, just on 32-dim SupCon
embeddings instead of raw/PCA-100 features.

Runs on GPU (src/supcon.py auto-selects cuda) for the SupCon fit; the MLP
head on the resulting 32-dim embeddings is cheap enough that CPU is fine.

To keep runtime bounded, reuses v15's already layer-selected result for
each feature variant (pca_mlp / raw_mlp's chosen layer) rather than
re-running a full CV layer sweep for SupCon too -- consistent with how the
original SupCon investigation fixed a single layer rather than re-sweeping
per config.

Usage: python v16_train_supcon_ranking.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.supcon import fit_supcon_projector  # noqa: E402
from src.combined_splits import make_combined_train_val_test_split  # noqa: E402
import v15_train_ranking_probe as v15  # noqa: E402

RESULTS_V15_PATH = ROOT / "data" / "v15_new_activations" / "training_results.json"
RESULTS_OUT_PATH = ROOT / "data" / "v15_new_activations" / "training_results_supcon.json"
OUT_DIR = ROOT / "models"
SEED = 0
FALLBACK_LAYER = 20
MLP_HIDDEN = (64,)
MLP_MAX_ITER = 500


class SupConPipe:
    """Duck-types v15's `pipe.transform(vecs)` interface around a frozen SupCon encoder.

    Holds the raw nn.Module (not fit_supcon_projector's `encode_fn` closure --
    joblib/pickle can't serialize a locally-defined closure function) and
    reimplements the forward pass directly so this whole object is picklable.

    `pre_pca`, when given, is applied AFTER scaling and BEFORE the SupCon
    projector -- this is what actually makes the "pca" feature variant
    different from "raw" (feeding the projector 100-dim PCA'd input instead
    of the full 4096-dim standardized activation).
    """

    def __init__(self, model, scaler, device, pre_pca=None):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.pre_pca = pre_pca

    def transform(self, X):
        import torch
        Xs = self.scaler.transform(X)
        if self.pre_pca is not None:
            Xs = self.pre_pca.transform(Xs)
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(Xs, dtype=np.float32), device=self.device)
            return self.model(Xt).cpu().numpy()


def best_layer_for(feature_variant: str, layers: list[int]) -> int:
    if RESULTS_V15_PATH.exists():
        r = json.loads(RESULTS_V15_PATH.read_text())
        key = f"{feature_variant}_mlp"
        if key in r.get("pooled", {}):
            return r["pooled"][key]["best_layer"]
    return FALLBACK_LAYER if FALLBACK_LAYER in layers else layers[len(layers) // 2]


def build_supcon_training_pool(train_ids, per_group_vecs, per_group_tiers, dataset_source_by_id, layer_idx):
    X, y, src = [], [], []
    for g in train_ids:
        vecs = per_group_vecs[g][layer_idx]
        tiers = per_group_tiers[g]
        for i in range(len(tiers)):
            X.append(vecs[i])
            y.append(int(tiers[i]))
            src.append(dataset_source_by_id[g])
    return np.array(X), np.array(y), np.array(src)


def fit_supcon_pipe_and_mlp(train_ids, per_group_vecs, per_group_tiers, dataset_source_by_id, layer_idx,
                             feature_variant):
    X_pool, y_pool, src_pool = build_supcon_training_pool(
        train_ids, per_group_vecs, per_group_tiers, dataset_source_by_id, layer_idx)
    scaler = StandardScaler().fit(X_pool)
    X_scaled = scaler.transform(X_pool)

    pre_pca = None
    if feature_variant == "pca":
        from sklearn.decomposition import PCA
        pre_pca = PCA(n_components=min(v15.PCA_DIM, X_scaled.shape[0] - 1), random_state=SEED).fit(X_scaled)
        X_scaled = pre_pca.transform(X_scaled)
    elif feature_variant != "raw":
        raise ValueError(feature_variant)

    encode_fn, supcon_model = fit_supcon_projector(
        X_scaled, y_pool, embed_dim=32, hidden_dim=None, temperature=0.1,
        weight_decay=1e-3, lr=0.01, epochs=500, seed=SEED,
    )
    device = next(supcon_model.parameters()).device
    pipe = SupConPipe(supcon_model, scaler, device, pre_pca=pre_pca)

    # MLP head on pairwise diffs of the frozen SupCon embeddings
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
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER, random_state=SEED, early_stopping=True)
    clf.fit(np.array(X), np.array(y))
    return pipe, {"clf": clf}


def eval_ranking_mlp(group_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, model):
    return v15.eval_ranking(group_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, model, "mlp")


def run_pooled(feature_variant, layers, per_group_vecs, per_group_tiers, ids, dataset_source, source_problem_id):
    layer = best_layer_for(feature_variant, layers.tolist())
    li = layers.tolist().index(layer)
    dataset_source_by_id = dict(zip(ids, dataset_source))

    labels_dummy = np.ones(len(ids), dtype=np.int64)
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    train_ids, val_ids, test_ids = ids[train_idx], ids[val_idx], ids[test_idx]

    pipe, model = fit_supcon_pipe_and_mlp(train_ids, per_group_vecs, per_group_tiers, dataset_source_by_id, li,
                                           feature_variant)
    val_hit, _, n_val = eval_ranking_mlp(val_ids, per_group_vecs, per_group_tiers, li, pipe, model)
    test_hit, _, n_test = eval_ranking_mlp(test_ids, per_group_vecs, per_group_tiers, li, pipe, model)
    return {"layer": layer, "val_hit_rate": val_hit, "n_val": n_val,
            "test_hit_rate": test_hit, "n_test": n_test, "pipe": pipe, "model": model}


def run_lodo(feature_variant, layers, per_group_vecs, per_group_tiers, ids, dataset_source, held_out_source):
    layer = best_layer_for(feature_variant, layers.tolist())
    li = layers.tolist().index(layer)
    dataset_source_by_id = dict(zip(ids, dataset_source))

    train_mask = dataset_source != held_out_source
    test_mask = dataset_source == held_out_source
    train_ids = ids[train_mask]
    test_ids = ids[test_mask]
    if len(test_ids) < 5 or len(train_ids) < 20:
        return None

    pipe, model = fit_supcon_pipe_and_mlp(train_ids, per_group_vecs, per_group_tiers, dataset_source_by_id, li,
                                           feature_variant)
    test_hit, _, n_test = eval_ranking_mlp(test_ids, per_group_vecs, per_group_tiers, li, pipe, model)
    return {"held_out_source": held_out_source, "layer": layer, "test_hit_rate": test_hit, "n_test": n_test}


def main():
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v15.load_groups()
    results = {"pooled": {}, "lodo": {}}

    for feature_variant in ("pca", "raw"):
        key = f"supcon_{feature_variant}_mlp"
        t0 = time.time()
        print(f"\n[v16] === POOLED: {key} (layer={best_layer_for(feature_variant, layers.tolist())}) ===", flush=True)
        r = run_pooled(feature_variant, layers, per_group_vecs, per_group_tiers, ids, dataset_source, source_problem_id)
        print(f"[v16] {key}: layer={r['layer']} val={r['val_hit_rate']:.3f}(n={r['n_val']}) "
              f"test={r['test_hit_rate']:.3f}(n={r['n_test']}) [{time.time()-t0:.0f}s]", flush=True)
        import joblib
        joblib.dump({"pipe": r["pipe"], "model": r["model"], "head": "mlp"},
                     OUT_DIR / f"rank_probe_v16_{key}_pooled.joblib")
        results["pooled"][key] = {k: v for k, v in r.items() if k not in ("pipe", "model")}

        print(f"[v16] === LODO: {key} ===", flush=True)
        lodo_results = []
        for held_out in ("clarifycodebench", "orchid_heval", "orchid_bcb"):
            t1 = time.time()
            lr = run_lodo(feature_variant, layers, per_group_vecs, per_group_tiers, ids, dataset_source, held_out)
            if lr is None:
                print(f"[v16] {key} LODO held_out={held_out}: skipped (too few rows)", flush=True)
                continue
            print(f"[v16] {key} LODO held_out={held_out}: layer={lr['layer']} "
                  f"test={lr['test_hit_rate']:.3f}(n={lr['n_test']}) [{time.time()-t1:.0f}s]", flush=True)
            lodo_results.append(lr)
        results["lodo"][key] = lodo_results

    RESULTS_OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[v16] saved results to {RESULTS_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
