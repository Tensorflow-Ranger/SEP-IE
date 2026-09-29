"""v19: SupCon + MLP ablation for v19's concatenation-format activations
(4 sources, stricter 3-way clean filter -- see v19_train_ranking_probe.py),
closing the caveat both `## v19` and the zero-shot-control section of
FINDINGS.md flagged: v17/v18 found SupCon contrastive pretraining makes
`raw` (4096-dim) features competitive with `pca` on LODO, but that was never
re-run against v19's larger/differently-filtered 4-source pool. Motivated by
the layer-sweep finding that plain nonlinearity (an MLP head on frozen
linear features) does NOT help LODO generalization -- SupCon is a different
lever: it changes what the FROZEN REPRESENTATION optimizes for (pulling
same-tier points together across sources, not just within one), rather than
adding capacity to the downstream classifier.

Same three-way structural pattern as v16 (inline-rewrite) and v18
(v17's concat format): frozen SupCon projector (contrast on tier-as-label)
+ small MLP pairwise-diff head, same win-counting inference as the plain
`mlp` head. Reuses v19_train_ranking_probe's already-selected best layer
per feature variant's _mlp config (from training_results.json) rather than
re-sweeping CV for SupCon too (same shortcut v16/v18 took) -- SupCon encoder
training + the pairwise MLP head both run on GPU (src/supcon.py's
fit_supcon_projector auto-selects cuda; the pairwise head's input is only
32-dim post-SupCon so sklearn's CPU MLPClassifier is fast regardless, unlike
the 4096-dim raw features that made the plain (non-SupCon) layer sweep slow
on CPU).

Adds BOTH v19 pooled-eval modes (proportional + equal_per_source), plus all
4 LODO folds (v18 only had 3, pre-bcb_expand).

Usage: python v19_train_supcon_ranking.py
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
from src.combined_splits import make_combined_train_val_test_split, make_equal_per_source_split  # noqa: E402
import v19_train_ranking_probe as v19  # noqa: E402

RESULTS_V19_PATH = ROOT / "data" / "v19_concat_activations" / "training_results.json"
RESULTS_OUT_PATH = ROOT / "data" / "v19_concat_activations" / "training_results_supcon.json"
OUT_DIR = ROOT / "models"
SEED = 0
FALLBACK_LAYER = 22
MLP_HIDDEN = (64,)
MLP_MAX_ITER = 500
SOURCES = v19.SOURCES


class SupConPipe:
    """Duck-types v19's `pipe.transform(vecs)` interface around a frozen SupCon encoder."""

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
    if RESULTS_V19_PATH.exists():
        r = json.loads(RESULTS_V19_PATH.read_text())
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
        pre_pca = PCA(n_components=min(v19.PCA_DIM, X_scaled.shape[0] - 1), random_state=SEED).fit(X_scaled)
        X_scaled = pre_pca.transform(X_scaled)
    elif feature_variant != "raw":
        raise ValueError(feature_variant)

    encode_fn, supcon_model = fit_supcon_projector(
        X_scaled, y_pool, embed_dim=32, hidden_dim=None, temperature=0.1,
        weight_decay=1e-3, lr=0.01, epochs=500, seed=SEED,
    )
    device = next(supcon_model.parameters()).device
    pipe = SupConPipe(supcon_model, scaler, device, pre_pca=pre_pca)

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
    return v19.eval_ranking(group_ids, per_group_vecs, per_group_tiers, layer_idx, pipe, model, "mlp")


def run_pooled_from_split(feature_variant, layers, per_group_vecs, per_group_tiers, ids, dataset_source,
                           train_idx, val_idx, test_idx, tag):
    layer = best_layer_for(feature_variant, layers.tolist())
    li = layers.tolist().index(layer)
    dataset_source_by_id = dict(zip(ids, dataset_source))
    train_ids, val_ids, test_ids = ids[train_idx], ids[val_idx], ids[test_idx]

    pipe, model = fit_supcon_pipe_and_mlp(train_ids, per_group_vecs, per_group_tiers, dataset_source_by_id, li,
                                           feature_variant)
    val_hit, _, n_val = eval_ranking_mlp(val_ids, per_group_vecs, per_group_tiers, li, pipe, model)
    test_hit, _, n_test = eval_ranking_mlp(test_ids, per_group_vecs, per_group_tiers, li, pipe, model)
    return {"split": tag, "layer": layer, "val_hit_rate": val_hit, "n_val": n_val,
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
    import torch
    print(f"[v19-supcon] cuda available: {torch.cuda.is_available()}", flush=True)

    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    labels_dummy = np.ones(len(ids), dtype=np.int64)
    prop_train_idx, prop_val_idx, prop_test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    eq_train_idx, eq_val_idx, eq_test_idx = make_equal_per_source_split(
        dataset_source, source_problem_id, seed=SEED,
    )

    results = {"pooled": {}, "pooled_equal_per_source": {}, "lodo": {}}

    for feature_variant in ("pca", "raw"):
        key = f"supcon_{feature_variant}_mlp"
        layer = best_layer_for(feature_variant, layers.tolist())

        for tag, (tr, va, te) in (
            ("proportional", (prop_train_idx, prop_val_idx, prop_test_idx)),
            ("equal_per_source", (eq_train_idx, eq_val_idx, eq_test_idx)),
        ):
            t0 = time.time()
            print(f"\n[v19-supcon] === POOLED[{tag}]: {key} (layer={layer}) ===", flush=True)
            r = run_pooled_from_split(feature_variant, layers, per_group_vecs, per_group_tiers, ids, dataset_source,
                                       tr, va, te, tag)
            print(f"[v19-supcon] {key} [{tag}]: layer={r['layer']} val={r['val_hit_rate']:.3f}(n={r['n_val']}) "
                  f"test={r['test_hit_rate']:.3f}(n={r['n_test']}) [{time.time()-t0:.0f}s]", flush=True)
            import joblib
            joblib.dump({"pipe": r["pipe"], "model": r["model"], "head": "mlp"},
                        OUT_DIR / f"rank_probe_v19_{key}_{tag}.joblib")
            slot = results["pooled"] if tag == "proportional" else results["pooled_equal_per_source"]
            slot[key] = {k: v for k, v in r.items() if k not in ("pipe", "model")}

        print(f"[v19-supcon] === LODO: {key} ===", flush=True)
        lodo_results = []
        for held_out in SOURCES:
            t1 = time.time()
            lr = run_lodo(feature_variant, layers, per_group_vecs, per_group_tiers, ids, dataset_source, held_out)
            if lr is None:
                print(f"[v19-supcon] {key} LODO held_out={held_out}: skipped (too few rows)", flush=True)
                continue
            print(f"[v19-supcon] {key} LODO held_out={held_out}: layer={lr['layer']} "
                  f"test={lr['test_hit_rate']:.3f}(n={lr['n_test']}) [{time.time()-t1:.0f}s]", flush=True)
            lodo_results.append(lr)
        results["lodo"][key] = lodo_results

    RESULTS_OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n[v19-supcon] saved results to {RESULTS_OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
