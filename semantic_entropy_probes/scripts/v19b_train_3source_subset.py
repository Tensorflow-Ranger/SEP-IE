"""v19b: isolate the effect of adding orchid_bcb_expand from the effect of
v19's stricter clean filter, by retraining on the SAME v19_concat_activations
(already extracted under the strict 3-way filter) restricted to just the 3
original sources (clarifycodebench, orchid_heval, orchid_bcb) -- i.e. what
v17 measured, but with v19's stricter more_important_than_label filter
applied instead of v17's looser leak+functional-only filter.

This isolates: v17 (3 src, loose filter) -> v19b (3 src, strict filter) ->
v19 (4 src, strict filter), so each arrow is exactly one change.

No new extraction or Bedrock calls needed -- just reload v19's already-built
per_group_vecs/per_group_tiers and filter dataset_source before training.
Reuses v19_train_ranking_probe.py's grid/CV/eval code unchanged.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.combined_splits import make_combined_train_val_test_split, report_split_composition  # noqa: E402
from v19_train_ranking_probe import (  # noqa: E402
    load_groups, cv_select_best_layer, fit_layer_probe, eval_ranking, run_lodo,
)

RESULTS_PATH = ROOT / "data" / "v19_concat_activations" / "training_results_3source_subset.json"
SEED = 0
SOURCES_3 = ("clarifycodebench", "orchid_heval", "orchid_bcb")


def main() -> None:
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = load_groups()
    mask = np.isin(dataset_source, SOURCES_3)
    ids, dataset_source, source_problem_id = ids[mask], dataset_source[mask], source_problem_id[mask]
    print(f"[v19b] restricted to 3 sources: {len(ids)} groups", flush=True)
    report_split_composition("3-source subset", np.arange(len(ids)), np.ones(len(ids), dtype=np.int64), dataset_source)

    results = {"pooled": {}, "lodo": {}}
    grid = [(fv, h) for fv in ("pca", "raw") for h in ("lr", "mlp")]

    labels_dummy = np.ones(len(ids), dtype=np.int64)
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    print(f"[v19b] pooled split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)

    for feature_variant, head in grid:
        key = f"{feature_variant}_{head}"
        t0 = time.time()
        train_ids, val_ids, test_ids = ids[train_idx], ids[val_idx], ids[test_idx]
        trainval_idx = np.concatenate([train_idx, val_idx])
        cv_mean = cv_select_best_layer(layers.tolist(), per_group_vecs, per_group_tiers, trainval_idx,
                                        ids, dataset_source, source_problem_id, feature_variant, head)
        best_layer = max(cv_mean, key=cv_mean.get)
        li = layers.tolist().index(best_layer)
        pipe, model = fit_layer_probe(train_ids, per_group_vecs, per_group_tiers, li, feature_variant, head)
        val_hit, _, n_val = eval_ranking(val_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
        test_hit, _, n_test = eval_ranking(test_ids, per_group_vecs, per_group_tiers, li, pipe, model, head)
        print(f"[v19b] {key} POOLED: best_layer={best_layer} cv={cv_mean[best_layer]:.3f} "
              f"val={val_hit:.3f}(n={n_val}) test={test_hit:.3f}(n={n_test}) [{time.time()-t0:.0f}s]", flush=True)
        results["pooled"][key] = {"best_layer": best_layer, "cv_hit_rate": cv_mean[best_layer],
                                   "val_hit_rate": val_hit, "n_val": n_val, "test_hit_rate": test_hit, "n_test": n_test}

        lodo_results = []
        for held_out in SOURCES_3:
            t1 = time.time()
            lr = run_lodo(layers, per_group_vecs, per_group_tiers, ids, dataset_source, source_problem_id,
                           feature_variant, head, held_out)
            if lr is None:
                continue
            print(f"[v19b] {key} LODO held_out={held_out}: best_layer={lr['best_layer']} "
                  f"cv={lr['cv_hit_rate']:.3f} test={lr['test_hit_rate']:.3f}(n={lr['n_test']}) "
                  f"[{time.time()-t1:.0f}s]", flush=True)
            lodo_results.append(lr)
        results["lodo"][key] = lodo_results

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[v19b] saved to {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
