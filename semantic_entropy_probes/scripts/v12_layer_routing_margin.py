"""Compare val-selected fixed-layer ranking vs. unsupervised per-group
confidence-margin routing vs. an oracle per-group layer ceiling.

Context: v12_train_ranking_probe.py sweeps LAYERS=[16,18,20,22,24,26], picks
one global layer by val hit_rate (currently layer 26), and uses that same
probe for every group at test time. This script asks: instead of one fixed
layer, can we route each group to whichever swept layer its own probe is
most CONFIDENT about (margin = top1 score - top2 score within that group,
using only the probe's own outputs, no labels), and does that beat the
fixed-layer baseline? We also compute the label-using oracle ceiling (best
possible layer per group) to see how much headroom a router could ever
capture.

Usage: python v12_layer_routing_margin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.combined_splits import make_combined_train_val_test_split  # noqa: E402
from v12_train_ranking_probe import load_groups  # noqa: E402

MODELS_DIR = ROOT / "models"
SEED = 0
# "second half of the network" among the swept layers (16..26 out of ~36 total)
ROUTING_LAYERS = [20, 22, 24, 26]


def group_scores(vecs_all_layers, layers, li_list, pipes_ws):
    """vecs_all_layers: (n_layers, n_candidates, dim) for one group.
    Returns {layer: scores (n_candidates,)}."""
    out = {}
    for li in li_list:
        layer = layers[li]
        pipe, w = pipes_ws[layer]
        z = pipe.transform(vecs_all_layers[li])
        out[layer] = z @ w
    return out


def main() -> None:
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = load_groups()
    layers = layers.tolist()
    li_by_layer = {layer: i for i, layer in enumerate(layers)}

    labels_dummy = np.ones(len(ids), dtype=np.int64)
    train_idx, val_idx, test_idx = make_combined_train_val_test_split(
        labels_dummy, dataset_source, source_problem_id, seed=SEED,
    )
    test_ids = ids[test_idx]

    pipes_ws = {}
    for layer in layers:
        d = joblib.load(MODELS_DIR / f"rank_probe_v12_pairwise_layer{layer}.joblib")
        pipes_ws[layer] = (d["pipe"], d["w"])

    fixed_best = 26  # from models/rank_probe_v12_pairwise.json, val-selected
    routing_li = [li_by_layer[l] for l in ROUTING_LAYERS]

    fixed_hits, routed_hits, oracle_hits = [], [], []
    n_layers_agree_with_oracle = 0

    for gid in test_ids:
        vecs = per_group_vecs[gid]  # (n_layers, n_candidates, dim)
        tiers = per_group_tiers[gid]
        best_tier = tiers.max()

        scores_by_layer = group_scores(vecs, layers, routing_li, pipes_ws)

        # fixed best-layer baseline
        s = scores_by_layer[fixed_best]
        top = np.argsort(-s)[0]
        fixed_hits.append(int(tiers[top] == best_tier))

        # unsupervised confidence-margin routing: pick layer with largest
        # top1-top2 score gap, using ONLY the probe's own scores (no labels)
        best_margin, best_layer_choice, best_layer_top = -np.inf, None, None
        for layer in ROUTING_LAYERS:
            s = scores_by_layer[layer]
            order = np.argsort(-s)
            margin = s[order[0]] - s[order[1]] if len(order) > 1 else 0.0
            if margin > best_margin:
                best_margin, best_layer_choice, best_layer_top = margin, layer, order[0]
        routed_hits.append(int(tiers[best_layer_top] == best_tier))

        # oracle ceiling: best possible outcome from ANY swept layer for this group
        oracle_hit = 0
        for layer in ROUTING_LAYERS:
            s = scores_by_layer[layer]
            top = np.argsort(-s)[0]
            if tiers[top] == best_tier:
                oracle_hit = 1
                break
        oracle_hits.append(oracle_hit)

    n = len(test_ids)
    print(f"[routing] n_test_groups={n}, routing layers={ROUTING_LAYERS}")
    print(f"[routing] fixed best-layer ({fixed_best}, val-selected):  hit_rate={np.mean(fixed_hits):.3f}")
    print(f"[routing] confidence-margin routed (unsupervised):        hit_rate={np.mean(routed_hits):.3f}")
    print(f"[routing] oracle per-group layer ceiling (uses labels):   hit_rate={np.mean(oracle_hits):.3f}")


if __name__ == "__main__":
    main()
