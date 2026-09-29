"""Apples-to-apples baseline for the v12 ranking-probe heval32 eval: score the
OLD deployed probes (rank1_lr_layer20 / rank2_mlp_layer22, both trained on the
i.i.d. binary ambiguous/clear objective) on the SAME 32 heval32 problems and
the SAME inline-rewrite prompt format used for the new v12 pairwise-ranking
probe (data/v12_ranking/heval32_activations/*.npz already has layers 20 and
22 extracted in that format -- no new extraction needed).

Replicates the live pipeline's actual selection rule (v10_round_worker.py /
v11_heval_batch_worker.py): p1 = rank1.predict_proba(class=1) at layer 20,
p2 = rank2.predict_proba(class=1) at layer 22, mean_p = (p1+p2)/2,
delta = mean_p - baseline_mean_p, candidates sorted ascending by delta (most
negative delta = biggest drop in "ambiguous" probability = best). Rank-1 hit
rate: does the top-ranked (most negative delta) candidate have the max tier
in its group?

Usage: python v12_eval_heval32_old_probes.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"
RANK1_LAYER, RANK2_LAYER = 20, 22

VARIANT_PATHS = {
    "vagueness": {
        "act_dir": ROOT / "data" / "v12_ranking" / "heval32_activations",
        "out": ROOT / "data" / "v12_ranking" / "heval32_old_probes_eval_results.json",
    },
    "lexical": {
        "act_dir": ROOT / "data" / "v12_ranking" / "heval32_lexical_activations",
        "out": ROOT / "data" / "v12_ranking" / "heval32_lexical_old_probes_eval_results.json",
    },
}


def proba_class1(pipe, x):
    classes = list(pipe.named_steps["clf"].classes_) if hasattr(pipe.named_steps["clf"], "classes_") else list(pipe.classes_)
    idx1 = classes.index(1)
    return pipe.predict_proba(x)[:, idx1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["vagueness", "lexical"], default="vagueness")
    args = ap.parse_args()
    ACT_DIR = VARIANT_PATHS[args.variant]["act_dir"]

    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)

    npz_paths = sorted(ACT_DIR.glob("*.npz"))
    print(f"[old-probes-heval32] {len(npz_paths)} problems found (rank1@layer{RANK1_LAYER}, rank2@layer{RANK2_LAYER})", flush=True)

    hits, ranks, skipped = [], [], []
    per_problem = []
    for npz_path in npz_paths:
        gid = npz_path.stem
        data = np.load(npz_path)
        layers = data["layers"].tolist()
        li1, li2 = layers.index(RANK1_LAYER), layers.index(RANK2_LAYER)
        tiers = data["tiers"]

        if len(set(tiers.tolist())) < 2:
            skipped.append(gid)
            continue

        baseline = data["baseline"]  # (n_layers, dim)
        base_x1 = baseline[li1][None, :]
        base_x2 = baseline[li2][None, :]
        base_p1 = proba_class1(rank1, base_x1)[0]
        base_p2 = proba_class1(rank2, base_x2)[0]
        base_mean = (base_p1 + base_p2) / 2

        cand = data["candidates"]  # (n_candidates, n_layers, dim)
        x1 = cand[:, li1, :]
        x2 = cand[:, li2, :]
        p1 = proba_class1(rank1, x1)
        p2 = proba_class1(rank2, x2)
        mean_p = (p1 + p2) / 2
        delta = mean_p - base_mean

        order = np.argsort(delta)  # ascending: most negative (best) first
        best_tier = tiers.max()
        top_idx = order[0]
        hit = int(tiers[top_idx] == best_tier)
        rank_of_best = 1 + int(np.where(tiers[order] == best_tier)[0][0])
        hits.append(hit)
        ranks.append(rank_of_best)
        per_problem.append({"id": gid, "hit": hit, "rank_of_best": rank_of_best,
                             "n_candidates": len(tiers), "best_tier": int(best_tier)})

    print(f"[old-probes-heval32] {len(skipped)} problems skipped (all-same-tier, no ranking signal): {skipped}", flush=True)
    print(f"[old-probes-heval32] rank-1 hit rate: {np.mean(hits):.3f} ({sum(hits)}/{len(hits)})", flush=True)
    print(f"[old-probes-heval32] mean rank of best: {np.mean(ranks):.2f}", flush=True)
    for row in per_problem:
        mark = "HIT " if row["hit"] else "MISS"
        print(f"[old-probes-heval32]   {mark} {row['id']}: rank={row['rank_of_best']}/{row['n_candidates']} "
              f"(best_tier={row['best_tier']})", flush=True)

    out_path = VARIANT_PATHS[args.variant]["out"]
    out_path.write_text(json.dumps({
        "rank1_layer": RANK1_LAYER, "rank2_layer": RANK2_LAYER,
        "hit_rate": float(np.mean(hits)), "mean_rank": float(np.mean(ranks)),
        "n_evaluated": len(hits), "n_skipped": len(skipped), "skipped_ids": skipped,
        "per_problem": per_problem,
    }, indent=2))
    print(f"[old-probes-heval32] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
