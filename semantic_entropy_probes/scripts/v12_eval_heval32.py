"""v12 heval32 eval: score the trained rank_probe_v12_pairwise on the 32-problem
heval32 held-out benchmark's activations (data/v12_ranking/heval32_activations/),
computing the same rank-1 hit rate / mean rank metric Stage E reports on its
own val/test splits -- except heval32 was never in the training pool at all
(excluded at Stage A), so this is a fully clean generalization check plus the
specific "score on the heval32 problems" number the user asked for.

Usage: python v12_eval_heval32.py
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

MODEL_PATH = ROOT / "models" / "rank_probe_v12_pairwise.joblib"
MODEL_JSON_PATH = ROOT / "models" / "rank_probe_v12_pairwise.json"

VARIANT_PATHS = {
    "vagueness": {
        "act_dir": ROOT / "data" / "v12_ranking" / "heval32_activations",
        "out": ROOT / "data" / "v12_ranking" / "heval32_eval_results.json",
    },
    "lexical": {
        "act_dir": ROOT / "data" / "v12_ranking" / "heval32_lexical_activations",
        "out": ROOT / "data" / "v12_ranking" / "heval32_lexical_eval_results.json",
    },
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["vagueness", "lexical"], default="vagueness")
    args = ap.parse_args()
    ACT_DIR = VARIANT_PATHS[args.variant]["act_dir"]

    meta = json.loads(MODEL_JSON_PATH.read_text())
    layer = meta["layer"]
    bundle = joblib.load(MODEL_PATH)
    pipe, w = bundle["pipe"], bundle["w"]

    npz_paths = sorted(ACT_DIR.glob("*.npz"))
    print(f"[v12-heval32-eval] {len(npz_paths)} problems found, using layer={layer}", flush=True)

    hits, ranks, skipped = [], [], []
    per_problem = []
    for npz_path in npz_paths:
        gid = npz_path.stem
        data = np.load(npz_path)
        layers = data["layers"].tolist()
        li = layers.index(layer)
        cand = data["candidates"][:, li, :]  # (n_candidates, dim)
        tiers = data["tiers"]

        if len(set(tiers.tolist())) < 2:
            skipped.append(gid)
            continue

        z = pipe.transform(cand)
        scores = z @ w
        order = np.argsort(-scores)
        best_tier = tiers.max()
        top_idx = order[0]
        hit = int(tiers[top_idx] == best_tier)
        rank_of_best = 1 + int(np.where(tiers[order] == best_tier)[0][0])
        hits.append(hit)
        ranks.append(rank_of_best)
        per_problem.append({"id": gid, "hit": hit, "rank_of_best": rank_of_best,
                             "n_candidates": len(tiers), "best_tier": int(best_tier)})

    print(f"[v12-heval32-eval] {len(skipped)} problems skipped (all-same-tier, no ranking signal): {skipped}", flush=True)
    print(f"[v12-heval32-eval] rank-1 hit rate: {np.mean(hits):.3f} ({sum(hits)}/{len(hits)})", flush=True)
    print(f"[v12-heval32-eval] mean rank of best: {np.mean(ranks):.2f}", flush=True)
    for row in per_problem:
        mark = "HIT " if row["hit"] else "MISS"
        print(f"[v12-heval32-eval]   {mark} {row['id']}: rank={row['rank_of_best']}/{row['n_candidates']} "
              f"(best_tier={row['best_tier']})", flush=True)

    out_path = VARIANT_PATHS[args.variant]["out"]
    out_path.write_text(json.dumps({
        "layer": layer,
        "hit_rate": float(np.mean(hits)),
        "mean_rank": float(np.mean(ranks)),
        "n_evaluated": len(hits),
        "n_skipped": len(skipped),
        "skipped_ids": skipped,
        "per_problem": per_problem,
    }, indent=2))
    print(f"[v12-heval32-eval] saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
