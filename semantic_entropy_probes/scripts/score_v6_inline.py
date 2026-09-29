"""Score rank1 (LR)/rank2 (MLP) on the inline-edit variant
(data/detail_addition_probe_test_v6_inline.jsonl,
activations_detail_test_v6_inline/) -- same 16 double-held-out prompts as
scripts/score_rank1_rank2_full_testset.py, but redundant/missing detail is
rewritten inline (same style as the source prose, no appended bullet, no
"Constraints" word, identical ending across all three variants). See
scripts/build_detail_addition_dataset_v6_inline.py's docstring.

Reports both raw P(ambiguous) and binary entropy H(p), same as the prior
runs, for direct comparison.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_activation_bundle

ACTS_DIR = ROOT / "activations_detail_test_v6_inline"
RESULTS_DIR = ROOT / "results_v6_inline"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"


def H(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(ACTS_DIR)
    id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}

    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    rank1_layer = json.load(open(ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.json"))["layer"]
    rank2_layer = json.load(open(ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.json"))["layer"]
    print(f"[score_v6] rank1 (LR) layer={rank1_layer}, rank2 (MLP) layer={rank2_layer}")

    base_ids = sorted({m["id"].rsplit("__", 1)[0] for m in meta_rows}, key=lambda x: int(x.split("_")[-1]))

    table = []
    for base in base_ids:
        row = {"base_id": base, "ambiguity_types": ",".join(meta_rows[id_to_row[f"{base}__baseline"]]["ambiguity_types"])}
        for variant in ("baseline", "redundant", "missing"):
            i = id_to_row[f"{base}__{variant}"]
            x1 = acts[i, rank1_layer, :].reshape(1, -1)
            x2 = acts[i, rank2_layer, :].reshape(1, -1)
            p1 = float(rank1.predict_proba(x1)[0, list(rank1.classes_).index(1)])
            p2 = float(rank2.predict_proba(x2)[0, list(rank2.classes_).index(1)])
            row[f"rank1_p_amb_{variant}"] = round(p1, 4)
            row[f"rank2_p_amb_{variant}"] = round(p2, 4)
        table.append(row)

    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f"\n=== rank1 (LR) / rank2 (MLP), P(ambiguous), n={len(table)}, INLINE edits (no appended text) ===")
    print(f"  {'base_id':14s} {'type':22s} {'LR b/r/m':20s} {'MLP b/r/m':20s}")
    for row in table:
        print(f"  {row['base_id']:14s} {row['ambiguity_types']:22s} "
              f"{row['rank1_p_amb_baseline']:.2f}/{row['rank1_p_amb_redundant']:.2f}/{row['rank1_p_amb_missing']:.2f}          "
              f"{row['rank2_p_amb_baseline']:.2f}/{row['rank2_p_amb_redundant']:.2f}/{row['rank2_p_amb_missing']:.2f}")

    def summarize(key):
        d_red = [row[f"{key}_redundant"] - row[f"{key}_baseline"] for row in table]
        d_mis = [row[f"{key}_missing"] - row[f"{key}_baseline"] for row in table]
        return {
            "mean_delta_redundant": float(np.mean(d_red)), "std_delta_redundant": float(np.std(d_red)),
            "n_redundant_down": int(sum(1 for d in d_red if d < 0)),
            "mean_delta_missing": float(np.mean(d_mis)), "std_delta_missing": float(np.std(d_mis)),
            "n_missing_down": int(sum(1 for d in d_mis if d < 0)),
            "n_total": len(d_red),
        }

    summary = {"rank1_lr_p_ambiguous": summarize("rank1_p_amb"), "rank2_mlp_p_ambiguous": summarize("rank2_p_amb")}

    print(f"\n=== Summary (n={len(table)}, negative = probe says 'less ambiguous') ===")
    for name, s in summary.items():
        print(f"  {name:24s} redundant: mean={s['mean_delta_redundant']:+.4f}+/-{s['std_delta_redundant']:.4f} "
              f"({s['n_redundant_down']}/{s['n_total']} down)   "
              f"missing: mean={s['mean_delta_missing']:+.4f}+/-{s['std_delta_missing']:.4f} "
              f"({s['n_missing_down']}/{s['n_total']} down)")

    # ---- entropy view ----
    entropy_table = []
    for row in table:
        e = {"base_id": row["base_id"]}
        for key in ("rank1_p_amb", "rank2_p_amb"):
            for v in ("baseline", "redundant", "missing"):
                e[f"{key}_H_{v}"] = round(float(H(np.array(row[f"{key}_{v}"]))), 4)
        entropy_table.append(e)

    def summarize_H(key):
        d_red = [row[f"{key}_H_redundant"] - row[f"{key}_H_baseline"] for row in entropy_table]
        d_mis = [row[f"{key}_H_missing"] - row[f"{key}_H_baseline"] for row in entropy_table]
        return {
            "mean_deltaH_redundant": float(np.mean(d_red)), "n_redundant_more_confident": int(sum(1 for d in d_red if d < 0)),
            "mean_deltaH_missing": float(np.mean(d_mis)), "n_missing_more_confident": int(sum(1 for d in d_mis if d < 0)),
            "n_total": len(d_red),
        }

    entropy_summary = {"rank1_lr": summarize_H("rank1_p_amb"), "rank2_mlp": summarize_H("rank2_p_amb")}
    print(f"\n=== Entropy H(p) shift (bits; negative = probe MORE confident) ===")
    for name, s in entropy_summary.items():
        print(f"  {name:14s} redundant: mean={s['mean_deltaH_redundant']:+.4f} ({s['n_redundant_more_confident']}/{s['n_total']} more confident)   "
              f"missing: mean={s['mean_deltaH_missing']:+.4f} ({s['n_missing_more_confident']}/{s['n_total']} more confident)")

    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"raw_p": summary, "entropy": entropy_summary}, f, indent=2)

    print(f"\n[score_v6] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
