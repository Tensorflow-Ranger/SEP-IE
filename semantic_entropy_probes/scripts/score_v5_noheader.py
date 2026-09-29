"""Score rank1 (LR)/rank2 (MLP) on the no-"Constraints"-header variant
(data/detail_addition_probe_test_v5_noheader.jsonl,
activations_detail_test_v5_noheader/) -- same 16 double-held-out prompts
and same bullet content as scripts/score_rank1_rank2_full_testset.py, with
every literal "Constraints" occurrence removed. See
scripts/build_detail_addition_dataset_v5_noheader.py's docstring for why.
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

ACTS_DIR = ROOT / "activations_detail_test_v5_noheader"
RESULTS_DIR = ROOT / "results_v5_noheader"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(ACTS_DIR)
    id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}

    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    rank1_layer = json.load(open(ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.json"))["layer"]
    rank2_layer = json.load(open(ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.json"))["layer"]
    print(f"[score_v5] rank1 (LR) layer={rank1_layer}, rank2 (MLP) layer={rank2_layer}")

    base_ids = sorted({m["id"].rsplit("__", 1)[0] for m in meta_rows}, key=lambda x: int(x.split("_")[-1]))

    table = []
    for base in base_ids:
        row = {"base_id": base, "ambiguity_types": ",".join(meta_rows[id_to_row[f"{base}__baseline"]]["ambiguity_types"])}
        for variant in ("baseline", "redundant", "missing"):
            i = id_to_row[f"{base}__{variant}"]
            x1 = acts[i, rank1_layer, :].reshape(1, -1)
            x2 = acts[i, rank2_layer, :].reshape(1, -1)
            p1 = rank1.predict_proba(x1)[0, list(rank1.classes_).index(1)]
            p2 = rank2.predict_proba(x2)[0, list(rank2.classes_).index(1)]
            row[f"rank1_p_amb_{variant}"] = round(float(p1), 4)
            row[f"rank2_p_amb_{variant}"] = round(float(p2), 4)
        table.append(row)

    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f"\n=== rank1 (LR) / rank2 (MLP), P(ambiguous), n={len(table)}, NO 'Constraints' header anywhere ===")
    print(f"  {'base_id':14s} {'type':22s} {'LR b/r/m':20s} {'MLP b/r/m':20s}")
    for row in table:
        print(f"  {row['base_id']:14s} {row['ambiguity_types']:22s} "
              f"{row['rank1_p_amb_baseline']:.2f}/{row['rank1_p_amb_redundant']:.2f}/{row['rank1_p_amb_missing']:.2f}          "
              f"{row['rank2_p_amb_baseline']:.2f}/{row['rank2_p_amb_redundant']:.2f}/{row['rank2_p_amb_missing']:.2f}")

    def summarize(key):
        d_red = [row[f"{key}_redundant"] - row[f"{key}_baseline"] for row in table]
        d_mis = [row[f"{key}_missing"] - row[f"{key}_baseline"] for row in table]
        return {
            "mean_delta_redundant": float(np.mean(d_red)),
            "std_delta_redundant": float(np.std(d_red)),
            "n_redundant_down": int(sum(1 for d in d_red if d < 0)),
            "mean_delta_missing": float(np.mean(d_mis)),
            "std_delta_missing": float(np.std(d_mis)),
            "n_missing_down": int(sum(1 for d in d_mis if d < 0)),
            "n_total": len(d_red),
        }

    summary = {"rank1_lr_p_ambiguous": summarize("rank1_p_amb"), "rank2_mlp_p_ambiguous": summarize("rank2_p_amb")}
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary (n={len(table)}, negative = probe says 'less ambiguous') ===")
    for name, s in summary.items():
        print(f"  {name:24s} redundant: mean={s['mean_delta_redundant']:+.4f}+/-{s['std_delta_redundant']:.4f} "
              f"({s['n_redundant_down']}/{s['n_total']} down)   "
              f"missing: mean={s['mean_delta_missing']:+.4f}+/-{s['std_delta_missing']:.4f} "
              f"({s['n_missing_down']}/{s['n_total']} down)")

    print(f"\n[score_v5] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
