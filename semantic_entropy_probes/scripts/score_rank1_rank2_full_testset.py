"""Score rank1 (LR, layer 20) and rank2 (MLP, layer 22) on EVERY K=1
clarifycodebench prompt whose source problem is genuinely unseen by their
saved pipelines -- i.e. in combined_splits' test fold, not train+val (the
saved rank1/rank2 pipelines are refit on train+val, see
scripts/save_best_models.py, so val counts as "seen" too).

This is the maximal such set: all 16 distinct source problems that have a
K=1 row in that test fold (verified in-conversation via
src.combined_splits.make_combined_train_val_test_split, seed=0). Pulled
from four activation caches built across this investigation:
  - activations_detail_test_v2/   (tag=binary rows: amb_task_4,26,49,87,
    118,198,212,279,321,417)
  - activations_detail_test_v2/   (tag=multiclass row amb_task_142, which
    also happens to be in rank1/2's test fold)
  - activations_detail_test_v3_extra/  (amb_task_282, amb_task_360)
  - activations_detail_test_v4_extra/  (amb_task_28, amb_task_125, amb_task_346)
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

RESULTS_DIR = ROOT / "results_rank1_rank2_full_testset"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"

SOURCES = [
    ROOT / "activations_detail_test_v2",
    ROOT / "activations_detail_test_v3_extra",
    ROOT / "activations_detail_test_v4_extra",
]

BASE_IDS = [
    "amb_task_4", "amb_task_26", "amb_task_49", "amb_task_87", "amb_task_118",
    "amb_task_198", "amb_task_212", "amb_task_279", "amb_task_321", "amb_task_417",
    "amb_task_142", "amb_task_282", "amb_task_360",
    "amb_task_28", "amb_task_125", "amb_task_346",
]


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    rank1_layer = json.load(open(ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.json"))["layer"]
    rank2_layer = json.load(open(ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.json"))["layer"]
    print(f"[score_full] rank1 (LR) layer={rank1_layer}, rank2 (MLP) layer={rank2_layer}")

    # pull activations + metadata for the 30 needed rows (16 base_ids x 3 variants)
    # from whichever source directory actually contains each id.
    pred = {}
    ambiguity_types = {}
    for src_dir in SOURCES:
        acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(src_dir)
        id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}
        for base in BASE_IDS:
            for variant in ("baseline", "redundant", "missing"):
                rid = f"{base}__{variant}"
                if rid not in id_to_row or rid in pred:
                    continue
                i = id_to_row[rid]
                x1 = acts[i, rank1_layer, :].reshape(1, -1)
                x2 = acts[i, rank2_layer, :].reshape(1, -1)
                p1 = rank1.predict_proba(x1)[0, list(rank1.classes_).index(1)]
                p2 = rank2.predict_proba(x2)[0, list(rank2.classes_).index(1)]
                pred[rid] = dict(rank1_p_ambiguous=float(p1), rank2_p_ambiguous=float(p2))
                ambiguity_types[base] = meta_rows[i]["ambiguity_types"]

    missing_rows = [f"{b}__{v}" for b in BASE_IDS for v in ("baseline", "redundant", "missing") if f"{b}__{v}" not in pred]
    if missing_rows:
        print(f"[score_full] WARNING: missing rows not found in any activation cache: {missing_rows}")

    table = []
    for b in BASE_IDS:
        if any(f"{b}__{v}" not in pred for v in ("baseline", "redundant", "missing")):
            continue
        row = {"base_id": b, "ambiguity_types": ",".join(ambiguity_types[b])}
        for v in ("baseline", "redundant", "missing"):
            row[f"rank1_p_amb_{v}"] = round(pred[f"{b}__{v}"]["rank1_p_ambiguous"], 4)
            row[f"rank2_p_amb_{v}"] = round(pred[f"{b}__{v}"]["rank2_p_ambiguous"], 4)
        table.append(row)

    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f"\n=== rank1 (LR) / rank2 (MLP), P(ambiguous), n={len(table)} double-held-out prompts ===")
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

    print(f"\n[score_full] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
