"""Score rank1 (LR)/rank2 (MLP) on the FULL inline-edit test: v6's 16 base
prompts (activations_detail_test_v6_inline/) plus v7's 24 additional base
prompts (activations_detail_test_v7_expand/) = 40 total, the entire pool of
K=1 clarifycodebench rows genuinely held out from rank1/rank2's saved
pipelines. Same inline-edit methodology throughout (see
scripts/build_detail_addition_dataset_v6_inline.py's docstring).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_activation_bundle

RESULTS_DIR = ROOT / "results_v7_combined"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"
SOURCES = [ROOT / "activations_detail_test_v6_inline", ROOT / "activations_detail_test_v7_expand"]


def H(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    rank1_layer = json.load(open(ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.json"))["layer"]
    rank2_layer = json.load(open(ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.json"))["layer"]
    print(f"[score_v7] rank1 (LR) layer={rank1_layer}, rank2 (MLP) layer={rank2_layer}")

    table = []
    for src_dir in SOURCES:
        acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(src_dir)
        id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}
        base_ids = sorted({m["id"].rsplit("__", 1)[0] for m in meta_rows}, key=lambda x: int(x.split("_")[-1]))
        for base in base_ids:
            row = {"base_id": base, "source": src_dir.name,
                   "ambiguity_types": ",".join(meta_rows[id_to_row[f"{base}__baseline"]]["ambiguity_types"])}
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

    print(f"\n=== rank1 (LR) / rank2 (MLP), P(ambiguous), n={len(table)}, INLINE edits (full pool) ===")
    print(f"  {'base_id':14s} {'type':22s} {'LR b/r/m':20s} {'MLP b/r/m':20s}")
    for row in table:
        print(f"  {row['base_id']:14s} {row['ambiguity_types']:22s} "
              f"{row['rank1_p_amb_baseline']:.2f}/{row['rank1_p_amb_redundant']:.2f}/{row['rank1_p_amb_missing']:.2f}          "
              f"{row['rank2_p_amb_baseline']:.2f}/{row['rank2_p_amb_redundant']:.2f}/{row['rank2_p_amb_missing']:.2f}")

    def deltas(key):
        d_red = np.array([row[f"{key}_redundant"] - row[f"{key}_baseline"] for row in table])
        d_mis = np.array([row[f"{key}_missing"] - row[f"{key}_baseline"] for row in table])
        return d_red, d_mis

    summary = {}
    print(f"\n=== Summary (n={len(table)}, negative = probe says 'less ambiguous') ===")
    for name, key in [("rank1_lr_p_ambiguous", "rank1_p_amb"), ("rank2_mlp_p_ambiguous", "rank2_p_amb")]:
        d_red, d_mis = deltas(key)
        diff = d_mis - d_red
        t = stats.ttest_rel(d_mis, d_red)
        w = stats.wilcoxon(d_mis, d_red)
        summary[name] = {
            "mean_delta_redundant": float(d_red.mean()), "std_delta_redundant": float(d_red.std()),
            "n_redundant_down": int((d_red < 0).sum()),
            "mean_delta_missing": float(d_mis.mean()), "std_delta_missing": float(d_mis.std()),
            "n_missing_down": int((d_mis < 0).sum()),
            "n_total": len(d_red),
            "mean_missing_minus_redundant": float(diff.mean()),
            "n_missing_lower_than_redundant": int((diff < 0).sum()),
            "paired_ttest_p": float(t.pvalue),
            "wilcoxon_p": float(w.pvalue),
        }
        s = summary[name]
        print(f"  {name:24s} redundant: mean={s['mean_delta_redundant']:+.4f}+/-{s['std_delta_redundant']:.4f} "
              f"({s['n_redundant_down']}/{s['n_total']} down)   "
              f"missing: mean={s['mean_delta_missing']:+.4f}+/-{s['std_delta_missing']:.4f} "
              f"({s['n_missing_down']}/{s['n_total']} down)")
        print(f"    paired (missing - redundant): mean={s['mean_missing_minus_redundant']:+.4f}, "
              f"{s['n_missing_lower_than_redundant']}/{s['n_total']} prompts where missing pulled lower, "
              f"paired-t p={s['paired_ttest_p']:.4f}, wilcoxon p={s['wilcoxon_p']:.4f}")

    # ---- entropy view ----
    entropy_summary = {}
    print(f"\n=== Entropy H(p) shift (bits; negative = probe MORE confident) ===")
    for name, key in [("rank1_lr", "rank1_p_amb"), ("rank2_mlp", "rank2_p_amb")]:
        Hb = H(np.array([row[f"{key}_baseline"] for row in table]))
        Hr = H(np.array([row[f"{key}_redundant"] for row in table]))
        Hm = H(np.array([row[f"{key}_missing"] for row in table]))
        d_red, d_mis = Hr - Hb, Hm - Hb
        entropy_summary[name] = {
            "mean_deltaH_redundant": float(d_red.mean()), "n_redundant_more_confident": int((d_red < 0).sum()),
            "mean_deltaH_missing": float(d_mis.mean()), "n_missing_more_confident": int((d_mis < 0).sum()),
            "n_total": len(d_red),
        }
        s = entropy_summary[name]
        print(f"  {name:14s} redundant: mean={s['mean_deltaH_redundant']:+.4f} ({s['n_redundant_more_confident']}/{s['n_total']} more confident)   "
              f"missing: mean={s['mean_deltaH_missing']:+.4f} ({s['n_missing_more_confident']}/{s['n_total']} more confident)")

    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"raw_p": summary, "entropy": entropy_summary}, f, indent=2)

    print(f"\n[score_v7] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
