"""Same n=40 hand-authored inline-edit test as scripts/score_v7_combined.py,
but scored with the layer-36 probes (rank1b_lr / rank2b_mlp) instead of
rank1 (layer 20)/rank2 (layer 22) -- layer 36 had the largest missing-vs-
redundant entropy shift in scripts/layer_sweep_confidence_jump.py.
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

RESULTS_DIR = ROOT / "results_layer36_v6v7"
RANK1B_PATH = ROOT / "models" / "rank1b_lr_layer36_pca100_ccb_heval.joblib"
RANK2B_PATH = ROOT / "models" / "rank2b_mlp_layer36_pca100_ccb_heval.joblib"
SOURCES = [ROOT / "activations_detail_test_v6_inline", ROOT / "activations_detail_test_v7_expand"]
LAYER = 36


def H(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rank1b = joblib.load(RANK1B_PATH)
    rank2b = joblib.load(RANK2B_PATH)
    print(f"[score_layer36] rank1b (LR) / rank2b (MLP), both layer={LAYER}")

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
                x = acts[i, LAYER, :].reshape(1, -1)
                p1 = float(rank1b.predict_proba(x)[0, list(rank1b.classes_).index(1)])
                p2 = float(rank2b.predict_proba(x)[0, list(rank2b.classes_).index(1)])
                row[f"rank1b_p_amb_{variant}"] = round(p1, 4)
                row[f"rank2b_p_amb_{variant}"] = round(p2, 4)
            table.append(row)

    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    print(f"\n=== rank1b (LR) / rank2b (MLP), layer {LAYER}, P(ambiguous), n={len(table)} ===")
    for row in table:
        print(f"  {row['base_id']:14s} {row['ambiguity_types']:22s} "
              f"{row['rank1b_p_amb_baseline']:.2f}/{row['rank1b_p_amb_redundant']:.2f}/{row['rank1b_p_amb_missing']:.2f}          "
              f"{row['rank2b_p_amb_baseline']:.2f}/{row['rank2b_p_amb_redundant']:.2f}/{row['rank2b_p_amb_missing']:.2f}")

    def deltas(key):
        d_red = np.array([row[f"{key}_redundant"] - row[f"{key}_baseline"] for row in table])
        d_mis = np.array([row[f"{key}_missing"] - row[f"{key}_baseline"] for row in table])
        return d_red, d_mis

    summary = {}
    print(f"\n=== Summary (n={len(table)}, negative = probe says 'less ambiguous') ===")
    for name, key in [("rank1b_lr_p_ambiguous", "rank1b_p_amb"), ("rank2b_mlp_p_ambiguous", "rank2b_p_amb")]:
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

    entropy_summary = {}
    print(f"\n=== Entropy H(p) shift (bits; negative = probe MORE confident) ===")
    for name, key in [("rank1b_lr", "rank1b_p_amb"), ("rank2b_mlp", "rank2b_p_amb")]:
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

    print(f"\n[score_layer36] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
