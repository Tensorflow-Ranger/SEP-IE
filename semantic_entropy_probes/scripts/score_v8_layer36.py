"""v8 (Qwen-generated inline edits, n=40) scored with the layer-36 probes
(rank1b_lr / rank2b_mlp) instead of rank1 (layer 20)/rank2 (layer 22).
Mirrors scripts/score_v8_qwen_generated.py's methodology exactly.
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

ACTS_DIR = ROOT / "activations_detail_test_v8_qwen_generated"
RESULTS_DIR = ROOT / "results_v8_layer36"
RANK1B_PATH = ROOT / "models" / "rank1b_lr_layer36_pca100_ccb_heval.joblib"
RANK2B_PATH = ROOT / "models" / "rank2b_mlp_layer36_pca100_ccb_heval.joblib"
DATA_PATH = ROOT / "data" / "detail_addition_probe_test_v8_qwen_generated.jsonl"
LAYER = 36


def H(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(ACTS_DIR)
    id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}

    no_change = {}
    for line in DATA_PATH.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "qwen_made_no_change" in r:
            no_change[r["id"]] = r["qwen_made_no_change"]

    rank1b = joblib.load(RANK1B_PATH)
    rank2b = joblib.load(RANK2B_PATH)
    print(f"[score_v8_layer36] rank1b (LR) / rank2b (MLP), both layer={LAYER}")

    base_ids = sorted({m["id"].rsplit("__", 1)[0] for m in meta_rows}, key=lambda x: int(x.split("_")[-1]))

    table = []
    for base in base_ids:
        row = {"base_id": base, "ambiguity_types": ",".join(meta_rows[id_to_row[f"{base}__baseline"]]["ambiguity_types"])}
        for variant in ("baseline", "redundant", "missing"):
            i = id_to_row[f"{base}__{variant}"]
            x = acts[i, LAYER, :].reshape(1, -1)
            p1 = float(rank1b.predict_proba(x)[0, list(rank1b.classes_).index(1)])
            p2 = float(rank2b.predict_proba(x)[0, list(rank2b.classes_).index(1)])
            row[f"rank1b_p_amb_{variant}"] = round(p1, 4)
            row[f"rank2b_p_amb_{variant}"] = round(p2, 4)
            if variant != "baseline":
                row[f"{variant}_qwen_no_change"] = no_change.get(f"{base}__{variant}", None)
        table.append(row)

    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)

    n_red_nochange = sum(1 for row in table if row.get("redundant_qwen_no_change"))
    n_mis_nochange = sum(1 for row in table if row.get("missing_qwen_no_change"))
    print(f"[score_v8_layer36] Qwen made NO change in: redundant {n_red_nochange}/{len(table)}, missing {n_mis_nochange}/{len(table)}")

    def deltas(key, subset=None):
        rows = table if subset is None else [r for r in table if r["base_id"] in subset]
        d_red = np.array([row[f"{key}_redundant"] - row[f"{key}_baseline"] for row in rows])
        d_mis = np.array([row[f"{key}_missing"] - row[f"{key}_baseline"] for row in rows])
        return d_red, d_mis

    def run_summary(name, key, subset=None, label="all"):
        d_red, d_mis = deltas(key, subset)
        diff = d_mis - d_red
        t = stats.ttest_rel(d_mis, d_red)
        w = stats.wilcoxon(d_mis, d_red)
        s = {
            "label": label,
            "mean_delta_redundant": float(d_red.mean()), "mean_delta_missing": float(d_mis.mean()),
            "n_total": len(d_red),
            "mean_missing_minus_redundant": float(diff.mean()),
            "n_missing_lower_than_redundant": int((diff < 0).sum()),
            "paired_ttest_p": float(t.pvalue), "wilcoxon_p": float(w.pvalue),
        }
        print(f"  [{label}] {name}: redundant mean={s['mean_delta_redundant']:+.4f} missing mean={s['mean_delta_missing']:+.4f} "
              f"diff mean={s['mean_missing_minus_redundant']:+.4f} t_p={s['paired_ttest_p']:.4f} w_p={s['wilcoxon_p']:.4f}")
        return s

    print(f"\n=== Summary (n={len(table)}, negative = probe says 'less ambiguous'), layer={LAYER} ===")
    summary = {}
    for name, key in [("rank1b_lr_p_ambiguous", "rank1b_p_amb"), ("rank2b_mlp_p_ambiguous", "rank2b_p_amb")]:
        summary[name] = run_summary(name, key, subset=None, label="all_n40")

    clean_ids = {row["base_id"] for row in table
                 if not row.get("redundant_qwen_no_change") and not row.get("missing_qwen_no_change")}
    print(f"\n=== Clean subset (drop no-op generations, n={len(clean_ids)}) ===")
    summary_clean = {}
    for name, key in [("rank1b_lr_p_ambiguous", "rank1b_p_amb"), ("rank2b_mlp_p_ambiguous", "rank2b_p_amb")]:
        summary_clean[name] = run_summary(name, key, subset=clean_ids, label=f"clean_n{len(clean_ids)}")

    entropy_summary = {}
    print(f"\n=== Entropy H(p) shift (bits), layer={LAYER} ===")
    for name, key in [("rank1b_lr", "rank1b_p_amb"), ("rank2b_mlp", "rank2b_p_amb")]:
        Hb = H(np.array([row[f"{key}_baseline"] for row in table]))
        Hr = H(np.array([row[f"{key}_redundant"] for row in table]))
        Hm = H(np.array([row[f"{key}_missing"] for row in table]))
        d_red, d_mis = Hr - Hb, Hm - Hb
        entropy_summary[name] = {"mean_deltaH_redundant": float(d_red.mean()), "mean_deltaH_missing": float(d_mis.mean())}
        print(f"  {name}: redundant dH={d_red.mean():+.4f}  missing dH={d_mis.mean():+.4f}")

    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "raw_p": summary, "raw_p_clean_subset": summary_clean, "entropy": entropy_summary,
            "n_redundant_no_change": n_red_nochange, "n_missing_no_change": n_mis_nochange,
        }, f, indent=2)

    print(f"\n[score_v8_layer36] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
