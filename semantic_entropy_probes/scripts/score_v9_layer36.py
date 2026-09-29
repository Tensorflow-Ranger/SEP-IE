"""v9 (self-elicited-ambiguity candidates, n=40 prompts x up to 4 candidates)
scored with the layer-36 probes (rank1b_lr / rank2b_mlp) instead of rank1
(layer 20)/rank2 (layer 22). Mirrors scripts/score_v9_self_ambiguity.py's
methodology exactly.
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

ACTS_DIR = ROOT / "activations_detail_test_v9_self_ambiguity"
RESULTS_DIR = ROOT / "results_v9_layer36"
RANK1B_PATH = ROOT / "models" / "rank1b_lr_layer36_pca100_ccb_heval.joblib"
RANK2B_PATH = ROOT / "models" / "rank2b_mlp_layer36_pca100_ccb_heval.joblib"
LAYER = 36


def H(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(ACTS_DIR)
    id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}

    rank1b = joblib.load(RANK1B_PATH)
    rank2b = joblib.load(RANK2B_PATH)
    print(f"[score_v9_layer36] rank1b (LR) / rank2b (MLP), both layer={LAYER}")

    def score(i):
        x = acts[i, LAYER, :].reshape(1, -1)
        p1 = float(rank1b.predict_proba(x)[0, list(rank1b.classes_).index(1)])
        p2 = float(rank2b.predict_proba(x)[0, list(rank2b.classes_).index(1)])
        return p1, p2

    base_ids = sorted({m["id"].rsplit("__", 1)[0] for m in meta_rows}, key=lambda x: int(x.split("_")[-1]))

    full_table, top_table = [], []
    for base in base_ids:
        bi = id_to_row[f"{base}__baseline"]
        p1_base, p2_base = score(bi)
        h1_base, h2_base = float(H(np.array(p1_base))), float(H(np.array(p2_base)))

        cand_rows = [m for m in meta_rows if m["id"].startswith(f"{base}__cand")]
        cand_results = []
        for m in cand_rows:
            ci = id_to_row[m["id"]]
            p1, p2 = score(ci)
            h1, h2 = float(H(np.array(p1))), float(H(np.array(p2)))
            d1, d2 = p1 - p1_base, p2 - p2_base
            mean_dP = (d1 + d2) / 2
            row = {
                "base_id": base, "candidate_idx": m["id"].rsplit("cand", 1)[-1],
                "rank1b_p_amb": round(p1, 4), "rank2b_p_amb": round(p2, 4),
                "rank1b_delta_p": round(d1, 4), "rank2b_delta_p": round(d2, 4),
                "mean_delta_p": round(mean_dP, 4),
                "rank1b_delta_H": round(h1 - h1_base, 4), "rank2b_delta_H": round(h2 - h2_base, 4),
                "both_probes_agree_less_ambiguous": bool(d1 < 0 and d2 < 0),
            }
            full_table.append(row)
            cand_results.append(row)

        if cand_results:
            best = min(cand_results, key=lambda r: r["mean_delta_p"])
            top_table.append({
                "base_id": base,
                "rank1b_p_amb_baseline": round(p1_base, 4), "rank2b_p_amb_baseline": round(p2_base, 4),
                "best_candidate_idx": best["candidate_idx"],
                "rank1b_p_amb_best": best["rank1b_p_amb"], "rank2b_p_amb_best": best["rank2b_p_amb"],
                "rank1b_delta_p": best["rank1b_delta_p"], "rank2b_delta_p": best["rank2b_delta_p"],
                "mean_delta_p": best["mean_delta_p"],
                "rank1b_delta_H": best["rank1b_delta_H"], "rank2b_delta_H": best["rank2b_delta_H"],
                "both_probes_agree": best["both_probes_agree_less_ambiguous"],
            })

    with (RESULTS_DIR / "all_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(full_table[0].keys()))
        w.writeheader()
        w.writerows(full_table)
    with (RESULTS_DIR / "top_candidate_per_prompt.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(top_table[0].keys()))
        w.writeheader()
        w.writerows(top_table)

    rank1_deltas = np.array([r["rank1b_delta_p"] for r in top_table])
    rank2_deltas = np.array([r["rank2b_delta_p"] for r in top_table])
    n_agree = sum(1 for r in top_table if r["both_probes_agree"])

    summary = {
        "n_prompts": len(top_table), "n_both_probes_agree_less_ambiguous": n_agree,
        "rank1b_lr": {
            "mean_delta_p_best_candidate": float(rank1_deltas.mean()), "n_down": int((rank1_deltas < 0).sum()),
            "ttest_vs_zero_p": float(stats.ttest_1samp(rank1_deltas, 0).pvalue),
            "wilcoxon_vs_zero_p": float(stats.wilcoxon(rank1_deltas).pvalue),
        },
        "rank2b_mlp": {
            "mean_delta_p_best_candidate": float(rank2_deltas.mean()), "n_down": int((rank2_deltas < 0).sum()),
            "ttest_vs_zero_p": float(stats.ttest_1samp(rank2_deltas, 0).pvalue),
            "wilcoxon_vs_zero_p": float(stats.wilcoxon(rank2_deltas).pvalue),
        },
    }

    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== v9 self-elicited (layer 36): best candidate per prompt (n={len(top_table)}) ===")
    for name in ("rank1b_lr", "rank2b_mlp"):
        s = summary[name]
        print(f"  {name}: mean_delta={s['mean_delta_p_best_candidate']:+.4f} ({s['n_down']}/{len(top_table)} down) "
              f"ttest_p={s['ttest_vs_zero_p']:.4f} wilcoxon_p={s['wilcoxon_vs_zero_p']:.4f}")
    print(f"  both probes agree 'less ambiguous' on best candidate: {n_agree}/{len(top_table)}")
    print(f"\n[score_v9_layer36] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
