"""Score rank1 (LR)/rank2 (MLP) on the v9 self-elicited-ambiguity test:
for each of the 40 held-out base prompts, Qwen3-8B itself listed up to 4
candidate ambiguities and, for each, an inline rewrite adding that candidate
clarification was generated (see gen_v9_self_ambiguity.py). This script:
  1. Scores baseline + every candidate rewrite with rank1 (LR) and rank2 (MLP).
  2. For each base prompt, picks the "top" candidate = the one with the
     largest mean P(ambiguous) drop across both probes (i.e. the candidate
     Qwen self-identified that moved the probes most toward "not ambiguous").
  3. Reports entropy shift too.
  4. Runs paired stats (best-candidate delta vs baseline) across all 40
     prompts, and compares against the known hand-authored/Qwen-rewritten
     "missing detail" and "redundant" deltas from v6/v7 (if available) for
     context.
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
RESULTS_DIR = ROOT / "results_v9_self_ambiguity"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"


def H(p):
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
    print(f"[score_v9] rank1 (LR) layer={rank1_layer}, rank2 (MLP) layer={rank2_layer}")

    def score(i):
        x1 = acts[i, rank1_layer, :].reshape(1, -1)
        x2 = acts[i, rank2_layer, :].reshape(1, -1)
        p1 = float(rank1.predict_proba(x1)[0, list(rank1.classes_).index(1)])
        p2 = float(rank2.predict_proba(x2)[0, list(rank2.classes_).index(1)])
        return p1, p2

    base_ids = sorted({m["id"].rsplit("__", 1)[0] for m in meta_rows}, key=lambda x: int(x.split("_")[-1]))

    full_table = []
    top_table = []
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
                "base_id": base,
                "candidate_idx": m["id"].rsplit("cand", 1)[-1],
                "rank1_p_amb": round(p1, 4), "rank2_p_amb": round(p2, 4),
                "rank1_delta_p": round(d1, 4), "rank2_delta_p": round(d2, 4),
                "mean_delta_p": round(mean_dP, 4),
                "rank1_delta_H": round(h1 - h1_base, 4), "rank2_delta_H": round(h2 - h2_base, 4),
                "both_probes_agree_less_ambiguous": bool(d1 < 0 and d2 < 0),
            }
            full_table.append(row)
            cand_results.append(row)

        if cand_results:
            best = min(cand_results, key=lambda r: r["mean_delta_p"])
            top_table.append({
                "base_id": base,
                "rank1_p_amb_baseline": round(p1_base, 4), "rank2_p_amb_baseline": round(p2_base, 4),
                "best_candidate_idx": best["candidate_idx"],
                "rank1_p_amb_best": best["rank1_p_amb"], "rank2_p_amb_best": best["rank2_p_amb"],
                "rank1_delta_p": best["rank1_delta_p"], "rank2_delta_p": best["rank2_delta_p"],
                "mean_delta_p": best["mean_delta_p"],
                "rank1_delta_H": best["rank1_delta_H"], "rank2_delta_H": best["rank2_delta_H"],
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

    print(f"\n=== v9 self-elicited: top candidate per prompt (n={len(top_table)}) ===")
    print(f"  {'base_id':14s} {'best_cand':10s} {'LR base->best':16s} {'MLP base->best':16s} {'agree':6s}")
    for row in top_table:
        print(f"  {row['base_id']:14s} {row['best_candidate_idx']:10s} "
              f"{row['rank1_p_amb_baseline']:.2f}->{row['rank1_p_amb_best']:.2f}       "
              f"{row['rank2_p_amb_baseline']:.2f}->{row['rank2_p_amb_best']:.2f}       "
              f"{str(row['both_probes_agree']):6s}")

    rank1_deltas = np.array([r["rank1_delta_p"] for r in top_table])
    rank2_deltas = np.array([r["rank2_delta_p"] for r in top_table])
    mean_deltas = np.array([r["mean_delta_p"] for r in top_table])
    n_agree = sum(1 for r in top_table if r["both_probes_agree"])

    summary = {
        "n_prompts": len(top_table),
        "n_both_probes_agree_less_ambiguous": n_agree,
        "rank1_lr": {
            "mean_delta_p_best_candidate": float(rank1_deltas.mean()),
            "n_down": int((rank1_deltas < 0).sum()),
            "ttest_vs_zero_p": float(stats.ttest_1samp(rank1_deltas, 0).pvalue),
            "wilcoxon_vs_zero_p": float(stats.wilcoxon(rank1_deltas).pvalue),
        },
        "rank2_mlp": {
            "mean_delta_p_best_candidate": float(rank2_deltas.mean()),
            "n_down": int((rank2_deltas < 0).sum()),
            "ttest_vs_zero_p": float(stats.ttest_1samp(rank2_deltas, 0).pvalue),
            "wilcoxon_vs_zero_p": float(stats.wilcoxon(rank2_deltas).pvalue),
        },
    }

    # ---- compare best-candidate delta vs the known v6/v7 hand-authored "missing"
    # and "redundant" deltas, when that results file is available ----
    v7_summary_path = ROOT / "results_v7_combined" / "summary.json"
    if v7_summary_path.exists():
        v7 = json.load(open(v7_summary_path))
        summary["comparison_to_v7_hand_authored"] = {
            "rank1_lr_v7_mean_delta_missing": v7["raw_p"]["rank1_lr_p_ambiguous"]["mean_delta_missing"],
            "rank1_lr_v7_mean_delta_redundant": v7["raw_p"]["rank1_lr_p_ambiguous"]["mean_delta_redundant"],
            "rank2_mlp_v7_mean_delta_missing": v7["raw_p"]["rank2_mlp_p_ambiguous"]["mean_delta_missing"],
            "rank2_mlp_v7_mean_delta_redundant": v7["raw_p"]["rank2_mlp_p_ambiguous"]["mean_delta_redundant"],
        }

    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Summary: does the BEST self-elicited candidate per prompt reduce P(ambiguous)? (n={len(top_table)}) ===")
    for name in ("rank1_lr", "rank2_mlp"):
        s = summary[name]
        print(f"  {name:10s} mean_delta={s['mean_delta_p_best_candidate']:+.4f} "
              f"({s['n_down']}/{len(top_table)} down) "
              f"ttest_vs_0_p={s['ttest_vs_zero_p']:.4f} wilcoxon_vs_0_p={s['wilcoxon_vs_zero_p']:.4f}")
    print(f"  both probes agree 'less ambiguous' on best candidate: {n_agree}/{len(top_table)}")
    if "comparison_to_v7_hand_authored" in summary:
        c = summary["comparison_to_v7_hand_authored"]
        print(f"\n  for context, v7 hand-authored/inline (n=40): "
              f"LR missing={c['rank1_lr_v7_mean_delta_missing']:+.4f} redundant={c['rank1_lr_v7_mean_delta_redundant']:+.4f}  "
              f"MLP missing={c['rank2_mlp_v7_mean_delta_missing']:+.4f} redundant={c['rank2_mlp_v7_mean_delta_redundant']:+.4f}")

    print(f"\n[score_v9] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
