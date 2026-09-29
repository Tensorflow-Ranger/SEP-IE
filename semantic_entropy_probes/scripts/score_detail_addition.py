"""Score the hand-built redundant-vs-missing-detail probe test
(data/detail_addition_probe_test.jsonl, activations_detail_test/) with the
three trained probes already on disk:

  - results_oatml_multiclass/probe.pkl: raw-4096-dim, layer 17, 3-class
    (K in {1,2,3}) LogisticRegression-equivalent (src/torch_probe.py).
    Reports full class distribution and E[K] = sum_k P(k)*k.
  - models/rank1_lr_layer20_pca100_ccb_heval.joblib: StandardScaler->PCA(100)
    ->LogisticRegression pipeline, raw layer-20 input, binary (0=clear,
    1=ambiguous). Reports P(ambiguous).
  - models/rank2_mlp_layer22_pca100_ccb_heval.joblib: same pipeline shape,
    torch MLP classifier head, raw layer-22 input, binary. Reports
    P(ambiguous).

For each of the 10 base prompts, prints/saves the baseline / +redundant /
+missing values from all three probes side by side, so a real ambiguity
signal should show +missing moving toward "less ambiguous" while +redundant
stays close to baseline.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_activation_bundle
from src.oatml_sep import expected_value

ACTS_DIR = ROOT / "activations_detail_test"
RESULTS_DIR = ROOT / "results_detail_test"
MULTICLASS_PROBE_PATH = ROOT / "results_oatml_multiclass" / "probe.pkl"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(ACTS_DIR)
    ids = [m["id"] for m in meta_rows]
    id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}

    with MULTICLASS_PROBE_PATH.open("rb") as f:
        mc = pickle.load(f)
    mc_model, mc_meta = mc["model"], mc["metadata"]
    mc_layer = mc_meta["layer"]
    mc_classes = np.array(mc_meta["classes"])
    print(f"[score] multiclass probe: layer={mc_layer} classes={mc_classes.tolist()}")

    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    with (ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.json").open() as f:
        rank1_layer = json.load(f)["layer"]
    with (ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.json").open() as f:
        rank2_layer = json.load(f)["layer"]
    print(f"[score] rank1 (LR) layer={rank1_layer}, rank2 (MLP) layer={rank2_layer}")

    X_mc = acts[:, mc_layer, :]
    X_r1 = acts[:, rank1_layer, :]
    X_r2 = acts[:, rank2_layer, :]

    mc_probs = mc_model.predict_proba(X_mc)  # (N, 3), columns ordered per mc_model.classes_
    mc_ek = expected_value(mc_probs, mc_model.classes_)

    r1_probs = rank1.predict_proba(X_r1)
    r1_p_amb = r1_probs[:, list(rank1.classes_).index(1)]

    r2_probs = rank2.predict_proba(X_r2)
    r2_p_amb = r2_probs[:, list(rank2.classes_).index(1)]

    per_row = {}
    for i, m in enumerate(meta_rows):
        per_row[m["id"]] = dict(
            base_id=m["id"].rsplit("__", 1)[0],
            variant=m["id"].rsplit("__", 1)[1],
            source_problem_id=m["source_problem_id"],
            ambiguity_types=m["ambiguity_types"],
            mc_expected_k=float(mc_ek[i]),
            mc_p_k1=float(mc_probs[i, list(mc_model.classes_).index(1)]),
            mc_p_k2=float(mc_probs[i, list(mc_model.classes_).index(2)]),
            mc_p_k3=float(mc_probs[i, list(mc_model.classes_).index(3)]),
            rank1_p_ambiguous=float(r1_p_amb[i]),
            rank2_p_ambiguous=float(r2_p_amb[i]),
        )

    base_ids = sorted({r["base_id"] for r in per_row.values()}, key=lambda b: int(b.split("_")[-1]))

    fieldnames = [
        "base_id", "ambiguity_types",
        "mc_expected_k_baseline", "mc_expected_k_redundant", "mc_expected_k_missing",
        "rank1_p_amb_baseline", "rank1_p_amb_redundant", "rank1_p_amb_missing",
        "rank2_p_amb_baseline", "rank2_p_amb_redundant", "rank2_p_amb_missing",
    ]
    table_rows = []
    for b in base_ids:
        rb = per_row[f"{b}__baseline"]
        rr = per_row[f"{b}__redundant"]
        rm = per_row[f"{b}__missing"]
        table_rows.append({
            "base_id": b,
            "ambiguity_types": ",".join(rb["ambiguity_types"]),
            "mc_expected_k_baseline": round(rb["mc_expected_k"], 4),
            "mc_expected_k_redundant": round(rr["mc_expected_k"], 4),
            "mc_expected_k_missing": round(rm["mc_expected_k"], 4),
            "rank1_p_amb_baseline": round(rb["rank1_p_ambiguous"], 4),
            "rank1_p_amb_redundant": round(rr["rank1_p_ambiguous"], 4),
            "rank1_p_amb_missing": round(rm["rank1_p_ambiguous"], 4),
            "rank2_p_amb_baseline": round(rb["rank2_p_ambiguous"], 4),
            "rank2_p_amb_redundant": round(rr["rank2_p_ambiguous"], 4),
            "rank2_p_amb_missing": round(rm["rank2_p_ambiguous"], 4),
        })

    with (RESULTS_DIR / "per_row_predictions.json").open("w", encoding="utf-8") as f:
        json.dump(per_row, f, indent=2)
    with (RESULTS_DIR / "comparison_table.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    # ---- summary stats: does +missing move down, does +redundant stay put? ----
    def deltas(key):
        d_red = [per_row[f"{b}__redundant"][key] - per_row[f"{b}__baseline"][key] for b in base_ids]
        d_mis = [per_row[f"{b}__missing"][key] - per_row[f"{b}__baseline"][key] for b in base_ids]
        return d_red, d_mis

    print("\n=== Per-prompt comparison ===")
    header = f"{'base_id':14s} {'type':22s} {'mcE[K] b/r/m':22s} {'r1 P(amb) b/r/m':22s} {'r2 P(amb) b/r/m':22s}"
    print(header)
    print("-" * len(header))
    for row in table_rows:
        print(
            f"{row['base_id']:14s} {row['ambiguity_types']:22s} "
            f"{row['mc_expected_k_baseline']:.2f}/{row['mc_expected_k_redundant']:.2f}/{row['mc_expected_k_missing']:.2f}      "
            f"{row['rank1_p_amb_baseline']:.2f}/{row['rank1_p_amb_redundant']:.2f}/{row['rank1_p_amb_missing']:.2f}         "
            f"{row['rank2_p_amb_baseline']:.2f}/{row['rank2_p_amb_redundant']:.2f}/{row['rank2_p_amb_missing']:.2f}"
        )

    print("\n=== Summary: mean delta from baseline (negative = probe says 'less ambiguous') ===")
    summary = {}
    for name, key in [
        ("multiclass E[K]", "mc_expected_k"),
        ("rank1 (LR) P(ambiguous)", "rank1_p_ambiguous"),
        ("rank2 (MLP) P(ambiguous)", "rank2_p_ambiguous"),
    ]:
        d_red, d_mis = deltas(key)
        n_red_down = sum(1 for d in d_red if d < 0)
        n_mis_down = sum(1 for d in d_mis if d < 0)
        summary[name] = {
            "mean_delta_redundant": float(np.mean(d_red)),
            "mean_delta_missing": float(np.mean(d_mis)),
            "n_redundant_moved_down": n_red_down,
            "n_missing_moved_down": n_mis_down,
            "n_total": len(d_red),
        }
        print(f"{name:26s} redundant: mean={np.mean(d_red):+.4f} ({n_red_down}/{len(d_red)} down)   "
              f"missing: mean={np.mean(d_mis):+.4f} ({n_mis_down}/{len(d_mis)} down)")

    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[score] wrote {RESULTS_DIR}/comparison_table.csv, per_row_predictions.json, summary.json")


if __name__ == "__main__":
    main()
