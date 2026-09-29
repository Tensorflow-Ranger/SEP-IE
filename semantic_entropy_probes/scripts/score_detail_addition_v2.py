"""v2 scoring: same probes as scripts/score_detail_addition.py, but each
probe is only scored against the subset of data/detail_addition_probe_test_v2.jsonl
that is actually in ITS OWN test split (tag="multiclass" for
results_oatml_multiclass/probe.pkl; tag="binary" for both
models/rank1_lr*.joblib and rank2_mlp*.joblib, which share the same split).
See scripts/build_detail_addition_dataset_v2.py's docstring for why this
replaces v1 (v1's items were mostly in-training-set, not held out).
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

ACTS_DIR = ROOT / "activations_detail_test_v2"
RESULTS_DIR = ROOT / "results_detail_test_v2"
MULTICLASS_PROBE_PATH = ROOT / "results_oatml_multiclass" / "probe.pkl"
RANK1_PATH = ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.joblib"
RANK2_PATH = ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.joblib"


def base_and_variant(row_id: str) -> tuple[str, str]:
    base, variant = row_id.rsplit("__", 1)
    return base, variant


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    acts, _labels, _groups, _token_lens, meta_rows = load_activation_bundle(ACTS_DIR)
    tag_by_id = {m["id"]: m.get("dataset_source") for m in meta_rows}  # dataset_source unused here; placeholder
    # tag isn't in metadata.jsonl's written fields (extract_activations only writes a fixed schema) --
    # reload the source dataset file directly to recover it.
    src_rows = {}
    with (ROOT / "data" / "detail_addition_probe_test_v2.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            src_rows[r["id"]] = r
    id_to_row = {m["id"]: i for i, m in enumerate(meta_rows)}

    mc_ids = [m["id"] for m in meta_rows if src_rows[m["id"]]["tag"] == "multiclass"]
    bin_ids = [m["id"] for m in meta_rows if src_rows[m["id"]]["tag"] == "binary"]
    print(f"[score_v2] {len(mc_ids)} multiclass-tag rows, {len(bin_ids)} binary-tag rows")

    with MULTICLASS_PROBE_PATH.open("rb") as f:
        mc = pickle.load(f)
    mc_model, mc_meta = mc["model"], mc["metadata"]
    mc_layer = mc_meta["layer"]
    print(f"[score_v2] multiclass probe: layer={mc_layer} classes={mc_meta['classes']}")

    rank1 = joblib.load(RANK1_PATH)
    rank2 = joblib.load(RANK2_PATH)
    with (ROOT / "models" / "rank1_lr_layer20_pca100_ccb_heval.json").open() as f:
        rank1_layer = json.load(f)["layer"]
    with (ROOT / "models" / "rank2_mlp_layer22_pca100_ccb_heval.json").open() as f:
        rank2_layer = json.load(f)["layer"]
    print(f"[score_v2] rank1 (LR) layer={rank1_layer}, rank2 (MLP) layer={rank2_layer}")

    # ---- multiclass probe, scored only on its own held-out subset ----
    mc_idx = [id_to_row[i] for i in mc_ids]
    X_mc = acts[mc_idx, mc_layer, :]
    mc_probs = mc_model.predict_proba(X_mc)
    mc_ek = expected_value(mc_probs, mc_model.classes_)
    mc_pred = {}
    for row_i, gid in enumerate(mc_idx):
        rid = meta_rows[gid]["id"]
        mc_pred[rid] = dict(
            expected_k=float(mc_ek[row_i]),
            p_k1=float(mc_probs[row_i, list(mc_model.classes_).index(1)]),
            p_k2=float(mc_probs[row_i, list(mc_model.classes_).index(2)]),
            p_k3=float(mc_probs[row_i, list(mc_model.classes_).index(3)]),
        )

    # ---- binary probes, scored only on their own held-out subset ----
    bin_idx = [id_to_row[i] for i in bin_ids]
    X_r1 = acts[bin_idx, rank1_layer, :]
    X_r2 = acts[bin_idx, rank2_layer, :]
    r1_probs = rank1.predict_proba(X_r1)
    r2_probs = rank2.predict_proba(X_r2)
    r1_amb_col = list(rank1.classes_).index(1)
    r2_amb_col = list(rank2.classes_).index(1)
    bin_pred = {}
    for row_i, gid in enumerate(bin_idx):
        rid = meta_rows[gid]["id"]
        bin_pred[rid] = dict(
            rank1_p_ambiguous=float(r1_probs[row_i, r1_amb_col]),
            rank2_p_ambiguous=float(r2_probs[row_i, r2_amb_col]),
        )

    # ---- assemble comparison tables ----
    def build_table(base_ids, pred_dict, metric_keys):
        rows = []
        for b in base_ids:
            row = {"base_id": b, "ambiguity_types": ",".join(src_rows[f"{b}__baseline"]["ambiguity_types"])}
            for key in metric_keys:
                for variant in ("baseline", "redundant", "missing"):
                    row[f"{key}_{variant}"] = round(pred_dict[f"{b}__{variant}"][key], 4)
            rows.append(row)
        return rows

    mc_base_ids = sorted({base_and_variant(i)[0] for i in mc_ids}, key=lambda x: int(x.split("_")[-1]))
    bin_base_ids = sorted({base_and_variant(i)[0] for i in bin_ids}, key=lambda x: int(x.split("_")[-1]))

    mc_table = build_table(mc_base_ids, mc_pred, ["expected_k"])
    bin_table = build_table(bin_base_ids, bin_pred, ["rank1_p_ambiguous", "rank2_p_ambiguous"])

    with (RESULTS_DIR / "multiclass_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mc_table[0].keys()))
        w.writeheader()
        w.writerows(mc_table)
    with (RESULTS_DIR / "binary_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(bin_table[0].keys()))
        w.writeheader()
        w.writerows(bin_table)
    with (RESULTS_DIR / "raw_predictions.json").open("w", encoding="utf-8") as f:
        json.dump({"multiclass": mc_pred, "binary": bin_pred}, f, indent=2)

    print("\n=== Multiclass probe (test-split only), E[K] baseline/redundant/missing ===")
    for row in mc_table:
        print(f"  {row['base_id']:14s} {row['ambiguity_types']:22s} "
              f"{row['expected_k_baseline']:.2f} / {row['expected_k_redundant']:.2f} / {row['expected_k_missing']:.2f}")

    print("\n=== Binary probes (test-split only), P(ambiguous) baseline/redundant/missing ===")
    print(f"  {'base_id':14s} {'type':22s} {'LR b/r/m':18s} {'MLP b/r/m':18s}")
    for row in bin_table:
        print(f"  {row['base_id']:14s} {row['ambiguity_types']:22s} "
              f"{row['rank1_p_ambiguous_baseline']:.2f}/{row['rank1_p_ambiguous_redundant']:.2f}/{row['rank1_p_ambiguous_missing']:.2f}      "
              f"{row['rank2_p_ambiguous_baseline']:.2f}/{row['rank2_p_ambiguous_redundant']:.2f}/{row['rank2_p_ambiguous_missing']:.2f}")

    def summarize(table, key):
        d_red = [row[f"{key}_redundant"] - row[f"{key}_baseline"] for row in table]
        d_mis = [row[f"{key}_missing"] - row[f"{key}_baseline"] for row in table]
        return {
            "mean_delta_redundant": float(np.mean(d_red)),
            "n_redundant_down": int(sum(1 for d in d_red if d < 0)),
            "mean_delta_missing": float(np.mean(d_mis)),
            "n_missing_down": int(sum(1 for d in d_mis if d < 0)),
            "n_total": len(d_red),
        }

    summary = {
        "multiclass_expected_k": summarize(mc_table, "expected_k"),
        "rank1_lr_p_ambiguous": summarize(bin_table, "rank1_p_ambiguous"),
        "rank2_mlp_p_ambiguous": summarize(bin_table, "rank2_p_ambiguous"),
    }
    with (RESULTS_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (held-out test-split only; negative = probe says 'less ambiguous') ===")
    for name, s in summary.items():
        print(f"  {name:26s} redundant: mean={s['mean_delta_redundant']:+.4f} "
              f"({s['n_redundant_down']}/{s['n_total']} down)   "
              f"missing: mean={s['mean_delta_missing']:+.4f} ({s['n_missing_down']}/{s['n_total']} down)")

    print(f"\n[score_v2] wrote results -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()
