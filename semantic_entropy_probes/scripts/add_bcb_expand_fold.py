"""Fill in the orchid_bcb_expand column of the LODO matrices, without
recomputing the clarifycodebench/orchid_heval/orchid_bcb columns already on
disk in results_combined/confound_report.json and results_combined/supcon_report.json.

Why this is safe to compute in isolation: orchid_bcb_expand's held-out fold
trains on {clarifycodebench, orchid_heval, orchid_bcb, humanevalcomm} -- the
exact same 1,608-row pool used before orchid_bcb_expand existed (it has zero
source_problem_id overlap with any other source, verified in
build_combined_dataset.py, so it contributes 0 rows to its own held-out
fold's training set). The other three folds' train pools now additionally
contain orchid_bcb_expand's 2,745 rows, but this script does not touch those
folds -- it only computes the new held_out="orchid_bcb_expand" fold and
merges it into the existing on-disk reports, per instruction to avoid
recomputing the numbers that don't change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.confound_tests import leave_one_dataset_out, _load_prompts_by_id
from scripts.train_supcon_probe import evaluate_lodo as supcon_evaluate_lodo
from src.data_utils import load_combined_activation_bundle

RESULTS_DIR = ROOT / "results_combined"
DATA_PATH = ROOT / "data" / "ambiguity_dataset_combined.jsonl"
HELD_OUT = ["orchid_bcb_expand"]
SEED = 0


def main() -> None:
    acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows = \
        load_combined_activation_bundle(ROOT / "activations_combined")
    prompts_by_id = _load_prompts_by_id(DATA_PATH)
    prompts = np.array([prompts_by_id[m["id"]] for m in meta_rows])

    with (RESULTS_DIR / "run_summary.json").open("r", encoding="utf-8") as f:
        run_summary = json.load(f)

    # --- confound_report.json: LR + MLP baseline, new held_out fold only ---
    with (RESULTS_DIR / "confound_report.json").open("r", encoding="utf-8") as f:
        confound_report = json.load(f)

    for tag, backend, best_layer in [
        ("lr", "torch", run_summary["best_layer_lr"]),
        ("mlp", "torch_mlp", run_summary["best_layer_mlp"]),
    ]:
        print(f"[add_bcb_expand_fold] === {tag} (layer {best_layer}) held_out=orchid_bcb_expand ===")
        X_layer = acts[:, best_layer, :]
        new_fold = leave_one_dataset_out(
            X_layer, labels, dataset_source, groups, prompts,
            backend=backend, seed=SEED, held_out_sources=HELD_OUT,
        )
        confound_report[tag]["lodo"].update(new_fold)

    with (RESULTS_DIR / "confound_report.json").open("w", encoding="utf-8") as f:
        json.dump(confound_report, f, indent=2)
    print(f"[add_bcb_expand_fold] updated {RESULTS_DIR / 'confound_report.json'}")

    # --- supcon_report.json: 12-config grid, new held_out fold only ---
    with (RESULTS_DIR / "supcon_report.json").open("r", encoding="utf-8") as f:
        supcon_report = json.load(f)

    layer = supcon_report["layer"]
    print(f"[add_bcb_expand_fold] === supcon grid (layer {layer}) held_out=orchid_bcb_expand ===")
    new_lodo = supcon_evaluate_lodo(acts, labels, groups, dataset_source, layer, SEED, held_out_sources=HELD_OUT)

    for key, new_val in new_lodo.items():
        existing = supcon_report["lodo"][key]
        new_fold_auroc = new_val["fold_aurocs"][0] if new_val["fold_aurocs"] else float("nan")
        existing["fold_aurocs"] = existing["fold_aurocs"] + [new_fold_auroc]
        existing["mean_auroc"] = float(np.nanmean(existing["fold_aurocs"]))

    best_lodo = max(supcon_report["lodo"].items(),
                     key=lambda kv: kv[1]["mean_auroc"] if not np.isnan(kv[1]["mean_auroc"]) else -1)
    supcon_report["best_lodo_config"] = best_lodo[0]
    supcon_report["lodo_fold_order"] = ["clarifycodebench", "orchid_heval", "orchid_bcb", "orchid_bcb_expand"]

    with (RESULTS_DIR / "supcon_report.json").open("w", encoding="utf-8") as f:
        json.dump(supcon_report, f, indent=2)
    print(f"[add_bcb_expand_fold] updated {RESULTS_DIR / 'supcon_report.json'}")
    print(f"[add_bcb_expand_fold] best LODO config: {best_lodo[0]} (mean_auroc={best_lodo[1]['mean_auroc']:.4f})")


if __name__ == "__main__":
    main()
