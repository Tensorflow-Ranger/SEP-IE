"""v20c: autonomous end-to-end driver -- full-scale round-2 AND round-3
curriculum-combinatorial probe pipeline, run unattended.

Stages (each individually resumable -- safe to kill and rerun):
  1. prepare round-2 branches (5-fold, offline)          -> v20c_prepare_branches_full.py
  2. elicit fresh candidates per branch (Bedrock)         -> v20c_branch_elicit_r2.py
  3. oracle-answer them (Bedrock)                         -> v20c_branch_oracle_r2.py
  4. extract activations (GPU)                            -> v20c_extract_r2.py
  5. train + eval round-2 probe vs. naive round-1 transfer, group-disjoint split
  6. prepare round-3 branches from round-2's branches (5-fold within round-2 data, offline)
  7. elicit / 8. oracle-answer / 9. extract for round-3 (Bedrock + GPU)
  10. train + eval round-3 probe vs. naive round-1/round-2 transfer

Writes data/v20_branches_full/FINAL_RESULTS.json at the end and after each
major stage (partial results so progress is visible if killed early).
Logs every stage's stdout to data/v20_branches_full/driver.log.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "v20_branches_full"
LOG_PATH = OUT_DIR / "driver.log"
RESULTS_PATH = OUT_DIR / "FINAL_RESULTS.json"
PY_CEGDS = "/home/ubuntu/miniforge3/envs/cegds/bin/python"
PY_PLAIN = sys.executable

OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def run(cmd, use_gpu_env=False):
    py = PY_CEGDS if use_gpu_env else PY_PLAIN
    full_cmd = [py, "-m", cmd]
    log(f"RUN: {' '.join(full_cmd)}")
    t0 = time.time()
    with open(LOG_PATH, "a") as logf:
        proc = subprocess.run(full_cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    log(f"DONE ({time.time()-t0:.0f}s) exit={proc.returncode}: {cmd}")
    if proc.returncode != 0:
        raise RuntimeError(f"{cmd} failed with exit {proc.returncode}, see {LOG_PATH}")


def save_results(partial):
    existing = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text())
    existing.update(partial)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2))
    log(f"saved partial results: {list(partial.keys())}")


def main():
    log("=== v20c autonomous pipeline starting ===")

    # --- ROUND 2 ---
    log("--- stage 1: prepare round-2 branches (5-fold, offline) ---")
    run("scripts.v20c_prepare_branches_full", use_gpu_env=True)  # needs sklearn (cegds env)

    log("--- stage 2: elicit round-2 candidates (Bedrock) ---")
    run("scripts.v20c_branch_elicit_r2")

    log("--- stage 3: oracle-answer round-2 candidates (Bedrock) ---")
    run("scripts.v20c_branch_oracle_r2")

    log("--- stage 4: extract round-2 activations (GPU) ---")
    run("scripts.v20c_extract_r2", use_gpu_env=True)

    log("--- stage 5: train + eval round-2 probe (naive vs. unweighted) ---")
    run("scripts.v20c_train_eval_round2", use_gpu_env=True)
    r2_results = json.loads((OUT_DIR / "round2_results.json").read_text())
    save_results({"round2": r2_results})
    log(f"round2 results: {r2_results}")

    log("--- stage 5b: train + eval round-2 probe (v20d reweighted schemes) ---")
    run("scripts.v20d_train_eval_round2_reweighted", use_gpu_env=True)
    r2_reweighted_results = json.loads((OUT_DIR / "round2_results_reweighted.json").read_text())
    save_results({"round2_reweighted": r2_reweighted_results})
    log(f"round2 reweighted results: {r2_reweighted_results}")
    log("=== stopping after round 2 per user instruction -- round 3 not run ===")

    log("=== v20c autonomous pipeline COMPLETE ===")
    save_results({"status": "COMPLETE"})


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        log(f"PIPELINE FAILED: {e}")
        save_results({"status": f"FAILED: {e}"})
        raise
