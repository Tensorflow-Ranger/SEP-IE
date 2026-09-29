"""Restrict the combined dataset to just clarifycodebench + orchid_heval.

Per instruction: drop orchid_bcb, orchid_bcb_expand, and humanevalcomm, and
rerun the same experiment battery on the 2-source subset. Reuses the
already-extracted activations_combined/ bundle (same prompts, same model
pass, same row order) instead of re-running extraction -- filtering rows out
of an existing activation tensor is exact and free; there's nothing a
re-extraction would recompute differently for the rows that are kept.

Outputs:
  data/ambiguity_dataset_ccb_heval.jsonl  -- filtered dataset rows
  activations_ccb_heval/activations.npy   -- filtered activation tensor
  activations_ccb_heval/metadata.jsonl    -- filtered metadata, same order
  activations_ccb_heval/extract_config.json -- copied unchanged (same model/config)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEEP_SOURCES = {"clarifycodebench", "orchid_heval"}

SRC_DATA = ROOT / "data" / "ambiguity_dataset_combined.jsonl"
DST_DATA = ROOT / "data" / "ambiguity_dataset_ccb_heval.jsonl"
SRC_ACT_DIR = ROOT / "activations_combined"
DST_ACT_DIR = ROOT / "activations_ccb_heval"


def main() -> None:
    # --- filter the dataset jsonl ---
    rows = []
    with SRC_DATA.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    kept_rows = [r for r in rows if r["dataset_source"] in KEEP_SOURCES]
    with DST_DATA.open("w", encoding="utf-8") as f:
        for r in kept_rows:
            f.write(json.dumps(r) + "\n")
    print(f"[build_ccb_heval_subset] {len(kept_rows)}/{len(rows)} rows kept -> {DST_DATA}")

    # --- filter the activation bundle, preserving row order ---
    acts = np.load(SRC_ACT_DIR / "activations.npy")
    meta_rows = []
    with (SRC_ACT_DIR / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta_rows.append(json.loads(line))
    assert acts.shape[0] == len(meta_rows)

    keep_mask = np.array([m["dataset_source"] in KEEP_SOURCES for m in meta_rows])
    acts_kept = acts[keep_mask]
    meta_kept = [m for m, k in zip(meta_rows, keep_mask) if k]

    DST_ACT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DST_ACT_DIR / "activations.npy", acts_kept)
    with (DST_ACT_DIR / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for m in meta_kept:
            f.write(json.dumps(m) + "\n")
    shutil.copy(SRC_ACT_DIR / "extract_config.json", DST_ACT_DIR / "extract_config.json")

    by_source = {}
    for m in meta_kept:
        by_source[m["dataset_source"]] = by_source.get(m["dataset_source"], 0) + 1
    print(f"[build_ccb_heval_subset] {acts_kept.shape[0]}/{acts.shape[0]} activation rows kept "
          f"-> {DST_ACT_DIR}, by source: {by_source}")


if __name__ == "__main__":
    main()
