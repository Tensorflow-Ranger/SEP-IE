"""Load extracted activations + metadata into arrays for probing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_activation_bundle(activations_dir: Path):
    """Return (acts, labels, groups, token_lens, meta_rows).

    acts: float32 array (N, n_states, hidden_size)
    labels: int array (N,)
    groups: str array (N,) -- source_problem_id, the split-grouping key
    token_lens: int array (N,) -- prompt token count (length-confound control)
    """
    activations_dir = Path(activations_dir)
    acts = np.load(activations_dir / "activations.npy").astype(np.float32)

    meta_rows = []
    with (activations_dir / "metadata.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta_rows.append(json.loads(line))

    # metadata.jsonl is written in the same order as activations.npy rows.
    labels = np.array([m["label"] for m in meta_rows], dtype=np.int64)
    groups = np.array([m["source_problem_id"] for m in meta_rows])
    token_lens = np.array([m["prompt_token_len"] for m in meta_rows], dtype=np.float64)

    assert acts.shape[0] == len(meta_rows), (
        f"activation rows ({acts.shape[0]}) != metadata rows ({len(meta_rows)})"
    )
    return acts, labels, groups, token_lens, meta_rows


def load_combined_activation_bundle(activations_dir: Path):
    """Like load_activation_bundle, plus dataset_source and ambiguity_type arrays.

    For the combined multi-source dataset (build_combined_dataset.py /
    src/combined_splits.py) -- `label` here is binary (0=clear, 1=ambiguous),
    and `dataset_source` identifies which of the three source datasets each
    row came from (see src/combined_splits.py.DATASET_SOURCES).
    """
    acts, labels, groups, token_lens, meta_rows = load_activation_bundle(activations_dir)
    dataset_source = np.array([m["dataset_source"] for m in meta_rows])
    ambiguity_type = np.array([m["ambiguity_type"] for m in meta_rows])
    return acts, labels, groups, token_lens, dataset_source, ambiguity_type, meta_rows
