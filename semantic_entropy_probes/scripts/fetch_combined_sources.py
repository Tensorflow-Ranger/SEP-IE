"""Download the raw source files for the combined binary ambiguity dataset.

Four sources (see build_combined_dataset.py for how they're merged):
  1. ClarifyCodeBench -- already built locally as data/ambiguity_dataset.jsonl
     (K in {0,1,2,3}); nothing to fetch here.
  2. Orchid (SII-YDD/Orchid on HuggingFace) -- configs Orchid-HEval,
     Orchid-BCB, and Orchid-BCB-Expand. Downloaded via huggingface_hub into
     the local HF cache; this script just resolves local paths and copies
     each data.jsonl file into data/raw/ for a stable, explicit reference.
     Orchid-BCB-Expand is a DIFFERENT, larger set of 976 BigCodeBench
     problems (task_ids 164-1139) from Orchid-BCB's 164 (task_ids 0-163) --
     zero task_id overlap, verified directly -- so it's treated as its own
     dataset_source/source_problem_id namespace, not merged into orchid_bcb.
     Its ambiguous-column names differ slightly: `Lexical-Expand_prompt`/
     `Vagueness-Expand_prompt` instead of `Lexical_prompt`/`Vagueness_prompt`.
  3. human-eval-comm (jie-jw-wu/human-eval-comm on GitHub) -- only the
     "incompleteness" modified problems (auto_modified_problems/
     heval_modified_incomplete.jsonl), plus the original HumanEval.jsonl
     (Benchmark/HumanEval.jsonl) needed to verify index alignment.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

HUMAN_EVAL_COMM_BASE = "https://raw.githubusercontent.com/jie-jw-wu/human-eval-comm/main"


def fetch_orchid() -> None:
    for cfg, fname in [
        ("Orchid-HEval", "Orchid-HEval/data.jsonl"),
        ("Orchid-BCB", "Orchid-BCB/data.jsonl"),
        ("Orchid-BCB-Expand", "Orchid-BCB-Expand/data.jsonl"),
    ]:
        cached = hf_hub_download(repo_id="SII-YDD/Orchid", filename=fname, repo_type="dataset")
        dest = RAW_DIR / f"{cfg.lower().replace('-', '_')}.jsonl"
        shutil.copy(cached, dest)
        print(f"[fetch] {cfg} -> {dest}")


def fetch_human_eval_comm() -> None:
    for rel, dest_name in [
        ("auto_modified_problems/heval_modified_incomplete.jsonl", "heval_modified_incomplete.jsonl"),
        ("Benchmark/HumanEval.jsonl", "HumanEval.jsonl"),
    ]:
        url = f"{HUMAN_EVAL_COMM_BASE}/{rel}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest = RAW_DIR / dest_name
        dest.write_bytes(resp.content)
        print(f"[fetch] {url} -> {dest}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetch_orchid()
    fetch_human_eval_comm()
    print("[fetch] done")


if __name__ == "__main__":
    main()
