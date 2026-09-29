"""Build the held-out set for the "easy problems" v11 loop: the 32 Orchid-HEval
problems that landed in the TEST split of the exact split used to save the
deployed probes (rank1_lr_layer20_pca100_ccb_heval / rank2_mlp_layer22_pca100_ccb_heval
-- src/combined_splits.py, seed=0, grouped by source_problem_id). Using the test
split (not train/val) matters: those are the only orchid_heval problems the probes
never saw during training, so scoring them isn't leaking train-set signal.

Each entry uses the Vagueness_prompt variant as `ambiguous_prompt` -- one of the
two ambiguity types (lexical, vagueness) that actually made it into the combined
probe-training dataset (semantic/syntactic were dropped, see FINDINGS.md), and
present for all 164 raw orchid_heval problems (vs. lexical's 157), so it's the
more complete/consistent choice across all 32 test-set problems specifically
(confirmed: all 4 variant types are present for all 32, but vagueness was picked
for consistency with what the probe actually saw at training time).

Unlike the AtCoder-style ClarifyCodeBench problems, every orchid_heval problem
already ships its own reference solution AND an explicit ambiguity_explanation --
"vagueness_missing_detail" here is Vagueness_ambiguity_explanation, the exact
detail the Vagueness_prompt variant omits relative to `clear_prompt`.

Output: data/v10_code_quality/held_out_orchid32.json
  {
    "heval_<N>": {
      "task_id": "HumanEval/<N>", "entry_point": ...,
      "ambiguous_prompt": <Vagueness_prompt>, "clear_prompt": <original prompt>,
      "missing_detail": <Vagueness_ambiguity_explanation>,
      "solution": ..., "test_case": [...]  # all confirmed relation=="==" for this 32
    }, ...
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.combined_splits import make_combined_train_val_test_split  # noqa: E402
from src.data_utils import load_combined_activation_bundle  # noqa: E402

RAW_PATH = ROOT / "data" / "raw" / "orchid_heval.jsonl"
ACTIVATIONS_DIR = ROOT / "activations_ccb_heval"
OUT_PATH = ROOT / "data" / "v10_code_quality" / "held_out_orchid32.json"
SEED = 0  # must match scripts/save_best_models.py -- the split the deployed probes were saved against


def test_set_orchid_heval_names() -> set[str]:
    """Reproduce the exact test split used when the deployed rank1/rank2 probes were
    saved (save_best_models.py: seed=0, grouped by source_problem_id), and return the
    orchid_heval problems ('HumanEval/N') that landed in TEST -- the only ones the
    probes never saw during training/val."""
    _, labels, groups, _, dataset_source, _, _ = load_combined_activation_bundle(ACTIVATIONS_DIR)
    _, _, test_idx = make_combined_train_val_test_split(labels, dataset_source, groups, seed=SEED)
    test_groups = set(groups[test_idx][dataset_source[test_idx] == "orchid_heval"].tolist())
    return {f"HumanEval/{g.split('::')[1]}" for g in test_groups}


def main() -> None:
    test_names = test_set_orchid_heval_names()

    rows = [json.loads(l) for l in open(RAW_PATH) if l.strip()]
    by_name = {r["name"]: r for r in rows}

    missing = test_names - set(by_name.keys())
    assert not missing, f"test-set problems missing from raw file: {missing}"

    held_out = {}
    for name in sorted(test_names, key=lambda n: int(n.split("/")[1])):
        r = by_name[name]
        n_str = name.split("/")[1]
        tcs = r["test_case"]
        assert all(tc.get("relation") == "==" for tc in tcs), \
            f"{name}: expected all-'==' test cases (verified earlier), found other relation"
        held_out[f"heval_{n_str}"] = {
            "task_id": name,
            "entry_point": r["entry_point"],
            "ambiguous_prompt": r["Vagueness_prompt"],
            "clear_prompt": r["prompt"],
            "missing_detail": r["Vagueness_ambiguity_explanation"],
            "solution": r["solution"],
            "test_case": tcs,
        }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    json.dump(held_out, open(OUT_PATH, "w"), indent=2)
    print(f"[build_orchid32] {len(held_out)} problems -> {OUT_PATH}")


if __name__ == "__main__":
    main()
