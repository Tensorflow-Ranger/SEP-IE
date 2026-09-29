"""v20e: small-sample validation of the v20c_branch_elicit_r2.py tier-definition
fix (see that script's docstring) -- re-run elicitation+tiering, WITH
`reference_text` (the dataset's own labeled ambiguity) now included, on a
small sample of branches that were previously labeled "easy" (no tier-2
candidate at all) under the old, label-blind "most important" tiering.

Does NOT touch production files (branches_round2.json, elicited_round2.jsonl)
-- reads branches_round2.json for branch_prompt/extra_context, backfills
reference_text from the v13 chunk files (same source
v20c_prepare_branches_full.py now carries forward for future runs), and
writes its own scratch output.

Output: data/v20_branches_full/v20e_relabel_test_sample.json
  [{"branch_idx":..., "id":..., "dataset_source":..., "old_max_tier":...,
    "new_max_tier":..., "old_candidates":[...], "new_candidates":[...]}, ...]

Usage: python scripts/v20e_relabel_test_sample.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from v19_extract_activations import CHUNK_DIR  # noqa: E402
import v20c_branch_elicit_r2 as elicit_r2  # noqa: E402 -- reuses the FIXED SYSTEM_PROMPT/build_user_prompt

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
ELICITED_PATH = ROOT / "data" / "v20_branches_full" / "elicited_round2.jsonl"
ACT_DIR = ROOT / "data" / "v20_branches_full" / "round2_activations"
OUT_PATH = ROOT / "data" / "v20_branches_full" / "v20e_relabel_test_sample.json"

SEED = 0
N_EASY_RANDOM = 20
KNOWN_EASY_HANDPICKED = [3096, 1103, 3488, 2319, 2786]  # manually inspected in this session


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def load_reference_text_by_id() -> dict:
    ref = {}
    for f in sorted(CHUNK_DIR.glob("chunk_*.json")):
        for g in json.loads(f.read_text()):
            ref[g["id"]] = g.get("reference_text", "")
    return ref


def main():
    env = {}
    for line in open(ROOT.parent / ".env"):
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = env["AWS"]

    branches = json.loads(BRANCHES_PATH.read_text())
    old_elicited = {d["branch_idx"]: d for d in read_jsonl(ELICITED_PATH) if "error" not in d}
    ref_by_id = load_reference_text_by_id()
    print(f"[v20e-test] {len(branches)} branches, {len(ref_by_id)} ids with reference_text", flush=True)

    # Find all "easy" branches (no tier-2 in the OLD elicited output) that have usable activations.
    easy_idxs = []
    for i in range(len(branches)):
        if not (ACT_DIR / f"{i}.npz").exists():
            continue
        d = np.load(ACT_DIR / f"{i}.npz")
        tiers = d["tiers"]
        if len(set(tiers.tolist())) < 2:
            continue
        if tiers.max() == 1:
            easy_idxs.append(i)

    rng = np.random.RandomState(SEED)
    random_sample = rng.choice([i for i in easy_idxs if i not in KNOWN_EASY_HANDPICKED],
                                size=N_EASY_RANDOM, replace=False).tolist()
    sample_idxs = KNOWN_EASY_HANDPICKED + random_sample
    print(f"[v20e-test] testing {len(sample_idxs)} previously-easy branches "
          f"({len(KNOWN_EASY_HANDPICKED)} hand-picked + {len(random_sample)} random)", flush=True)

    results = []
    for idx in sample_idxs:
        branch = dict(branches[idx])
        branch["reference_text"] = ref_by_id.get(branch["id"], "")
        old = old_elicited.get(idx, {})
        old_cands = old.get("candidates", [])
        old_max_tier = max((c.get("tier", 0) for c in old_cands), default=None)

        rt = boto3.client("bedrock-runtime", region_name=elicit_r2.REGION)
        user_prompt = elicit_r2.build_user_prompt(branch)
        try:
            raw = elicit_r2.call_deepseek(rt, user_prompt)
            parsed = elicit_r2.extract_json(raw)
            new_cands = parsed["candidates"]
            new_max_tier = max((c.get("tier", 0) for c in new_cands), default=None)
            print(f"[v20e-test] idx={idx} id={branch['id']:35s} had_ref_text={bool(branch['reference_text'])} "
                  f"old_max_tier={old_max_tier} -> new_max_tier={new_max_tier}", flush=True)
            results.append({
                "branch_idx": idx, "id": branch["id"], "dataset_source": branch["dataset_source"],
                "had_reference_text": bool(branch["reference_text"]),
                "old_max_tier": old_max_tier, "new_max_tier": new_max_tier,
                "old_candidates": old_cands, "new_candidates": new_cands,
            })
        except Exception as e:  # noqa: BLE001
            print(f"[v20e-test] idx={idx} id={branch['id']} ERROR {e}", flush=True)
            results.append({"branch_idx": idx, "id": branch["id"], "error": str(e)})

    OUT_PATH.write_text(json.dumps(results, indent=2))

    ok = [r for r in results if "error" not in r]
    n_recovered = sum(1 for r in ok if r["new_max_tier"] == 2)
    n_missing_ref = sum(1 for r in ok if not r["had_reference_text"])
    print(f"\n[v20e-test] {n_recovered}/{len(ok)} previously-easy branches now have a tier-2 candidate "
          f"({n_missing_ref} had NO reference_text available at all, e.g. new bcb_expand/needs-elicitation rows)",
          flush=True)
    print(f"[v20e-test] saved -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
