"""Within a SINGLE prompt, do its different candidate modifications (Qwen's
up-to-4 self-elicited ambiguities, scored in v9) have a genuinely
differentiated effect on P(ambiguous) -- or is any apparent ordering just
noise?

Two checks for "is this a real ranking, not noise":
  1. Cross-probe agreement: rank1 (LR, layer 20) and rank2 (MLP, layer 22)
     are independently trained, different architectures, different layers.
     If they agree on which candidate was most/least effective WITHIN a
     prompt, that ordering reflects a shared, real signal in the
     activations -- not idiosyncratic noise in one probe.
  2. Null-effect floor: 7/160 v9 candidates are literal no-ops (Qwen
     returned the original text unchanged, byte-for-byte identical prompt).
     Since extraction is a deterministic forward pass, these MUST show
     delta_p == 0 exactly -- any non-zero here would indicate a pipeline
     bug, and the magnitude of genuine effects can be compared against this
     hard floor.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    v9_all = list(csv.DictReader(open(ROOT / "results_v9_self_ambiguity" / "all_candidates.csv")))
    gen_rows = [json.loads(l) for l in open(ROOT / "data" / "detail_addition_probe_test_v9_self_ambiguity.jsonl")]
    cand_text = {r["id"]: r.get("candidate_text", "") for r in gen_rows}
    no_change = {r["id"]: r.get("qwen_made_no_change", False) for r in gen_rows}

    by_base = defaultdict(list)
    for r in v9_all:
        r["rank1_delta_p"] = float(r["rank1_delta_p"])
        r["rank2_delta_p"] = float(r["rank2_delta_p"])
        r["mean_delta_p"] = float(r["mean_delta_p"])
        cid = f"{r['base_id']}__cand{r['candidate_idx']}"
        r["candidate_text"] = cand_text.get(cid, "")
        r["no_change"] = no_change.get(cid, False)
        by_base[r["base_id"]].append(r)

    # ---- null-effect floor: no-op candidates ----
    noop_deltas = [r["mean_delta_p"] for cands in by_base.values() for r in cands if r["no_change"]]
    real_deltas = [r["mean_delta_p"] for cands in by_base.values() for r in cands if not r["no_change"]]
    print(f"=== null-effect floor: no-op candidates (n={len(noop_deltas)}) ===")
    print(f"  mean|delta_p| for no-ops: {np.mean(np.abs(noop_deltas)):.5f} (should be ~0, sanity check)")
    print(f"  mean|delta_p| for real edits (n={len(real_deltas)}): {np.mean(np.abs(real_deltas)):.4f}")
    print(f"  -> real edits move the probe {np.mean(np.abs(real_deltas)) / max(np.mean(np.abs(noop_deltas)), 1e-9):.0f}x "
          f"more than the floor (or floor is exactly 0)\n")

    # ---- within-prompt spread and cross-probe rank agreement ----
    spreads = []
    top1_agree = 0
    spearman_rhos = []
    n_full4 = 0
    for base, cands in by_base.items():
        if len(cands) < 2:
            continue
        deltas = [c["mean_delta_p"] for c in cands]
        spreads.append(max(deltas) - min(deltas))

        r1 = [c["rank1_delta_p"] for c in cands]
        r2 = [c["rank2_delta_p"] for c in cands]
        best1 = cands[int(np.argmin(r1))]["candidate_idx"]
        best2 = cands[int(np.argmin(r2))]["candidate_idx"]
        if best1 == best2:
            top1_agree += 1
        if len(cands) == 4:
            n_full4 += 1
            rho, _ = stats.spearmanr(r1, r2)
            if not np.isnan(rho):
                spearman_rhos.append(rho)

    print(f"=== within-prompt spread across candidates (n={len(spreads)} prompts) ===")
    print(f"  mean spread (max-min delta_p among that prompt's candidates): {np.mean(spreads):.4f}")
    print(f"  median spread: {np.median(spreads):.4f}, min: {np.min(spreads):.4f}, max: {np.max(spreads):.4f}")
    print(f"  -> spread is {np.mean(spreads) / max(np.mean(np.abs(noop_deltas)), 1e-9):.0f}x the no-op floor "
          f"(or floor is exactly 0) -- candidates within a prompt genuinely differ, not just noise\n")

    print(f"=== cross-probe agreement on WITHIN-PROMPT ranking (n={len(by_base)} prompts) ===")
    print(f"  top-1 agreement (LR's best candidate == MLP's best candidate): {top1_agree}/{len(by_base)} "
          f"({100*top1_agree/len(by_base):.0f}%)  [chance level for 4 candidates = 25%]")
    if spearman_rhos:
        print(f"  mean within-prompt Spearman rho(LR ranking, MLP ranking) over {len(spearman_rhos)} "
              f"prompts with all 4 candidates: {np.mean(spearman_rhos):+.3f} "
              f"(median {np.median(spearman_rhos):+.3f})")
        print(f"  fraction of prompts with rho > 0 (probes agree on relative order): "
              f"{sum(1 for r in spearman_rhos if r > 0)}/{len(spearman_rhos)}")

    # ---- show a few concrete examples: best vs worst candidate text + effect ----
    print(f"\n=== example prompts: best vs worst self-elicited candidate ===")
    examples = sorted(by_base.items(), key=lambda kv: -(max(c["mean_delta_p"] for c in kv[1]) - min(c["mean_delta_p"] for c in kv[1])))[:4]
    for base, cands in examples:
        cands_sorted = sorted(cands, key=lambda c: c["mean_delta_p"])
        best, worst = cands_sorted[0], cands_sorted[-1]
        print(f"\n  {base} (spread={worst['mean_delta_p']-best['mean_delta_p']:.4f}):")
        print(f"    BEST  (delta={best['mean_delta_p']:+.4f}): {best['candidate_text'][:140]}")
        print(f"    WORST (delta={worst['mean_delta_p']:+.4f}): {worst['candidate_text'][:140]}")


if __name__ == "__main__":
    main()
