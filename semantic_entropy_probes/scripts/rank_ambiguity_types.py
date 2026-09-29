"""Is there a clear ranking among the different KINDS of prompt changes --
do some ambiguity_types (ClarifyCodeBench's own taxonomy: Terminology,
Output Format, Behavior, Edge Cases, Comparison Rules, Indices & Ranges,
Collection Semantics, Numerical Precision) get more effectively "resolved"
(bigger P(ambiguous) drop) than others when the real clarifying detail is
added? And within v9's self-elicited candidates, does listing ORDER predict
effectiveness (does Qwen tend to name the most impactful ambiguity first)?

Uses layer 20/22 (rank1/rank2) results exclusively -- the layer shown to be
the robust choice across all three test constructions (hand-authored,
Qwen-generated, Qwen self-elicited), NOT the layer-36 variants which turned
out not to generalize.

Deliberately checks robustness ACROSS constructions (v7 hand-authored, v8
Qwen-generated, v9 self-elicited-best) rather than reporting a ranking from
just one run, given the layer-36 lesson that a single-run pattern can be a
construction artifact.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def load_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    v7 = load_csv(ROOT / "results_v7_combined" / "comparison_table.csv")
    v8 = load_csv(ROOT / "results_v8_qwen_generated" / "comparison_table.csv")
    v9_top = load_csv(ROOT / "results_v9_self_ambiguity" / "top_candidate_per_prompt.csv")
    v9_all = load_csv(ROOT / "results_v9_self_ambiguity" / "all_candidates.csv")

    base_to_type = {r["base_id"]: r["ambiguity_types"] for r in v7}

    # per-prompt combined "missing pull" (avg of LR/MLP delta vs baseline), each construction
    def per_prompt_delta_missing(rows, r1_col, r2_col, base_col):
        out = {}
        for r in rows:
            base = r[base_col]
            base_p = base_to_type.get(base)
            d1 = float(r[f"rank1_p_amb_missing"]) - float(r[f"rank1_p_amb_baseline"])
            d2 = float(r[f"rank2_p_amb_missing"]) - float(r[f"rank2_p_amb_baseline"])
            out[base] = (d1 + d2) / 2
        return out

    v7_delta = per_prompt_delta_missing(v7, "rank1", "rank2", "base_id")
    v8_delta = per_prompt_delta_missing(v8, "rank1", "rank2", "base_id")
    v9_delta = {r["base_id"]: float(r["mean_delta_p"]) for r in v9_top}  # already avg of rank1/rank2, best-of-4

    # ================= ranking by ambiguity_type =================
    by_type_v7, by_type_v8, by_type_v9 = defaultdict(list), defaultdict(list), defaultdict(list)
    for base, t in base_to_type.items():
        by_type_v7[t].append(v7_delta[base])
        by_type_v8[t].append(v8_delta[base])
        by_type_v9[t].append(v9_delta[base])

    types = sorted(base_to_type.values(), key=lambda t: -list(base_to_type.values()).count(t))
    types = sorted(set(types))

    print("=== mean P(ambiguous) delta on 'missing' condition, by ambiguity_type ===")
    print("(more negative = more effectively resolved; n = # prompts of that type, out of 40 total)\n")
    rows_out = []
    for t in types:
        n = len(by_type_v7[t])
        m7, m8, m9 = mean(by_type_v7[t]), mean(by_type_v8[t]), mean(by_type_v9[t])
        avg_v7v8 = (m7 + m8) / 2
        consistent_sign = (m7 < 0) == (m8 < 0) == (m9 < 0)
        rows_out.append({"type": t, "n": n, "v7_hand": m7, "v8_qwen": m8, "v9_best_of_4": m9,
                          "avg_v7_v8": avg_v7v8, "consistent_direction_all3": consistent_sign})

    rows_out.sort(key=lambda r: r["avg_v7_v8"])
    print(f"  {'type':22s} {'n':3s} {'v7 hand':>9s} {'v8 qwen':>9s} {'v9 best4':>9s} {'avg(v7,v8)':>11s}  consistent?")
    for r in rows_out:
        print(f"  {r['type']:22s} {r['n']:3d} {r['v7_hand']:+9.4f} {r['v8_qwen']:+9.4f} {r['v9_best_of_4']:+9.4f} "
              f"{r['avg_v7_v8']:+11.4f}  {r['consistent_direction_all3']}")

    # Spearman correlation between v7 and v8 per-type means (do the two authorship
    # variants agree on WHICH types are most resolvable, even if not perfectly)
    common_types = [t for t in types if len(by_type_v7[t]) >= 2]  # need >=2 for a meaningful per-type mean
    if len(common_types) >= 3:
        v7_means = [mean(by_type_v7[t]) for t in common_types]
        v8_means = [mean(by_type_v8[t]) for t in common_types]
        rho, p = stats.spearmanr(v7_means, v8_means)
        print(f"\n  Spearman rho between v7-hand-authored and v8-qwen-generated per-type ranking "
              f"(n_types={len(common_types)}, types with >=2 prompts only): rho={rho:.3f}, p={p:.3f}")

    # ================= per-prompt ranking across all 40, each construction =================
    print("\n=== per-prompt ranking (all 40), sorted by avg(v7,v8) delta_missing, most-resolved first ===")
    per_prompt = []
    for base, t in base_to_type.items():
        per_prompt.append({"base_id": base, "type": t, "v7": v7_delta[base], "v8": v8_delta[base], "v9_best": v9_delta[base]})
    per_prompt.sort(key=lambda r: (r["v7"] + r["v8"]) / 2)
    print(f"  {'base_id':14s} {'type':22s} {'v7':>8s} {'v8':>8s} {'v9_best':>8s}")
    for r in per_prompt[:10]:
        print(f"  {r['base_id']:14s} {r['type']:22s} {r['v7']:+8.4f} {r['v8']:+8.4f} {r['v9_best']:+8.4f}")
    print("  ...")
    for r in per_prompt[-10:]:
        print(f"  {r['base_id']:14s} {r['type']:22s} {r['v7']:+8.4f} {r['v8']:+8.4f} {r['v9_best']:+8.4f}")

    rho_v7v9, p_v7v9 = stats.spearmanr([r["v7"] for r in per_prompt], [r["v9_best"] for r in per_prompt])
    rho_v7v8, p_v7v8 = stats.spearmanr([r["v7"] for r in per_prompt], [r["v8"] for r in per_prompt])
    print(f"\n  per-prompt Spearman rho: v7 vs v8 = {rho_v7v8:.3f} (p={p_v7v8:.4f}); "
          f"v7 vs v9-best = {rho_v7v9:.3f} (p={p_v7v9:.4f})")

    # ================= does elicitation ORDER predict effectiveness (v9)? =================
    print("\n=== v9: does candidate listing order (0=first Qwen named) predict effect size? ===")
    by_idx = defaultdict(list)
    for r in v9_all:
        by_idx[int(r["candidate_idx"])].append(float(r["mean_delta_p"]))
    for idx in sorted(by_idx):
        vals = by_idx[idx]
        print(f"  candidate_idx={idx}: n={len(vals)}, mean_delta_p={mean(vals):+.4f}, "
              f"n_negative={sum(1 for v in vals if v < 0)}/{len(vals)}")

    # within-prompt: for each prompt, rank its candidates 0..3 by effect size (1=biggest drop),
    # then correlate that rank against candidate_idx (listing order)
    within_prompt_ranks = []  # (candidate_idx, effect_rank)
    by_base = defaultdict(list)
    for r in v9_all:
        by_base[r["base_id"]].append((int(r["candidate_idx"]), float(r["mean_delta_p"])))
    for base, cands in by_base.items():
        cands_sorted = sorted(cands, key=lambda x: x[1])  # most negative (best) first
        for effect_rank, (cidx, _delta) in enumerate(cands_sorted):
            within_prompt_ranks.append((cidx, effect_rank))
    if within_prompt_ranks:
        idxs = [x[0] for x in within_prompt_ranks]
        eranks = [x[1] for x in within_prompt_ranks]
        rho_order, p_order = stats.spearmanr(idxs, eranks)
        print(f"\n  within-prompt Spearman rho(candidate_idx, effect_rank) = {rho_order:.3f} (p={p_order:.4f})")
        print(f"  (effect_rank 0 = most effective candidate in that prompt; "
              f"positive rho would mean 'later-listed = less effective', i.e. Qwen front-loads the best one)")

    with (ROOT / "results_v9_self_ambiguity" / "ambiguity_type_ranking.json").open("w") as f:
        json.dump({"by_type": rows_out}, f, indent=2)
    print(f"\n[rank_ambiguity_types] wrote results_v9_self_ambiguity/ambiguity_type_ranking.json")


if __name__ == "__main__":
    main()
