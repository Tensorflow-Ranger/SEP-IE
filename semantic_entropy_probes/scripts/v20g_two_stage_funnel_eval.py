"""v20g: honest two-stage funnel eval, per user request -- don't score probe_1
out-of-distribution on round-2 branches. Instead:

  1. Score probe_1 on its OWN native round-1 pools, restricted to the same
     held-out test_groups used throughout v20c/v20d/v20e (244 groups) -- this
     reproduces the ~0.62 in-domain number FINDINGS.md cites (line 3552-3554).
  2. For every test group probe_1 GETS WRONG, look up the round-2 branch that
     applies EXACTLY that wrong candidate (matched by question text, not just
     "some top-3-wrong branch") and score it with the v20e-fixed round-2
     probe (probe_2r, fixed_w10=0.15, trained on train-side groups only).
  3. Final hit rate = (probe_1 hits) + (probe_1 misses recovered by probe_2r
     on the matching branch) -- a real two-round funnel, not two independent
     evaluations on mismatched test distributions.

Caveat surfaced directly, not hidden: branches_round2.json's top-3-wrong
candidates per group were selected by a 5-fold-CV probe (v20c_prepare_branches_full.py),
NOT by this script's own probe_1 (trained on the exact 977/244 split). If
probe_1's actual top-1 wrong pick isn't among those 3 stored branches, that
miss can't be chained to a round-2 branch and is counted as unrecoverable
(reported explicitly, not silently dropped).

Output: data/v20_branches_full/two_stage_funnel_results.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v19_train_ranking_probe as v19  # noqa: E402
from v19_extract_activations import (  # noqa: E402
    JUDGED_PATH, CHUNK_DIR, V12_JUDGED_PATH, load_answers_by_question,
)

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
ACT_DIR = ROOT / "data" / "v20_branches_full" / "round2_activations"
OUT_PATH = ROOT / "data" / "v20_branches_full" / "two_stage_funnel_results.json"
LAYER = 22
SEED = 0
TEST_FRAC = 0.2
FIXED_W10 = 0.15


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fit_pairwise_lr(pools, weight_of=None):
    all_vecs = np.concatenate([v for v, _ in pools], axis=0)
    pipe = make_pipeline(StandardScaler()).fit(all_vecs)
    X, y, w = [], [], []
    for vecs, tiers in pools:
        z = pipe.transform(vecs)
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    pw = weight_of((int(tiers[i]), int(tiers[j]))) if weight_of else 1.0
                    X.append(z[i] - z[j]); y.append(1); w.append(pw)
                    X.append(z[j] - z[i]); y.append(0); w.append(pw)
    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(
        np.array(X), np.array(y), sample_weight=np.array(w)
    )
    return pipe, clf.coef_[0]


def reconstruct_question_order(gid, judged_rows, v12_judged):
    """Exactly replays v19_extract_activations.py's filtering loop so index i
    here matches index i in per_group_vecs[gid]/per_group_tiers[gid]."""
    answers_by_q = load_answers_by_question(gid, v12_judged)
    qs = []
    for c in judged_rows[gid]["judged"]["candidates"]:
        q = c["question"].strip()
        if answers_by_q.get(q) is None:
            continue
        qs.append(q)
    return qs


def main():
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    li = layers.tolist().index(LAYER)

    branches = json.loads(BRANCHES_PATH.read_text())
    branches_by_group = defaultdict(list)  # gid -> [(branch_list_idx, removed_question)]
    for i, b in enumerate(branches):
        npz_path = ACT_DIR / f"{i}.npz"
        if not npz_path.exists():
            continue
        branches_by_group[b["id"]].append((i, b["removed_question"].strip()))

    groups_with_branches = sorted(branches_by_group.keys())
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_branches))
    n_test = max(1, int(len(groups_with_branches) * TEST_FRAC))
    test_groups = [g for g in np.array(groups_with_branches)[perm[:n_test]].tolist()]
    train_groups_r2 = [g for g in groups_with_branches if g not in set(test_groups)]
    print(f"[v20g] {len(groups_with_branches)} groups w/ usable branches -> "
          f"train={len(train_groups_r2)} test={len(test_groups)}", flush=True)

    # --- Stage 1: probe_1, trained + scored on native round-1 pools only ---
    train_pools_r1 = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_groups_r2
                       if g in per_group_vecs and len(set(per_group_tiers[g].tolist())) >= 2]
    pipe1, w1 = fit_pairwise_lr(train_pools_r1)
    print(f"[v20g] probe_1 trained on {len(train_pools_r1)} round-1 pools", flush=True)

    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}

    test_results = []
    n_no_native_pool = 0
    for g in test_groups:
        if g not in per_group_vecs or len(set(per_group_tiers[g].tolist())) < 2:
            n_no_native_pool += 1
            continue
        vecs = per_group_vecs[g][li]
        tiers = per_group_tiers[g]
        z = pipe1.transform(vecs)
        scores = z @ w1
        top1 = int(np.argmax(scores))
        best_tier = int(tiers.max())
        hit1 = int(tiers[top1] == best_tier)
        test_results.append({"id": g, "hit1": hit1, "top1_idx": top1, "n_candidates": len(tiers)})

    n1 = len(test_results)
    hit1_count = sum(r["hit1"] for r in test_results)
    print(f"[v20g] probe_1 native round-1 hit rate: {hit1_count}/{n1} = {hit1_count/n1:.4f} "
          f"({n_no_native_pool} test groups skipped, no valid native pool)", flush=True)

    # --- Stage 2: probe_2r, trained on TRAIN groups' round-2 branches only ---
    train_pools_r2 = []
    for g in train_groups_r2:
        for branch_idx, _q in branches_by_group[g]:
            d = np.load(ACT_DIR / f"{branch_idx}.npz")
            vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
            tiers = d["tiers"]
            train_pools_r2.append((vecs, tiers))
    weight_of = lambda t: FIXED_W10 if t == (1, 0) else 1.0  # noqa: E731
    pipe2r, w2r = fit_pairwise_lr(train_pools_r2, weight_of=weight_of)
    print(f"[v20g] probe_2r trained on {len(train_pools_r2)} round-2 branches "
          f"({len(train_groups_r2)} train groups)", flush=True)

    # --- Chain: for every probe_1 MISS, find the branch matching its actual
    # top-1 wrong question, score with probe_2r ---
    n_recovered = 0
    n_unrecovered_no_branch = 0
    n_miss = 0
    for r in test_results:
        if r["hit1"]:
            continue
        n_miss += 1
        gid = r["id"]
        qs = reconstruct_question_order(gid, judged_rows, v12_judged)
        if r["top1_idx"] >= len(qs):
            n_unrecovered_no_branch += 1
            r["round2_available"] = False
            continue
        wrong_q = qs[r["top1_idx"]]
        match = next((idx for idx, bq in branches_by_group[gid] if bq == wrong_q), None)
        if match is None:
            n_unrecovered_no_branch += 1
            r["round2_available"] = False
            continue
        d = np.load(ACT_DIR / f"{match}.npz")
        vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
        tiers = d["tiers"]
        z = pipe2r.transform(vecs)
        scores = z @ w2r
        top1_r2 = int(np.argmax(scores))
        hit2 = int(tiers[top1_r2] == tiers.max())
        r["round2_available"] = True
        r["hit2"] = hit2
        n_recovered += hit2

    final_hits = hit1_count + n_recovered
    print(f"[v20g] {n_miss} probe_1 misses; {n_miss - n_unrecovered_no_branch} had a matching round-2 "
          f"branch ({n_unrecovered_no_branch} unrecoverable -- no matching branch/activation)", flush=True)
    print(f"[v20g] round-2 recovered {n_recovered}/{n_miss - n_unrecovered_no_branch} of the "
          f"chainable misses", flush=True)
    print(f"[v20g] FINAL two-stage hit rate: {final_hits}/{n1} = {final_hits/n1:.4f} "
          f"(vs. probe_1-alone {hit1_count/n1:.4f})", flush=True)

    out = {
        "n_test_groups_scored": n1,
        "n_test_groups_skipped_no_native_pool": n_no_native_pool,
        "probe_1_native_round1_hit_rate": hit1_count / n1,
        "probe_1_hits": hit1_count,
        "n_misses": n_miss,
        "n_misses_unrecoverable_no_matching_branch": n_unrecovered_no_branch,
        "n_misses_recovered_by_round2": n_recovered,
        "final_two_stage_hit_rate": final_hits / n1,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"[v20g] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
