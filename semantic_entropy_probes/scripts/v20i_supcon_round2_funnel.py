"""v20i: does SupCon help the round-2 stage of the two-stage funnel?

Direct follow-up to v20g_two_stage_funnel_eval.py (probe_1 native round-1 hit
rate 160/244=0.6557, chained to plain pairwise-LR probe_2r on the 83
chainable misses -> final 192/244=0.7869). Everything about the funnel is
IDENTICAL (same 977/244 split, same probe_1, same miss-chaining-by-exact-
question-match) except stage 2's training method, which is swapped for the
SupCon recipe already validated in `## v19 SupCon` (FINDINGS.md:3111) and
piloted for round-depth in `v20_train_round_supcon.py`: a small (embed_dim=32,
no hidden layer) contrastive projection head trained on tier-as-label over
raw activations (src/supcon.py's `fit_supcon_projector`, plain variant -- not
cross_source_only, matching v19's winning config), frozen, then an MLP
pairwise-diff head on top, scored by win-counting (sum of pairwise P(i>j)),
exactly as v20_train_round_supcon.py's eval_ranking "mlp" branch does.

Two round-2 stage-2 variants are compared against v20g's plain-LR baselines:
  - supcon_raw_mlp            (unweighted pairs)   vs. v20g's probe_2  (0.517 on its own test slice)
  - supcon_raw_mlp_reweighted (1,0-pairs downsampled to 15% of count,
    approximating v20d/e's fixed_w10=0.15 pairwise LR reweight -- sklearn's
    MLPClassifier has no sample_weight support, so downsampling is the
    closest available analog) vs. v20g's probe_2r (0.540 / funnel-chained 0.787)

Usage: python v20i_supcon_round2_funnel.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import v19_train_ranking_probe as v19  # noqa: E402
from v19_extract_activations import (  # noqa: E402
    JUDGED_PATH, V12_JUDGED_PATH, load_answers_by_question,
)
from src.supcon import fit_supcon_projector  # noqa: E402

BRANCHES_PATH = ROOT / "data" / "v20_branches_full" / "branches_round2.json"
ACT_DIR = ROOT / "data" / "v20_branches_full" / "round2_activations"
OUT_PATH = ROOT / "data" / "v20_branches_full" / "supcon_round2_funnel_results.json"
LAYER = 22
SEED = 0
TEST_FRAC = 0.2
FIXED_W10 = 0.15
MLP_HIDDEN = (64,)
MLP_MAX_ITER = 500


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fit_pairwise_lr(pools):
    """Unchanged from v20g -- refits probe_1 identically for stage-1 parity."""
    all_vecs = np.concatenate([v for v, _ in pools], axis=0)
    pipe = make_pipeline(StandardScaler()).fit(all_vecs)
    X, y = [], []
    for vecs, tiers in pools:
        z = pipe.transform(vecs)
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    X.append(z[i] - z[j]); y.append(1)
                    X.append(z[j] - z[i]); y.append(0)
    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    return pipe, clf.coef_[0]


def reconstruct_question_order(gid, judged_rows, v12_judged):
    answers_by_q = load_answers_by_question(gid, v12_judged)
    qs = []
    for c in judged_rows[gid]["judged"]["candidates"]:
        q = c["question"].strip()
        if answers_by_q.get(q) is None:
            continue
        qs.append(q)
    return qs


class SupConPipe:
    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device

    def transform(self, X):
        Xs = self.scaler.transform(X)
        with torch.no_grad():
            Xt = torch.tensor(np.asarray(Xs, dtype=np.float32), device=self.device)
            return self.model(Xt).cpu().numpy()


def fit_supcon_round2(train_pools_r2, downsample_10_frac=None, cross_source_only=False):
    """train_pools_r2: list of (vecs, tiers, source) per branch. Fits a SupCon
    projector on tier-as-label over ALL candidate rows, then an MLP
    pairwise-diff head on the frozen embeddings.
    `cross_source_only`: use src/supcon.py's cross-source-positives-only
    variant (`## v19 cross-source-only SupCon`) -- positives for an anchor
    are same-tier rows from a DIFFERENT dataset_source only, forcing "tier 2"
    (say) to mean the same thing across clarifycodebench/orchid_heval/
    orchid_bcb/orchid_bcb_expand rather than letting same-source same-tier
    pairs dominate (bcb_expand alone is the majority of round-2 branches).
    `downsample_10_frac`: if set, randomly keep only this fraction of
    (tier=1,tier=0)-type pairs before fitting the MLP head (analog of
    v20d/e's fixed_w10 pairwise-LR reweight, since MLPClassifier has no
    sample_weight)."""
    X_pool, y_pool, src_pool = [], [], []
    for vecs, tiers, source in train_pools_r2:
        for i in range(len(tiers)):
            X_pool.append(vecs[i]); y_pool.append(int(tiers[i])); src_pool.append(source)
    X_pool, y_pool, src_pool = np.array(X_pool), np.array(y_pool), np.array(src_pool)
    scaler = StandardScaler().fit(X_pool)
    X_scaled = scaler.transform(X_pool)

    encode_fn, supcon_model = fit_supcon_projector(
        X_scaled, y_pool, embed_dim=32, hidden_dim=None, temperature=0.1,
        weight_decay=1e-3, lr=0.01, epochs=500, seed=SEED,
        sources_train=src_pool if cross_source_only else None,
        cross_source_only=cross_source_only,
    )
    device = next(supcon_model.parameters()).device
    pipe = SupConPipe(supcon_model, scaler, device)

    rng = np.random.RandomState(SEED)
    X, y = [], []
    for vecs, tiers, _source in train_pools_r2:
        z = pipe.transform(vecs)
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] <= tiers[j]:
                    continue
                pair_type = (int(tiers[i]), int(tiers[j]))
                if downsample_10_frac is not None and pair_type == (1, 0):
                    if rng.random_sample() >= downsample_10_frac:
                        continue
                X.append(z[i] - z[j]); y.append(1)
                X.append(z[j] - z[i]); y.append(0)
    clf = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, max_iter=MLP_MAX_ITER,
                         random_state=SEED, early_stopping=True).fit(np.array(X), np.array(y))
    return pipe, clf, len(y)


def score_win_counting(pipe, clf, vecs):
    z = pipe.transform(vecs)
    n = len(vecs)
    diffs, pair_idx = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diffs.append(z[i] - z[j]); pair_idx.append(i)
    proba = clf.predict_proba(np.array(diffs))[:, 1]
    scores = np.zeros(n)
    for p, i in zip(proba, pair_idx):
        scores[i] += p
    return scores


def main():
    ids, dataset_source, source_problem_id, layers, per_group_vecs, per_group_tiers = v19.load_groups()
    li = layers.tolist().index(LAYER)

    branches = json.loads(BRANCHES_PATH.read_text())
    branches_by_group = defaultdict(list)
    for i, b in enumerate(branches):
        if not (ACT_DIR / f"{i}.npz").exists():
            continue
        branches_by_group[b["id"]].append((i, b["removed_question"].strip()))

    groups_with_branches = sorted(branches_by_group.keys())
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(groups_with_branches))
    n_test = max(1, int(len(groups_with_branches) * TEST_FRAC))
    test_groups = [g for g in np.array(groups_with_branches)[perm[:n_test]].tolist()]
    train_groups_r2 = [g for g in groups_with_branches if g not in set(test_groups)]
    print(f"[v20i] {len(groups_with_branches)} groups w/ branches -> "
          f"train={len(train_groups_r2)} test={len(test_groups)}", flush=True)

    # --- Stage 1: identical to v20g, refit for parity ---
    train_pools_r1 = [(per_group_vecs[g][li], per_group_tiers[g]) for g in train_groups_r2
                       if g in per_group_vecs and len(set(per_group_tiers[g].tolist())) >= 2]
    pipe1, w1 = fit_pairwise_lr(train_pools_r1)

    judged_rows = {r["id"]: r for r in read_jsonl(JUDGED_PATH)}
    v12_judged = {r["id"]: r for r in read_jsonl(V12_JUDGED_PATH)}

    test_results = []
    for g in test_groups:
        if g not in per_group_vecs or len(set(per_group_tiers[g].tolist())) < 2:
            continue
        vecs = per_group_vecs[g][li]
        tiers = per_group_tiers[g]
        z = pipe1.transform(vecs)
        scores = z @ w1
        top1 = int(np.argmax(scores))
        best_tier = int(tiers.max())
        hit1 = int(tiers[top1] == best_tier)
        test_results.append({"id": g, "hit1": hit1, "top1_idx": top1})

    n1 = len(test_results)
    hit1_count = sum(r["hit1"] for r in test_results)
    print(f"[v20i] probe_1 native round-1 hit rate: {hit1_count}/{n1} = {hit1_count/n1:.4f} "
          f"(parity check vs. v20g's 160/244=0.6557)", flush=True)

    # --- Stage 2 train pools (round-2 branches, train groups only) ---
    train_pools_r2 = []
    for g in train_groups_r2:
        for branch_idx, _q in branches_by_group[g]:
            d = np.load(ACT_DIR / f"{branch_idx}.npz")
            vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
            tiers = d["tiers"]
            source = branches[branch_idx]["dataset_source"]
            train_pools_r2.append((vecs, tiers, source))
    n_sources = len(set(s for _, _, s in train_pools_r2))
    print(f"[v20i] {len(train_pools_r2)} round-2 train branches, {n_sources} dataset_source values", flush=True)

    misses = [r for r in test_results if not r["hit1"]]
    # Resolve chainable branches once, shared across both SupCon variants.
    chain_targets = {}
    n_unrecoverable = 0
    for r in misses:
        gid = r["id"]
        qs = reconstruct_question_order(gid, judged_rows, v12_judged)
        if r["top1_idx"] >= len(qs):
            n_unrecoverable += 1
            continue
        wrong_q = qs[r["top1_idx"]]
        match = next((idx for idx, bq in branches_by_group[gid] if bq == wrong_q), None)
        if match is None:
            n_unrecoverable += 1
            continue
        chain_targets[gid] = match
    print(f"[v20i] {len(misses)} misses; {len(chain_targets)} chainable, "
          f"{n_unrecoverable} unrecoverable (parity check vs. v20g's 83/1)", flush=True)

    results = {}
    variants = (
        ("supcon_raw_mlp", None, False),
        ("supcon_raw_mlp_reweighted", FIXED_W10, False),
        ("supcon_xsrc_raw_mlp", None, True),
        ("supcon_xsrc_raw_mlp_reweighted", FIXED_W10, True),
    )
    for variant_name, downsample, xsrc in variants:
        print(f"\n[v20i] === fitting {variant_name} (downsample_10_frac={downsample}, "
              f"cross_source_only={xsrc}) ===", flush=True)
        pipe2, clf2, n_pairs = fit_supcon_round2(train_pools_r2, downsample_10_frac=downsample,
                                                  cross_source_only=xsrc)
        print(f"[v20i] {variant_name} trained on {n_pairs} pairwise-diff rows", flush=True)

        n_recovered = 0
        for gid, branch_idx in chain_targets.items():
            d = np.load(ACT_DIR / f"{branch_idx}.npz")
            vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
            tiers = d["tiers"]
            scores = score_win_counting(pipe2, clf2, vecs)
            top1 = int(np.argmax(scores))
            hit2 = int(tiers[top1] == tiers.max())
            n_recovered += hit2

        final_hits = hit1_count + n_recovered
        print(f"[v20i] {variant_name}: round-2 recovered {n_recovered}/{len(chain_targets)} chainable misses",
              flush=True)
        print(f"[v20i] {variant_name}: FINAL two-stage hit rate: {final_hits}/{n1} = {final_hits/n1:.4f} "
              f"(vs. probe_1-alone {hit1_count/n1:.4f}, vs. v20g plain-LR funnel 0.7869)", flush=True)
        results[variant_name] = {
            "n_recovered": n_recovered,
            "n_chainable": len(chain_targets),
            "final_hits": final_hits,
            "final_two_stage_hit_rate": final_hits / n1,
        }

    out = {
        "n_test_groups_scored": n1,
        "probe_1_native_round1_hit_rate": hit1_count / n1,
        "probe_1_hits": hit1_count,
        "n_misses": len(misses),
        "n_misses_chainable": len(chain_targets),
        "n_misses_unrecoverable": n_unrecoverable,
        "variants": results,
        "comparison_v20g_plain_lr_funnel": {
            "final_two_stage_hit_rate": 0.7869, "n_recovered": 32, "n_chainable": 83,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[v20i] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
