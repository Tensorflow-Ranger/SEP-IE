"""v20j: full GPU round-2 ablation grid -- plain MLP (no SupCon) + SupCon +
cross-source-only SupCon, each x the same 3 pairwise-weighting schemes v20d
used for the plain-LR round-2 probe (unweighted / inv_freq / fixed_w10=0.15).

Stage 1 (probe_1) is UNCHANGED from v20g/v20i (sklearn pairwise LR, CPU --
it's tiny and fast, and changing it would shift which of the 244 test groups
count as "misses", breaking comparability with every number reported so far:
v20g plain-LR funnel 0.7869, v20h zero-shot 0.8484, v20i SupCon 0.7910/0.8156).
Only stage 2 (the round-2 branch probe) is redone here, fully on GPU:

  - Feature encoders (fit ONCE per type, reused across all 3 weighting
    variants since encoding is independent of pair-weighting):
      * plain      : StandardScaler only (no SupCon) -- isolates whether
                     nonlinearity ALONE (an MLP head, no representation
                     learning) helps, vs. needing SupCon's contrastive
                     pretraining (mirrors FINDINGS.md's raw_lr vs raw_mlp
                     vs supcon_raw_mlp three-way split in `## v19 SupCon`).
      * supcon     : plain SupCon projector (`## v19 SupCon`), same-tier
                     positives from anywhere.
      * supcon_xsrc: cross-source-only SupCon (`## v19 cross-source-only
                     SupCon`), positives restricted to a different
                     dataset_source, same tier.
  - Pairwise head: single hidden layer (64 units) MLP, trained on GPU with
    Adam + a per-pair sample weight via CrossEntropyLoss(reduction="none"),
    replacing the previous CPU sklearn MLPClassifier (which has no
    sample_weight -- v20i had to approximate fixed_w10 via pair-downsampling
    as a workaround). This now applies the EXACT same weight_of() functions
    v20d used for probe_2/2i/2r, no approximation needed.
  - Weighting schemes (identical definitions to
    v20d_train_eval_round2_reweighted.py's `fit_pairwise_lr`):
      * unweighted: all pairs weight 1.0
      * inv_freq  : weight(pair_type) = mean_count / count(pair_type)
      * fixed_w10 : weight((1,0)) = 0.15, else 1.0

All branch encoding is batched into ONE GPU forward pass per feature type
(previous v20i called `pipe.transform()` inside a per-branch Python loop --
~2857 separate small GPU round-trips per variant -- which is almost
certainly what triggered the transient NVML/CUDACachingAllocator crash
partway through v20i's cross-source run). A short retry-with-backoff wraps
each GPU fit/encode call as a defense against that same flakiness recurring.

Usage: python v20j_gpu_round2_ablation.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
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
OUT_PATH = ROOT / "data" / "v20_branches_full" / "gpu_round2_ablation_results.json"
LAYER = 22
SEED = 0
TEST_FRAC = 0.2
FIXED_W10 = 0.15
HEAD_HIDDEN = 64
HEAD_EPOCHS = 400
HEAD_LR = 0.01
HEAD_WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def with_retry(fn, tries=3, base_delay=2.0):
    for attempt in range(tries):
        try:
            return fn()
        except RuntimeError as e:
            if attempt == tries - 1:
                raise
            print(f"[v20j] RuntimeError on attempt {attempt+1}/{tries} ({e}); "
                  f"clearing cache and retrying...", flush=True)
            torch.cuda.empty_cache()
            time.sleep(base_delay * (attempt + 1))


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fit_pairwise_lr(pools):
    """Stage-1 probe_1 -- unchanged from v20g/v20i, sklearn CPU (fast, and
    kept exactly as-is for parity with every prior report)."""
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


def fit_supcon_pipe(train_pools_r2, cross_source_only: bool):
    X_pool, y_pool, src_pool = [], [], []
    for vecs, tiers, source in train_pools_r2:
        for i in range(len(tiers)):
            X_pool.append(vecs[i]); y_pool.append(int(tiers[i])); src_pool.append(source)
    X_pool, y_pool, src_pool = np.array(X_pool), np.array(y_pool), np.array(src_pool)
    scaler = StandardScaler().fit(X_pool)
    X_scaled = scaler.transform(X_pool)

    def _fit():
        return fit_supcon_projector(
            X_scaled, y_pool, embed_dim=32, hidden_dim=None, temperature=0.1,
            weight_decay=1e-3, lr=0.01, epochs=500, seed=SEED, device=DEVICE,
            sources_train=src_pool if cross_source_only else None,
            cross_source_only=cross_source_only,
        )
    encode_fn, supcon_model = with_retry(_fit)
    device = next(supcon_model.parameters()).device
    return SupConPipe(supcon_model, scaler, device)


class PlainPipe:
    def __init__(self, scaler):
        self.scaler = scaler

    def transform(self, X):
        return self.scaler.transform(X)


def fit_plain_pipe(train_pools_r2):
    all_vecs = np.concatenate([vecs for vecs, _tiers, _src in train_pools_r2], axis=0)
    return PlainPipe(StandardScaler().fit(all_vecs))


def batch_encode(pipe, pools):
    """One GPU forward pass for ALL branches, instead of per-branch calls."""
    lens = [len(vecs) for vecs, _tiers, _src in pools]
    all_vecs = np.concatenate([vecs for vecs, _tiers, _src in pools], axis=0)
    all_z = with_retry(lambda: pipe.transform(all_vecs))
    out, idx = [], 0
    for l in lens:
        out.append(all_z[idx:idx + l])
        idx += l
    return out


def count_pair_types(pools):
    c = Counter()
    for _vecs, tiers, _src in pools:
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] > tiers[j]:
                    c[(int(tiers[i]), int(tiers[j]))] += 1
    return c


def weight_fn_for(scheme, type_counts):
    if scheme == "unweighted":
        return lambda t: 1.0
    if scheme == "inv_freq":
        mean_count = np.mean(list(type_counts.values()))
        return lambda t: mean_count / type_counts[t]
    if scheme == "fixed_w10":
        return lambda t: FIXED_W10 if t == (1, 0) else 1.0
    raise ValueError(scheme)


def build_pairwise_diffs(zs, pools, weight_of):
    X, y, w = [], [], []
    for z, (_vecs, tiers, _src) in zip(zs, pools):
        n = len(tiers)
        for i in range(n):
            for j in range(n):
                if tiers[i] <= tiers[j]:
                    continue
                pw = weight_of((int(tiers[i]), int(tiers[j])))
                X.append(z[i] - z[j]); y.append(1); w.append(pw)
                X.append(z[j] - z[i]); y.append(0); w.append(pw)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(w, dtype=np.float32)


class TorchPairwiseMLPHead:
    """1-hidden-layer MLP pairwise-diff head, trained full-batch on GPU with
    per-sample weights (CrossEntropyLoss(reduction="none"), manually
    weighted+averaged) -- the thing sklearn's MLPClassifier cannot do."""

    def __init__(self, in_dim, hidden_dim=HEAD_HIDDEN, weight_decay=HEAD_WEIGHT_DECAY,
                 lr=HEAD_LR, epochs=HEAD_EPOCHS, seed=SEED, device=DEVICE):
        self.device = device
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2),
        ).to(device)
        self.weight_decay = weight_decay
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y, w):
        def _fit():
            Xt = torch.tensor(X, device=self.device)
            yt = torch.tensor(y, device=self.device)
            wt = torch.tensor(w, device=self.device)
            opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.model.train()
            for _ in range(self.epochs):
                opt.zero_grad()
                logits = self.model(Xt)
                loss_per = nn.functional.cross_entropy(logits, yt, reduction="none")
                loss = (loss_per * wt).sum() / wt.sum()
                loss.backward()
                opt.step()
            self.model.eval()
            return self
        return with_retry(_fit)

    @torch.no_grad()
    def predict_proba(self, X):
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=self.device)
        return torch.softmax(self.model(Xt), dim=1).cpu().numpy()


def score_win_counting(pipe, head, vecs):
    z = pipe.transform(vecs)
    n = len(vecs)
    diffs, pair_idx = [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diffs.append(z[i] - z[j]); pair_idx.append(i)
    proba = head.predict_proba(np.array(diffs, dtype=np.float32))[:, 1]
    scores = np.zeros(n)
    for p, i in zip(proba, pair_idx):
        scores[i] += p
    return scores


def main():
    print(f"[v20j] device={DEVICE}", flush=True)
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

    # --- Stage 1 (unchanged) ---
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
    print(f"[v20j] probe_1 native round-1 hit rate: {hit1_count}/{n1} = {hit1_count/n1:.4f} "
          f"(parity check vs. 160/244=0.6557)", flush=True)

    # --- Round-2 train pools (with dataset_source, for cross-source SupCon) ---
    train_pools_r2 = []
    for g in train_groups_r2:
        for branch_idx, _q in branches_by_group[g]:
            d = np.load(ACT_DIR / f"{branch_idx}.npz")
            vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
            tiers = d["tiers"]
            source = branches[branch_idx]["dataset_source"]
            train_pools_r2.append((vecs, tiers, source))
    print(f"[v20j] {len(train_pools_r2)} round-2 train branches", flush=True)
    type_counts = count_pair_types(train_pools_r2)
    print(f"[v20j] train pair-type counts: {type_counts}", flush=True)

    misses = [r for r in test_results if not r["hit1"]]
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
    print(f"[v20j] {len(misses)} misses; {len(chain_targets)} chainable, "
          f"{n_unrecoverable} unrecoverable (parity check vs. 83/1)", flush=True)

    feature_types = (
        ("plain", lambda: fit_plain_pipe(train_pools_r2)),
        ("supcon", lambda: fit_supcon_pipe(train_pools_r2, cross_source_only=False)),
        ("supcon_xsrc", lambda: fit_supcon_pipe(train_pools_r2, cross_source_only=True)),
    )
    weighting_schemes = ("unweighted", "inv_freq", "fixed_w10")

    all_results = {}
    for feat_name, fit_pipe_fn in feature_types:
        print(f"\n[v20j] ##### feature encoder: {feat_name} #####", flush=True)
        pipe = fit_pipe_fn()
        zs_train = batch_encode(pipe, train_pools_r2)
        in_dim = zs_train[0].shape[1]

        for scheme in weighting_schemes:
            variant_name = f"{feat_name}_mlp_{scheme}"
            weight_of = weight_fn_for(scheme, type_counts)
            X, y, w = build_pairwise_diffs(zs_train, train_pools_r2, weight_of)
            head = TorchPairwiseMLPHead(in_dim=in_dim).fit(X, y, w)
            print(f"[v20j] {variant_name}: trained on {len(y)} pairwise-diff rows (in_dim={in_dim})",
                  flush=True)

            n_recovered = 0
            for gid, branch_idx in chain_targets.items():
                d = np.load(ACT_DIR / f"{branch_idx}.npz")
                vecs = d["candidates"][:, list(d["layers"]).index(LAYER), :]
                tiers = d["tiers"]
                scores = score_win_counting(pipe, head, vecs)
                top1 = int(np.argmax(scores))
                hit2 = int(tiers[top1] == tiers.max())
                n_recovered += hit2

            final_hits = hit1_count + n_recovered
            print(f"[v20j] {variant_name}: round-2 recovered {n_recovered}/{len(chain_targets)} "
                  f"chainable misses -> FINAL {final_hits}/{n1} = {final_hits/n1:.4f}", flush=True)
            all_results[variant_name] = {
                "n_recovered": n_recovered, "n_chainable": len(chain_targets),
                "final_hits": final_hits, "final_two_stage_hit_rate": final_hits / n1,
            }
        del pipe
        torch.cuda.empty_cache()

    out = {
        "n_test_groups_scored": n1,
        "probe_1_native_round1_hit_rate": hit1_count / n1,
        "probe_1_hits": hit1_count,
        "n_misses": len(misses),
        "n_misses_chainable": len(chain_targets),
        "n_misses_unrecoverable": n_unrecoverable,
        "train_pair_type_counts": {f"{k[0]}>{k[1]}": v for k, v in type_counts.items()},
        "variants": all_results,
        "reference": {
            "v20g_plain_lr_fixed_w10": 0.7869,
            "v20h_zero_shot_qwen": 0.8484,
            "v20i_supcon_downsample_approx_fixed_w10": 0.8156,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n[v20j] === SUMMARY (final two-stage hit rate, n={n1}) ===", flush=True)
    for name, r in sorted(all_results.items(), key=lambda kv: -kv[1]["final_two_stage_hit_rate"]):
        print(f"  {name:35s} {r['final_two_stage_hit_rate']:.4f} "
              f"({r['n_recovered']}/{r['n_chainable']} recovered)", flush=True)
    print(f"[v20j] saved to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
