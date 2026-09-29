"""v20c stage 6: build round-3 branches from round-2's own branches (fresh
elicitation again, same "no candidate depletion, full pool replacement"
principle as round 2) -- for every usable round-2 branch, train a probe on
OTHER folds' round-2 branches (reusing each branch's inherited `fold` field
from round-2 prep, so no group ever leaks into the probe used to pick its
own top-3-wrong), score that branch's own candidate pool, take its top-3
wrong picks, and for each build a round-3 branch_prompt = round-2's
branch_prompt + a second chained "Clarification: {answer}".

Output: data/v20_branches_full/branches_round3.json
  [{"parent_branch_idx":..., "id":..., "fold":..., "removed_question":...,
    "branch_prompt":..., "extra_context":...}, ...]
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
from v19_extract_activations import CONCAT_TEMPLATE  # noqa: E402

OUT_DIR = ROOT / "data" / "v20_branches_full"
BRANCHES_R2_PATH = OUT_DIR / "branches_round2.json"
ELICITED_R2_PATH = OUT_DIR / "elicited_round2.jsonl"
ANSWERS_R2_PATH = OUT_DIR / "answers_round2.jsonl"
ACT_R2_DIR = OUT_DIR / "round2_activations"
OUT_PATH = OUT_DIR / "branches_round3.json"
LAYER = 22
N_BRANCH = 3


def read_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def fit_pairwise_lr(pools):
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


def main():
    branches_r2 = json.loads(BRANCHES_R2_PATH.read_text())
    elicited = {d["branch_idx"]: d for d in read_jsonl(ELICITED_R2_PATH) if "error" not in d}
    answers = {d["branch_idx"]: d for d in read_jsonl(ANSWERS_R2_PATH) if "error" not in d}

    # load activations + reconstruct aligned (question, answer, tier) per usable branch
    branch_data = {}  # branch_idx -> {"vecs":..., "tiers":..., "cands":[{"question","answer","tier"}]}
    for i in range(len(branches_r2)):
        npz_path = ACT_R2_DIR / f"{i}.npz"
        if not npz_path.exists() or i not in elicited or i not in answers:
            continue
        d = np.load(npz_path)
        li = list(d["layers"]).index(LAYER)
        vecs = d["candidates"][:, li, :]
        tiers = d["tiers"]

        ans_by_q = {a["question"].strip(): a["answer"] for a in answers[i]["answers"]
                    if "question" in a and "answer" in a}
        filtered = []
        for c in elicited[i]["candidates"]:
            q = c["question"].strip()
            ans = ans_by_q.get(q)
            if ans is None:
                continue
            filtered.append({"question": q, "answer": ans, "tier": c["tier"]})
        if len(filtered) != len(tiers):
            continue
        branch_data[i] = {"vecs": vecs, "tiers": tiers, "cands": filtered}

    print(f"[v20c-prepare-r3] {len(branch_data)} usable round-2 branches "
          f"(of {len(branches_r2)} total)", flush=True)

    # group by fold (inherited from the parent group's round-2 fold assignment)
    by_fold = defaultdict(list)
    for i in branch_data:
        by_fold[branches_r2[i]["fold"]].append(i)
    folds = sorted(by_fold)
    print(f"[v20c-prepare-r3] fold sizes: {[(f, len(by_fold[f])) for f in folds]}", flush=True)

    r3_branches = []
    n_skipped = 0
    for fold_k in folds:
        test_idxs = by_fold[fold_k]
        train_idxs = [i for i in branch_data if i not in test_idxs]
        if len(train_idxs) < 10:
            print(f"[v20c-prepare-r3] fold {fold_k}: too few training branches ({len(train_idxs)}), skipping", flush=True)
            n_skipped += len(test_idxs)
            continue
        train_pools = [(branch_data[i]["vecs"], branch_data[i]["tiers"]) for i in train_idxs]
        pipe_k, w_k = fit_pairwise_lr(train_pools)

        for i in test_idxs:
            bd = branch_data[i]
            vecs, tiers, cands = bd["vecs"], bd["tiers"], bd["cands"]
            if len(set(tiers.tolist())) < 2:
                n_skipped += 1
                continue
            z = pipe_k.transform(vecs)
            scores = z @ w_k
            best_tier = tiers.max()
            wrong_idx = [j for j in range(len(tiers)) if tiers[j] != best_tier]
            wrong_idx_sorted = sorted(wrong_idx, key=lambda j: -scores[j])[:N_BRANCH]

            parent = branches_r2[i]
            for j in wrong_idx_sorted:
                wrong_ans = cands[j]["answer"]
                branch_prompt_r3 = CONCAT_TEMPLATE.format(prompt=parent["branch_prompt"], answer=wrong_ans)
                r3_branches.append({
                    "parent_branch_idx": i, "id": parent["id"], "fold": fold_k,
                    "dataset_source": parent["dataset_source"],
                    "removed_question": cands[j]["question"], "removed_answer": wrong_ans,
                    "removed_tier": int(tiers[j]),
                    "branch_prompt": branch_prompt_r3,
                    "extra_context": parent["extra_context"],
                })

    OUT_PATH.write_text(json.dumps(r3_branches, indent=2))
    print(f"\n[v20c-prepare-r3] wrote {len(r3_branches)} round-3 branches "
          f"({n_skipped} round-2 branches skipped) to {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
