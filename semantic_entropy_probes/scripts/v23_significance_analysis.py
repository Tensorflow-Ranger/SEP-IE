"""v23: statistical significance for the v21/v22 code-vs-tests ablations --
does round-1 -> round-2 clarification actually move functional pass rate by
more than noise, does the trained probe's selection beat zero-shot
Qwen/DeepSeek self-rank, and (2nd pass, added per the user's request) does
swapping the CODEGEN model from Qwen3-8B to DeepSeek V3.2 change any of this,
on the SAME 244-group test set / SAME candidate pools / SAME selections --
only which model writes the code differs.

Two tests per comparison (paired, since every comparison is on the identical
244 groups): exact McNemar's test (binomial test on discordant pairs) and a
paired bootstrap (10,000 resamples) 95% CI on the pass-rate difference.

Usage: python v23_significance_analysis.py
Runs on whichever results files exist yet -- prints what it has and what
it's still waiting on, rather than failing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
PIPELINES = {
    "probe+qwen_code": ROOT / "data" / "v21_code_vs_tests" / "results_fixed_env.json",
    "probe+deepseek_code": ROOT / "data" / "v21_code_vs_tests" / "results_deepseek_codegen_fixed_env.json",
    "qwen_selfrank+qwen_code": ROOT / "data" / "v22_selector_ablation" / "qwen_selfrank" / "results_fixed_env.json",
    "qwen_selfrank+deepseek_code": ROOT / "data" / "v22_selector_ablation" / "qwen_selfrank" / "results_deepseek_codegen_fixed_env.json",
    "deepseek_selfrank+qwen_code": ROOT / "data" / "v22_selector_ablation" / "deepseek_selfrank" / "results_fixed_env.json",
    "deepseek_selfrank+deepseek_code": ROOT / "data" / "v22_selector_ablation" / "deepseek_selfrank" / "results_deepseek_codegen_fixed_env.json",
}
N_BOOTSTRAP = 10000
SEED = 0


def load(name: str) -> dict[str, dict] | None:
    p = PIPELINES[name]
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    return {r["id"]: r for r in rows}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    p = binomtest(min(a_only, b_only), n, 0.5, alternative="two-sided").pvalue
    return a_only, b_only, p


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot=N_BOOTSTRAP, seed=SEED) -> tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs[i] = b[idx].mean() - a[idx].mean()
    return float(b.mean() - a.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def compare(name_a: str, key_a: str, rows_a: dict, name_b: str, key_b: str, rows_b: dict, common_ids: list[str]):
    a = np.array([rows_a[g][key_a] for g in common_ids], dtype=bool)
    b = np.array([rows_b[g][key_b] for g in common_ids], dtype=bool)
    n = len(common_ids)
    a_only, b_only, p = mcnemar_exact(a, b)
    diff, lo, hi = paired_bootstrap_ci(a.astype(float), b.astype(float))
    sig = "**SIGNIFICANT (p<0.05)**" if p < 0.05 else "not significant"
    print(f"\n{name_a}:{key_a} ({a.mean():.4f}) vs {name_b}:{key_b} ({b.mean():.4f})  n={n}")
    print(f"  McNemar: {name_a}-only-pass={a_only}, {name_b}-only-pass={b_only}, p={p:.4f} -> {sig}")
    print(f"  Bootstrap diff ({name_b}-{name_a}): {diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"{'  <- excludes 0' if lo > 0 or hi < 0 else ''}")


def main():
    loaded = {name: load(name) for name in PIPELINES}
    for name, rows in loaded.items():
        status = f"{len(rows)} groups" if rows is not None else "NOT YET AVAILABLE"
        print(f"[v23] {name}: {status}")

    available = {name: rows for name, rows in loaded.items() if rows is not None}
    if not available:
        print("\n[v23] nothing available yet.")
        return
    ids = sorted(next(iter(available.values())).keys())

    print(f"\n{'='*76}\n[v23] WITHIN-PIPELINE: does round-2 clarification beat round-1-only?\n{'='*76}")
    for name, rows in available.items():
        compare(name, "round1_pass", rows, name, "round2_pass", rows, sorted(set(ids) & set(rows.keys())))

    print(f"\n{'='*76}\n[v23] SELECTOR COMPARISON (within qwen-codegen): probe vs zero-shot self-rank\n{'='*76}")
    probe_q = available.get("probe+qwen_code")
    if probe_q:
        for name in ("qwen_selfrank+qwen_code", "deepseek_selfrank+qwen_code"):
            rows = available.get(name)
            if not rows:
                continue
            common = sorted(set(probe_q.keys()) & set(rows.keys()))
            compare(name, "round1_pass", rows, "probe+qwen_code", "round1_pass", probe_q, common)
            compare(name, "round2_pass", rows, "probe+qwen_code", "round2_pass", probe_q, common)

    print(f"\n{'='*76}\n[v23] SELECTOR COMPARISON (within deepseek-codegen): probe vs zero-shot self-rank\n{'='*76}")
    probe_d = available.get("probe+deepseek_code")
    if probe_d:
        for name in ("qwen_selfrank+deepseek_code", "deepseek_selfrank+deepseek_code"):
            rows = available.get(name)
            if not rows:
                continue
            common = sorted(set(probe_d.keys()) & set(rows.keys()))
            compare(name, "round1_pass", rows, "probe+deepseek_code", "round1_pass", probe_d, common)
            compare(name, "round2_pass", rows, "probe+deepseek_code", "round2_pass", probe_d, common)

    print(f"\n{'='*76}\n[v23] CODEGEN MODEL COMPARISON (same selector/prompts): Qwen3-8B vs DeepSeek V3.2 as coder\n{'='*76}")
    for selector in ("probe", "qwen_selfrank", "deepseek_selfrank"):
        rows_q = available.get(f"{selector}+qwen_code")
        rows_d = available.get(f"{selector}+deepseek_code")
        if not (rows_q and rows_d):
            continue
        common = sorted(set(rows_q.keys()) & set(rows_d.keys()))
        for key in ("round1_pass", "round2_pass"):
            compare(f"{selector}+qwen_code", key, rows_q, f"{selector}+deepseek_code", key, rows_d, common)

    missing = [name for name, rows in loaded.items() if rows is None]
    if missing:
        print(f"\n[v23] Still waiting on: {missing}. Re-run this script once they finish for the full picture.")


if __name__ == "__main__":
    main()
