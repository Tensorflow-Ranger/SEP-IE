"""Recompute every pass-rate table used in the paper using the FRACTIONAL
test-case pass rate (round{N}_n_pass / round{N}_n_total per group), instead
of FINDINGS.md's binary all-or-nothing round{N}_pass metric.

Motivation: FINDINGS.md's v21-27 hit1-vs-pass-rate analysis used the binary
metric throughout and found the aggregate correlation non-significant. A
direct spot-check (see FINDINGS.md-adjacent discussion, not written up there)
showed the fractional metric recovers a real, statistically significant
round-1 effect the binary metric was masking. This script makes that
recomputation a reusable, checkable artifact instead of a one-off shell
calculation.

Usage: python scripts/paper_recompute_fractional_pass.py
Output: prints all tables to stdout and writes them to
        data/paper_tables/fractional_pass_tables.md
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "paper_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINES = {
    "trained probe (SupCon)": ROOT / "data/v21_code_vs_tests/results_deepseek_codegen_fixed_env.json",
    "Qwen3-8B zero-shot self-rank": ROOT / "data/v22_selector_ablation/qwen_selfrank/results_deepseek_codegen_fixed_env.json",
    "DeepSeek V3.2 zero-shot self-rank": ROOT / "data/v22_selector_ablation/deepseek_selfrank/results_deepseek_codegen_fixed_env.json",
}
NO_CLARIFICATION_FILE = ROOT / "data/v25_baseline_no_clarification/results_fixed_env.json"

N_BOOTSTRAP = 10_000
RNG_SEED = 0


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def frac(row: dict, rnd: int) -> float | None:
    n = row.get(f"round{rnd}_n_total")
    p = row.get(f"round{rnd}_n_pass")
    if n is None or n == 0:
        return None
    return p / n


def bootstrap_gap_ci(a: list[float], b: list[float], n_boot: int = N_BOOTSTRAP, seed: int = RNG_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        sa = [rng.choice(a) for _ in a]
        sb = [rng.choice(b) for _ in b]
        diffs.append(mean(sa) - mean(sb))
    diffs.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = int(0.975 * n_boot)
    return diffs[lo_idx], diffs[hi_idx]


def two_sample_report(a: list[float], b: list[float]) -> dict:
    """a = hit1/treatment group, b = miss/control group."""
    ma, mb = mean(a), mean(b)
    gap_pp = 100 * (ma - mb)
    u, p_mw = stats.mannwhitneyu(a, b, alternative="two-sided")
    t, p_t = stats.ttest_ind(a, b, equal_var=False)
    lo, hi = bootstrap_gap_ci(a, b)
    return {
        "n_a": len(a), "mean_a": ma,
        "n_b": len(b), "mean_b": mb,
        "gap_pp": gap_pp,
        "mannwhitney_p": p_mw,
        "welch_t_p": p_t,
        "bootstrap_ci_pp": (100 * lo, 100 * hi),
    }


def fmt_row(label: str, r: dict) -> str:
    sig = "significant (CI excludes 0)" if r["bootstrap_ci_pp"][0] * r["bootstrap_ci_pp"][1] > 0 else "not significant (CI includes 0)"
    return (
        f"| {label} | {r['n_a']} | {100*r['mean_a']:.1f}% | {r['n_b']} | {100*r['mean_b']:.1f}% "
        f"| {r['gap_pp']:+.1f}pp | MWU p={r['mannwhitney_p']:.3f} | Welch-t p={r['welch_t_p']:.3f} "
        f"| bootstrap 95% CI [{r['bootstrap_ci_pp'][0]:+.1f}, {r['bootstrap_ci_pp'][1]:+.1f}]pp ({sig}) |"
    )


def table_hit1_vs_pass(rnd: int, exclude_ccb: bool) -> list[str]:
    d = load(PIPELINES["trained probe (SupCon)"])
    lines = []
    sources = [None, "orchid_bcb", "orchid_heval", "orchid_bcb_expand"]
    if not exclude_ccb:
        sources.append("clarifycodebench")
    header = ("source", "n(hit1=1)", "mean_frac(hit1=1)", "n(hit1=0)", "mean_frac(hit1=0)", "gap", "MWU p", "Welch-t p", "bootstrap 95% CI")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for source in sources:
        rows = [r for r in d if (source is None or r["source"] == source)]
        if source is None:
            rows = [r for r in rows if not exclude_ccb or r["source"] != "clarifycodebench"]
        h1 = [frac(r, rnd) for r in rows if r.get("hit1") == 1 and frac(r, rnd) is not None]
        h0 = [frac(r, rnd) for r in rows if r.get("hit1") == 0 and frac(r, rnd) is not None]
        if not h1 or not h0:
            continue
        label = source or ("ALL Orchid combined" if exclude_ccb else "ALL sources")
        lines.append(fmt_row(label, two_sample_report(h1, h0)))
    return lines


def table_pipeline_comparison() -> list[str]:
    lines = ["| pipeline | round-1 mean fractional pass rate | round-1+2 mean fractional pass rate |",
             "|---|---|---|"]
    for name, path in PIPELINES.items():
        d = load(path)
        r1 = [frac(r, 1) for r in d if frac(r, 1) is not None]
        r2 = [frac(r, 2) for r in d if frac(r, 2) is not None]
        lines.append(f"| {name} | {100*mean(r1):.1f}% (n={len(r1)}) | {100*mean(r2):.1f}% (n={len(r2)}) |")
    return lines


def paired_report(treatment: list[float], control: list[float]) -> dict:
    """treatment/control are matched 1:1 (same group order) -- use paired tests."""
    assert len(treatment) == len(control)
    diffs = [t - c for t, c in zip(treatment, control)]
    mean_diff_pp = 100 * mean(diffs)
    # Wilcoxon signed-rank on the paired differences (drop exact-zero diffs per convention)
    try:
        w, p_w = stats.wilcoxon(treatment, control)
    except ValueError:
        w, p_w = float("nan"), float("nan")
    t, p_t = stats.ttest_rel(treatment, control)
    rng = random.Random(RNG_SEED)
    n = len(diffs)
    boot_means = []
    for _ in range(N_BOOTSTRAP):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(mean(sample))
    boot_means.sort()
    lo = boot_means[int(0.025 * N_BOOTSTRAP)]
    hi = boot_means[int(0.975 * N_BOOTSTRAP)]
    return {
        "n": n,
        "mean_treatment": mean(treatment),
        "mean_control": mean(control),
        "mean_diff_pp": mean_diff_pp,
        "wilcoxon_p": p_w,
        "paired_t_p": p_t,
        "bootstrap_ci_pp": (100 * lo, 100 * hi),
    }


def fmt_paired_row(label: str, r: dict) -> str:
    sig = "significant (CI excludes 0)" if r["bootstrap_ci_pp"][0] * r["bootstrap_ci_pp"][1] > 0 else "not significant (CI includes 0)"
    return (
        f"| {label} | {r['n']} | {100*r['mean_treatment']:.1f}% | {100*r['mean_control']:.1f}% "
        f"| {r['mean_diff_pp']:+.1f}pp | Wilcoxon p={r['wilcoxon_p']:.4f} | paired-t p={r['paired_t_p']:.4f} "
        f"| bootstrap 95% CI [{r['bootstrap_ci_pp'][0]:+.1f}, {r['bootstrap_ci_pp'][1]:+.1f}]pp ({sig}) |"
    )


def table_no_clarification_vs_any() -> list[str]:
    """Same 244 groups under two conditions -- paired test, not two-sample."""
    no_clar = load(NO_CLARIFICATION_FILE)
    probe = load(PIPELINES["trained probe (SupCon)"])
    assert [r["id"] for r in no_clar] == [r["id"] for r in probe], "group order must match for pairing"

    no_clar_frac = [r["n_pass"] / r["n_total"] for r in no_clar]
    r1_frac = [frac(r, 1) for r in probe]
    r12_frac = [frac(r, 2) for r in probe]

    lines = ["| comparison | n | mean_frac(treatment) | mean_frac(no-clar) | mean diff | Wilcoxon p | paired-t p | bootstrap 95% CI |",
             "|" + "---|" * 8]
    lines.append(fmt_paired_row("round-1 (probe) vs. no-clarification", paired_report(r1_frac, no_clar_frac)))
    lines.append(fmt_paired_row("round-1+2 (probe) vs. no-clarification", paired_report(r12_frac, no_clar_frac)))
    return lines


def main():
    out = []
    out.append("# Fractional test-case pass-rate tables (recomputed from raw per-group JSON)\n")
    out.append("Generated by `scripts/paper_recompute_fractional_pass.py`. All values are the mean of "
                "`round{N}_n_pass / round{N}_n_total` across groups (fractional test-case pass rate), "
                "NOT the binary all-or-nothing `round{N}_pass` metric FINDINGS.md's own tables used.\n")

    out.append("\n## Pipeline comparison (round-1 and round-1+2, all 244 groups)\n")
    out += table_pipeline_comparison()

    out.append("\n\n## Hit1 vs. pass-rate gap, round-1, all sources\n")
    out += table_hit1_vs_pass(rnd=1, exclude_ccb=False)

    out.append("\n\n## Hit1 vs. pass-rate gap, round-1, Orchid sources only (excludes clarifycodebench)\n")
    out += table_hit1_vs_pass(rnd=1, exclude_ccb=True)

    out.append("\n\n## Hit1 vs. pass-rate gap, round-1+2, Orchid sources only\n")
    out += table_hit1_vs_pass(rnd=2, exclude_ccb=True)

    out.append("\n\n## No-clarification baseline vs. any clarification (probe pipeline)\n")
    out += table_no_clarification_vs_any()

    text = "\n".join(out) + "\n"
    print(text)
    (OUT_DIR / "fractional_pass_tables.md").write_text(text)
    print(f"\nWritten to {OUT_DIR / 'fractional_pass_tables.md'}")


if __name__ == "__main__":
    main()
