"""Plot per-layer AUROC (test + bootstrapped 90% CI, and grouped-CV band) for
the OATML-style binary probe (K=1 "low" vs K in {2,3} "high").

One chart, one job: does the probe's AUROC for this binary split vary
systematically with layer depth. Single y-axis (AUROC), ordinal x-axis
(layer index). Categorical hues follow the fixed palette order.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

SERIES_BLUE = "#2a78d6"
SERIES_BLUE_BAND = "#9ec5f4"
SERIES_BLUE_CI = "#6da7ec"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_oatml")
    args = ap.parse_args()

    rows = _read_csv(args.results_dir / "layer_metrics.csv")
    summary = json.loads((args.results_dir / "run_summary.json").read_text())

    layers = [int(r["layer"]) for r in rows]
    test_auroc = [float(r["test_auroc"]) for r in rows]
    ci_low = [float(r["test_auroc_ci_low"]) for r in rows]
    ci_high = [float(r["test_auroc_ci_high"]) for r in rows]
    cv_mean = [float(r["cv_auroc_mean"]) for r in rows]
    cv_std = [float(r["cv_auroc_std"]) for r in rows]
    cv_lo = [m - s for m, s in zip(cv_mean, cv_std)]
    cv_hi = [m + s for m, s in zip(cv_mean, cv_std)]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(layers, cv_lo, cv_hi, color=SERIES_BLUE_BAND, alpha=0.35,
                     linewidth=0, label="5-fold grouped CV AUROC (mean ± std)")
    ax.errorbar(layers, test_auroc, yerr=[
        [t - lo for t, lo in zip(test_auroc, ci_low)],
        [hi - t for t, hi in zip(test_auroc, ci_high)],
    ], color=SERIES_BLUE, linewidth=2.0, marker="o", markersize=5,
        ecolor=SERIES_BLUE_CI, elinewidth=1.2, capsize=2,
        label="Held-out test AUROC (90% bootstrap CI)")

    ax.axhline(0.5, color=INK_MUTED, linewidth=1.5, linestyle="--", label="Chance (AUROC = 0.5)")

    threshold = summary["split_threshold"]
    dist = summary["binary_label_distribution"]
    ax.set_xlabel("Layer (0 = embedding output)", color=INK_SECONDARY)
    ax.set_ylabel("AUROC", color=INK_SECONDARY)
    ax.set_title("OATML-style binary probe: K=1 (\"low\") vs K∈{2,3} (\"high\")\n"
                 f"best_split threshold≈{threshold:.2f} — {dist.get('0', '?')} low / {dist.get('1', '?')} high",
                 color=INK_PRIMARY, fontsize=12, loc="left")

    ax.set_ylim(0.2, 1.02)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)

    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = args.results_dir / "layer_auroc.png"
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"[plot_oatml_results] wrote {out_path}")


if __name__ == "__main__":
    main()
