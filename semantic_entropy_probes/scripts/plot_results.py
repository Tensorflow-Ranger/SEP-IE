"""Plot per-layer probe Spearman correlation (test split + grouped-CV band) vs. baselines.

One chart, one job: does the probe's predicted ambiguity degree track the
true count K systematically with layer depth, and how does it compare to
chance (rho=0) and to a length-only control. Single y-axis (Spearman rho),
ordinal x-axis (layer index, 0 = embedding output). Categorical hues follow
the fixed palette order (blue = probe, orange = length-only baseline); chance
is a muted reference line, not a data series.
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

# Palette (see dataviz skill reference/palette.md) -- fixed categorical order.
SERIES_BLUE = "#2a78d6"
SERIES_BLUE_BAND = "#9ec5f4"  # step 200, for the CV std ribbon
SERIES_ORANGE = "#eb6834"
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
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = ap.parse_args()

    layer_metrics = _read_csv(args.results_dir / "layer_metrics.csv")
    layer_cv = _read_csv(args.results_dir / "layer_cv_spearman.csv")
    baselines = json.loads((args.results_dir / "baselines.json").read_text())

    layers = [int(r["layer"]) for r in layer_metrics]
    test_spearman = [float(r["test_spearman"]) for r in layer_metrics]
    cv_mean = [float(r["cv_spearman_mean"]) for r in layer_cv]
    cv_std = [float(r["cv_spearman_std"]) for r in layer_cv]
    cv_lo = [m - s for m, s in zip(cv_mean, cv_std)]
    cv_hi = [m + s for m, s in zip(cv_mean, cv_std)]

    length_rho = baselines["length_only"]["test_spearman"]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(layers, cv_lo, cv_hi, color=SERIES_BLUE_BAND, alpha=0.45,
                     linewidth=0, label="5-fold grouped CV Spearman (mean ± std)")
    ax.plot(layers, test_spearman, color=SERIES_BLUE, linewidth=2.0, marker="o",
             markersize=5, label="Held-out test Spearman")

    ax.axhline(0.0, color=INK_MUTED, linewidth=1.5, linestyle="--", label="Chance (rho = 0)")
    if length_rho == length_rho:  # not NaN
        ax.axhline(length_rho, color=SERIES_ORANGE, linewidth=1.8, linestyle="--",
                   label=f"Token-length-only baseline (rho = {length_rho:.2f})")

    ax.set_xlabel("Layer (0 = embedding output)", color=INK_SECONDARY)
    ax.set_ylabel("Spearman correlation (predicted E[K] vs. true K)", color=INK_SECONDARY)
    ax.set_title("Does Qwen3-8B's residual stream track ambiguity DEGREE?\n"
                 "Per-layer multinomial probe, last prompt token, frozen activations",
                 color=INK_PRIMARY, fontsize=12, loc="left")

    ax.set_ylim(-0.3, 1.02)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)

    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SECONDARY)

    fig.tight_layout()
    out_path = args.results_dir / "layer_spearman.png"
    fig.savefig(out_path, facecolor=SURFACE)
    print(f"[plot_results] wrote {out_path}")


if __name__ == "__main__":
    main()
