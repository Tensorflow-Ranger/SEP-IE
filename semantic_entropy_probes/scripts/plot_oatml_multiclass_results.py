"""Plot per-layer Spearman (+ grouped-CV band) for the OATML-style 3-class
probe (raw K in {1,2,3}, not binarized -- see train_oatml_style_probe.py
--mode multiclass). Same chart shape as plot_results.py's Spearman plot, so
the two are visually comparable side by side.
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
    ap.add_argument("--results-dir", type=Path, default=ROOT / "results_oatml_multiclass")
    args = ap.parse_args()

    rows = _read_csv(args.results_dir / "layer_metrics.csv")
    summary = json.loads((args.results_dir / "run_summary.json").read_text())

    layers = [int(r["layer"]) for r in rows]
    test_spearman = [float(r["test_spearman"]) for r in rows]
    cv_mean = [float(r["cv_spearman_mean"]) for r in rows]
    cv_std = [float(r["cv_spearman_std"]) for r in rows]
    cv_lo = [m - s for m, s in zip(cv_mean, cv_std)]
    cv_hi = [m + s for m, s in zip(cv_mean, cv_std)]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(layers, cv_lo, cv_hi, color=SERIES_BLUE_BAND, alpha=0.45,
                     linewidth=0, label="5-fold grouped CV Spearman (mean ± std)")
    ax.plot(layers, test_spearman, color=SERIES_BLUE, linewidth=2.0, marker="o",
             markersize=5, label="Held-out test Spearman")

    ax.axhline(0.0, color=INK_MUTED, linewidth=1.5, linestyle="--", label="Chance (rho = 0)")

    backend = summary.get("backend", "?")
    ax.set_xlabel("Layer (0 = embedding output)", color=INK_SECONDARY)
    ax.set_ylabel("Spearman correlation (predicted E[K] vs. true K)", color=INK_SECONDARY)
    ax.set_title("OATML-style 3-class probe: raw K∈{1,2,3}, not binarized\n"
                 f"backend={backend}, per-layer LogisticRegression, TBG token",
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
    print(f"[plot_oatml_multiclass_results] wrote {out_path}")


if __name__ == "__main__":
    main()
