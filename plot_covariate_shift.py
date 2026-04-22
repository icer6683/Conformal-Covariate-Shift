#!/usr/bin/env python3
"""
plot_covariate_shift.py
=======================
Illustrates the covariate shift between a chosen test sector and all remaining
sectors (the calibration pool), using the saved S&P 500 data.

Works for any sector present in the saved data — Technology, Energy, Healthcare,
Financials, etc. — by passing --test_sector.

Usage
-----
  # Technology vs. rest  (primary paper figure)
  python plot_covariate_shift.py --npz sp500_20240201_20240328.npz

  # Energy vs. rest
  python plot_covariate_shift.py --npz sp500_20240201_20240328.npz --test_sector Energy

  # Healthcare vs. rest, saved as PDF
  python plot_covariate_shift.py --npz sp500_20240201_20240328.npz \\
      --test_sector Healthcare --save results/covariate_shift_healthcare.pdf

Options
-------
  --npz          Path to .npz data file (required)
  --json         Path to .json metadata (inferred from --npz if omitted)
  --test_sector  Sector to compare against all others. Default: Technology
  --save         Output path (PNG or PDF). Default: results/covariate_shift.png
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from finance_data import load_stored, filter_by_sector


# ── Clip percentiles to remove extreme outliers for cleaner visuals ──────────
CLIP_PCT = 1.0   # clip bottom and top 1% of each covariate


def clip_covariate(arr: np.ndarray, pct: float = CLIP_PCT) -> np.ndarray:
    lo = np.percentile(arr, pct)
    hi = np.percentile(arr, 100 - pct)
    return arr[(arr >= lo) & (arr <= hi)]


def kde_plot(ax, data, color, label, alpha_fill=0.25, lw=2.0):
    """Plot a KDE curve with a shaded fill on ax."""
    kde   = gaussian_kde(data, bw_method="scott")
    x_lo  = np.percentile(data, 0.5)
    x_hi  = np.percentile(data, 99.5)
    xs    = np.linspace(x_lo, x_hi, 400)
    ys    = kde(xs)
    ax.plot(xs, ys, color=color, lw=lw, label=label)
    ax.fill_between(xs, ys, alpha=alpha_fill, color=color)


def main():
    parser = argparse.ArgumentParser(description="Plot covariate shift: Tech vs non-Tech")
    parser.add_argument("--npz",  required=True,  help="Path to .npz data file")
    parser.add_argument("--json", default=None,   help="Path to .json metadata (inferred if omitted)")
    parser.add_argument("--save", default=None,   help="Save figure to this path (PDF or PNG)")
    parser.add_argument("--test_sector", default="Technology",
                        help="Sector held out as 'test'. Default: Technology")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    npz_path  = Path(args.npz)
    json_path = Path(args.json) if args.json else npz_path.with_suffix(".json")
    result    = load_stored(npz_path, json_path)

    cov_names = result["cov_names"]
    n_cov     = len(cov_names)

    # Pretty labels for each covariate
    pretty = {
        "OvernightGapReturn":    "Overnight Gap Return\n$(Open_t - Close_{t-1})/Close_{t-1}$",
        "Above52wLowReturn":     "Above 52-Week Low Return\n$(Open_t - Low_{52w})/Low_{52w}$",
        "TurnoverRatio_lag1":    "Turnover Ratio (lag 1)\n$Volume_{t-1}/(Shares \cdot Close_{t-1})$",
        "DailyRangeReturn_lag1": "Daily Range Return (lag 1)\n$(High - Low)_{t-1}/Close_{t-1}$",
    }

    # ── Split into Tech vs non-Tech ───────────────────────────────────────────
    tech     = filter_by_sector(result, [args.test_sector])
    sectors  = [m["sector"] for m in result["meta"]]
    non_idx  = [i for i, s in enumerate(sectors) if s.lower() != args.test_sector.lower()]
    non_tech = {
        "X":         result["X"][non_idx],
        "Y":         result["Y"][non_idx],
        "dates":     result["dates"],
        "tickers":   [result["tickers"][i] for i in non_idx],
        "meta":      [result["meta"][i]    for i in non_idx],
        "cov_names": cov_names,
    }

    n_tech    = tech["X"].shape[0]
    n_nontech = non_tech["X"].shape[0]
    L         = result["X"].shape[1]
    date0     = result["dates"][0]
    date1     = result["dates"][-1]

    print(f"Tech tickers    : {n_tech}")
    print(f"Non-tech tickers: {n_nontech}")
    print(f"Date range      : {date0} → {date1}")
    print(f"Covariates      : {cov_names}")

    # ── Flatten: (n_series, L, n_cov) -> (n_series*L, n_cov) ─────────────────
    X_tech    = tech["X"].reshape(-1, n_cov)      # (n*L, n_cov)
    X_nontech = non_tech["X"].reshape(-1, n_cov)  # (n*L, n_cov)

    # ── Plot ──────────────────────────────────────────────────────────────────
    ncols = min(n_cov, 2)
    nrows = (n_cov + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    TECH_COLOR    = "#2166ac"   # blue
    NONTECH_COLOR = "#d6604d"   # red-orange

    for k, ax in enumerate(axes):
        if k >= n_cov:
            ax.set_visible(False)
            continue

        t_vals  = clip_covariate(X_tech[:, k])
        nt_vals = clip_covariate(X_nontech[:, k])

        kde_plot(ax, nt_vals, color=NONTECH_COLOR,
                 label=f"Non-{args.test_sector}  (n={n_nontech})")
        kde_plot(ax, t_vals,  color=TECH_COLOR,
                 label=f"{args.test_sector}  (n={n_tech})")

        name  = cov_names[k]
        label = pretty.get(name, name)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Compute a simple KL-divergence proxy (histogram-based) per covariate
    kl_lines = []
    for k in range(n_cov):
        t_vals  = clip_covariate(X_tech[:, k])
        nt_vals = clip_covariate(X_nontech[:, k])
        lo = min(t_vals.min(), nt_vals.min())
        hi = max(t_vals.max(), nt_vals.max())
        bins = np.linspace(lo, hi, 50)
        p, _ = np.histogram(t_vals,  bins=bins, density=True)
        q, _ = np.histogram(nt_vals, bins=bins, density=True)
        eps  = 1e-10
        p    = p + eps
        q    = q + eps
        p   /= p.sum()
        q   /= q.sum()
        kl   = float(np.sum(p * np.log(p / q)))
        kl_lines.append(f"  {cov_names[k]:28s}  KL(Tech ‖ Non-Tech) = {kl:.4f}")

    print("\nDistribution divergence (histogram-based KL):")
    for line in kl_lines:
        print(line)

    fig.suptitle(
        f"Covariate shift: {args.test_sector} vs. Non-{args.test_sector} stocks\n"
        f"S&P 500  |  {date0} → {date1}",
        fontsize=12, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"\n[SAVED] {out}")
    else:
        out = Path("results/covariate_shift.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"\n[SAVED] {out}")


if __name__ == "__main__":
    main()
