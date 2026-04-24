#!/usr/bin/env python3
"""
multi_seed_medical.py
=====================
Run the medical sepsis experiment across multiple random seeds and aggregate
results. Each seed independently subsamples n_traincal patients from the
9264-patient TrainCal pool and n_test patients from the 5827-patient Test
pool, so variance reflects genuine patient-mix uncertainty.

Usage
-----
  # noshift, 10 seeds
  python medical/multi_seed_medical.py \\
      --pkl medical/sepsis_experiment_data_nacl_target.pkl \\
      --n_seeds 10 --n_traincal 1000 --n_test 500 \\
      --save_json results/medical/json/medical_ms_noshift.json \\
      --save_plot results/medical/pdf/medical_ms_noshift.pdf

  # with shift (LR + ACI), 10 seeds
  python medical/multi_seed_medical.py \\
      --pkl medical/sepsis_experiment_data_nacl_target.pkl \\
      --n_seeds 10 --n_traincal 1000 --n_test 500 --with_shift \\
      --save_json results/medical/json/medical_ms_shift.json \\
      --save_plot results/medical/pdf/medical_ms_shift.pdf

  # LR only (gamma=0), 10 seeds
  python medical/multi_seed_medical.py \\
      --pkl medical/sepsis_experiment_data_nacl_target.pkl \\
      --n_seeds 10 --n_traincal 1000 --n_test 500 --with_shift \\
      --gamma_grid 0.0 \\
      --save_json results/medical/json/medical_ms_LRonly.json \\
      --save_plot results/medical/pdf/medical_ms_LRonly.pdf
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medical.medical_conformal import (
    GAMMA_GRID,
    load_data,
    run_medical_experiment,
)

_C_COV    = "#2166ac"
_C_TARGET = "#d6604d"
_C_WIDTH  = "#4dac26"
_C_VAR    = "#7b2d8b"


def run_all_seeds(data, n_seeds, base_seed, n_traincal, n_test,
                  cal_frac, alpha, gamma_grid, with_shift):
    """Run one experiment per seed; return list of per-seed result dicts."""
    results = []
    for i in range(n_seeds):
        seed = base_seed + i
        print(f"  [seed {i+1:3d}/{n_seeds}]  seed={seed}", flush=True)
        r = run_medical_experiment(
            data=data,
            cal_frac=cal_frac,
            alpha=alpha,
            seed=seed,
            gamma_grid=gamma_grid,
            with_shift=with_shift,
            n_traincal=n_traincal,
            n_test=n_test,
            verbose=False,
        )
        results.append(r)
        print(f"             coverage={r['overall_coverage']:.4f}  "
              f"mean_width={np.mean(r['width_by_time']):.2f}", flush=True)
    return results


def aggregate(results, alpha, config):
    """Aggregate per-seed results into a single dict matching the synthetic schema."""
    n_seeds = len(results)
    T = len(results[0]["coverage_by_time"])   # 23 prediction hours
    hours = results[0]["hours"]               # [1..23]

    by_time = {}
    for t in range(T):
        covs   = [r["coverage_by_time"][t] for r in results]
        widths = [r["width_by_time"][t]     for r in results]
        by_time[str(t + 1)] = {
            "coverage_mean":   float(np.mean(covs)),
            "coverage_std":    float(np.std(covs)),
            "coverage_median": float(np.median(covs)),
            "coverage_q25":    float(np.percentile(covs, 25)),
            "coverage_q75":    float(np.percentile(covs, 75)),
            "coverage_min":    float(np.min(covs)),
            "coverage_max":    float(np.max(covs)),
            "width_mean":      float(np.mean(widths)),
            "width_std":       float(np.std(widths)),
        }

    overall_covs   = [r["overall_coverage"] for r in results]
    overall_widths = [np.mean(r["width_by_time"]) for r in results]

    target = 1.0 - alpha
    T3 = max(1, T // 3)
    early = [by_time[str(h + 1)]["coverage_mean"] for h in range(T3)]
    late  = [by_time[str(h + 1)]["coverage_mean"] for h in range(T - T3, T)]

    overall = {
        "coverage_mean":       float(np.mean(overall_covs)),
        "coverage_std":        float(np.std(overall_covs)),
        "coverage_se":         float(np.std(overall_covs) / np.sqrt(n_seeds)),
        "width_mean":          float(np.mean(overall_widths)),
        "width_std":           float(np.std(overall_widths)),
        "early_coverage_mean": float(np.mean(early)),
        "late_coverage_mean":  float(np.mean(late)),
        "coverage_degradation": float(np.mean(early) - np.mean(late)),
    }

    return {
        "n_seeds":    n_seeds,
        "hours":      hours,
        "config":     config,
        "by_time":    by_time,
        "overall":    overall,
        "per_seed_coverage": overall_covs,
        "per_seed_width":    overall_widths,
    }


def plot_aggregated(agg, save_path=None):
    hours  = agg["hours"]            # [1..23]
    T      = len(hours)
    target = 1.0 - agg["config"]["alpha"]
    n_seed = agg["n_seeds"]
    cfg    = agg["config"]
    shift_str = "LR + ACI" if cfg["with_shift"] and cfg["gamma_grid"] != [0.0] \
                else ("LR only, γ=0" if cfg["with_shift"] else "uniform + ACI")

    cov_means = [agg["by_time"][str(h)]["coverage_mean"] for h in hours]
    cov_q25   = [agg["by_time"][str(h)]["coverage_q25"]  for h in hours]
    cov_q75   = [agg["by_time"][str(h)]["coverage_q75"]  for h in hours]
    cov_stds  = [agg["by_time"][str(h)]["coverage_std"]  for h in hours]
    w_means   = [agg["by_time"][str(h)]["width_mean"]    for h in hours]
    w_stds    = [agg["by_time"][str(h)]["width_std"]     for h in hours]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Medical Multi-Seed Results (n={n_seed} seeds)  —  "
        f"Sepsis ICU  |  {shift_str}  |  "
        f"TrainCal={cfg['n_traincal']}  Test={cfg['n_test']}  α={cfg['alpha']}",
        fontsize=11, fontweight="bold",
    )

    def _clean(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Coverage over time + IQR
    ax = axes[0, 0]
    ax.plot(hours, cov_means, color=_C_COV, lw=2, label="Mean coverage")
    ax.fill_between(hours, cov_q25, cov_q75, alpha=0.25, color=_C_COV, label="IQR (25–75%)")
    ax.axhline(target, color=_C_TARGET, ls="--", lw=1.8, label=f"Target ({target:.0%})")
    ax.set_ylim(max(0, min(cov_q25) - 0.05), 1.02)
    ax.set_xlabel("Prediction hour", fontsize=10)
    ax.set_ylabel("Coverage rate", fontsize=10)
    ax.set_title(f"Coverage over Time  ({n_seed} seeds)", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); _clean(ax)

    # Width over time ± std
    ax = axes[0, 1]
    wm = np.array(w_means); ws = np.array(w_stds)
    ax.plot(hours, wm, color=_C_WIDTH, lw=2)
    ax.fill_between(hours, wm - ws, wm + ws, alpha=0.25, color=_C_WIDTH)
    ax.set_xlabel("Prediction hour", fontsize=10)
    ax.set_ylabel("Mean interval width (mL/hr)", fontsize=10)
    ax.set_title("Prediction Interval Width", fontsize=10)
    ax.grid(True, alpha=0.25); _clean(ax)

    # Coverage boxplots at early / mid / late
    ax = axes[0, 2]
    idx_early = 0; idx_mid = T // 2; idx_late = T - 1
    dists = []
    for hi in [hours[idx_early], hours[idx_mid], hours[idx_late]]:
        key = str(hi)
        dists.append([
            agg["by_time"][key]["coverage_q25"],
            agg["by_time"][key]["coverage_mean"] - agg["by_time"][key]["coverage_std"],
            agg["by_time"][key]["coverage_mean"],
            agg["by_time"][key]["coverage_mean"] + agg["by_time"][key]["coverage_std"],
            agg["by_time"][key]["coverage_q75"],
        ])
    # Use per-seed per-hour data for proper boxplots if available
    # (we only stored summary stats, so approximate with ±std bars)
    bp_vals = [[agg["by_time"][str(hours[i])]["coverage_mean"]] for i in [idx_early, idx_mid, idx_late]]
    ax.bar([0, 1, 2], [v[0] for v in bp_vals], color=_C_COV, alpha=0.6,
           tick_label=[f"h={hours[idx_early]}", f"h={hours[idx_mid]}", f"h={hours[idx_late]}"])
    ax.errorbar([0, 1, 2], [v[0] for v in bp_vals],
                yerr=[agg["by_time"][str(hours[i])]["coverage_std"] for i in [idx_early, idx_mid, idx_late]],
                fmt="none", color="black", capsize=5)
    ax.axhline(target, color=_C_TARGET, ls="--", lw=1.8)
    ax.set_ylim(max(0, min(v[0] for v in bp_vals) - 0.1), 1.02)
    ax.set_ylabel("Coverage rate", fontsize=10)
    ax.set_title("Coverage at Early / Mid / Late Hour  (mean ± std)", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y"); _clean(ax)

    # Per-seed overall coverage histogram
    ax = axes[1, 0]
    per_seed = agg["per_seed_coverage"]
    ax.hist(per_seed, bins=min(n_seed, 10), color=_C_COV, alpha=0.7, edgecolor="white")
    ax.axvline(target, color=_C_TARGET, ls="--", lw=1.8, label=f"Target {target:.0%}")
    ax.axvline(np.mean(per_seed), color=_C_COV, lw=2, label=f"Mean {np.mean(per_seed):.3f}")
    ax.set_xlabel("Overall coverage per seed", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of Per-Seed Coverage", fontsize=10)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25); _clean(ax)

    # Coverage variability across seeds over time
    ax = axes[1, 1]
    ax.plot(hours, cov_stds, color=_C_VAR, lw=2)
    ax.set_xlabel("Prediction hour", fontsize=10)
    ax.set_ylabel("Std dev of coverage across seeds", fontsize=10)
    ax.set_title("Coverage Variability Across Seeds", fontsize=10)
    ax.grid(True, alpha=0.25); _clean(ax)

    # Summary table
    ax = axes[1, 2]
    ax.axis("off")
    ov = agg["overall"]
    table_data = [
        ["Overall coverage",   f"{ov['coverage_mean']:.3f} ± {ov['coverage_std']:.3f}"],
        ["Target coverage",    f"{target:.3f}"],
        ["Coverage error",     f"{ov['coverage_mean'] - target:+.3f}"],
        ["Std error (seeds)",  f"{ov['coverage_se']:.4f}"],
        ["Early coverage",     f"{ov['early_coverage_mean']:.3f}"],
        ["Late coverage",      f"{ov['late_coverage_mean']:.3f}"],
        ["Degradation",        f"{ov['coverage_degradation']:+.3f}"],
        ["Mean width (mL/hr)", f"{ov['width_mean']:.1f} ± {ov['width_std']:.1f}"],
        ["Seeds",              f"{n_seed}"],
    ]
    tbl = ax.table(cellText=table_data, cellLoc="left",
                   colWidths=[0.58, 0.42], loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.9)
    for i in range(len(table_data)):
        tbl[(i, 0)].set_facecolor("#e8eef4")
        tbl[(i, 1)].set_facecolor("#f5f7fa")
    ax.set_title("Summary Statistics", fontweight="bold", fontsize=10, pad=12)

    fig.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  [Plot] Saved to {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True,
                        help="Path to sepsis_experiment_data_nacl_target.pkl")
    parser.add_argument("--n_seeds",    type=int,   default=10)
    parser.add_argument("--base_seed",  type=int,   default=1000,
                        help="Seeds used will be base_seed, base_seed+1, ...")
    parser.add_argument("--n_traincal", type=int,   default=1000)
    parser.add_argument("--n_test",     type=int,   default=500)
    parser.add_argument("--cal_frac",   type=float, default=0.5)
    parser.add_argument("--alpha",      type=float, default=0.1)
    parser.add_argument("--gamma_grid", type=float, nargs="+", default=None,
                        help="ACI step-size candidates (default: GAMMA_GRID from medical_conformal)")
    parser.add_argument("--with_shift", action="store_true", default=False)
    parser.add_argument("--save_json",  default=None)
    parser.add_argument("--save_plot",  default=None)
    args = parser.parse_args()

    gamma_grid = args.gamma_grid if args.gamma_grid is not None else GAMMA_GRID

    print(f"Loading {args.pkl} ...")
    data = load_data(args.pkl)
    n_tc_full = len(data["patient_trajectory_list_traincal"])
    n_te_full = len(data["patient_trajectory_list_test"])
    print(f"Full pool: {n_tc_full} TrainCal, {n_te_full} Test")
    print(f"Subsampling: {args.n_traincal} TrainCal, {args.n_test} Test per seed")
    print(f"Condition: {'with_shift (LR+ACI)' if args.with_shift else 'noshift (uniform+ACI)'}  "
          f"gamma_grid={gamma_grid}")
    print(f"Running {args.n_seeds} seeds (base_seed={args.base_seed}) ...")
    print()

    t0 = datetime.now()
    per_seed = run_all_seeds(
        data=data,
        n_seeds=args.n_seeds,
        base_seed=args.base_seed,
        n_traincal=args.n_traincal,
        n_test=args.n_test,
        cal_frac=args.cal_frac,
        alpha=args.alpha,
        gamma_grid=gamma_grid,
        with_shift=args.with_shift,
    )
    elapsed = datetime.now() - t0
    print(f"\nAll seeds complete in {elapsed}.")

    config = {
        "n_traincal":  args.n_traincal,
        "n_test":      args.n_test,
        "cal_frac":    args.cal_frac,
        "alpha":       args.alpha,
        "gamma_grid":  gamma_grid,
        "with_shift":  args.with_shift,
        "base_seed":   args.base_seed,
        "n_seeds":     args.n_seeds,
    }
    agg = aggregate(per_seed, args.alpha, config)

    ov = agg["overall"]
    target = 1.0 - args.alpha
    print(f"\n{'='*55}")
    print(f"  SUMMARY ({args.n_seeds} seeds)")
    print(f"{'='*55}")
    print(f"  Overall coverage : {ov['coverage_mean']:.4f} ± {ov['coverage_std']:.4f}  "
          f"(target {target:.2f},  error {ov['coverage_mean']-target:+.4f})")
    print(f"  Std error        : {ov['coverage_se']:.4f}")
    print(f"  Early coverage   : {ov['early_coverage_mean']:.4f}")
    print(f"  Late coverage    : {ov['late_coverage_mean']:.4f}")
    print(f"  Degradation      : {ov['coverage_degradation']:+.4f}")
    print(f"  Mean width       : {ov['width_mean']:.2f} ± {ov['width_std']:.2f} mL/hr")
    print(f"{'='*55}")

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"  [Results] Saved to {out}")

    plot_aggregated(agg, save_path=args.save_plot)


if __name__ == "__main__":
    main()
