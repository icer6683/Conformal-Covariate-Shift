#!/usr/bin/env python3
"""
=============================================================================
FINANCE CONFORMAL PREDICTION  —  OnlineConformalPredictor on S&P 500 data
=============================================================================

STATUS: LEGACY / UNUSED.
  Has never been run to produce saved JSON output. The paper baselines are
  produced by running finance_conformal.py with AdaptedCAFHT ablations
  (aci_stepsize=0.0 for LR-only; uniform weights for ACI-only) on the same
  covariate model. This script is retained for reference because it provides
  an AR(1)-only (no-covariate) baseline, but it is structurally different
  from the covariate-model AdaptedCAFHT baselines and is not used in the
  current experimental plan.

Mirrors the structure of finance_conformal.py but replaces AdaptedCAFHT with
OnlineConformalPredictor (adaptive_conformal.py).

KEY DIFFERENCES vs finance_conformal.py
----------------------------------------
  Prediction model : AR(1) on Y only  (finance_conformal uses a linear
                     covariate model Y_t ~ X_t)
  Calibration      : unweighted empirical quantile of AR(1) residuals
  Shift correction : none  (no logistic-regression density-ratio weights)
  ACI              : off by default; enable with --with_aci
  Online update    : off by default; enable with --adaptive_update to append
                     observed test residuals to the score window each step

USAGE
-----
  # Basic (AR(1) split-conformal, no adaptation)
  python finance_adaptive.py --npz sp500_20240201_20240328.npz --test_sector Technology

  # With ACI — fair comparison to finance_conformal.py (which always uses ACI)
  python finance_adaptive.py --npz sp500_20240701_20240830.npz --test_sector Technology --with_aci

  # With online score update (the "adaptive" part of OnlineConformalPredictor)
  python finance_adaptive.py --npz sp500_20240701_20240830.npz  --test_sector Technology --adaptive_update

  # Save results
  python finance_adaptive.py --npz sp500_20240201_20240328.npz --test_sector Technology \
      --with_aci --save_json results/finance_adaptive_tech.json --save_plot results/finance_adaptive_tech.png

FULL OPTIONS
------------
  --npz              Path to .npz data file (required)
  --test_sector      Sector held out as test set (required)
  --cal_frac         Fraction of non-test tickers used for calibration. Default: 0.5
  --alpha            Miscoverage level. Default: 0.1  (targets 90% coverage)
  --window_size      Sliding window size for adaptive_update mode. Default: None (unlimited)
  --with_aci         Enable per-series ACI alpha adjustment (mirrors finance_conformal.py)
  --gamma            ACI step-size gamma. Default: 0.005
  --adaptive_update  Append observed test residuals to score window after each time step
  --seed             Random seed. Default: 42
  --save_plot        Path to save the output figure
  --save_json        Path to save results as JSON
=============================================================================
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from finance.finance_data import load_stored
from core.adaptive_conformal import OnlineConformalPredictor

# ── Shared plot style ────────────────────────────────────────────────────────
_C_COV    = "#2166ac"   # coverage line  (blue)
_C_TARGET = "#d6604d"   # target line    (red-orange)
_C_WIDTH  = "#4dac26"   # width line     (green)

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =============================================================================
# Main experiment
# =============================================================================

def run_adaptive_finance_experiment(
    result,
    test_sector,
    cal_frac=0.5,
    alpha=0.1,
    seed=42,
    window_size=None,
    with_aci=False,
    gamma=0.005,
    adaptive_update=False,
    mixed=False,
    mixed_test_frac=0.15,
):
    """
    mixed=True  : ignore test_sector; randomly draw mixed_test_frac of all tickers
                  as test (no sector bias → null/no-shift baseline).
    mixed=False : hold out test_sector as test (sector covariate shift experiment).
    """
    Y       = result["Y"]
    dates   = result["dates"]
    meta    = result["meta"]
    tickers = result["tickers"]

    n_series, L, _ = Y.shape
    T = L - 1

    rng = np.random.default_rng(seed)

    if mixed:
        # ── Mixed mode: random test split, no sector filtering ────────────────
        n_test  = max(10, int(n_series * mixed_test_frac))
        all_idx = rng.permutation(n_series)
        test_mask           = np.zeros(n_series, dtype=bool)
        test_mask[all_idx[:n_test]] = True
        display_sector = "Mixed (all sectors)"
    else:
        test_mask = np.array([m["sector"].lower() == test_sector.lower() for m in meta])
        n_test    = int(test_mask.sum())
        display_sector = test_sector
        if n_test == 0:
            available = sorted({m["sector"] for m in meta})
            raise ValueError(
                f"No tickers found for sector '{test_sector}'.\nAvailable: {available}"
            )

    n_other = int((~test_mask).sum())
    if n_other == 0:
        raise ValueError("All tickers belong to the test set.")

    other_idx = rng.permutation(np.where(~test_mask)[0])
    n_cal     = int(n_other * cal_frac)
    n_train   = n_other - n_cal
    if n_train == 0:
        raise ValueError(f"cal_frac={cal_frac} leaves no training tickers.")

    train_idx = other_idx[:n_train]
    cal_idx   = other_idx[n_train:]
    test_idx  = np.where(test_mask)[0]

    Y_train = Y[train_idx]   # (n_train, L, 1)
    Y_cal   = Y[cal_idx]     # (n_cal,   L, 1)
    Y_test  = Y[test_idx]    # (n_test,  L, 1)
    test_tickers = [tickers[i] for i in test_idx]

    print(f"\n{'='*62}")
    print(f"  Finance Adaptive Experiment  (OnlineConformalPredictor)")
    print(f"{'='*62}")
    print(f"  Total tickers   : {n_series}")
    print(f"  Test set        : {display_sector}  ({n_test} tickers)")
    print(f"  Train           : {n_train} tickers")
    print(f"  Cal             : {n_cal} tickers")
    print(f"  Time steps      : {L}  [{dates[0]} -> {dates[-1]}]")
    print(f"  Alpha           : {alpha}  (target = {1-alpha:.0%})")
    print(f"  Window size     : {window_size if window_size else 'unlimited'}")
    print(f"  With ACI        : {with_aci}  (gamma={gamma})")
    print(f"  Adaptive update : {adaptive_update}")
    print()

    predictor = OnlineConformalPredictor(alpha=alpha, window_size=window_size)

    # Per-series ACI alpha (only used when with_aci=True)
    alpha_t = np.full(n_test, alpha, dtype=float)

    coverage_by_time = []
    width_by_time    = []
    all_covered      = []
    first_true       = []
    first_lower      = []
    first_upper      = []

    for t in range(T):

        # ── fit AR(1) on training data up to time t+1 ────────────────────────
        # Mirrors: linear_model.fit(Y_train[:, :t+2, :], X_train[:, :t+2, :])
        predictor.fit_ar_model(Y_train[:, :t+2, :])

        # ── build calibration scores ──────────────────────────────────────────
        # Accumulate AR(1) residuals over all calibration tickers and all steps
        # s = 1..t+1.  Mirrors finance_conformal.py's double loop over (i, s),
        # starting at s=1 because AR(1) requires a previous value (s=0 has none).
        cal_scores = []
        for i in range(n_cal):
            for s in range(1, t + 2):   # s = 1, 2, ..., t+1
                y_prev    = float(Y_cal[i, s - 1, 0])
                y_true_c  = float(Y_cal[i, s, 0])
                y_pred_c  = predictor.ar_intercept_ + predictor.ar_coef_ * y_prev
                cal_scores.append(abs(y_true_c - y_pred_c))

        # Inject scores directly into predictor (bypasses the fit loop in calibrate())
        predictor.conformity_scores = cal_scores

        # ── ACI state for this time step ──────────────────────────────────────
        if with_aci:
            alpha_used = alpha_t.copy()
            alpha_next = alpha_t.copy()

        covered_t = []
        width_t   = []

        for i in range(n_test):
            y_prev = float(Y_test[i, t, 0])
            y_true = float(Y_test[i, t + 1, 0])
            y_pred = predictor.ar_intercept_ + predictor.ar_coef_ * y_prev

            # ── quantile ──────────────────────────────────────────────────────
            if with_aci:
                a = float(np.clip(alpha_used[i], 1e-6, 1.0 - 1e-6))
                n = len(predictor.conformity_scores)
                q_level = np.ceil((n + 1) * (1.0 - a)) / n
                q_level = min(q_level, 1.0)
                q = (float(np.quantile(predictor.conformity_scores, q_level))
                     if predictor.conformity_scores
                     else 2.0 * predictor.model_params["noise_std"])
            else:
                q = predictor.get_current_quantile()

            lo, hi  = y_pred - q, y_pred + q
            covered = int(lo <= y_true <= hi)

            covered_t.append(covered)
            width_t.append(hi - lo)

            if i == 0:
                first_true.append(y_true)
                first_lower.append(lo)
                first_upper.append(hi)

            # ── ACI update ────────────────────────────────────────────────────
            if with_aci:
                err = 0 if covered else 1
                alpha_next[i] = alpha_used[i] + gamma * (alpha - err)

        if with_aci:
            alpha_t = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)

        # ── optional online update: append test residuals to score window ─────
        # This is the distinguishing "online" feature of OnlineConformalPredictor.
        # Each observed test residual is added to the window so future quantiles
        # reflect the actual test distribution, not just calibration.
        if adaptive_update:
            for i in range(n_test):
                y_prev    = float(Y_test[i, t, 0])
                y_true    = float(Y_test[i, t + 1, 0])
                y_pred_up = predictor.ar_intercept_ + predictor.ar_coef_ * y_prev
                new_score = abs(y_true - y_pred_up)
                predictor.conformity_scores.append(new_score)
                if window_size is not None and len(predictor.conformity_scores) > window_size:
                    predictor.conformity_scores.pop(0)

        coverage_by_time.append(float(np.mean(covered_t)))
        width_by_time.append(float(np.mean(width_t)))
        all_covered.extend(covered_t)

        if (t + 1) % 10 == 0 or t == T - 1:
            print(f"  [t={t+1:3d}/{T}]  coverage={np.mean(covered_t):.3f}  "
                  f"width={np.mean(width_t):.4f}")

    overall_coverage = float(np.mean(all_covered))
    target = 1.0 - alpha
    print(f"\n  Overall coverage : {overall_coverage:.4f}  "
          f"(target = {target:.4f},  error = {overall_coverage - target:+.4f})")
    print(f"  Mean width       : {float(np.mean(width_by_time)):.4f}")

    return {
        "coverage_by_time":  coverage_by_time,
        "width_by_time":     width_by_time,
        "overall_coverage":  overall_coverage,
        "target_coverage":   target,
        "dates":             [str(dates[t + 1]) for t in range(T)],
        "first_test_ticker": test_tickers[0],
        "first_test_series": {
            "true":  first_true,
            "lower": first_lower,
            "upper": first_upper,
        },
        "config": {
            "test_sector":      display_sector,
            "n_train":          int(n_train),
            "n_cal":            int(n_cal),
            "n_test":           int(n_test),
            "L":                int(L),
            "alpha":            alpha,
            "seed":             seed,
            "window_size":      window_size,
            "with_aci":         with_aci,
            "gamma":            gamma,
            "adaptive_update":  adaptive_update,
            "mixed":            mixed,
        },
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results, save_path=None):
    dates   = results["dates"]
    cov_t   = results["coverage_by_time"]
    width_t = results["width_by_time"]
    target  = results["target_coverage"]
    cfg     = results["config"]
    first   = results["first_test_series"]
    ticker  = results["first_test_ticker"]

    x          = np.arange(len(dates))
    aci_str    = "ACI" if cfg["with_aci"] else "no ACI"
    online_str = "  |  +online_update" if cfg["adaptive_update"] else ""
    mixed_tag  = "  |  mixed test set" if cfg.get("mixed") else ""
    fig, axes  = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"OnlineConformalPredictor  |  Test: {cfg['test_sector']}  |  "
        f"{aci_str}{online_str}{mixed_tag}\n"
        f"alpha={cfg['alpha']}  |  "
        f"train/cal/test = {cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} tickers",
        fontsize=11, fontweight="bold",
    )

    # ── Coverage over time ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, cov_t, color=_C_COV, linewidth=1.8, label="Empirical coverage")
    ax.axhline(target, color=_C_TARGET, linestyle="--", linewidth=1.8,
               label=f"Target ({target:.0%})")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Coverage rate", fontsize=10)
    ax.set_title(f"Coverage over Time  ({cfg['test_sector']})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.text(0.02, 0.05,
            f"Overall: {results['overall_coverage']:.3f}  "
            f"(error {results['overall_coverage'] - target:+.3f})",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    _style_ax(ax)

    # ── Width over time ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, width_t, color=_C_WIDTH, linewidth=1.8)
    ax.set_ylabel("Mean interval width", fontsize=10)
    ax.set_title("Prediction Interval Width over Time", fontsize=10)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── First ticker prediction interval ─────────────────────────────────────
    ax        = axes[1, 0]
    true_arr  = np.array(first["true"])
    lower_arr = np.array(first["lower"])
    upper_arr = np.array(first["upper"])
    y_min     = float(np.percentile(true_arr, 1))
    y_max     = float(np.percentile(true_arr, 99))
    margin    = (y_max - y_min) * 0.3
    ax.fill_between(x, lower_arr, upper_arr, alpha=0.25, color=_C_COV,
                    label="Prediction interval")
    ax.plot(x, true_arr, color="black", linewidth=1.5, label="Actual return")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_ylabel("Intraday return", fontsize=10)
    ax.set_title(f"{ticker}  —  Return vs. Prediction Interval", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── Cumulative coverage ───────────────────────────────────────────────────
    ax      = axes[1, 1]
    cum_cov = np.cumsum(cov_t) / np.arange(1, len(cov_t) + 1)
    ax.plot(x, cum_cov, color=_C_COV, linewidth=1.8, label="Cumulative coverage")
    ax.axhline(target, color=_C_TARGET, linestyle="--", linewidth=1.8,
               label=f"Target ({target:.0%})")
    ax.set_ylabel("Cumulative coverage rate", fontsize=10)
    ax.set_title("Cumulative Coverage", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── Shared x-axis ticks (dates) ───────────────────────────────────────────
    tick_every = max(1, len(dates) // 10)
    tick_pos   = x[::tick_every]
    tick_lbl   = [dates[i] for i in tick_pos]
    for row in axes:
        for ax in row:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("Date", fontsize=10)

    fig.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  [Plot] Saved to {out}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run OnlineConformalPredictor on S&P 500 finance data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--npz",             required=True,
                        help="Path to .npz data file")
    parser.add_argument("--json",            default=None,
                        help="Path to companion .json (inferred from --npz if omitted)")
    parser.add_argument("--test_sector",     default="Technology",
                        help="GICS sector to hold out as test set (ignored when --mixed is set). "
                             "Default: Technology")
    parser.add_argument("--cal_frac",        type=float, default=0.5)
    parser.add_argument("--alpha",           type=float, default=0.1)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--window_size",     type=int,   default=None,
                        help="Sliding window for adaptive_update mode (None = unlimited)")
    parser.add_argument("--with_aci",        action="store_true", default=False,
                        help="Enable per-series ACI alpha adjustment")
    parser.add_argument("--gamma",           type=float, default=0.005,
                        help="ACI step-size (only used with --with_aci)")
    parser.add_argument("--adaptive_update", action="store_true", default=False,
                        help="Append observed test residuals to score window each step")
    parser.add_argument("--mixed",           action="store_true", default=False,
                        help="Mixed-sector test set: randomly draw tickers from ALL "
                             "sectors as test (null/no-shift baseline). "
                             "Overrides --test_sector.")
    parser.add_argument("--mixed_test_frac", type=float, default=0.15,
                        help="Fraction of all tickers used as test in --mixed mode. "
                             "Default: 0.15")
    parser.add_argument("--save_plot",       default=None)
    parser.add_argument("--save_json",       default=None)
    args = parser.parse_args()

    npz_path  = Path(args.npz)
    json_path = Path(args.json) if args.json else npz_path.with_suffix(".json")

    print(f"Loading {npz_path} ...")
    result = load_stored(npz_path, json_path)

    results = run_adaptive_finance_experiment(
        result=result,
        test_sector=args.test_sector,
        cal_frac=args.cal_frac,
        alpha=args.alpha,
        seed=args.seed,
        window_size=args.window_size,
        with_aci=args.with_aci,
        gamma=args.gamma,
        adaptive_update=args.adaptive_update,
        mixed=args.mixed,
        mixed_test_frac=args.mixed_test_frac,
    )

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Results] Saved to {out}")

    plot_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()
