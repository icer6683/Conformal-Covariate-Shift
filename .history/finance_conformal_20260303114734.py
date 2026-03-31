#!/usr/bin/env python3
"""
=============================================================================
FINANCE CONFORMAL PREDICTION
=============================================================================

Runs AdaptedCAFHT conformal prediction on real S&P 500 data loaded via
finance_data.py.

The only difference from the synthetic experiments is that fit_ar_model()
is replaced by fit_linear_model(), which predicts:

    Close_t ≈ β₀ + β₁·Open_t + β₂·OvernightGap_t + β₃·Volume_lag1_t + ...

The split is cross-sectional (by ticker):
  - Train tickers  : fit the linear regression model
  - Cal tickers    : compute conformity scores to set the quantile
  - Test tickers   : evaluate empirical coverage over time

USAGE:
  # First pull data (once):
  python finance_data.py --pull --start 2024-01-01 --end 2024-04-01

  # Then run:
  python finance_conformal.py --npz sp500_20240102_20240229.npz 
  python finance_conformal.py --npz sp500_20240102_20240229.npz  --sector Technology
  python finance_conformal.py --npz sp500_20240102_20240229.npz  --alpha 0.05

=============================================================================
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from finance_data import load_stored, filter_by_sector, filter_by_industry
from algorithm import AdaptedCAFHT


# ============================================================
#  Linear regression model (drop-in replacement for AR model)
# ============================================================

class LinearCovariateModel:
    """
    Fits Close_t = β₀ + X_t @ β  via OLS across all train (series, timestep) pairs.

    Covariates in X are already correctly aligned in finance_data.py:
      - Open, OvernightGap : same-day, no lookahead
      - Volume_lag1, DailyRange_lag1, VWAPproxy_lag1 : lagged, no lookahead

    After fitting, noise_std is set on the AdaptedCAFHT instance so that
    fallback paths (empty calibration set) behave sensibly.
    """

    def __init__(self, cov_names: list):
        self.cov_names = cov_names
        self.beta = None        # (n_cov + 1,) — intercept first
        self.noise_std = 1.0

    def fit(self, Y_train: np.ndarray, X_train: np.ndarray):
        """
        Parameters
        ----------
        Y_train : (n_train, L, 1)
        X_train : (n_train, L, n_cov)
        """
        n, L, n_cov = X_train.shape

        y = Y_train[:, :, 0].reshape(-1)                    # (n*L,)
        X = X_train.reshape(-1, n_cov)                      # (n*L, n_cov)
        X_design = np.hstack([np.ones((len(y), 1)), X])     # (n*L, n_cov+1)

        self.beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        resid = y - X_design @ self.beta
        self.noise_std = float(np.std(resid, ddof=X_design.shape[1]))

        print(f"  [Model] Fitted on {n} series × {L} steps")
        print(f"  [Model] Residual std : {self.noise_std:.4f}")
        print(f"  [Model] Coefficients :")
        print(f"            intercept = {self.beta[0]:.4f}")
        for name, coef in zip(self.cov_names, self.beta[1:]):
            print(f"            {name:22s} = {coef:.4f}")

    def predict(self, x_t: np.ndarray) -> float:
        """Predict Close for a single covariate vector x_t of shape (n_cov,)."""
        return float(self.beta[0] + x_t @ self.beta[1:])


# ============================================================
#  Experiment runner
# ============================================================

def run_finance_experiment(
    result: dict,
    test_sector: str,
    cal_frac: float = 0.5,
    alpha: float = 0.1,
    seed: int = 42,
    aci_stepsize: float = 0.005,
) -> dict:
    """
    Run AdaptedCAFHT on finance data with a linear covariate model.

    Parameters
    ----------
    result       : dict from load_stored() or filter_by_*()
    test_sector   : fraction of tickers used to fit the linear model
    cal_frac     : fraction of tickers used to calibrate conformal scores
    alpha        : miscoverage level (target coverage = 1 - alpha)
    seed         : RNG seed for ticker shuffle
    aci_stepsize : ACI step size (mirrors algorithm.py usage)

    Returns
    -------
    dict with keys: coverage_by_time, width_by_time, overall_coverage,
                    target_coverage, dates, config
    """
    Y         = result["Y"]           # (n_series, L, 1)
    X         = result["X"]           # (n_series, L, n_cov)
    dates     = result["dates"]       # (L,)
    cov_names = result["cov_names"]

    n_series, L, _ = Y.shape
    rng = np.random.default_rng(seed)

    # ---- Ticker split ----------------------------------------
    idx     = rng.permutation(n_series)
    n_train = int(n_series * train_frac)
    n_cal   = int(n_series * cal_frac)
    n_test  = n_series - n_train - n_cal

    if n_test <= 0:
        raise ValueError(
            f"No test tickers left: n_series={n_series}, "
            f"train_frac={train_frac}, cal_frac={cal_frac}."
        )

    train_idx = idx[:n_train]
    cal_idx   = idx[n_train : n_train + n_cal]
    test_idx  = idx[n_train + n_cal :]

    Y_train, X_train = Y[train_idx], X[train_idx]
    Y_cal,   X_cal   = Y[cal_idx],   X[cal_idx]
    Y_test,  X_test  = Y[test_idx],  X[test_idx]

    print(f"\n{'='*62}")
    print(f"  Finance Conformal Experiment  (AdaptedCAFHT)")
    print(f"{'='*62}")
    print(f"  Total tickers   : {n_series}")
    print(f"  Train           : {n_train}  ({train_frac:.0%})")
    print(f"  Cal             : {n_cal}    ({cal_frac:.0%})")
    print(f"  Test            : {n_test}   ({1-train_frac-cal_frac:.0%})")
    print(f"  Time steps      : {L}  [{dates[0]} → {dates[-1]}]")
    print(f"  Covariates      : {cov_names}")
    print(f"  Alpha           : {alpha}  (target = {1-alpha:.0%})")
    print(f"  ACI stepsize    : {aci_stepsize}")
    print()

    # ---- Step 1: Fit linear model ----------------------------
    linear_model = LinearCovariateModel(cov_names)
    linear_model.fit(Y_train, X_train)

    # ---- Step 2: Initialise AdaptedCAFHT --------------------
    # We use AdaptedCAFHT exactly as in the synthetic experiments.
    # The only change: conformity scores come from the linear model
    # rather than from fit_ar_model(). Everything else — _weighted_quantile,
    # ACI alpha updates, density-ratio weighting — is inherited unchanged.
    predictor = AdaptedCAFHT(alpha=alpha)
    predictor.noise_std = linear_model.noise_std   # keeps fallback sensible

    # ---- Step 3: Calibrate -----------------------------------
    # Scores: |Close_t - linear_pred_t| over all (cal_ticker, timestep) pairs.
    cal_scores = []
    for i in range(n_cal):
        for t in range(L):
            y_true = float(Y_cal[i, t, 0])
            y_pred = linear_model.predict(X_cal[i, t, :])
            cal_scores.append(abs(y_true - y_pred))

    predictor._scores  = np.array(cal_scores, dtype=float)
    predictor._weights = np.ones(len(cal_scores), dtype=float)
    predictor._q       = None   # computed on-demand inside _weighted_quantile

    print(f"  [Cal]  {n_cal} series × {L} steps = {len(cal_scores)} scores, "
          f"median = {np.median(predictor._scores):.4f}")

    # ---- Step 4: Test with ACI alpha updates -----------------
    print(f"  [Test] Running on {n_test} series × {L} time steps...")

    # Per-series ACI alpha tracking — identical to algorithm.py usage
    alpha_t = np.full(n_test, alpha, dtype=float)

    coverage_by_time = []
    width_by_time    = []
    all_covered      = []

    for t in range(L):
        covered_t = []
        width_t   = []

        for i in range(n_test):
            x_t    = X_test[i, t, :]
            y_true = float(Y_test[i, t, 0])
            y_pred = linear_model.predict(x_t)

            # Conformal quantile at this series' current alpha level
            a = float(np.clip(alpha_t[i], 1e-6, 1 - 1e-6))
            q = predictor._weighted_quantile(
                predictor._scores, predictor._weights, 1.0 - a
            )

            lo = y_pred - q
            hi = y_pred + q

            covered = int(lo <= y_true <= hi)
            covered_t.append(covered)
            width_t.append(hi - lo)

            # ACI update: nudge alpha toward achieving target coverage
            alpha_t[i] += aci_stepsize * (alpha - (1 - covered))

        coverage_by_time.append(float(np.mean(covered_t)))
        width_by_time.append(float(np.mean(width_t)))
        all_covered.extend(covered_t)

    overall_coverage = float(np.mean(all_covered))
    target = 1.0 - alpha

    print(f"\n  Overall coverage : {overall_coverage:.4f}  "
          f"(target = {target:.4f},  error = {overall_coverage - target:+.4f})")
    print(f"  Mean width       : {np.mean(width_by_time):.4f}")

    return {
        "coverage_by_time": coverage_by_time,
        "width_by_time":    width_by_time,
        "overall_coverage": overall_coverage,
        "target_coverage":  target,
        "dates":            [str(d) for d in dates],
        "config": {
            "n_train":      int(n_train),
            "n_cal":        int(n_cal),
            "n_test":       int(n_test),
            "L":            int(L),
            "alpha":        alpha,
            "aci_stepsize": aci_stepsize,
            "seed":         seed,
            "cov_names":    cov_names,
        },
    }


# ============================================================
#  Plotting
# ============================================================

def plot_results(results: dict, save_path: str = None):
    dates   = results["dates"]
    cov_t   = results["coverage_by_time"]
    width_t = results["width_by_time"]
    target  = results["target_coverage"]
    cfg     = results["config"]

    x = np.arange(len(dates))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Finance Conformal Prediction (AdaptedCAFHT)  |  "
        f"α={cfg['alpha']}  |  "
        f"train/cal/test = {cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} tickers",
        fontsize=12, fontweight='bold'
    )

    # --- Coverage ---
    axes[0].plot(x, cov_t, 'b-', linewidth=1.5, label='Empirical coverage')
    axes[0].axhline(target, color='red', linestyle='--', linewidth=2,
                    label=f'Target ({target:.0%})')
    if len(cov_t) >= 10:
        roll = np.convolve(cov_t, np.ones(10) / 10, mode='valid')
        axes[0].plot(x[9:], roll, 'navy', linewidth=2, alpha=0.6,
                     label='10-day rolling avg')
    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel('Coverage rate')
    axes[0].set_title('Coverage over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].text(
        0.02, 0.05,
        f"Overall: {results['overall_coverage']:.3f}  "
        f"(error {results['overall_coverage'] - target:+.3f})",
        transform=axes[0].transAxes, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    # --- Width ---
    axes[1].plot(x, width_t, 'g-', linewidth=1.5)
    axes[1].set_ylabel('Mean interval width')
    axes[1].set_title('Prediction Interval Width over Time')
    axes[1].grid(True, alpha=0.3)

    # Shared x-axis date labels
    tick_every = max(1, len(dates) // 10)
    tick_pos   = x[::tick_every]
    tick_lbl   = [dates[i] for i in tick_pos]
    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Date')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved to {save_path}")

    plt.show()


# ============================================================
#  CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run AdaptedCAFHT conformal prediction on S&P 500 finance data.",
    )

    # Data
    parser.add_argument("--npz", required=True,
                        help="Path to .npz file saved by finance_data.py")
    parser.add_argument("--json", default=None,
                        help="Companion .json file (inferred from --npz if omitted)")
    parser.add_argument("--sector",   default=None,
                        help="Filter to a sector, e.g. 'Technology'")
    parser.add_argument("--industry", default=None,
                        help="Filter to an industry, e.g. 'Semiconductors'")

    # Split
    parser.add_argument("--train_frac", type=float, default=0.6,
                        help="Fraction of tickers for training (default: 0.6)")
    parser.add_argument("--cal_frac",   type=float, default=0.2,
                        help="Fraction of tickers for calibration (default: 0.2)")

    # Conformal
    parser.add_argument("--alpha",        type=float, default=0.1,
                        help="Miscoverage level (default: 0.1)")
    parser.add_argument("--aci_stepsize", type=float, default=0.005,
                        help="ACI step size (default: 0.005)")
    parser.add_argument("--seed",         type=int,   default=42,
                        help="RNG seed for ticker shuffle (default: 42)")

    # Output
    parser.add_argument("--save_plot", default=None,
                        help="Path to save the plot PNG")
    parser.add_argument("--save_json", default=None,
                        help="Path to save results JSON")

    args = parser.parse_args()

    # ---- Load ------------------------------------------------
    npz_path  = Path(args.npz)
    json_path = Path(args.json) if args.json else npz_path.with_suffix(".json")
    print(f"Loading {npz_path} ...")
    result = load_stored(npz_path, json_path)

    if args.sector:
        print(f"Filtering to sector: {args.sector}")
        result = filter_by_sector(result, [args.sector])
    if args.industry:
        print(f"Filtering to industry: {args.industry}")
        result = filter_by_industry(result, [args.industry])

    # ---- Run -------------------------------------------------
    results = run_finance_experiment(
        result       = result,
        train_frac   = args.train_frac,
        cal_frac     = args.cal_frac,
        alpha        = args.alpha,
        seed         = args.seed,
        aci_stepsize = args.aci_stepsize,
    )

    # ---- Save JSON -------------------------------------------
    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Results] Saved to {out}")

    # ---- Plot ------------------------------------------------
    plot_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()