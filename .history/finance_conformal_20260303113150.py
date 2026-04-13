#!/usr/bin/env python3
"""
=============================================================================
FINANCE CONFORMAL PREDICTION
=============================================================================

Runs AdaptedCAFHT conformal prediction on real S&P 500 data loaded via
finance_data.py.

The split is by ticker (cross-sectional), mirroring the synthetic experiments:
  - Train tickers   : fit the linear regression model
  - Cal tickers     : compute conformity scores to set the quantile
  - Test tickers    : evaluate empirical coverage over time

The prediction model replaces the AR model with a linear regression:
  Close_t = β₀ + β₁·Open_t + β₂·OvernightGap_t + β₃·Volume_lag1_t + ...

USAGE:
  # First pull data (once):
  python finance_data.py --pull --start 2024-01-01 --end 2024-03-01

  # Then run conformal experiment:
  python finance_conformal.py --npz sp500_20240102_20240229.npz
  python finance_conformal.py --npz sp500_20240102_20240229.npz --alpha 0.1 --sector Technology
  python finance_conformal.py --npz sp500_20240102_20240229.npz --train_frac 0.6 --cal_frac 0.2

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
#  Linear regression model (replaces AR model in AdaptedCAFHT)
# ============================================================

class LinearCovariateModel:
    """
    Predicts Close_t as a linear combination of covariates at time t.

        Close_t ≈ β₀ + X_t @ β

    X_t may include same-day covariates (Open, OvernightGap) and
    lagged covariates (Volume_lag1, DailyRange_lag1, VWAPproxy_lag1).
    All are already correctly aligned in finance_data.py.

    Parameters
    ----------
    cov_names : list[str]
        Names of the covariate columns, in the same order as X[..., :].
    """

    def __init__(self, cov_names: list):
        self.cov_names = cov_names
        self.beta = None          # shape (n_cov + 1,)  — includes intercept
        self.noise_std = 1.0

    def fit(self, Y_train: np.ndarray, X_train: np.ndarray):
        """
        Fit via OLS.

        Parameters
        ----------
        Y_train : (n_series, L, 1)
        X_train : (n_series, L, n_cov)
        """
        n, L, _ = Y_train.shape
        n_cov = X_train.shape[2]

        # Flatten all (series, timestep) pairs
        y = Y_train[:, :, 0].reshape(-1)           # (n*L,)
        X = X_train.reshape(-1, n_cov)              # (n*L, n_cov)

        # Add intercept column
        ones = np.ones((X.shape[0], 1))
        X_design = np.hstack([ones, X])             # (n*L, n_cov+1)

        # OLS
        self.beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)

        # Residual std
        y_hat = X_design @ self.beta
        resid = y - y_hat
        self.noise_std = float(np.std(resid, ddof=X_design.shape[1]))

        print(f"  [Model] Fitted linear model on {n} series × {L} steps")
        print(f"  [Model] Residual std: {self.noise_std:.4f}")
        print(f"  [Model] Coefficients:")
        print(f"          intercept = {self.beta[0]:.4f}")
        for name, coef in zip(self.cov_names, self.beta[1:]):
            print(f"          {name:20s} = {coef:.4f}")

    def predict(self, x_t: np.ndarray) -> float:
        """
        Predict Close at a single time step.

        Parameters
        ----------
        x_t : (n_cov,) array — covariates at time t
        """
        if self.beta is None:
            raise RuntimeError("Model not fitted yet.")
        return float(self.beta[0] + x_t @ self.beta[1:])

    def residual_std(self) -> float:
        return self.noise_std


# ============================================================
#  Patched predictor — wraps AdaptedCAFHT with linear model
# ============================================================

class FinanceCAFHT(AdaptedCAFHT):
    """
    AdaptedCAFHT with the AR model replaced by a linear covariate model.

    Only the three methods that touch the model are overridden:
      - fit_ar_model  →  fit_linear_model
      - calibrate     →  uses X covariates for scores
      - predict_with_interval  →  uses X covariates for prediction
    """

    def __init__(self, alpha=0.1, cov_names=None):
        super().__init__(alpha=alpha)
        self.linear_model = LinearCovariateModel(cov_names or [])
        self._cal_X = None   # stored during calibrate() for convenience

    # ----------------------------------------------------------
    #  Model fitting
    # ----------------------------------------------------------

    def fit_linear_model(self, Y_train: np.ndarray, X_train: np.ndarray):
        """Fit the linear covariate model and sync noise_std into parent."""
        self.linear_model.fit(Y_train, X_train)
        # Keep noise_std in sync so fallback paths in parent work correctly
        self.noise_std = self.linear_model.noise_std

    # Override parent method so old call sites don't silently use AR
    def fit_ar_model(self, Y_subset):
        raise NotImplementedError(
            "Use fit_linear_model(Y_train, X_train) for finance data."
        )

    # ----------------------------------------------------------
    #  Calibration
    # ----------------------------------------------------------

    def calibrate_with_X(self, cal_Y: np.ndarray, cal_X: np.ndarray):
        """
        Compute conformity scores using the linear model.

        Parameters
        ----------
        cal_Y : (n_cal, L, 1)
        cal_X : (n_cal, L, n_cov)
        """
        n_cal, L, _ = cal_Y.shape
        scores = []

        for i in range(n_cal):
            for t in range(L):
                y_true = float(cal_Y[i, t, 0])
                x_t = cal_X[i, t, :]
                y_pred = self.linear_model.predict(x_t)
                scores.append(abs(y_true - y_pred))

        self._scores = np.array(scores, dtype=float)
        self._weights = np.ones_like(self._scores)
        self._q = None

        print(f"  [Cal]   Calibrated on {n_cal} series × {L} steps "
              f"({len(scores)} scores), "
              f"median score = {np.median(self._scores):.4f}")

    # ----------------------------------------------------------
    #  Prediction
    # ----------------------------------------------------------

    def predict_finance(self, x_t: np.ndarray, alpha_level: float = None) -> tuple:
        """
        Predict Close_t and return a conformal prediction interval.

        Parameters
        ----------
        x_t : (n_cov,) covariate vector at time t
        alpha_level : float, optional — overrides self.alpha (for ACI)

        Returns
        -------
        (pred, lower, upper)
        """
        pred = self.linear_model.predict(x_t)

        a = self.alpha if alpha_level is None else float(alpha_level)
        a = float(np.clip(a, 1e-6, 1.0 - 1e-6))

        if self._scores is None or self._scores.size == 0:
            q = 2.0 * self.noise_std
        else:
            q = self._weighted_quantile(self._scores, self._weights, 1.0 - a)

        return float(pred), float(pred - q), float(pred + q)


# ============================================================
#  Experiment runner
# ============================================================

def run_finance_experiment(
    result: dict,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
    alpha: float = 0.1,
    seed: int = 42,
    aci_stepsize: float = 0.005,
    use_aci: bool = False,
) -> dict:
    """
    Run conformal prediction experiment on finance data.

    Ticker split (cross-sectional):
      train_frac  →  fit linear model
      cal_frac    →  calibrate conformal quantile
      1 - train_frac - cal_frac  →  test (evaluate coverage)

    Parameters
    ----------
    result : dict
        Output of load_stored() or filter_by_*.
    train_frac, cal_frac : float
        Fractions of tickers for training and calibration.
    alpha : float
        Miscoverage level (target coverage = 1 - alpha).
    seed : int
        RNG seed for the ticker shuffle.
    aci_stepsize : float
        Step size for adaptive conformal (only used if use_aci=True).
    use_aci : bool
        If True, use ACI online alpha update during testing.

    Returns
    -------
    dict with keys: coverage_by_time, width_by_time, overall, config
    """
    Y = result["Y"]          # (n_series, L, 1)
    X = result["X"]          # (n_series, L, n_cov)
    dates = result["dates"]  # (L,)
    cov_names = result["cov_names"]

    n_series, L, _ = Y.shape
    rng = np.random.default_rng(seed)

    # ---- Ticker split ----------------------------------------
    idx = rng.permutation(n_series)
    n_train = int(n_series * train_frac)
    n_cal   = int(n_series * cal_frac)
    n_test  = n_series - n_train - n_cal

    if n_test <= 0:
        raise ValueError(
            f"No test series: n_series={n_series}, "
            f"train_frac={train_frac}, cal_frac={cal_frac}"
        )

    train_idx = idx[:n_train]
    cal_idx   = idx[n_train : n_train + n_cal]
    test_idx  = idx[n_train + n_cal :]

    Y_train, X_train = Y[train_idx], X[train_idx]
    Y_cal,   X_cal   = Y[cal_idx],   X[cal_idx]
    Y_test,  X_test  = Y[test_idx],  X[test_idx]

    print(f"\n{'='*62}")
    print(f"  Finance Conformal Experiment")
    print(f"{'='*62}")
    print(f"  Tickers total   : {n_series}")
    print(f"  Train tickers   : {n_train}  ({train_frac:.0%})")
    print(f"  Cal tickers     : {n_cal}    ({cal_frac:.0%})")
    print(f"  Test tickers    : {n_test}   ({1-train_frac-cal_frac:.0%})")
    print(f"  Time steps (L)  : {L}  [{dates[0]} → {dates[-1]}]")
    print(f"  Covariates      : {cov_names}")
    print(f"  Alpha           : {alpha}  (target coverage = {1-alpha:.0%})")
    print(f"  ACI             : {use_aci}")
    print()

    # ---- Fit -------------------------------------------------
    predictor = FinanceCAFHT(alpha=alpha, cov_names=cov_names)
    predictor.fit_linear_model(Y_train, X_train)

    # ---- Calibrate -------------------------------------------
    predictor.calibrate_with_X(Y_cal, X_cal)

    # ---- Test: evaluate coverage at each time step -----------
    print(f"\n  Running test on {n_test} series × {L} time steps...")

    coverage_by_time = []   # fraction of test series covered at each t
    width_by_time    = []   # mean interval width at each t
    all_covered      = []   # flat list of 0/1 coverage indicators

    # ACI: per-series alpha tracking
    alpha_t = np.full(n_test, alpha)

    for t in range(L):
        covered_t = []
        width_t   = []

        for i in range(n_test):
            x_t   = X_test[i, t, :]
            y_true = float(Y_test[i, t, 0])

            a = float(alpha_t[i]) if use_aci else alpha
            pred, lo, hi = predictor.predict_finance(x_t, alpha_level=a)

            covered = int(lo <= y_true <= hi)
            covered_t.append(covered)
            width_t.append(hi - lo)

            # ACI update
            if use_aci:
                alpha_t[i] += aci_stepsize * (alpha - (1 - covered))

        coverage_by_time.append(float(np.mean(covered_t)))
        width_by_time.append(float(np.mean(width_t)))
        all_covered.extend(covered_t)

    overall_coverage = float(np.mean(all_covered))
    target = 1.0 - alpha

    print(f"\n  Overall coverage : {overall_coverage:.4f}  "
          f"(target = {target:.4f}, "
          f"error = {overall_coverage - target:+.4f})")
    print(f"  Mean width       : {np.mean(width_by_time):.4f}")

    return {
        "coverage_by_time": coverage_by_time,
        "width_by_time":    width_by_time,
        "overall_coverage": overall_coverage,
        "target_coverage":  target,
        "dates":            list(dates),
        "config": {
            "n_train": int(n_train),
            "n_cal":   int(n_cal),
            "n_test":  int(n_test),
            "L":       int(L),
            "alpha":   alpha,
            "use_aci": use_aci,
            "seed":    seed,
            "cov_names": cov_names,
        },
    }


# ============================================================
#  Plotting
# ============================================================

def plot_results(results: dict, save_path: str = None):
    """Plot coverage and interval width over time."""
    dates    = results["dates"]
    cov_t    = results["coverage_by_time"]
    width_t  = results["width_by_time"]
    target   = results["target_coverage"]
    cfg      = results["config"]

    # Use integer indices if dates can't be parsed cleanly
    x = np.arange(len(dates))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(
        f"Finance Conformal Prediction  |  "
        f"α={cfg['alpha']}  |  "
        f"train/cal/test = {cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} tickers  |  "
        f"ACI={cfg['use_aci']}",
        fontsize=12, fontweight='bold'
    )

    # --- Coverage ---
    axes[0].plot(x, cov_t, 'b-', linewidth=1.5, label='Empirical coverage')
    axes[0].axhline(target, color='red', linestyle='--', linewidth=2,
                    label=f'Target ({target:.0%})')

    # Rolling 10-day average
    if len(cov_t) >= 10:
        roll = np.convolve(cov_t, np.ones(10)/10, mode='valid')
        axes[0].plot(x[9:], roll, 'navy', linewidth=2, alpha=0.6,
                     label='10-day rolling avg')

    axes[0].set_ylim(0.5, 1.05)
    axes[0].set_ylabel('Coverage rate')
    axes[0].set_title('Coverage over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Annotate overall coverage
    axes[0].text(
        0.02, 0.05,
        f"Overall: {results['overall_coverage']:.3f}  "
        f"(error {results['overall_coverage']-target:+.3f})",
        transform=axes[0].transAxes, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    # --- Width ---
    axes[1].plot(x, width_t, 'g-', linewidth=1.5)
    axes[1].set_ylabel('Mean interval width')
    axes[1].set_title('Prediction Interval Width over Time')
    axes[1].grid(True, alpha=0.3)

    # Tick labels: show ~10 evenly spaced dates
    tick_every = max(1, len(dates) // 10)
    tick_positions = x[::tick_every]
    tick_labels    = [dates[i] for i in tick_positions]
    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
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
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    parser.add_argument("--npz", required=True,
                        help="Path to .npz file saved by finance_data.py")
    parser.add_argument("--json", default=None,
                        help="Path to companion .json (inferred from --npz if omitted)")
    parser.add_argument("--sector",   default=None,
                        help="Filter to a single sector, e.g. 'Technology'")
    parser.add_argument("--industry", default=None,
                        help="Filter to a single industry, e.g. 'Semiconductors'")

    # Split
    parser.add_argument("--train_frac", type=float, default=0.6,
                        help="Fraction of tickers for training (default: 0.6)")
    parser.add_argument("--cal_frac", type=float, default=0.2,
                        help="Fraction of tickers for calibration (default: 0.2)")

    # Conformal
    parser.add_argument("--alpha", type=float, default=0.1,
                        help="Miscoverage level (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for ticker shuffle (default: 42)")
    parser.add_argument("--aci", action="store_true",
                        help="Use adaptive conformal inference (ACI) during testing")
    parser.add_argument("--aci_stepsize", type=float, default=0.005,
                        help="ACI step size (default: 0.005)")

    # Output
    parser.add_argument("--save_plot", default=None,
                        help="Path to save the plot (e.g. results/finance_plot.png)")
    parser.add_argument("--save_json", default=None,
                        help="Path to save results JSON")

    args = parser.parse_args()

    # ---- Load data -------------------------------------------
    npz_path  = Path(args.npz)
    json_path = Path(args.json) if args.json else npz_path.with_suffix(".json")

    print(f"Loading data from {npz_path} ...")
    result = load_stored(npz_path, json_path)

    # Optional filtering
    if args.sector:
        print(f"Filtering to sector: {args.sector}")
        result = filter_by_sector(result, [args.sector])
    if args.industry:
        print(f"Filtering to industry: {args.industry}")
        result = filter_by_industry(result, [args.industry])

    # ---- Run experiment --------------------------------------
    results = run_finance_experiment(
        result=result,
        train_frac=args.train_frac,
        cal_frac=args.cal_frac,
        alpha=args.alpha,
        seed=args.seed,
        aci_stepsize=args.aci_stepsize,
        use_aci=args.aci,
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