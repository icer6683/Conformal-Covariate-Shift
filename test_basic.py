#!/usr/bin/env python3
"""
=============================================================================
TEST BASIC CONFORMAL COVERAGE 
=============================================================================

PURPOSE:
  Generate data using ts_generator.py and test the *basic* conformal predictor
  (AR(1) on Y only, no shift correction), with coverage visualization.
  Supports two covariate modes:
    - static  : time-invariant X (Poisson)
    - dynamic : time-varying X_t following its own AR(1)

WHAT THIS SHOWS:
  Since the basic method ignores X (both static and dynamic), coverage will
  typically degrade as β grows and/or under covariate shift.

USAGE:
  # default (static X, no shift)
  python test_basic.py

  # tighter α and more series
  python test_basic.py --alpha 0.05 --n_series 1000

  # dynamic X with test-set covariate shift
  python test_basic.py --covariate_mode dynamic --with_shift \
      --x_rate 0.6 --x_rate_shift 0.9 --beta 1.0

EXAMPLE USAGE:

static + basic + no shift:
    python test_basic.py --n_series 1000
static + basic + with shift:
    python test_basic.py --with_shift --n_series 1000

dynamic + basic + no shift:
    python test_basic.py --n_series 1000 --covariate_mode dynamic
dynamic + basic + with shift:
    python test_basic.py --with_shift --n_series 1000 --covariate_mode dynamic

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator
from basic_conformal import BasicConformalPredictor


def run_coverage_experiment(
    generator: TimeSeriesGenerator,
    predictor: BasicConformalPredictor,
    *,
    n_series: int = 500,
    covariate_mode: str = "static",
    # Y model params
    ar_coef: float = 0.7,
    beta: float = 1.0,
    noise_std: float = 0.2,
    trend_coef: float = 0.0,
    # Static-X generation params
    covar_rate: float = 1.0,
    covar_rate_shift: float | None = None,
    # Dynamic-X generation params
    x_rate: float = 0.7,
    x_trend: float = 0.0,
    x_noise_std: float = 0.2,
    x0_lambda: float = 1.0,
    # Dynamic-X shift params (if None, default to generation values)
    x_rate_shift: float | None = None,
    x_trend_shift: float | None = None,
    x_noise_std_shift: float | None = None,
    x0_lambda_shift: float | None = None,
    # Experiment options
    with_shift: bool = False,
    n_train: int = 200,
    n_cal: int = 100,
):
    """
    Run a coverage experiment (fit on original distribution, optionally test under shift).

    Args:
        generator: TimeSeriesGenerator configured with T, d=1, seed.
        predictor: BasicConformalPredictor instance (AR(1) on Y only).
        n_series:  Number of test series to evaluate.
        covariate_mode: 'static' or 'dynamic'.
        ar_coef, beta, noise_std, trend_coef: Y process parameters.
        covar_rate:        (static) Poisson rate for X on train/cal/test generation.
        covar_rate_shift:  (static) Poisson rate for X under TEST shift (if with_shift).
        x_rate, x_trend, x_noise_std, x0_lambda: (dynamic) X_t AR(1) generation params.
        x_rate_shift, x_trend_shift, x_noise_std_shift, x0_lambda_shift:
            (dynamic) X_t AR(1) params under TEST shift (defaults to gen values if None).
        with_shift: If True, apply covariate shift on the TEST series only.
        n_train, n_cal: Sizes for training and calibration sets.

    Returns:
        coverage_history: np.ndarray of shape (n_series,), 1=covered, 0=miss
        interval_widths:  np.ndarray of shape (n_series,)
    """
    mode = covariate_mode.lower()
    assert mode in {"static", "dynamic"}, "covariate_mode must be 'static' or 'dynamic'"

    if with_shift:
        print(f"Running coverage experiment with TEST covariate shift on {n_series} series...")
    else:
        print(f"Running coverage experiment on {n_series} series (NO TEST SHIFT)...")
    print(f"Covariate mode: {mode}")

    # -----------------------
    # Generate training data (NO SHIFT)
    # -----------------------
    print("Generating training data...")
    if mode == "static":
        train_Y, _ = generator.generate_with_poisson_covariate(
            n=n_train,
            ar_coef=ar_coef,
            beta=beta,
            covar_rate=covar_rate,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
        )
    else:
        train_Y, _ = generator.generate_with_dynamic_covariate(
            n=n_train,
            ar_coef=ar_coef,
            beta=beta,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
            x_rate=x_rate,
            x_trend=x_trend,
            x_noise_std=x_noise_std,
            x0_lambda=x0_lambda,
        )

    # -----------------------
    # Generate calibration data (NO SHIFT)
    # -----------------------
    print("Generating calibration data...")
    if mode == "static":
        cal_Y, _ = generator.generate_with_poisson_covariate(
            n=n_cal,
            ar_coef=ar_coef,
            beta=beta,
            covar_rate=covar_rate,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
        )
    else:
        cal_Y, _ = generator.generate_with_dynamic_covariate(
            n=n_cal,
            ar_coef=ar_coef,
            beta=beta,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
            x_rate=x_rate,
            x_trend=x_trend,
            x_noise_std=x_noise_std,
            x0_lambda=x0_lambda,
        )

    # -----------------------
    # Fit model & calibrate
    # -----------------------
    print("Fitting AR(1) model on training data...")
    predictor.fit_ar_model(train_Y)

    print("Calibrating conformal predictor on calibration data...")
    predictor.calibrate(cal_Y)

    # -----------------------
    # Test series loop
    # -----------------------
    print("Evaluating coverage on TEST data...")
    coverage_history = []
    interval_widths = []

    # Resolve shift params defaults for dynamic mode
    xr_s = x_rate if x_rate_shift is None else x_rate_shift
    xt_s = x_trend if x_trend_shift is None else x_trend_shift
    xn_s = x_noise_std if x_noise_std_shift is None else x_noise_std_shift
    xl_s = x0_lambda if x0_lambda_shift is None else x0_lambda_shift

    for i in range(n_series):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_series} series...")

        # Generate one original test series (NO SHIFT) in the chosen mode
        if mode == "static":
            Y_orig, X_orig = generator.generate_with_poisson_covariate(
                n=1,
                ar_coef=ar_coef,
                beta=beta,
                covar_rate=covar_rate,
                noise_std=noise_std,
                initial_mean=0.0,
                initial_std=1.0,
                trend_coef=trend_coef,
            )
        else:
            Y_orig, X_orig = generator.generate_with_dynamic_covariate(
                n=1,
                ar_coef=ar_coef,
                beta=beta,
                noise_std=noise_std,
                initial_mean=0.0,
                initial_std=1.0,
                trend_coef=trend_coef,
                x_rate=x_rate,
                x_trend=x_trend,
                x_noise_std=x_noise_std,
                x0_lambda=x0_lambda,
            )

        # Optionally apply TEST covariate shift via unified function
        if with_shift:
            if mode == "static":
                Y_shift, _ = generator.introduce_covariate_shift(
                    original_Y=Y_orig,
                    original_X=X_orig,  # shape (1,)
                    covariate_mode="static",
                    model_params={
                        "ar_coef": ar_coef,
                        "beta": beta,
                        "noise_std": noise_std,
                        "trend_coef": trend_coef,
                    },
                    shift_params={"shift_rate": covar_rate_shift if covar_rate_shift is not None else 3.0},
                )
            else:
                Y_shift, _ = generator.introduce_covariate_shift(
                    original_Y=Y_orig,
                    original_X=X_orig,  # shape (1, T+1)
                    covariate_mode="dynamic",
                    model_params={
                        "ar_coef": ar_coef,
                        "beta": beta,
                        "noise_std": noise_std,
                        "trend_coef": trend_coef,
                    },
                    shift_params={
                        "x_rate_shift": xr_s,
                        "x_trend_shift": xt_s,
                        "x_noise_std_shift": xn_s,
                        "x0_lambda_shift": xl_s,
                        "x0_redraw": True,  # per spec
                    },
                )
            test_series = Y_shift[0]  # (T+1, 1)
        else:
            test_series = Y_orig[0]   # (T+1, 1)

        # Conformal prediction for last step
        pred, lower, upper = predictor.predict_with_interval(test_series[:-1])
        true_value = test_series[-1, 0]

        covered = (lower <= true_value <= upper)
        coverage_history.append(1 if covered else 0)
        interval_widths.append(upper - lower)

    coverage_history = np.array(coverage_history)
    interval_widths = np.array(interval_widths)

    return coverage_history, interval_widths




def compute_moving_average_coverage(coverage_history, window_size=100):
    """
    Compute moving-average coverage from a binary 1/0 array.

    - Clamps window_size to [1, n] to avoid empty windows.
    - Returns an empty array if coverage_history is empty.
    - Uses convolution for efficiency.
    """
    cov = np.asarray(coverage_history, dtype=float)
    n = cov.size
    if n == 0:
        return np.array([], dtype=float)

    w = max(1, min(int(window_size), n))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(cov, kernel, mode='valid')

def plot_coverage_like_notebook(
    coverage_history,
    target_coverage,
    window_size=100,
    covariate_mode="static",
    with_shift=False,
    title=None
):
    """
    Plot moving-average coverage vs. the target coverage.

    Title auto-includes covariate mode ("Static X" or "Dynamic X_t")
    and whether test shift is applied ("with Shift" / "no Shift").

    - Uses effective window size w = clamp(window_size, 1, n).
    - Handles empty inputs gracefully.
    - Y-limits adapt to data (clamped to [0, 1]).
    """
    cov = np.asarray(coverage_history, dtype=float)
    n = cov.size
    if n == 0:
        print("Warning: coverage_history is empty; nothing to plot.")
        return np.array([], dtype=float)

    # Effective window used by the computation
    w = max(1, min(int(window_size), n))
    moving_coverage = compute_moving_average_coverage(cov, w)
    if moving_coverage.size == 0:
        print("Warning: not enough data for the chosen window; nothing to plot.")
        return moving_coverage

    # Build contextual title if not provided
    mode_str = "Dynamic $X_t$" if str(covariate_mode).lower() == "dynamic" else "Static X"
    shift_str = "with Shift" if with_shift == True else "no Shift"
    if title is None:
        title = f"Basic Conformal Prediction Coverage — {mode_str}, {shift_str}"

    time_indices = np.arange(moving_coverage.size)

    plt.figure(figsize=(12, 6))

    # Moving average coverage
    plt.plot(time_indices, moving_coverage, 'b-', linewidth=2,
             label=f'Moving Coverage (window={w})')

    # Target coverage line
    plt.axhline(y=target_coverage, color='red', linestyle='--', linewidth=2,
                label=f'Target Coverage ({target_coverage:.1%})')

    # Formatting
    plt.xlabel('Time Series Index')
    plt.ylabel('Coverage Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Adaptive y-limits (clamped to [0, 1])
    ymin, ymax = moving_coverage.min(), moving_coverage.max()
    pad = 0.05
    y_low  = max(0.0, ymin - pad)
    y_high = min(1.0, ymax + pad)
    if y_high - y_low < 0.2:  # ensure a minimum span
        mid = 0.5 * (y_low + y_high)
        y_low  = max(0.0, mid - 0.1)
        y_high = min(1.0, mid + 0.1)
    plt.ylim([y_low, y_high])

    # Stats box
    mean_cov = float(np.mean(moving_coverage))
    std_cov  = float(np.std(moving_coverage))
    plt.text(0.02, 0.02, f'Mean: {mean_cov:.1%}\nStd: {std_cov:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return moving_coverage



def main():
    parser = argparse.ArgumentParser(description='Test basic conformal prediction coverage')
    # Experiment sizes & basics
    parser.add_argument('--n_series', type=int, default=500, help='Number of test series')
    parser.add_argument('--n_train',  type=int, default=200, help='Number of training series')
    parser.add_argument('--n_cal',    type=int, default=100, help='Number of calibration series')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--window_size', type=int, default=100, help='Moving average window')
    parser.add_argument('--T', type=int, default=25, help='Time series length (T+1 points)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Y model params
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR(1) coefficient for Y')
    parser.add_argument('--beta', type=float, default=1.0, help='Covariate effect β on Y')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Std dev of Y noise')
    parser.add_argument('--trend_coef', type=float, default=0.0, help='Linear trend for Y')

    # Covariate mode
    parser.add_argument('--covariate_mode', choices=['static', 'dynamic'], default='static',
                        help='Use time-invariant X (static) or time-varying X_t (dynamic)')
    parser.add_argument('--with_shift', action='store_true',
                        help='Apply covariate shift on the TEST set')

    # Static-X params (generation + shift)
    parser.add_argument('--covar_rate', type=float, default=1.0, help='Poisson rate for X (static mode)')
    parser.add_argument('--covar_rate_shift', type=float, default=3.0,
                        help='Shifted Poisson rate for TEST X (static mode)')

    # Dynamic-X params (generation)
    parser.add_argument('--x_rate',      type=float, default=0.7, help='ρ_X for dynamic X_t')
    parser.add_argument('--x_trend',     type=float, default=0.0, help='trend_X for dynamic X_t')
    parser.add_argument('--x_noise_std', type=float, default=0.2, help='Std dev of η_t for X_t')
    parser.add_argument('--x0_lambda',   type=float, default=1.0, help='Poisson rate for X₀ (dynamic)')

    # Dynamic-X params (shift) — default to generation values if not provided
    parser.add_argument('--x_rate_shift',      type=float, default=None, help='Shifted ρ_X for TEST X_t')
    parser.add_argument('--x_trend_shift',     type=float, default=None, help='Shifted trend_X for TEST X_t')
    parser.add_argument('--x_noise_std_shift', type=float, default=None, help='Shifted X noise std for TEST')
    parser.add_argument('--x0_lambda_shift',   type=float, default=None, help='Shifted Poisson rate for X₀ (TEST)')

    args = parser.parse_args()

    print("BASIC CONFORMAL PREDICTION COVERAGE TEST")
    print("="*50)
    print("Replicating code.ipynb-style coverage analysis with new generator…")
    print("Parameters:")
    print(f"  Target coverage : {1-args.alpha:.1%} (α={args.alpha})")
    print(f"  Series (train/cal/test): {args.n_train}/{args.n_cal}/{args.n_series}")
    print(f"  Series length    : {args.T + 1}")
    print(f"  AR coef (Y)      : {args.ar_coef}")
    print(f"  β (X→Y)          : {args.beta}")
    print(f"  Y noise std      : {args.noise_std}")
    print(f"  Y trend          : {args.trend_coef}")
    print(f"  Covariate mode   : {args.covariate_mode}")
    if args.covariate_mode == 'static':
        print(f"  X (static) rate  : {args.covar_rate}"
              f"{' → '+str(args.covar_rate_shift) if args.with_shift else ''}")
    else:
        xr_s = args.x_rate if args.x_rate_shift is None else args.x_rate_shift
        xt_s = args.x_trend if args.x_trend_shift is None else args.x_trend_shift
        xn_s = args.x_noise_std if args.x_noise_std_shift is None else args.x_noise_std_shift
        xl_s = args.x0_lambda if args.x0_lambda_shift is None else args.x0_lambda_shift
        print(f"  X (dynamic) gen  : ρ_X={args.x_rate}, trend_X={args.x_trend}, σ_η={args.x_noise_std}, λ_X0={args.x0_lambda}")
        if args.with_shift:
            print(f"  X (dynamic) TEST : ρ_X→{xr_s}, trend_X→{xt_s}, σ_η→{xn_s}, λ_X0→{xl_s} (X₀ re-drawn)")
    print(f"  With TEST shift  : {args.with_shift}")
    print(f"  Window size      : {args.window_size}")
    print(f"  Seed             : {args.seed}")

    # Initialize generator and predictor
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)
    predictor = BasicConformalPredictor(alpha=args.alpha)

    # Run coverage experiment
    coverage_history, interval_widths = run_coverage_experiment(
        generator=generator,
        predictor=predictor,
        n_series=args.n_series,
        covariate_mode=args.covariate_mode,
        ar_coef=args.ar_coef,
        beta=args.beta,
        noise_std=args.noise_std,
        trend_coef=args.trend_coef,
        covar_rate=args.covar_rate,
        covar_rate_shift=args.covar_rate_shift,
        x_rate=args.x_rate,
        x_trend=args.x_trend,
        x_noise_std=args.x_noise_std,
        x0_lambda=args.x0_lambda,
        x_rate_shift=args.x_rate_shift,
        x_trend_shift=args.x_trend_shift,
        x_noise_std_shift=args.x_noise_std_shift,
        x0_lambda_shift=args.x0_lambda_shift,
        with_shift=args.with_shift,
        n_train=args.n_train,
        n_cal=args.n_cal,
    )

    # Compute overall statistics
    overall_coverage = float(np.mean(coverage_history)) if coverage_history.size else float('nan')
    target_coverage = 1 - args.alpha

    print(f"\n" + "="*40)
    print("OVERALL COVERAGE RESULTS")
    print("="*40)
    print(f"Target coverage     : {target_coverage:.1%}")
    print(f"Actual coverage     : {overall_coverage:.1%}")
    print(f"Coverage error      : {overall_coverage - target_coverage:+.1%}")
    print(f"Coverage std        : {np.std(coverage_history):.3f}")
    print(f"Mean interval width : {np.mean(interval_widths):.4f}")

    # Create main coverage plot
    print("\nCreating coverage plot…")
    moving_coverage = plot_coverage_like_notebook(
        coverage_history,
        target_coverage,
        args.window_size,
        covariate_mode=args.covariate_mode,
        with_shift=args.with_shift
    )

    # Summary statistics for moving coverage
    if moving_coverage.size:
        print(f"\nMOVING AVERAGE COVERAGE STATISTICS:")
        print(f"  Mean coverage : {np.mean(moving_coverage):.1%}")
        print(f"  Std coverage  : {np.std(moving_coverage):.3f}")
        print(f"  Min coverage  : {np.min(moving_coverage):.1%}")
        print(f"  Max coverage  : {np.max(moving_coverage):.1%}")
        coverage_within_range = np.abs(np.mean(moving_coverage) - target_coverage) < 0.05
    else:
        print("\nMOVING AVERAGE COVERAGE STATISTICS: (insufficient data)")
        coverage_within_range = False

    # Final assessment
    print(f"\nFinal Assessment:")
    print(f"  Coverage near target: {'✓' if coverage_within_range else '✗'}")
    print(f"  Algorithm working   : {'✓' if coverage_within_range else '✗'}")


if __name__ == "__main__":
    main()
