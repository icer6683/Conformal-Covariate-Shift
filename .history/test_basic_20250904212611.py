#!/usr/bin/env python3
"""
=============================================================================
TEST BASIC CONFORMAL COVERAGE - TIME-BASED ANALYSIS
=============================================================================

PURPOSE:
  Generate data using ts_generator.py and test the *basic* conformal predictor
  (AR(1) on Y only, no shift correction), with TIME-BASED coverage visualization.
  Shows how coverage changes as prediction horizon increases (t=1,2,3,...,T).
  
  Supports two covariate modes:
    - static  : time-invariant X (Poisson)
    - dynamic : time-varying X_t following its own AR(1)

WHAT THIS SHOWS:
  Coverage rate and interval width as a function of prediction time step t.
  Since the basic method ignores X (both static and dynamic), coverage will
  typically degrade as β grows and/or under covariate shift.

USAGE:
  # default (static X, no shift)
  python test_basic_time.py

  # tighter α and more series
  python test_basic_time.py --alpha 0.05 --n_series 1000

  # dynamic X with test-set covariate shift
  python test_basic_time.py --covariate_mode dynamic --with_shift \
      --x_rate 0.6 --x_rate_shift 0.9 --beta 1.0

EXAMPLE USAGE:

static + basic + no shift:
    python test_basic_time.py --n_series 1000
static + basic + with shift:
    python test_basic_time.py --with_shift --n_series 1000

dynamic + basic + no shift:
    python test_basic_time.py --n_series 1000 --covariate_mode dynamic
dynamic + basic + with shift:
    python test_basic_time.py --with_shift --n_series 1000 --covariate_mode dynamic

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator
from basic_conformal import BasicConformalPredictor


def run_time_based_coverage_experiment(
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
    Run a time-based coverage experiment: evaluate coverage at each time step t.

    Returns:
        results_by_time: dict with keys = time steps, values = coverage stats
    """
    mode = covariate_mode.lower()
    assert mode in {"static", "dynamic"}, "covariate_mode must be 'static' or 'dynamic'"

    if with_shift:
        print(f"Running TIME-BASED coverage experiment with TEST covariate shift on {n_series} series...")
    else:
        print(f"Running TIME-BASED coverage experiment on {n_series} series (NO TEST SHIFT)...")
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
    # Fit model & calibrate (using ALL available data)
    # -----------------------
    print("Fitting AR(1) model on training data...")
    predictor.fit_ar_model(train_Y)

    print("Calibrating conformal predictor on calibration data...")
    predictor.calibrate(cal_Y)

    # -----------------------
    # Generate test data
    # -----------------------
    print("Generating test data...")
    
    # Resolve shift params defaults for dynamic mode
    xr_s = x_rate if x_rate_shift is None else x_rate_shift
    xt_s = x_trend if x_trend_shift is None else x_trend_shift
    xn_s = x_noise_std if x_noise_std_shift is None else x_noise_std_shift
    xl_s = x0_lambda if x0_lambda_shift is None else x0_lambda_shift

    test_series_list = []
    
    for i in range(n_series):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_series} test series...")

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
        
        test_series_list.append(test_series)

    # Convert to numpy array
    test_data = np.array(test_series_list)  # Shape: (n_series, T+1, 1)
    
    # -----------------------
    # TIME-BASED EVALUATION
    # -----------------------
    print("\nEvaluating coverage by time step...")
    
    n_test, T_plus_1, d = test_data.shape
    T = T_plus_1 - 1  # Number of prediction steps possible
    
    results_by_time = {}
    
    # For each time step t, predict step t+1
    for t in range(T):  # t = 0, 1, 2, ..., T-1 (predicting t+1)
        print(f"  Evaluating predictions at time step {t+1}...")
        
        coverage_results = []
        predictions = []
        intervals = []
        
        for i in range(n_test):
            series = test_data[i]
            
            # Use data up to time t to predict time t+1
            input_series = series[:t+1]  # Y_0, Y_1, ..., Y_t
            true_value = series[t+1, 0]   # Y_{t+1}
            
            # Get prediction and interval
            pred, lower, upper = predictor.predict_with_interval(input_series)
            
            # Check coverage
            covered = (lower <= true_value <= upper)
            coverage_results.append(covered)
            
            predictions.append(pred)
            intervals.append([lower, upper])
        
        coverage_results = np.array(coverage_results)
        predictions = np.array(predictions)
        intervals = np.array(intervals)
        
        # Store results for this time step
        results_by_time[t+1] = {
            'coverage_rate': np.mean(coverage_results),
            'coverage_std': np.std(coverage_results),
            'interval_width': np.mean(intervals[:, 1] - intervals[:, 0]),
            'width_std': np.std(intervals[:, 1] - intervals[:, 0]),
            'n_predictions': len(coverage_results),
            'predictions': predictions,
            'intervals': intervals,
            'coverage_history': coverage_results
        }
        
        print(f"    Time {t+1}: Coverage = {np.mean(coverage_results):.1%}, "
              f"Width = {np.mean(intervals[:, 1] - intervals[:, 0]):.3f}")
    
    return results_by_time


def plot_time_based_results(results_by_time, target_coverage, covariate_mode="static", with_shift=False):
    """Plot coverage results by time step."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract time-based data
    time_steps = sorted(results_by_time.keys())
    coverage_rates = [results_by_time[t]['coverage_rate'] for t in time_steps]
    interval_widths = [results_by_time[t]['interval_width'] for t in time_steps]
    
    # Build title
    mode_str = "Dynamic $X_t$" if str(covariate_mode).lower() == "dynamic" else "Static X"
    shift_str = "with Shift" if with_shift else "no Shift"
    main_title = f"Time-Based Coverage Analysis — {mode_str}, {shift_str}"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    # Plot 1: Coverage rate by time step
    axes[0, 0].plot(time_steps, coverage_rates, 'bo-', linewidth=2, markersize=6)
    axes[0, 0].axhline(y=target_coverage, color='red', linestyle='--', linewidth=2,
                       label=f'Target ({target_coverage:.1%})')
    axes[0, 0].set_xlabel('Time Step t')
    axes[0, 0].set_ylabel('Coverage Rate')
    axes[0, 0].set_title('Coverage Rate vs. Time Step')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Interval width by time step  
    axes[0, 1].plot(time_steps, interval_widths, 'go-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Time Step t')
    axes[0, 1].set_ylabel('Average Interval Width')
    axes[0, 1].set_title('Prediction Interval Width vs. Time Step')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Coverage distribution across time steps
    all_coverage_by_time = []
    time_labels = []
    
    for t in time_steps:
        coverage_hist = results_by_time[t]['coverage_history']
        all_coverage_by_time.extend(coverage_hist)
        time_labels.extend([t] * len(coverage_hist))
    
    # Create a scatter plot showing coverage at each time step
    axes[1, 0].scatter(time_labels, all_coverage_by_time, alpha=0.6, s=20)
    axes[1, 0].axhline(y=target_coverage, color='red', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('Time Step t')
    axes[1, 0].set_ylabel('Coverage (1=covered, 0=miss)')
    axes[1, 0].set_title('Individual Coverage Results by Time Step')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([-0.1, 1.1])
    
    # Plot 4: Moving average of coverage rate
    window_size = min(5, len(coverage_rates))
    if window_size > 1:
        # Simple moving average
        moving_avg = []
        for i in range(len(coverage_rates) - window_size + 1):
            avg = np.mean(coverage_rates[i:i + window_size])
            moving_avg.append(avg)
        
        moving_time_steps = time_steps[window_size-1:]
        
        axes[1, 1].plot(time_steps, coverage_rates, 'b-', alpha=0.5, linewidth=1, label='Raw Coverage')
        axes[1, 1].plot(moving_time_steps, moving_avg, 'r-', linewidth=3, 
                       label=f'Moving Avg (window={window_size})')
        axes[1, 1].axhline(y=target_coverage, color='red', linestyle='--', alpha=0.7,
                          label=f'Target ({target_coverage:.1%})')
    else:
        axes[1, 1].plot(time_steps, coverage_rates, 'b-', linewidth=2, label='Coverage Rate')
        axes[1, 1].axhline(y=target_coverage, color='red', linestyle='--',
                          label=f'Target ({target_coverage:.1%})')
    
    axes[1, 1].set_xlabel('Time Step t')
    axes[1, 1].set_ylabel('Coverage Rate')
    axes[1, 1].set_title('Coverage Rate Trend Over Time')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test basic conformal prediction with time-based coverage analysis')
    # Experiment sizes & basics
    parser.add_argument('--n_series', type=int, default=500, help='Number of test series')
    parser.add_argument('--n_train',  type=int, default=200, help='Number of training series')
    parser.add_argument('--n_cal',    type=int, default=100, help='Number of calibration series')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--T', type=int, default=100, help='Time series length (T+1 points)')
    parser.add_argument('--seed', type=int, default=32, help='Random seed')

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

    print("BASIC CONFORMAL PREDICTION - TIME-BASED COVERAGE ANALYSIS")
    print("="*65)
    print("Analyzing how coverage changes as prediction horizon increases...")
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
    print(f"  Seed             : {args.seed}")

    # Initialize generator and predictor
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)
    predictor = BasicConformalPredictor(alpha=args.alpha)

    # Run time-based coverage experiment
    results_by_time = run_time_based_coverage_experiment(
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
    all_coverage = []
    all_widths = []
    for time_results in results_by_time.values():
        all_coverage.extend(time_results['coverage_history'])
        all_widths.extend([time_results['interval_width']] * time_results['n_predictions'])
    
    overall_coverage = float(np.mean(all_coverage)) if all_coverage else float('nan')
    target_coverage = 1 - args.alpha

    print(f"\n" + "="*50)
    print("OVERALL COVERAGE RESULTS")
    print("="*50)
    print(f"Target coverage     : {target_coverage:.1%}")
    print(f"Overall coverage    : {overall_coverage:.1%}")
    print(f"Coverage error      : {overall_coverage - target_coverage:+.1%}")
    print(f"Coverage std        : {np.std(all_coverage):.3f}")
    print(f"Mean interval width : {np.mean(all_widths):.4f}")

    print(f"\nCOVERAGE BY TIME STEP:")
    print("-" * 40)
    for time_step in sorted(results_by_time.keys()):
        time_results = results_by_time[time_step]
        print(f"Time {time_step:2d}: {time_results['coverage_rate']:.1%} "
              f"(width: {time_results['interval_width']:.3f}, "
              f"n={time_results['n_predictions']})")

    # Create time-based coverage plots
    print("\nCreating time-based coverage plots...")
    plot_time_based_results(
        results_by_time,
        target_coverage,
        covariate_mode=args.covariate_mode,
        with_shift=args.with_shift
    )

    # Final assessment
    time_steps = list(results_by_time.keys())
    coverage_rates = [results_by_time[t]['coverage_rate'] for t in time_steps]
    
    early_coverage = np.mean(coverage_rates[:len(coverage_rates)//3]) if coverage_rates else 0
    late_coverage = np.mean(coverage_rates[-len(coverage_rates)//3:]) if coverage_rates else 0
    coverage_degradation = early_coverage - late_coverage
    
    print(f"\nTIME-BASED ANALYSIS:")
    print(f"  Early coverage (first 1/3): {early_coverage:.1%}")
    print(f"  Late coverage (last 1/3):   {late_coverage:.1%}")
    print(f"  Coverage degradation:        {coverage_degradation:+.1%}")
    
    coverage_near_target = abs(overall_coverage - target_coverage) < 0.05
    print(f"\nFinal Assessment:")
    print(f"  Coverage near target: {'✓' if coverage_near_target else '✗'}")
    print(f"  Degradation < 10%:    {'✓' if abs(coverage_degradation) < 0.10 else '✗'}")


if __name__ == "__main__":
    main()