#!/usr/bin/env python3
"""
=============================================================================
TEST CONFORMAL COVERAGE - TIME-BASED ANALYSIS 
=============================================================================

PURPOSE:
  Generate data using ts_generator.py and test conformal predictors with 
  TIME-BASED coverage visualization. Shows how coverage changes as prediction 
  horizon increases (t=1,2,3,...,T).
  
  Supports three predictor modes:
    - basic     : BasicConformalPredictor (AR(1) on Y only, no adaptation)
    - adaptive  : OnlineConformalPredictor (AR(1) on Y only, adapts over time)
    - algorithm : AdaptedCAFHT from the paper (weighted conformal with likelihood ratios)
  
  Supports two covariate modes:
    - static  : time-invariant X (Poisson)
    - dynamic : time-varying X_t following its own AR(1)

WHAT THIS SHOWS:
  Coverage rate and interval width as a function of prediction time step t.
  Compares how basic vs adaptive vs algorithm methods handle covariate shift scenarios.

USAGE:
  # Basic predictor with default settings
  python test_conformal.py --predictor basic

  # Adaptive predictor with default settings  
  python test_conformal.py --predictor adaptive
  
  # Algorithm predictor with default settings
  python test_conformal.py --predictor algorithm

  # Dynamic covariates with test-set shift using algorithm predictor
  python test_conformal.py --predictor algorithm --covariate_mode dynamic \
      --with_shift --x_rate 0.6 --x_rate_shift 0.9 --beta 1.0

EXAMPLE USAGE:

Basic predictor examples:
    python test_conformal.py --predictor basic --n_series 1000
    python test_conformal.py --predictor basic --with_shift --n_series 1000

Adaptive predictor examples:
    python test_conformal.py --predictor adaptive --n_series 300
    python test_conformal.py --predictor adaptive --with_shift --n_series 300

Algorithm predictor examples:
    python test_conformal.py --predictor algorithm --n_series 300
    python test_conformal.py --predictor algorithm --with_shift --n_series 300

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator
from basic_conformal import BasicConformalPredictor
from adaptive_conformal import OnlineConformalPredictor
from algorithm_conformal import AdaptedCAFHT


def run_time_based_coverage_experiment(
    generator: TimeSeriesGenerator,
    predictor,  # Can be BasicConformalPredictor, OnlineConformalPredictor, or AlgorithmConformalPredictor
    predictor_type: str,  # 'basic', 'adaptive', or 'algorithm'
    *,
    n_series: int,
    covariate_mode: str,
    # Y model params
    ar_coef: float,
    beta: float,
    noise_std: float,
    trend_coef: float,
    # Static-X generation params
    covar_rate: float,
    covar_rate_shift: float,
    # Dynamic-X generation params
    x_rate: float,
    x_trend: float,
    x_noise_std: float,
    x0_lambda: float,
    # Dynamic-X shift params (if None, default to generation values)
    x_rate_shift: float,
    x_trend_shift: float,
    x_noise_std_shift: float,
    x0_lambda_shift: float,
    # Experiment options
    with_shift: bool,
    n_train: int,
    n_cal: int,
):
    """
    Run a time-based coverage experiment: evaluate coverage at each time step t.

    Returns:
        results_by_time: dict with keys = time steps, values = coverage stats
    """
    mode = covariate_mode.lower()
    assert mode in {"static", "dynamic"}, "covariate_mode must be 'static' or 'dynamic'"
    assert predictor_type in {"basic", "adaptive", "algorithm"}, "predictor_type must be 'basic', 'adaptive', or 'algorithm'"

    if with_shift:
        print(f"Running TIME-BASED coverage experiment with TEST covariate shift on {n_series} series...")
    else:
        print(f"Running TIME-BASED coverage experiment on {n_series} series (NO TEST SHIFT)...")
    print(f"Predictor type: {predictor_type}")
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

    # Reset adaptation for algorithm predictor at the start of testing
    if predictor_type == "algorithm" and hasattr(predictor, 'reset_adaptation'):
        predictor.reset_adaptation()

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
    
    # ALWAYS USE THE FIRST TEST SERIES
    example_idx = 0  # Always use the first series
    example_series = test_data[example_idx].copy()  # Shape: (T+1, 1)
    print(f"Storing first test series (index 0) for visualization")
    
    # -----------------------
    # TIME-BASED EVALUATION
    # -----------------------
    print("\nEvaluating coverage by time step...")
    
    n_test, T_plus_1, d = test_data.shape
    T = T_plus_1 - 1  # Number of prediction steps possible
    
    results_by_time = {}
    
    # Store the example series index and full data
    results_by_time['example_idx'] = example_idx
    results_by_time['example_full_series'] = example_series
    
    # INITIALIZE THE LISTS HERE - BEFORE THE LOOP
    results_by_time['example_predictions'] = []
    results_by_time['example_lower_bounds'] = []
    results_by_time['example_upper_bounds'] = []
    results_by_time['example_true_values'] = []
    
    # For each time step t, predict step t+1
    for t in range(T):  # t = 0, 1, 2, ..., T-1 (predicting t+1)
        print(f"  Evaluating predictions at time step {t+1}...")
        
        coverage_results = []
        predictions = []
        intervals = []
        
        # Re-fit and re-calibrate for algorithm predictor to handle shift better
        if t >= 1 and predictor_type == "algorithm":
            # Use increasing amount of data as time progresses
            predictor.fit_ar_model(train_Y[:,:t+2,:])
            predictor.calibrate(cal_Y[:,:t+2,:])

        for i in range(n_test):
            series = test_data[i]
            
            # Use data up to time t to predict time t+1
            input_series = series[:t+1]  # Y_0, Y_1, ..., Y_t
            true_value = series[t+1, 0]   # Y_{t+1}
            
            # Get prediction and interval - handle different predictor types
            if predictor_type == "basic":
                pred, lower, upper = predictor.predict_with_interval(input_series)
            elif predictor_type == "adaptive":
                pred, lower, upper = predictor.predict_with_interval(
                    input_series
                )
            else:  # algorithm
                pred, lower, upper = predictor.predict_with_interval(
                    input_series
                )
            
            # Check coverage
            covered = (lower <= true_value <= upper)
            coverage_results.append(covered)
            
            predictions.append(pred)
            intervals.append([lower, upper])
            
            # STORE DATA FOR THE FIRST SERIES (i == 0)
            if i == example_idx:  # This will always be 0
                results_by_time['example_predictions'].append(pred)
                results_by_time['example_lower_bounds'].append(lower)
                results_by_time['example_upper_bounds'].append(upper)
                results_by_time['example_true_values'].append(true_value)
        
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


def plot_time_based_results(results_by_time, target_coverage, predictor_type="basic", 
                          covariate_mode="static", with_shift=False):
    """Plot coverage results by time step."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 for 5 plots
    
    # Extract time-based data (exclude non-integer keys)
    time_steps = sorted([k for k in results_by_time.keys() if isinstance(k, int)])
    coverage_rates = [results_by_time[t]['coverage_rate'] for t in time_steps]
    interval_widths = [results_by_time[t]['interval_width'] for t in time_steps]
    
    # Build title
    predictor_str = predictor_type.capitalize()
    if predictor_type == "algorithm":
        predictor_str = "Algorithm (AdaptedCAFHT)"
    mode_str = "Dynamic $X_t$" if str(covariate_mode).lower() == "dynamic" else "Static X"
    shift_str = "with Shift" if with_shift else "no Shift"
    main_title = f"Time-Based Coverage Analysis — {predictor_str}, {mode_str}, {shift_str}"
    fig.suptitle(main_title, fontsize=14, fontweight='bold')
    
    # Plot 1: Coverage rate by time step (all series averaged)
    axes[0, 0].plot(time_steps, coverage_rates, 'b-', linewidth=2)
    axes[0, 0].set_ylim(0.8, 1)
    axes[0, 0].axhline(y=target_coverage, color='red', linestyle='--', linewidth=2,
                    label=f'Target ({target_coverage:.1%})')
    axes[0, 0].set_xlabel('Time Step t')
    axes[0, 0].set_ylabel('Coverage Rate')
    axes[0, 0].set_title('Coverage Rate vs. Time Step (All Series)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Interval width by time step (all series averaged)
    axes[0, 1].plot(time_steps, interval_widths, 'g-', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Time Step t')
    axes[0, 1].set_ylabel('Average Interval Width')
    axes[0, 1].set_title('Prediction Interval Width vs. Time Step (All Series)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: First test series with actual true values
    if 'example_true_values' in results_by_time:
        # Get the stored example data (always index 0)
        true_values = results_by_time['example_true_values']
        lower_bounds = results_by_time['example_lower_bounds']
        upper_bounds = results_by_time['example_upper_bounds']
        
        plot_steps = time_steps[:len(true_values)]
        
        # Plot the three lines
        axes[0, 2].plot(plot_steps, lower_bounds, 'b--', linewidth=1.5, alpha=0.7, label='Lower Bound')
        axes[0, 2].plot(plot_steps, upper_bounds, 'b--', linewidth=1.5, alpha=0.7, label='Upper Bound')
        axes[0, 2].plot(plot_steps, true_values, 'r-', linewidth=2, label='True Values')
        
        # Shade the prediction interval
        axes[0, 2].fill_between(plot_steps, lower_bounds, upper_bounds, 
                            alpha=0.2, color='blue')
        
        axes[0, 2].set_xlabel('Time Step t')
        axes[0, 2].set_ylabel('Value')
        axes[0, 2].set_title('First Test Series: Predictions and True Values')
        axes[0, 2].legend(loc='best')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Running average coverage for first test series
    if 'example_true_values' in results_by_time:
        # Calculate coverage for the first series at each time step
        first_series_coverage = []
        for i, t in enumerate(plot_steps):
            covered = (lower_bounds[i] <= true_values[i] <= upper_bounds[i])
            first_series_coverage.append(1 if covered else 0)
        
        # Calculate running average
        running_avg_coverage = []
        for i in range(len(first_series_coverage)):
            running_avg = np.mean(first_series_coverage[:i+1])
            running_avg_coverage.append(running_avg)
        
        axes[1, 0].plot(plot_steps, running_avg_coverage, 'b-', linewidth=2)
        axes[1, 0].set_ylim(0.7, 1.05)
        axes[1, 0].axhline(y=target_coverage, color='red', linestyle='--', linewidth=2,
                        label=f'Target ({target_coverage:.1%})')
        axes[1, 0].set_xlabel('Time Step t')
        axes[1, 0].set_ylabel('Running Average Coverage')
        axes[1, 0].set_title('First Test Series: Running Average Coverage')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add final coverage rate text
        final_coverage = running_avg_coverage[-1] if running_avg_coverage else 0
        axes[1, 0].text(0.02, 0.05, f'Final Coverage: {final_coverage:.1%}', 
                       transform=axes[1, 0].transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 5: Interval width for first test series only
    if 'example_true_values' in results_by_time:
        # Calculate interval widths for the first series
        first_series_widths = [upper_bounds[i] - lower_bounds[i] for i in range(len(plot_steps))]
        
        axes[1, 1].plot(plot_steps, first_series_widths, 'g-', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Time Step t')
        axes[1, 1].set_ylabel('Interval Width')
        axes[1, 1].set_title('First Test Series: Interval Width Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add text showing mean width for this series
        mean_width = np.mean(first_series_widths)
        axes[1, 1].text(0.02, 0.95, f'Mean Width: {mean_width:.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide the unused 6th subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Test conformal prediction with time-based coverage analysis')
    
    # MAIN TOGGLE: Predictor type
    parser.add_argument('--predictor', choices=['basic', 'adaptive', 'algorithm'], default='basic',
                        help='Choose predictor type: basic (fixed), adaptive (online), or algorithm (AdaptedCAFHT)')
    
    # Experiment sizes & basics
    parser.add_argument('--n_series', type=int, default=None, 
                        help='Number of test series (default varies by predictor)')
    parser.add_argument('--n_train', type=int, default=None,
                        help='Number of training series (default varies by predictor)')
    parser.add_argument('--n_cal', type=int, default=None,
                        help='Number of calibration series (default varies by predictor)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--T', type=int, default=None,
                        help='Time series length (default varies by predictor)')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')

    # Adaptive-specific parameters
    parser.add_argument('--window_size', type=int, default=800,
                        help='Window size for adaptive predictor (only used with --predictor adaptive)')

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
    parser.add_argument('--covar_rate_shift', type=float, default=None,
                        help='Shifted Poisson rate for TEST X (default varies by predictor)')

    # Dynamic-X params (generation)
    parser.add_argument('--x_rate', type=float, default=0.7, help='ρ_X for dynamic X_t')
    parser.add_argument('--x_trend', type=float, default=0.0, help='trend_X for dynamic X_t')
    parser.add_argument('--x_noise_std', type=float, default=0.2, help='Std dev of η_t for X_t')
    parser.add_argument('--x0_lambda', type=float, default=1.0, help='Poisson rate for X₀ (dynamic)')

    # Dynamic-X params (shift) — default to generation values if not provided
    parser.add_argument('--x_rate_shift', type=float, default=None, help='Shifted ρ_X for TEST X_t')
    parser.add_argument('--x_trend_shift', type=float, default=None, help='Shifted trend_X for TEST X_t')
    parser.add_argument('--x_noise_std_shift', type=float, default=None, help='Shifted X noise std for TEST')
    parser.add_argument('--x0_lambda_shift', type=float, default=None, help='Shifted Poisson rate for X₀ (TEST)')

    args = parser.parse_args()

    # Set predictor-specific defaults
    if args.predictor == "basic":
        defaults = {
            'n_series': 600,
            'n_train': 1200,
            'n_cal': 200,
            'T': 200,
            'covar_rate_shift': 3.0
        }
    elif args.predictor == "adaptive":
        defaults = {
            'n_series': 300,
            'n_train': 600,
            'n_cal': 100,
            'T': 40,
            'covar_rate_shift': 4.0
        }
    else:  # algorithm
        defaults = {
            'n_series': 300,
            'n_train': 600,
            'n_cal': 150,  # Slightly more for better gamma selection
            'T': 40,
            'covar_rate_shift': 3.5
        }

    # Apply defaults where arguments weren't provided
    for key, default_val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default_val)

    print(f"{args.predictor.upper()} CONFORMAL PREDICTION - TIME-BASED COVERAGE ANALYSIS")
    print("="*65)
    print("Analyzing how coverage changes as prediction horizon increases...")
    print("Parameters:")
    print(f"  Predictor type   : {args.predictor}")
    if args.predictor == "algorithm":
        print(f"                     (AdaptedCAFHT with weighted quantiles)")
    print(f"  Target coverage  : {1-args.alpha:.1%} (α={args.alpha})")
    print(f"  Series (train/cal/test): {args.n_train}/{args.n_cal}/{args.n_series}")
    print(f"  Series length    : {args.T + 1}")
    if args.predictor == "adaptive":
        print(f"  Window size      : {args.window_size}")
    print(f"  AR coef (Y)      : {args.ar_coef}")
    print(f"  β (X→Y)          : {args.beta}")
    print(f"  Y noise std      : {args.noise_std}")
    print(f"  Y trend          : {args.trend_coef}")
    print(f"  Covariate mode   : {args.covariate_mode}")
    if args.covariate_mode == 'static':
        shift_info = f" -> {args.covar_rate_shift}" if args.with_shift else ""
        print(f"  X (static) rate  : {args.covar_rate}{shift_info}")
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

    # Initialize generator
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)

    # Initialize appropriate predictor
    if args.predictor == "basic":
        predictor = BasicConformalPredictor(alpha=args.alpha)
    elif args.predictor == "adaptive":
        predictor = OnlineConformalPredictor(alpha=args.alpha, window_size=args.window_size)
    else:  # algorithm
        predictor = AlgorithmConformalPredictor(alpha=args.alpha)

    # Run time-based coverage experiment
    results_by_time = run_time_based_coverage_experiment(
        generator=generator,
        predictor=predictor,
        predictor_type=args.predictor,
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
    for key, value in results_by_time.items():
        # Skip non-integer keys (metadata keys)
        if not isinstance(key, int):
            continue
        
        time_results = value
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
    # Filter to only get integer keys (time steps)
    time_steps = sorted([k for k in results_by_time.keys() if isinstance(k, int)])
    for time_step in time_steps:
        time_results = results_by_time[time_step]
        print(f"Time {time_step:2d}: {time_results['coverage_rate']:.1%} "
              f"(width: {time_results['interval_width']:.3f}, "
              f"n={time_results['n_predictions']})")

    # Create time-based coverage plots
    print("\nCreating time-based coverage plots...")
    plot_time_based_results(
        results_by_time,
        target_coverage,
        predictor_type=args.predictor,
        covariate_mode=args.covariate_mode,
        with_shift=args.with_shift
    )

    # Final assessment
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