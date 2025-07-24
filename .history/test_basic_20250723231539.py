#!/usr/bin/env python3
"""
=============================================================================
TEST BASIC CONFORMAL COVERAGE - code.ipynb Style
=============================================================================

PURPOSE: Generate data from ts_generator.py and test basic conformal prediction
         with coverage visualization

This creates moving average coverage plots,
showing coverage rates over time to validate the conformal prediction algorithm.

USAGE:
  python test_basic.py
  python test_basic.py --alpha 0.05 --n_series 1000
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator
from basic_conformal import BasicConformalPredictor

def run_coverage_experiment(generator, predictor, n_series=500, ar_coef=0.7, noise_std=0.2, 
                           with_shift=False, shift_time=12, shift_amount=2.0):
    """
    Run coverage experiment similar to code.ipynb.
    
    Args:
        with_shift: If True, apply covariate shift to test data
        shift_time: When the shift occurs in the time series
        shift_amount: Magnitude of the covariate shift
    """
    if with_shift:
        print(f"Running coverage experiment with COVARIATE SHIFT on {n_series} time series...")
        print(f"  Shift occurs at t={shift_time}, magnitude={shift_amount}")
    else:
        print(f"Running coverage experiment on {n_series} time series (NO SHIFT)...")
    
    # Generate training data for model fitting (NO SHIFT)
    print("Generating training data...")
    train_data = generator.generate_ar_process(n=200, ar_coef=ar_coef, noise_std=noise_std)
    
    # Generate calibration data (NO SHIFT)
    print("Generating calibration data...")
    cal_data = generator.generate_ar_process(n=100, ar_coef=ar_coef, noise_std=noise_std)
    
    # Fit model and calibrate on original distribution
    print("Fitting AR model...")
    predictor.fit_ar_model(train_data)
    
    print("Calibrating conformal predictor...")
    predictor.calibrate(cal_data)
    
    # Generate test series one by one and track coverage
    if with_shift:
        print("Testing coverage on SHIFTED data...")
    else:
        print("Testing coverage on original data...")
        
    coverage_history = []
    interval_widths = []
    
    for i in range(n_series):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_series} series...")
        
        if with_shift:
            # Generate original series then apply covariate shift
            original_series = generator.generate_ar_process(n=1, ar_coef=ar_coef, noise_std=noise_std)[0]
            
            # Apply covariate shift using ts_generator method
            original_batch = original_series[np.newaxis, :]  # Add batch dimension
            _, shifted_batch = generator.introduce_covariate_shift(
                original_batch,
                shift_time=shift_time,
                shift_params={'shift_amount': shift_amount}
            )
            test_series = shifted_batch[0]  # Remove batch dimension
        else:
            # Generate normal test series (no shift)
            test_series = generator.generate_ar_process(n=1, ar_coef=ar_coef, noise_std=noise_std)[0]
        
        # Make prediction with interval (using predictor trained on original data)
        pred, lower, upper = predictor.predict_with_interval(test_series[:-1])
        true_value = test_series[-1, 0]
        
        # Check coverage (like your error1[i] = 0 if covered else 1)
        covered = (lower <= true_value <= upper)
        coverage_history.append(1 if covered else 0)  # 1 = covered, 0 = miss
        
        # Track interval width
        interval_widths.append(upper - lower)
    
    coverage_history = np.array(coverage_history)
    interval_widths = np.array(interval_widths)
    
    return coverage_history, interval_widths

def compute_moving_average_coverage(coverage_history, window_size=100):
    """
    Compute moving average coverage like in code.ipynb.
    
    Similar to your:
    moving_average1 = np.zeros(len(error1) - local_window + 1)
    for i in range(len(error1) - local_window + 1):
        moving_average1[i] = np.mean(error1[i:i + local_window])
    """
    if len(coverage_history) < window_size:
        window_size = len(coverage_history) // 2
    
    moving_coverage = np.zeros(len(coverage_history) - window_size + 1)
    
    for i in range(len(coverage_history) - window_size + 1):
        # Note: coverage_history has 1=covered, 0=miss
        # So mean gives coverage rate directly
        moving_coverage[i] = np.mean(coverage_history[i:i + window_size])
    
    return moving_coverage

def plot_comparison_coverage(coverage_no_shift, coverage_with_shift, target_coverage, window_size=100):
    """
    Plot comparison of coverage with and without covariate shift.
    
    This demonstrates that basic conformal prediction fails under covariate shift.
    """
    # Compute moving averages
    moving_no_shift = compute_moving_average_coverage(coverage_no_shift, window_size)
    moving_with_shift = compute_moving_average_coverage(coverage_with_shift, window_size)
    
    # Create time indices
    time_indices = np.arange(max(len(moving_no_shift), len(moving_with_shift)))
    
    plt.figure(figsize=(12, 6))
    
    # Plot both coverage lines
    if len(moving_no_shift) > 0:
        plt.plot(time_indices[:len(moving_no_shift)], moving_no_shift, 'b-', linewidth=2, 
                 label=f'No Shift (Mean: {np.mean(moving_no_shift):.1%})')
    
    if len(moving_with_shift) > 0:
        plt.plot(time_indices[:len(moving_with_shift)], moving_with_shift, 'r-', linewidth=2, 
                 label=f'With Covariate Shift (Mean: {np.mean(moving_with_shift):.1%})')
    
    # Plot target coverage line
    plt.axhline(y=target_coverage, color='black', linestyle='--', linewidth=2,
                label=f'Target Coverage ({target_coverage:.1%})')
    
    # Formatting
    plt.xlabel('Time Series Index')
    plt.ylabel('Coverage Rate')
    plt.title('Basic Conformal Prediction: Coverage With vs Without Covariate Shift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.6, 1.0])
    
    # Add text showing the failure
    if len(moving_with_shift) > 0:
        coverage_drop = np.mean(moving_no_shift) - np.mean(moving_with_shift)
        plt.text(0.02, 0.02, f'Coverage Drop: {coverage_drop:+.1%}\n'
                              f'Shift Failure: {"✓" if coverage_drop > 0.05 else "✗"}', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return moving_no_shift, moving_with_shift

def plot_coverage_like_notebook(coverage_history, target_coverage, window_size=100, title="Basic Conformal Prediction Coverage"):
    """
    Create coverage plot similar to code.ipynb notebook.
    
    Mimics your:
    plt.plot(dates[255:-250], 1-moving_average1)
    plt.plot(dates[255:-250], np.zeros(len(moving_average1))+0.9)
    plt.title("Local Volatility Coverage")
    """
    # Compute moving average coverage
    moving_coverage = compute_moving_average_coverage(coverage_history, window_size)
    
    # Create time indices (like your dates)
    time_indices = np.arange(len(moving_coverage))
    
    plt.figure(figsize=(12, 6))
    
    # Plot moving average coverage (like your coverage rate plot)
    plt.plot(time_indices, moving_coverage, 'b-', linewidth=2, 
             label=f'Moving Coverage (window={window_size})')
    
    # Plot target coverage line (like your horizontal line at 0.9)
    plt.axhline(y=target_coverage, color='red', linestyle='--', linewidth=2,
                label=f'Target Coverage ({target_coverage:.1%})')
    
    # Formatting like your notebook
    plt.xlabel('Time Series Index')
    plt.ylabel('Coverage Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.7, 1.0])  # Similar range to your plots
    
    # Add statistics text box
    mean_coverage = np.mean(moving_coverage)
    std_coverage = np.std(moving_coverage)
    plt.text(0.02, 0.02, f'Mean: {mean_coverage:.1%}\nStd: {std_coverage:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return moving_coverage
    """
    Plot comparison of coverage with and without covariate shift.
    
    This demonstrates that basic conformal prediction fails under covariate shift.
    """
    # Compute moving averages
    moving_no_shift = compute_moving_average_coverage(coverage_no_shift, window_size)
    moving_with_shift = compute_moving_average_coverage(coverage_with_shift, window_size)
    
    # Create time indices
    time_indices = np.arange(max(len(moving_no_shift), len(moving_with_shift)))
    
    plt.figure(figsize=(12, 6))
    
    # Plot both coverage lines
    if len(moving_no_shift) > 0:
        plt.plot(time_indices[:len(moving_no_shift)], moving_no_shift, 'b-', linewidth=2, 
                 label=f'No Shift (Mean: {np.mean(moving_no_shift):.1%})')
    
    if len(moving_with_shift) > 0:
        plt.plot(time_indices[:len(moving_with_shift)], moving_with_shift, 'r-', linewidth=2, 
                 label=f'With Covariate Shift (Mean: {np.mean(moving_with_shift):.1%})')
    
    # Plot target coverage line
    plt.axhline(y=target_coverage, color='black', linestyle='--', linewidth=2,
                label=f'Target Coverage ({target_coverage:.1%})')
    
    # Formatting
    plt.xlabel('Time Series Index')
    plt.ylabel('Coverage Rate')
    plt.title('Basic Conformal Prediction: Coverage With vs Without Covariate Shift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.6, 1.0])
    
    # Add text showing the failure
    if len(moving_with_shift) > 0:
        coverage_drop = np.mean(moving_no_shift) - np.mean(moving_with_shift)
        plt.text(0.02, 0.02, f'Coverage Drop: {coverage_drop:+.1%}\n'
                              f'Shift Failure: {"✓" if coverage_drop > 0.05 else "✗"}', 
                 transform=plt.gca().transAxes, 
                 bbox=dict(boxstyle="round", facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return moving_no_shift, moving_with_shift
    """
    Create coverage plot similar to code.ipynb notebook.
    
    Mimics your:
    plt.plot(dates[255:-250], 1-moving_average1)
    plt.plot(dates[255:-250], np.zeros(len(moving_average1))+0.9)
    plt.title("Local Volatility Coverage")
    """
    # Compute moving average coverage
    moving_coverage = compute_moving_average_coverage(coverage_history, window_size)
    
    # Create time indices (like your dates)
    time_indices = np.arange(len(moving_coverage))
    
    plt.figure(figsize=(12, 6))
    
    # Plot moving average coverage (like your coverage rate plot)
    plt.plot(time_indices, moving_coverage, 'b-', linewidth=2, 
             label=f'Moving Coverage (window={window_size})')
    
    # Plot target coverage line (like your horizontal line at 0.9)
    plt.axhline(y=target_coverage, color='red', linestyle='--', linewidth=2,
                label=f'Target Coverage ({target_coverage:.1%})')
    
    # Formatting like your notebook
    plt.xlabel('Time Series Index')
    plt.ylabel('Coverage Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.7, 1.0])  # Similar range to your plots
    
    # Add statistics text box
    mean_coverage = np.mean(moving_coverage)
    std_coverage = np.std(moving_coverage)
    plt.text(0.02, 0.02, f'Mean: {mean_coverage:.1%}\nStd: {std_coverage:.3f}', 
             transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return moving_coverage

def main():
    parser = argparse.ArgumentParser(description='Test basic conformal prediction coverage')
    parser.add_argument('--n_series', type=int, default=500, help='Number of test series')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR coefficient')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--window_size', type=int, default=100, help='Moving average window')
    parser.add_argument('--T', type=int, default=25, help='Time series length')
    parser.add_argument('--shift_time', type=int, default=12, help='When covariate shift occurs')
    parser.add_argument('--shift_amount', type=float, default=2.0, help='Magnitude of covariate shift')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("BASIC CONFORMAL PREDICTION: COVARIATE SHIFT FAILURE DEMO")
    print("="*60)
    print("Demonstrating that basic conformal prediction fails under covariate shift...")
    print(f"Parameters:")
    print(f"  Target coverage: {1-args.alpha:.1%} (α = {args.alpha})")
    print(f"  Test series: {args.n_series}")
    print(f"  Moving window: {args.window_size}")
    print(f"  AR coefficient: {args.ar_coef}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Series length: {args.T + 1}")
    print(f"  Shift time: {args.shift_time}")
    print(f"  Shift amount: {args.shift_amount}")
    
    # Initialize generator
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)
    target_coverage = 1 - args.alpha
    
    # Test 1: No covariate shift (should work)
    print(f"\n{'='*50}")
    print("TEST 1: NO COVARIATE SHIFT (SHOULD WORK)")
    print("="*50)
    
    predictor1 = BasicConformalPredictor(alpha=args.alpha)
    coverage_no_shift, widths_no_shift = run_coverage_experiment(
        generator=generator,
        predictor=predictor1,
        n_series=args.n_series,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std,
        with_shift=False
    )
    
    # Results for no shift
    coverage_no_shift_mean = np.mean(coverage_no_shift)
    print(f"\nResults (No Shift):")
    print(f"  Target coverage: {target_coverage:.1%}")
    print(f"  Actual coverage: {coverage_no_shift_mean:.1%}")
    print(f"  Coverage error: {coverage_no_shift_mean - target_coverage:+.1%}")
    
    # Test 2: With covariate shift (should fail)
    print(f"\n{'='*50}")
    print("TEST 2: WITH COVARIATE SHIFT (SHOULD FAIL)")
    print("="*50)
    
    predictor2 = BasicConformalPredictor(alpha=args.alpha)
    coverage_with_shift, widths_with_shift = run_coverage_experiment(
        generator=generator,
        predictor=predictor2,
        n_series=args.n_series,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std,
        with_shift=True,
        shift_time=args.shift_time,
        shift_amount=args.shift_amount
    )
    
    # Results for with shift
    coverage_with_shift_mean = np.mean(coverage_with_shift)
    print(f"\nResults (With Shift):")
    print(f"  Target coverage: {target_coverage:.1%}")
    print(f"  Actual coverage: {coverage_with_shift_mean:.1%}")
    print(f"  Coverage error: {coverage_with_shift_mean - target_coverage:+.1%}")
    
    # Compare results
    coverage_drop = coverage_no_shift_mean - coverage_with_shift_mean
    print(f"\n{'='*50}")
    print("COVARIATE SHIFT IMPACT")
    print("="*50)
    print(f"Coverage without shift: {coverage_no_shift_mean:.1%}")
    print(f"Coverage with shift: {coverage_with_shift_mean:.1%}")
    print(f"Coverage drop: {coverage_drop:+.1%}")
    
    if coverage_drop > 0.05:
        print(f"✓ COVARIATE SHIFT PROBLEM DEMONSTRATED")
        print(f"  Basic conformal prediction fails under covariate shift!")
        print(f"  This motivates the need for your weighted conformal method.")
    else:
        print(f"⚠ Covariate shift impact not clearly demonstrated")
        print(f"  Try increasing --shift_amount or more test series")
    
    # Create comparison plot
    print(f"\nCreating comparison plot...")
    moving_no_shift, moving_with_shift = plot_comparison_coverage(
        coverage_no_shift, 
        coverage_with_shift, 
        target_coverage, 
        args.window_size
    )
    
    print(f"\nExperiment completed!")
    print(f"This demonstrates why your research (weighted conformal prediction) is needed.")

if __name__ == "__main__":
    main()