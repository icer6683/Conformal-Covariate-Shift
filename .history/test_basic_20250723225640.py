#!/usr/bin/env python3
"""
=============================================================================
TEST BASIC CONFORMAL COVERAGE - code.ipynb Style
=============================================================================

PURPOSE: Generate data from ts_generator.py and test basic conformal prediction
         with coverage visualization similar to code.ipynb notebook.

This creates moving average coverage plots like your GARCH volatility example,
showing coverage rates over time to validate the conformal prediction algorithm.

USAGE:
  python test_basic_coverage.py
  python test_basic_coverage.py --alpha 0.05 --n_series 1000
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator
from basic_conformal import BasicConformalPredictor

def run_coverage_experiment(generator, predictor, n_series=500, ar_coef=0.7, noise_std=0.2):
    """
    Run coverage experiment similar to code.ipynb.
    
    This generates many time series and tests coverage over time,
    similar to your volatility forecasting experiment.
    """
    print(f"Running coverage experiment on {n_series} time series...")
    
    # Generate training data for model fitting
    print("Generating training data...")
    train_data = generator.generate_ar_process(n=200, ar_coef=ar_coef, noise_std=noise_std)
    
    # Generate calibration data  
    print("Generating calibration data...")
    cal_data = generator.generate_ar_process(n=100, ar_coef=ar_coef, noise_std=noise_std)
    
    # Fit model and calibrate
    print("Fitting AR model...")
    predictor.fit_ar_model(train_data)
    
    print("Calibrating conformal predictor...")
    predictor.calibrate(cal_data)
    
    # Generate test series one by one and track coverage
    print("Testing coverage over time...")
    coverage_history = []
    interval_widths = []
    
    for i in range(n_series):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_series} series...")
        
        # Generate single test series
        test_series = generator.generate_ar_process(n=1, ar_coef=ar_coef, noise_std=noise_std)[0]
        
        # Make prediction with interval
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

def main():
    parser = argparse.ArgumentParser(description='Test basic conformal prediction coverage')
    parser.add_argument('--n_series', type=int, default=500, help='Number of test series')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR coefficient')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--window_size', type=int, default=100, help='Moving average window')
    parser.add_argument('--T', type=int, default=25, help='Time series length')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("BASIC CONFORMAL PREDICTION COVERAGE TEST")
    print("="*50)
    print("Replicating code.ipynb style coverage analysis...")
    print(f"Parameters:")
    print(f"  Target coverage: {1-args.alpha:.1%} (α = {args.alpha})")
    print(f"  Test series: {args.n_series}")
    print(f"  Moving window: {args.window_size}")
    print(f"  AR coefficient: {args.ar_coef}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Series length: {args.T + 1}")
    
    # Initialize generator and predictor
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)
    predictor = BasicConformalPredictor(alpha=args.alpha)
    
    # Run coverage experiment
    coverage_history, interval_widths = run_coverage_experiment(
        generator=generator,
        predictor=predictor,
        n_series=args.n_series,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std
    )
    
    # Compute overall statistics
    overall_coverage = np.mean(coverage_history)
    target_coverage = 1 - args.alpha
    
    print(f"\n" + "="*40)
    print("OVERALL COVERAGE RESULTS")
    print("="*40)
    print(f"Target coverage: {target_coverage:.1%}")
    print(f"Actual coverage: {overall_coverage:.1%}")
    print(f"Coverage error: {overall_coverage - target_coverage:+.1%}")
    print(f"Coverage std: {np.std(coverage_history):.3f}")
    print(f"Mean interval width: {np.mean(interval_widths):.4f}")
    
    # Create main coverage plot (like code.ipynb)
    print(f"\nCreating coverage plot...")
    moving_coverage = plot_coverage_like_notebook(
        coverage_history, 
        target_coverage, 
        args.window_size,
        "Basic Conformal Prediction Coverage"
    )
    
    # Summary statistics like your notebook
    print(f"\nMOVING AVERAGE COVERAGE STATISTICS:")
    print(f"  Mean coverage: {np.mean(moving_coverage):.1%}")
    print(f"  Std coverage: {np.std(moving_coverage):.3f}")
    print(f"  Min coverage: {np.min(moving_coverage):.1%}")
    print(f"  Max coverage: {np.max(moving_coverage):.1%}")
    
    # Final assessment
    coverage_within_range = np.abs(np.mean(moving_coverage) - target_coverage) < 0.05
    print(f"\nFinal Assessment:")
    print(f"  Coverage near target: {'✓' if coverage_within_range else '✗'}")
    print(f"  Algorithm working: {'✓' if coverage_within_range else '✗'}")

if __name__ == "__main__":
    main()