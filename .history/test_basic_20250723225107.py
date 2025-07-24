#!/usr/bin/env python3
"""
=============================================================================
TEST BASIC CONFORMAL COVERAGE - code.ipynb Style
=============================================================================

PURPOSE: Generate data from ts_generator.py and test basic conformal prediction
         with coverage visualization.

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

def plot_detailed_results(coverage_history, interval_widths, target_coverage, window_size=100):
    """
    Additional plots for detailed analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Raw coverage over time (like your error1 plot)
    axes[0, 0].plot(coverage_history, 'o-', alpha=0.6, markersize=2)
    axes[0, 0].axhline(y=target_coverage, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Series Index')
    axes[0, 0].set_ylabel('Coverage (1=covered, 0=miss)')
    axes[0, 0].set_title('Point-wise Coverage Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Moving average coverage (main result)
    moving_coverage = compute_moving_average_coverage(coverage_history, window_size)
    axes[0, 1].plot(moving_coverage, 'b-', linewidth=2)
    axes[0, 1].axhline(y=target_coverage, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Series Index')
    axes[0, 1].set_ylabel('Moving Average Coverage')
    axes[0, 1].set_title(f'Local Coverage Rate (window={window_size})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Coverage histogram (distribution of local coverage)
    axes[1, 0].hist(moving_coverage, bins=30, alpha=0.7, density=True)
    axes[1, 0].axvline(target_coverage, color='red', linestyle='--', label=f'Target ({target_coverage:.1%})')
    axes[1, 0].axvline(np.mean(moving_coverage), color='green', linestyle='--', label=f'Mean ({np.mean(moving_coverage):.1%})')
    axes[1, 0].set_xlabel('Coverage Rate')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Local Coverage Rates')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Interval widths over time
    axes[1, 1].plot(interval_widths, 'g-', alpha=0.7)
    axes[1, 1].axhline(y=np.mean(interval_widths), color='orange', linestyle='--', 
                      label=f'Mean ({np.mean(interval_widths):.3f})')
    axes[1, 1].set_xlabel('Series Index')
    axes[1, 1].set_ylabel('Interval Width')
    axes[1, 1].set_title('Prediction Interval Widths')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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
    print(f"\nCreating coverage plots...")
    moving_coverage = plot_coverage_like_notebook(
        coverage_history, 
        target_coverage, 
        args.window_size,
        "Basic Conformal Prediction Coverage"
    )
    
    # Additional detailed plots
    plot_detailed_results(coverage_history, interval_widths, target_coverage, args.window_size)
    
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