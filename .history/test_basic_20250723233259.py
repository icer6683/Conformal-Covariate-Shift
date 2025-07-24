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

def run_online_coverage_experiment(generator, predictor, n_timesteps=500, ar_coef=0.7, noise_std=0.2, 
                                   with_shift=False, shift_time=250, shift_amount=2.0, initial_window=50):
    """
    Run ONLINE coverage experiment where t increases with each iteration.
    
    This generates ONE long time series and makes sequential predictions:
    - t=initial_window: predict Y_t using Y_0,...,Y_{t-1}  
    - t=initial_window+1: predict Y_{t+1} using Y_0,...,Y_t
    - etc.
    
    Args:
        n_timesteps: Total length of time series
        initial_window: How many points to use for initial training
        shift_time: When covariate shift occurs (absolute time)
        with_shift: Whether to apply covariate shift
    """
    if with_shift:
        print(f"Running ONLINE experiment with COVARIATE SHIFT...")
        print(f"  Total timesteps: {n_timesteps}")
        print(f"  Shift occurs at t={shift_time}, magnitude={shift_amount}")
    else:
        print(f"Running ONLINE experiment (NO SHIFT)...")
        print(f"  Total timesteps: {n_timesteps}")
    
    # Generate ONE long time series
    print("Generating long time series...")
    if with_shift:
        # Generate original long series
        # Note: generator.T controls the time series length, so we need to temporarily change it
        original_T = generator.T
        generator.T = n_timesteps - 1  # Set desired length
        
        original_long = generator.generate_ar_process(n=1, ar_coef=ar_coef, noise_std=noise_std)[0]
        
        # Apply covariate shift at specified time
        original_batch = original_long[np.newaxis, :]
        _, shifted_batch = generator.introduce_covariate_shift(
            original_batch,
            shift_time=shift_time,
            shift_params={'shift_amount': shift_amount}
        )
        long_series = shifted_batch[0]
        
        # Restore original T
        generator.T = original_T
        
        print(f"  Covariate shift applied at t={shift_time}")
        print(f"  Values before shift (t={shift_time-2}:{shift_time}): {original_long[shift_time-2:shift_time, 0]}")
        print(f"  Values after shift (t={shift_time}:{shift_time+2}): {long_series[shift_time:shift_time+2, 0]}")
        
    else:
        # Generate normal long series
        original_T = generator.T
        generator.T = n_timesteps - 1
        
        long_series = generator.generate_ar_process(n=1, ar_coef=ar_coef, noise_std=noise_std)[0]
        
        # Restore original T
        generator.T = original_T
    
    # Fit initial model on first part of series (before any potential shift)
    initial_train_data = long_series[:initial_window//2][np.newaxis, :]  # Add batch dimension
    initial_cal_data = long_series[initial_window//2:initial_window][np.newaxis, :]
    
    print("Fitting initial AR model...")
    predictor.fit_ar_model(initial_train_data)
    
    print("Initial calibration...")
    predictor.calibrate(initial_cal_data)
    
    # Online prediction phase
    print(f"Starting online predictions from t={initial_window}...")
    coverage_history = []
    interval_widths = []
    prediction_errors = []
    
    for t in range(initial_window, n_timesteps):
        if (t - initial_window + 1) % 100 == 0:
            print(f"  Online step {t - initial_window + 1}/{n_timesteps - initial_window}...")
        
        # Use all data up to time t-1 to predict time t
        historical_data = long_series[:t]  # Y_0, Y_1, ..., Y_{t-1}
        
        # Make prediction for time t
        pred, lower, upper = predictor.predict_with_interval(historical_data[np.newaxis, :])
        true_value = long_series[t, 0]  # Y_t
        
        # Check coverage
        covered = (lower <= true_value <= upper)
        coverage_history.append(1 if covered else 0)
        interval_widths.append(upper - lower)
        prediction_errors.append(abs(true_value - pred))
        
        # Debug output around shift time
        if with_shift and abs(t - shift_time) <= 2:
            print(f"    t={t} (shift at {shift_time}): pred={pred:.3f}, true={true_value:.3f}, "
                  f"interval=[{lower:.3f}, {upper:.3f}], covered={'✓' if covered else '✗'}")
    
    coverage_history = np.array(coverage_history)
    interval_widths = np.array(interval_widths)
    prediction_errors = np.array(prediction_errors)
    
    # Return results with time information
    results = {
        'coverage_history': coverage_history,
        'interval_widths': interval_widths, 
        'prediction_errors': prediction_errors,
        'time_indices': np.arange(initial_window, n_timesteps),
        'shift_time': shift_time if with_shift else None,
        'initial_window': initial_window
    }
    
    return results
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
            
            # Debug: Print detailed information about the shift
            if i < 3:
                print(f"  Debug series {i}:")
                print(f"    Shift time: {shift_time}, Shift amount: {shift_amount}")
                print(f"    Original series shape: {original_series.shape}")
                print(f"    Shifted series shape: {test_series.shape}")
                print(f"    Original values at shift time: {original_series[shift_time:shift_time+3, 0]}")
                print(f"    Shifted values at shift time: {test_series[shift_time:shift_time+3, 0]}")
                print(f"    Difference at shift time: {test_series[shift_time:shift_time+3, 0] - original_series[shift_time:shift_time+3, 0]}")
                print(f"    Original last value (prediction target): {original_series[-1, 0]:.3f}")
                print(f"    Shifted last value (prediction target): {test_series[-1, 0]:.3f}")
                print(f"    Target difference: {test_series[-1, 0] - original_series[-1, 0]:.3f}")
                
                # Check if the shift affects the prediction input
                print(f"    Prediction input (original): {original_series[:-1, 0]}")
                print(f"    Prediction input (shifted): {test_series[:-1, 0]}")
                input_diff = np.mean(np.abs(test_series[:-1, 0] - original_series[:-1, 0]))
                print(f"    Average input difference: {input_diff:.3f}")
        else:
            # Generate normal test series (no shift)
            test_series = generator.generate_ar_process(n=1, ar_coef=ar_coef, noise_std=noise_std)[0]
        
        # Make prediction with interval (using predictor trained on original data)
        pred, lower, upper = predictor.predict_with_interval(test_series[:-1])
        true_value = test_series[-1, 0]
        
        # Debug: Print prediction details for first few series
        if i < 3 and with_shift:
            print(f"    Prediction: {pred:.3f}")
            print(f"    True value: {true_value:.3f}")
            print(f"    Interval: [{lower:.3f}, {upper:.3f}]")
            print(f"    Interval width: {upper - lower:.3f}")
            print(f"    Covered: {lower <= true_value <= upper}")
        
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

def plot_online_comparison_coverage(results_no_shift, results_with_shift, target_coverage, window_size=50):
    """
    Plot online coverage comparison showing the effect of covariate shift over time.
    
    This shows coverage degradation AFTER the shift occurs.
    """
    # Compute moving averages
    coverage_no_shift = results_no_shift['coverage_history']
    coverage_with_shift = results_with_shift['coverage_history']
    
    moving_no_shift = compute_moving_average_coverage(coverage_no_shift, window_size)
    moving_with_shift = compute_moving_average_coverage(coverage_with_shift, window_size)
    
    # Time indices (actual time steps)
    time_no_shift = results_no_shift['time_indices'][:len(moving_no_shift)]
    time_with_shift = results_with_shift['time_indices'][:len(moving_with_shift)]
    
    plt.figure(figsize=(14, 6))
    
    # Plot both coverage lines over actual time
    plt.plot(time_no_shift, moving_no_shift, 'b-', linewidth=2, 
             label=f'No Shift (Mean: {np.mean(moving_no_shift):.1%})')
    
    plt.plot(time_with_shift, moving_with_shift, 'r-', linewidth=2, 
             label=f'With Covariate Shift (Mean: {np.mean(moving_with_shift):.1%})')
    
    # Mark the shift time
    if results_with_shift['shift_time'] is not None:
        plt.axvline(x=results_with_shift['shift_time'], color='orange', linestyle=':', 
                   linewidth=3, label=f'Shift at t={results_with_shift["shift_time"]}')
    
    # Plot target coverage line
    plt.axhline(y=target_coverage, color='black', linestyle='--', linewidth=2,
                label=f'Target Coverage ({target_coverage:.1%})')
    
    # Formatting
    plt.xlabel('Time Step (t)')
    plt.ylabel('Coverage Rate')
    plt.title('Online Conformal Prediction: Coverage Before/After Covariate Shift')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim([0.6, 1.0])
    
    # Add text showing the failure
    coverage_drop = np.mean(moving_no_shift) - np.mean(moving_with_shift)
    shift_time = results_with_shift['shift_time']
    
    # Analyze coverage before and after shift
    if shift_time is not None:
        # Coverage before shift (first half)
        before_shift_idx = time_with_shift < shift_time
        after_shift_idx = time_with_shift >= shift_time + window_size  # Allow time for effect
        
        if np.any(before_shift_idx) and np.any(after_shift_idx):
            coverage_before = np.mean(moving_with_shift[before_shift_idx])
            coverage_after = np.mean(moving_with_shift[after_shift_idx])
            shift_effect = coverage_before - coverage_after
            
            plt.text(0.02, 0.02, f'Coverage Drop: {coverage_drop:+.1%}\n'
                                  f'Before Shift: {coverage_before:.1%}\n'
                                  f'After Shift: {coverage_after:.1%}\n'
                                  f'Shift Effect: {shift_effect:+.1%}', 
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
    print("TEST 1: ONLINE PREDICTION - NO COVARIATE SHIFT")
    print("="*50)
    
    predictor1 = BasicConformalPredictor(alpha=args.alpha)
    results_no_shift = run_online_coverage_experiment(
        generator=generator,
        predictor=predictor1,
        n_timesteps=args.n_series,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std,
        with_shift=False,
        initial_window=100
    )
    
    # Results for no shift
    coverage_no_shift_mean = np.mean(coverage_no_shift)
    print(f"\nResults (No Shift):")
    print(f"  Target coverage: {target_coverage:.1%}")
    print(f"  Actual coverage: {coverage_no_shift_mean:.1%}")
    print(f"  Coverage error: {coverage_no_shift_mean - target_coverage:+.1%}")
    
    # Test 2: With covariate shift (should fail)
    print(f"\n{'='*50}")
    print("TEST 2: ONLINE PREDICTION - WITH COVARIATE SHIFT")
    print("="*50)
    
    predictor2 = BasicConformalPredictor(alpha=args.alpha)
    results_with_shift = run_online_coverage_experiment(
        generator=generator,
        predictor=predictor2,
        n_timesteps=args.n_series,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std,
        with_shift=True,
        shift_time=args.shift_time,
        shift_amount=args.shift_amount,
        initial_window=100
    )
    
    # Results for with shift
    coverage_with_shift_mean = np.mean(coverage_with_shift)
    print(f"\nResults (With Shift):")
    print(f"  Target coverage: {target_coverage:.1%}")
    print(f"  Actual coverage: {coverage_with_shift_mean:.1%}")
    print(f"  Coverage error: {coverage_with_shift_mean - target_coverage:+.1%}")
    
    # Compare results
    coverage_drop = coverage_no_shift_mean - coverage_with_shift_mean
    width_increase = np.mean(widths_with_shift) - np.mean(widths_no_shift)
    
    print(f"\n{'='*50}")
    print("COVARIATE SHIFT IMPACT")
    print("="*50)
    print(f"Coverage without shift: {coverage_no_shift_mean:.1%}")
    print(f"Coverage with shift: {coverage_with_shift_mean:.1%}")
    print(f"Coverage drop: {coverage_drop:+.1%}")
    print(f"")
    print(f"Interval width without shift: {np.mean(widths_no_shift):.4f}")
    print(f"Interval width with shift: {np.mean(widths_with_shift):.4f}")
    print(f"Width change: {width_increase:+.4f}")
    
    if coverage_drop > 0.05:
        print(f"✓ COVARIATE SHIFT PROBLEM DEMONSTRATED")
        print(f"  Basic conformal prediction fails under covariate shift!")
        print(f"  Coverage drops significantly while intervals stay narrow.")
        print(f"  This motivates the need for your weighted conformal method.")
    elif coverage_with_shift_mean > 0.95:
        print(f"⚠ SUSPICIOUSLY HIGH COVERAGE DETECTED")
        print(f"  Coverage of {coverage_with_shift_mean:.1%} suggests intervals may be too wide.")
        print(f"  Check: Are intervals reasonable? Is shift being applied correctly?")
        print(f"  Efficient conformal prediction should have coverage ≈ target, not 100%")
    else:
        print(f"⚠ Covariate shift impact not clearly demonstrated")
        print(f"  Try increasing --shift_amount or more test series")
    
    # Create online comparison plot
    print(f"\nCreating online comparison plot...")
    moving_no_shift, moving_with_shift = plot_online_comparison_coverage(
        results_no_shift, 
        results_with_shift, 
        target_coverage, 
        args.window_size
    )
    
    print(f"\nExperiment completed!")
    print(f"This demonstrates why your research (weighted conformal prediction) is needed.")

if __name__ == "__main__":
    main()