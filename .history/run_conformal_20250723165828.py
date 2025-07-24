"""
=============================================================================
TEST BASIC CONFORMAL PREDICTION (NO COVARIATE SHIFT)
=============================================================================

PURPOSE: Test the basic conformal prediction algorithm on data with NO covariate shift
         to validate that the algorithm achieves proper coverage before testing 
         with covariate shift.

This follows the approach from code.ipynb:
1. Generate time series data (no shift)
2. Train a simple forecasting model 
3. Compute conformity scores (prediction errors)
4. Apply conformal prediction to get prediction bands
5. Test coverage on new data (should achieve ~90% coverage)

USAGE:
  python run_conformal.py
  python run_conformal.py --alpha 0.05 --n_test 200
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator
from algorithm import AdaptedCAFHT

def simple_ar_forecast(train_data, ar_coef=0.7, noise_std=0.2):
    """
    Simple AR(1) forecasting model: Y_t = ar_coef * Y_{t-1} + noise
    Returns predictions for the next time step given historical data.
    """
    predictions = []
    
    for i in range(train_data.shape[0]):
        series = train_data[i, :-1, 0]  # All but last point
        # Simple AR(1) prediction: next value = ar_coef * last_value
        pred = ar_coef * series[-1]
        predictions.append(pred)
    
    return np.array(predictions)

def compute_conformity_scores(true_values, predictions):
    """
    Compute conformity scores as absolute residuals.
    This is similar to your notebook: |actual - predicted|
    """
    return np.abs(true_values - predictions)

def test_coverage_with_adapted_cafht(generator, n_train=500, n_test=100, alpha=0.1, 
                                    ar_coef=0.7, noise_std=0.2):
    """
    Test your Adapted CAFHT algorithm on data with NO covariate shift.
    
    This uses your actual algorithm from algorithm.py but displays results
    similar to the code.ipynb notebook style.
    """
    print(f"Testing Adapted CAFHT with α={alpha} (target coverage: {1-alpha:.1%})")
    
    # Generate training data (larger dataset for calibration)
    train_data = generator.generate_ar_process(
        n=n_train, ar_coef=ar_coef, noise_std=noise_std
    )
    
    # Generate test data (NO SHIFT - uniform likelihood ratios)
    test_data = generator.generate_ar_process(
        n=n_test, ar_coef=ar_coef, noise_std=noise_std
    )
    
    # Initialize your Adapted CAFHT algorithm
    algorithm = AdaptedCAFHT(alpha=alpha)
    
    print(f"Training data: {train_data.shape}")
    print(f"Test data: {test_data.shape}")
    
    # Test coverage on each test series
    coverage_history = []
    band_widths = []
    
    for t in range(n_test):
        print(f"Processing test series {t+1}/{n_test}...", end='\r')
        
        # Split test series: observed vs true future  
        test_series = test_data[t, :-1, :]  # Y_0 to Y_{T-1} (observed)
        true_future = test_data[t, :, :]    # Y_0 to Y_T (true values)
        
        # Use uniform likelihood ratios (no covariate shift)
        n_cal2 = train_data.shape[0] // 2  # Size of calibration split
        uniform_ratios = np.ones(n_cal2)  # All weights = 1 (no shift)
        
        try:
            # Run your Adapted CAFHT algorithm
            prediction_bands, online_stats = algorithm.predict_online(
                D_cal=train_data,
                test_series=test_series,
                likelihood_ratios=uniform_ratios,
                true_future=true_future
            )
            
            # Extract results
            coverage_rate = online_stats['final_coverage']
            avg_width = online_stats['average_width']
            
            coverage_history.append(coverage_rate)
            band_widths.append(avg_width)
            
            # Print first few for debugging
            if t < 5:
                print(f"\nSeries {t}: Coverage = {coverage_rate:.3f}, Width = {avg_width:.3f}")
                
        except Exception as e:
            print(f"\nError on series {t}: {e}")
            continue
    
    print(f"\nProcessed {len(coverage_history)} test series successfully")
    
    coverage_history = np.array(coverage_history)
    band_widths = np.array(band_widths)
    
    return coverage_history, band_widths

def plot_coverage_results(coverage_history, target_coverage, window_size=50):
    """
    Plot coverage results over time (similar to your notebook's moving average plot).
    """
    # Compute moving average coverage
    moving_coverage = []
    for i in range(len(coverage_history) - window_size + 1):
        moving_coverage.append(np.mean(coverage_history[i:i + window_size]))
    
    moving_coverage = np.array(moving_coverage)
    
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Coverage over time
    plt.subplot(1, 2, 1)
    plt.plot(coverage_history.astype(int), 'o-', alpha=0.7, markersize=3, label='Coverage')
    plt.axhline(y=target_coverage, color='red', linestyle='--', label=f'Target ({target_coverage:.1%})')
    plt.xlabel('Test Point')
    plt.ylabel('Coverage (1=covered, 0=not covered)')
    plt.title('Point-wise Coverage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Moving average coverage
    plt.subplot(1, 2, 2)
    plt.plot(moving_coverage, 'b-', linewidth=2, label=f'Moving avg (window={window_size})')
    plt.axhline(y=target_coverage, color='red', linestyle='--', label=f'Target ({target_coverage:.1%})')
    plt.xlabel('Test Point')
    plt.ylabel('Coverage Rate')
    plt.title('Local Coverage Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return moving_coverage

def main():
    parser = argparse.ArgumentParser(description='Test basic conformal prediction')
    parser.add_argument('--n_train', type=int, default=500, help='Training series')
    parser.add_argument('--n_test', type=int, default=200, help='Test points')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR coefficient')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Noise std')
    parser.add_argument('--window_size', type=int, default=100, help='Rolling window size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("TESTING BASIC CONFORMAL PREDICTION (NO COVARIATE SHIFT)")
    print("="*65)
    print("Goal: Validate that the algorithm achieves target coverage on regular time series")
    print(f"Parameters:")
    print(f"  Target coverage: {1-args.alpha:.1%} (α = {args.alpha})")
    print(f"  Training points: {args.n_train}")
    print(f"  Test points: {args.n_test}")
    print(f"  AR coefficient: {args.ar_coef}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Note: No covariate shift, so no weighting needed")
    
    # Initialize generator
    generator = TimeSeriesGenerator(T=25, d=1, seed=args.seed)
    
    # Test coverage using your Adapted CAFHT algorithm
    coverage_history, band_widths = test_coverage_with_adapted_cafht(
        generator=generator,
        n_train=args.n_train,
        n_test=args.n_test,
        alpha=args.alpha,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std
    )
    
    # Analyze results
    final_coverage = np.mean(coverage_history)
    target_coverage = 1 - args.alpha
    
    print(f"\n" + "="*50)
    print("COVERAGE RESULTS")
    print("="*50)
    print(f"Target coverage: {target_coverage:.1%}")
    print(f"Actual coverage: {final_coverage:.1%}")
    print(f"Coverage error: {final_coverage - target_coverage:+.1%}")
    print(f"Coverage std: {np.std(coverage_history):.3f}")
    print(f"Band width (mean): {np.mean(band_widths):.4f}")
    print(f"Band width (std): {np.std(band_widths):.4f}")
    
    # Plot results (similar to your notebook)
    moving_coverage = plot_coverage_results(coverage_history, target_coverage)
    
    # Additional statistics
    print(f"\nTest completed on {len(coverage_history)} time series")

if __name__ == "__main__":
    main()