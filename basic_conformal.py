#!/usr/bin/env python3
"""
=============================================================================
BASIC CONFORMAL PREDICTION FOR TIME SERIES
=============================================================================

PURPOSE: Simple, clean implementation of basic conformal prediction for AR time series.
         This is the standard method WITHOUT any covariate shift correction.

ALGORITHM:
1. Train a forecasting model on training data
2. Compute conformity scores (prediction errors) on calibration data  
3. Calculate the (1-α) quantile of conformity scores
4. For new predictions: prediction ± quantile = prediction interval

THEORY: Works for stationary AR processes because residuals are exchangeable,
        giving valid marginal coverage P(Y ∈ Ĉ(X)) ≥ 1-α

USAGE:
  python basic_conformal.py
  python basic_conformal.py --alpha 0.05 --n_test 200
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator

class BasicConformalPredictor:
    """
    Simple conformal prediction for time series.
    
    This implements the standard conformal prediction algorithm:
    1. Fit model on training data
    2. Compute conformity scores on calibration data
    3. Use quantile of scores for prediction intervals
    """
    
    def __init__(self, alpha=0.1):
        """
        Args:
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        self.quantile = None
        self.model_params = None
    
    def fit_ar_model(self, train_data):
        """
        Fit simple AR(1) model: Y_t = ar_coef * Y_{t-1} + noise
        
        Args:
            train_data: Training time series of shape (n, T+1, d)
        
        Returns:
            dict: Model parameters (ar_coef, noise_std)
        """
        # Extract features (Y_{t-1}) and targets (Y_t)
        features = train_data[:, :-1, 0].flatten()  # All Y_{t-1}
        targets = train_data[:, 1:, 0].flatten()   # All Y_t
        
        # Simple least squares: Y_t = ar_coef * Y_{t-1} + noise
        ar_coef = np.dot(features, targets) / np.dot(features, features)
        
        # Compute residuals and noise std
        predictions = ar_coef * features
        residuals = targets - predictions
        noise_std = np.std(residuals)
        
        self.model_params = {
            'ar_coef': ar_coef,
            'noise_std': noise_std
        }
        
        return self.model_params
    
    def predict_ar(self, series):
        """
        Make AR(1) prediction: Y_t = ar_coef * Y_{t-1}
        
        Args:
            series: Time series of shape (T+1, d)
        
        Returns:
            float: Prediction for next time step
        """
        if self.model_params is None:
            raise ValueError("Model not fitted. Call fit_ar_model first.")
        
        last_value = series[-1, 0]  # Y_{t-1}
        prediction = self.model_params['ar_coef'] * last_value
        
        return prediction
    
    def calibrate(self, calibration_data):
        """
        Compute conformity scores on calibration data.
        
        Args:
            calibration_data: Calibration time series of shape (n_cal, T+1, d)
        """
        if self.model_params is None:
            raise ValueError("Model not fitted. Call fit_ar_model first.")
        
        conformity_scores = []
        
        for i in range(calibration_data.shape[0]):
            series = calibration_data[i]
            
            # Predict last time step using all previous points
            prediction = self.predict_ar(series[:-1])
            true_value = series[-1, 0]
            
            # Conformity score = absolute residual
            score = abs(true_value - prediction)
            conformity_scores.append(score)
        
        conformity_scores = np.array(conformity_scores)
        
        # Compute quantile for prediction intervals
        self.quantile = np.quantile(conformity_scores, 1 - self.alpha)
        
        print(f"Calibration completed:")
        print(f"  Conformity scores - Mean: {np.mean(conformity_scores):.4f}, Std: {np.std(conformity_scores):.4f}")
        print(f"  {1-self.alpha:.1%} quantile: {self.quantile:.4f}")
        
        return conformity_scores
    
    def predict_with_interval(self, series):
        """
        Make prediction with conformal prediction interval.
        
        Args:
            series: Time series of shape (T, d) (without the target point)
        
        Returns:
            tuple: (prediction, lower_bound, upper_bound)
        """
        if self.quantile is None:
            raise ValueError("Model not calibrated. Call calibrate first.")
        
        # Make point prediction
        prediction = self.predict_ar(series)
        
        # Add conformal prediction interval
        lower_bound = prediction - self.quantile
        upper_bound = prediction + self.quantile
        
        return prediction, lower_bound, upper_bound
    
    def evaluate_coverage(self, test_data):
        """
        Evaluate coverage on test data.
        
        Args:
            test_data: Test time series of shape (n_test, T+1, d)
        
        Returns:
            dict: Coverage statistics
        """
        coverage_results = []
        predictions = []
        intervals = []
        
        for i in range(test_data.shape[0]):
            series = test_data[i]
            true_value = series[-1, 0]
            
            # Get prediction and interval
            pred, lower, upper = self.predict_with_interval(series[:-1])
            
            # Check coverage
            covered = (lower <= true_value <= upper)
            coverage_results.append(covered)
            
            predictions.append(pred)
            intervals.append([lower, upper])
            
            # Print first few for debugging
            if i < 5:
                print(f"Series {i}: pred={pred:.3f}, true={true_value:.3f}, "
                      f"interval=[{lower:.3f}, {upper:.3f}], covered={'✓' if covered else '✗'}")
        
        coverage_results = np.array(coverage_results)
        predictions = np.array(predictions)
        intervals = np.array(intervals)
        
        results = {
            'coverage_rate': np.mean(coverage_results),
            'coverage_std': np.std(coverage_results),
            'interval_width': np.mean(intervals[:, 1] - intervals[:, 0]),
            'width_std': np.std(intervals[:, 1] - intervals[:, 0]),
            'target_coverage': 1 - self.alpha,
            'predictions': predictions,
            'intervals': intervals,
            'coverage_history': coverage_results
        }
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Basic conformal prediction for time series')
    parser.add_argument('--n_train', type=int, default=200, help='Training series')
    parser.add_argument('--n_cal', type=int, default=100, help='Calibration series')
    parser.add_argument('--n_test', type=int, default=100, help='Test series')
    parser.add_argument('--T', type=int, default=25, help='Time series length')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='True AR coefficient')
    parser.add_argument('--noise_std', type=float, default=0.2, help='True noise std')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("BASIC CONFORMAL PREDICTION FOR TIME SERIES")
    print("="*50)
    print(f"Parameters:")
    print(f"  Target coverage: {1-args.alpha:.1%} (α = {args.alpha})")
    print(f"  Training series: {args.n_train}")
    print(f"  Calibration series: {args.n_cal}")
    print(f"  Test series: {args.n_test}")
    print(f"  Time series length: {args.T + 1}")
    print(f"  True AR coef: {args.ar_coef}")
    print(f"  True noise std: {args.noise_std}")
    
    # Initialize generator
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)
    
    # Generate data
    print(f"\nGenerating data...")
    train_data = generator.generate_ar_process(
        n=args.n_train, ar_coef=args.ar_coef, noise_std=args.noise_std
    )
    
    cal_data = generator.generate_ar_process(
        n=args.n_cal, ar_coef=args.ar_coef, noise_std=args.noise_std
    )
    
    test_data = generator.generate_ar_process(
        n=args.n_test, ar_coef=args.ar_coef, noise_std=args.noise_std
    )
    
    # Initialize conformal predictor
    predictor = BasicConformalPredictor(alpha=args.alpha)
    
    # Step 1: Fit AR model
    print(f"\nStep 1: Fitting AR(1) model...")
    model_params = predictor.fit_ar_model(train_data)
    print(f"  Estimated AR coef: {model_params['ar_coef']:.4f} (true: {args.ar_coef})")
    print(f"  Estimated noise std: {model_params['noise_std']:.4f} (true: {args.noise_std})")
    
    # Step 2: Calibrate conformal predictor
    print(f"\nStep 2: Calibrating conformal predictor...")
    conformity_scores = predictor.calibrate(cal_data)
    
    # Step 3: Evaluate on test data
    print(f"\nStep 3: Evaluating coverage on test data...")
    results = predictor.evaluate_coverage(test_data)
    
    # Print results
    print(f"\n" + "="*40)
    print("COVERAGE RESULTS")
    print("="*40)
    print(f"Target coverage: {results['target_coverage']:.1%}")
    print(f"Actual coverage: {results['coverage_rate']:.1%}")
    print(f"Coverage error: {results['coverage_rate'] - results['target_coverage']:+.1%}")
    print(f"Coverage std: {results['coverage_std']:.3f}")
    print(f"Interval width (mean): {results['interval_width']:.4f}")
    print(f"Interval width (std): {results['width_std']:.4f}")
    
    # Plot results
    plot_results(results, conformity_scores, args.alpha)

def plot_results(results, conformity_scores, alpha):
    """Plot coverage and conformity score results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Coverage over time
    axes[0, 0].plot(results['coverage_history'].astype(int), 'o-', alpha=0.7, markersize=3)
    axes[0, 0].axhline(y=results['target_coverage'], color='red', linestyle='--', 
                       label=f'Target ({results["target_coverage"]:.1%})')
    axes[0, 0].set_xlabel('Test Series')
    axes[0, 0].set_ylabel('Coverage (1=covered, 0=miss)')
    axes[0, 0].set_title('Coverage Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Moving average coverage
    window_size = min(20, len(results['coverage_history']) // 5)
    if window_size > 0:
        moving_avg = []
        for i in range(len(results['coverage_history']) - window_size + 1):
            moving_avg.append(np.mean(results['coverage_history'][i:i + window_size]))
        
        axes[0, 1].plot(moving_avg, 'b-', linewidth=2)
        axes[0, 1].axhline(y=results['target_coverage'], color='red', linestyle='--',
                          label=f'Target ({results["target_coverage"]:.1%})')
        axes[0, 1].set_xlabel('Test Series')
        axes[0, 1].set_ylabel('Moving Average Coverage')
        axes[0, 1].set_title(f'Local Coverage (window={window_size})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Conformity scores distribution
    axes[1, 0].hist(conformity_scores, bins=20, alpha=0.7, density=True)
    axes[1, 0].axvline(np.quantile(conformity_scores, 1-alpha), color='red', linestyle='--',
                      label=f'{1-alpha:.1%} quantile')
    axes[1, 0].set_xlabel('Conformity Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Conformity Scores')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Interval widths
    widths = results['intervals'][:, 1] - results['intervals'][:, 0]
    axes[1, 1].plot(widths, 'g-', alpha=0.7)
    axes[1, 1].axhline(y=np.mean(widths), color='orange', linestyle='--',
                      label=f'Mean ({np.mean(widths):.3f})')
    axes[1, 1].set_xlabel('Test Series')
    axes[1, 1].set_ylabel('Interval Width')
    axes[1, 1].set_title('Prediction Interval Widths')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()