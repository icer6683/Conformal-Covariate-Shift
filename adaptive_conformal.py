#!/usr/bin/env python3
"""
=============================================================================
ONLINE CONFORMAL PREDICTION FOR TIME SERIES
=============================================================================

PURPOSE: Implementation of online/adaptive conformal prediction for AR time series.
         Updates conformity scores and quantiles as new observations arrive.

ALGORITHM:
1. Train a forecasting model on training data (done once)
2. Initialize conformity scores with calibration data
3. For each new prediction:
   - Calculate interval using current quantile
   - After observing true value, update conformity scores
   - Recalculate quantile for next prediction

THEORY: Provides adaptive coverage that adjusts to distribution shifts,
        though coverage guarantee differs from standard conformal prediction.

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from ts_generator import TimeSeriesGenerator


class OnlineConformalPredictor:
    """
    Online/Adaptive conformal prediction for time series.
    
    Updates conformity scores with each new observation to adapt
    to changing distributions over time.
    """
    
    def __init__(self, alpha=0.1, window_size=100):
        """
        Args:
            alpha: Miscoverage level (e.g., 0.1 for 90% coverage)
            window_size: Size of sliding window for conformity scores
                        (None for growing window without limit)
        """
        self.alpha = alpha
        self.window_size = window_size
        self.conformity_scores = []
        self.model_params = None
        self.quantile_history = []  # Track how quantile evolves
        self.ar_intercept_ = None
        self.ar_coef_ = None
        self.covar_coef_ = None
        self.residual_quantile = None
    
    def fit_ar_model(self, train_Y, train_X=None):
        """
        Fit ARX(1) model WITH INTERCEPT:
            Y_{t+1} = ar_intercept + ar_coef * Y_t + covar_coef * X_t + noise

        Args:
            train_Y: Training time series of shape (n, L, d)
            train_X: Optional covariate series aligned with Y, shape (n, L) or (n, L, 1)

        Returns:
            dict: Model parameters (ar_intercept, ar_coef, covar_coef, noise_std)
        """
        # Extract features (Y_t) and targets (Y_{t+1})
        Y_prev = train_Y[:, :-1, 0].reshape(-1)
        Y_next = train_Y[:, 1:, 0].reshape(-1)

        # Build design matrix
        ones = np.ones_like(Y_prev)
        cols = [ones, Y_prev]

        # Handle X if provided
        covar_coef = 0.0
        if train_X is not None:
            X_arr = np.asarray(train_X)
            if X_arr.ndim == 3 and X_arr.shape[2] == 1:
                X_arr = X_arr[:, :, 0]
            # use X_t to predict Y_{t+1}
            X_prev = X_arr[:, :-1].reshape(-1)
            cols.append(X_prev)
        X_design = np.column_stack(cols)

        # Least squares fit
        coeffs, *_ = np.linalg.lstsq(X_design, Y_next, rcond=None)
        ar_intercept = float(coeffs[0])
        ar_coef = float(coeffs[1])
        if len(coeffs) > 2:
            covar_coef = float(coeffs[2])
        else:
            covar_coef = 0.0

        # Residuals and noise std
        fitted = X_design @ coeffs
        residuals = Y_next - fitted
        noise_std = float(np.std(residuals))

        # Store as attributes for compatibility
        self.ar_intercept_ = ar_intercept
        self.ar_coef_ = ar_coef
        self.covar_coef_ = covar_coef

        self.model_params = {
            'ar_intercept': ar_intercept,
            'ar_coef': ar_coef,
            'covar_coef': covar_coef,
            'noise_std': noise_std
        }
        return self.model_params
    
    def predict_ar(self, series, x_series=None):
        """
        Make ARX(1) prediction WITH INTERCEPT:
            Y_{t+1} = ar_intercept + ar_coef * Y_t + covar_coef * X_t

        Args:
            series: Time series of shape (L, d)
            x_series: Optional covariate series aligned with series, shape (L,) or (L,1)

        Returns:
            float: Prediction for next time step
        """
        if self.model_params is None:
            raise ValueError("Model not fitted. Call fit_ar_model first.")

        y_t = series[-1, 0]
        x_t = 0.0
        if x_series is not None:
            x_arr = np.asarray(x_series)
            if x_arr.ndim == 2 and x_arr.shape[1] == 1:
                x_arr = x_arr[:, 0]
            x_t = float(x_arr[-1])

        pred = (self.model_params['ar_intercept']
                + self.model_params['ar_coef'] * y_t
                + self.model_params.get('covar_coef', 0.0) * x_t)
        return float(pred)
    
    def calibrate(self, calibration_Y, calibration_X=None):
        """
        Initialize conformity scores using calibration data.

        Args:
            calibration_Y: Calibration time series of shape (n_cal, L, d)
            calibration_X: Optional covariate series of shape (n_cal, L) or (n_cal, L, 1)
        """
        if self.model_params is None:
            raise ValueError("Model not fitted. Call fit_ar_model first.")

        conformity_scores = []
        n_cal = calibration_Y.shape[0]

        for i in range(n_cal):
            series = calibration_Y[i]
            x_series = None if calibration_X is None else calibration_X[i]

            # Predict last time step using all previous points
            prediction = self.predict_ar(series[:-1], None if x_series is None else x_series[:-1])
            true_value = series[-1, 0]

            # Conformity score = absolute residual
            score = abs(true_value - prediction)
            conformity_scores.append(score)

        # Initialize the conformity score buffer
        self.conformity_scores = conformity_scores

        # Calculate initial quantile
        initial_quantile = np.quantile(self.conformity_scores, 1 - self.alpha)
        self.residual_quantile = initial_quantile  # Store for compatibility

        print(f"Online calibration completed:")
        print(f"  Initial scores: {len(self.conformity_scores)} samples")
        print(f"  Initial {1-self.alpha:.1%} quantile: {initial_quantile:.4f}")
        print(f"  Window size: {self.window_size if self.window_size else 'unlimited (growing)'}")

        return conformity_scores
    
    def get_current_quantile(self):
        """
        Calculate current quantile from conformity scores.
        
        Returns:
            float: Current quantile for prediction intervals
        """
        if not self.conformity_scores:
            # Fallback to model-based estimate if no scores yet
            if self.model_params:
                return 2 * self.model_params['noise_std']
            else:
                return 1.0  # Default fallback
        
        # Calculate quantile with finite-sample correction
        n = len(self.conformity_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)  # Cap at 1.0
        
        return np.quantile(self.conformity_scores, q_level)
    
    def predict_with_interval(self, series, x_series=None):
        """
        Make prediction with adaptive conformal prediction interval.

        Args:
            series: Time series of shape (L, d) (without the target point)
            x_series: Optional covariate series aligned with series, shape (L,) or (L,1)

        Returns:
            tuple: (prediction, lower_bound, upper_bound)
        """
        # Make point prediction
        prediction = self.predict_ar(series, x_series)

        # Get current quantile
        current_quantile = self.get_current_quantile()
        self.quantile_history.append(current_quantile)

        # Create prediction interval
        lower_bound = prediction - current_quantile
        upper_bound = prediction + current_quantile

        return prediction, lower_bound, upper_bound
    
    def evaluate_coverage(self, test_data, test_X=None, adaptive=True):
        """
        Evaluate coverage on test data with optional online updates.

        Args:
            test_data: Test time series of shape (n_test, L, d)
            test_X: Optional covariate series of shape (n_test, L) or (n_test, L, 1)
            adaptive: If True, update conformity scores online

        Returns:
            dict: Coverage statistics including quantile evolution
        """
        coverage_results = []
        predictions = []
        intervals = []
        quantiles_over_time = []
        interval_widths = []

        n_test = test_data.shape[0]
        for i in range(n_test):
            series = test_data[i]
            x_series = None if test_X is None else test_X[i]
            true_value = series[-1, 0]

            # Get prediction and interval
            pred, lower, upper = self.predict_with_interval(series[:-1], None if x_series is None else x_series[:-1])

            # Check coverage
            covered = (lower <= true_value <= upper)
            coverage_results.append(covered)

            predictions.append(pred)
            intervals.append([lower, upper])
            interval_widths.append(upper - lower)
            quantiles_over_time.append(self.get_current_quantile())

            # Update scores if adaptive mode is enabled
            if adaptive:
                new_score = abs(true_value - pred)
                self.conformity_scores.append(new_score)
                if self.window_size is not None and len(self.conformity_scores) > self.window_size:
                    self.conformity_scores.pop(0)

            if i < 5 or (i > 0 and i % 50 == 0):
                print(f"Series {i}: pred={pred:.3f}, true={true_value:.3f}, "
                      f"interval=[{lower:.3f}, {upper:.3f}], "
                      f"width={upper-lower:.3f}, covered={'✓' if covered else '✗'}")

        coverage_results = np.array(coverage_results)
        predictions = np.array(predictions)
        intervals = np.array(intervals)

        results = {
            'coverage_rate': np.mean(coverage_results),
            'coverage_std': np.std(coverage_results),
            'interval_width': np.mean(interval_widths),
            'width_std': np.std(interval_widths),
            'target_coverage': 1 - self.alpha,
            'predictions': predictions,
            'intervals': intervals,
            'coverage_history': coverage_results,
            'quantile_history': np.array(quantiles_over_time),
            'interval_widths': np.array(interval_widths),
            'adaptive': adaptive
        }
        return results


def plot_online_results(results, alpha):
    """Plot online conformal prediction results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Plot 1: Coverage over time
    axes[0, 0].plot(results['coverage_history'].astype(int), 'o-', alpha=0.7, markersize=2)
    axes[0, 0].axhline(y=results['target_coverage'], color='red', linestyle='--', 
                       label=f'Target ({results["target_coverage"]:.1%})')
    axes[0, 0].set_xlabel('Test Series')
    axes[0, 0].set_ylabel('Coverage (1=covered, 0=miss)')
    axes[0, 0].set_title('Coverage Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Moving average coverage
    window_size = min(50, len(results['coverage_history']) // 5)
    if window_size > 0:
        moving_avg = np.convolve(results['coverage_history'], 
                                 np.ones(window_size)/window_size, mode='valid')
        
        axes[0, 1].plot(moving_avg, 'b-', linewidth=2)
        axes[0, 1].axhline(y=results['target_coverage'], color='red', linestyle='--',
                          label=f'Target ({results["target_coverage"]:.1%})')
        axes[0, 1].set_xlabel('Test Series')
        axes[0, 1].set_ylabel('Moving Average Coverage')
        axes[0, 1].set_title(f'Local Coverage (window={window_size})')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Quantile evolution
    axes[0, 2].plot(results['quantile_history'], 'g-', alpha=0.7)
    axes[0, 2].set_xlabel('Test Series')
    axes[0, 2].set_ylabel('Quantile Value')
    axes[0, 2].set_title('Quantile Evolution (Adaptive)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Interval width evolution
    axes[1, 0].plot(results['interval_widths'], 'purple', alpha=0.7)
    axes[1, 0].axhline(y=np.mean(results['interval_widths']), 
                       color='orange', linestyle='--',
                       label=f'Mean ({np.mean(results["interval_widths"]):.3f})')
    axes[1, 0].set_xlabel('Test Series')
    axes[1, 0].set_ylabel('Interval Width')
    axes[1, 0].set_title('Prediction Interval Width Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Cumulative coverage rate
    cumulative_coverage = np.cumsum(results['coverage_history']) / np.arange(1, len(results['coverage_history'])+1)
    axes[1, 1].plot(cumulative_coverage, 'b-', linewidth=2)
    axes[1, 1].axhline(y=results['target_coverage'], color='red', linestyle='--',
                       label=f'Target ({results["target_coverage"]:.1%})')
    axes[1, 1].set_xlabel('Test Series')
    axes[1, 1].set_ylabel('Cumulative Coverage Rate')
    axes[1, 1].set_title('Cumulative Coverage')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Width vs Coverage correlation
    # Group into bins to show relationship
    n_bins = min(20, len(results['coverage_history']) // 10)
    if n_bins > 2:
        bin_size = len(results['coverage_history']) // n_bins
        bin_coverage = []
        bin_width = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = min((i+1) * bin_size, len(results['coverage_history']))
            bin_coverage.append(np.mean(results['coverage_history'][start_idx:end_idx]))
            bin_width.append(np.mean(results['interval_widths'][start_idx:end_idx]))
        
        axes[1, 2].scatter(bin_width, bin_coverage, alpha=0.7)
        axes[1, 2].axhline(y=results['target_coverage'], color='red', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('Average Interval Width')
        axes[1, 2].set_ylabel('Coverage Rate')
        axes[1, 2].set_title('Width vs Coverage (binned)')
        axes[1, 2].grid(True, alpha=0.3)
    
    mode_str = "Online/Adaptive" if results.get('adaptive', True) else "Static"
    plt.suptitle(f'{mode_str} Conformal Prediction Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Online conformal prediction for time series')
    parser.add_argument('--n_train', type=int, default=200, help='Training series')
    parser.add_argument('--n_cal', type=int, default=100, help='Calibration series')
    parser.add_argument('--n_test', type=int, default=200, help='Test series')
    parser.add_argument('--T', type=int, default=25, help='Time series length')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--window_size', type=int, default=100, 
                       help='Sliding window size (0 for unlimited/growing window)')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='True AR coefficient')
    parser.add_argument('--noise_std', type=float, default=0.2, help='True noise std')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_adapt', action='store_true', 
                       help='Disable online updates (for comparison)')
    
    args = parser.parse_args()
    
    # Handle window_size = 0 as unlimited
    window_size = None if args.window_size == 0 else args.window_size
    
    print("ONLINE CONFORMAL PREDICTION FOR TIME SERIES")
    print("="*50)
    print(f"Parameters:")
    print(f"  Target coverage: {1-args.alpha:.1%} (α = {args.alpha})")
    print(f"  Window size: {window_size if window_size else 'unlimited (growing)'}")
    print(f"  Training series: {args.n_train}")
    print(f"  Calibration series: {args.n_cal}")
    print(f"  Test series: {args.n_test}")
    print(f"  Time series length: {args.T + 1}")
    print(f"  True AR coef: {args.ar_coef}")
    print(f"  True noise std: {args.noise_std}")
    print(f"  Adaptive mode: {'DISABLED' if args.no_adapt else 'ENABLED'}")
    
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
    
    # Initialize online conformal predictor
    predictor = OnlineConformalPredictor(alpha=args.alpha, window_size=window_size)
    
    # Step 1: Fit AR model (done once, not updated online)
    print(f"\nStep 1: Fitting AR(1) model with intercept...")
    model_params = predictor.fit_ar_model(train_data)
    print(f"  Estimated AR intercept: {model_params['ar_intercept']:.4f}")
    print(f"  Estimated AR coef: {model_params['ar_coef']:.4f} (true: {args.ar_coef})")
    print(f"  Estimated covariate coef: {model_params.get('covar_coef', 0.0):.4f}")
    print(f"  Estimated noise std: {model_params['noise_std']:.4f} (true: {args.noise_std})")
    
    # Step 2: Initialize with calibration data
    print(f"\nStep 2: Initializing with calibration data...")
    conformity_scores = predictor.calibrate(cal_data)
    
    # Step 3: Evaluate on test data with online updates
    print(f"\nStep 3: Evaluating on test data {'WITHOUT' if args.no_adapt else 'WITH'} online updates...")
    results = predictor.evaluate_coverage(test_data, adaptive=not args.no_adapt)
    
    # Print results
    print(f"\n" + "="*40)
    print("COVERAGE RESULTS")
    print("="*40)
    print(f"Target coverage: {results['target_coverage']:.1%}")
    print(f"Actual coverage: {results['coverage_rate']:.1%}")
    print(f"Coverage error: {results['coverage_rate'] - results['target_coverage']:+.1%}")
    print(f"Coverage std: {results['coverage_std']:.3f}")
    print(f"Initial interval width: {results['interval_widths'][0]:.4f}")
    print(f"Final interval width: {results['interval_widths'][-1]:.4f}")
    print(f"Mean interval width: {results['interval_width']:.4f}")
    print(f"Width std: {results['width_std']:.4f}")
    
    if not args.no_adapt:
        print(f"Quantile adaptation:")
        print(f"  Initial quantile: {results['quantile_history'][0]:.4f}")
        print(f"  Final quantile: {results['quantile_history'][-1]:.4f}")
        print(f"  Quantile std: {np.std(results['quantile_history']):.4f}")
    
    # Plot results
    plot_online_results(results, args.alpha)


if __name__ == "__main__":
    main()