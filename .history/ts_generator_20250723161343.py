#!/usr/bin/env python3
"""
=============================================================================
UNIVARIATE TIME SERIES GENERATOR WITH VISUALIZATION
=============================================================================

PURPOSE: Generate univariate AR(1) time series with covariate shift and 
         comprehensive visualization for understanding conformal prediction concepts.

MODEL: Y_t = α * Y_{t-1} + trend * t + ε_t
       where α is AR coefficient, trend is linear trend, ε_t ~ N(0, σ²)

COVARIATE SHIFT: Modifies distribution of Y_{0...T-1} while preserving 
                 P(Y_T | Y_{0...T-1}) by regenerating Y_T with same AR(1) model

KEY FEATURES:
- Univariate AR(1) time series generation
- Extensive 6-panel visualization showing covariate shift effects
- Statistical summaries and model verification
- Multiple shift types: mean_shift, scale_shift, selection_bias, time_varying
- Likelihood ratio computation with visualization
- Terminal output with detailed analysis

DEFAULT VALUES:
- T=30 (time series length), d=1 (univariate), n_train=100, n_test=50
- ar_coef=0.7, noise_std=0.2, shift_amount=2.0, shift_time=0
- seed=42, save_plot=False

USAGE:
  python ts_generator.py                           # Basic visualization (shift at t=0)
  python ts_generator.py --save_plot               # Save plots to file
  python ts_generator.py --shift_time 15 --T 30   # Mid-series shift at t=15
  python ts_generator.py --shift_amount 3.0 --T 50 # Larger shift, longer series

OUTPUT: Comprehensive plots showing original vs shifted time series, 
        distributions, and preserved conditional relationships
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional, Callable
import argparse

class TimeSeriesGenerator:
    """
    Generate time series data for conformal prediction with covariate shift.
    """
    
    def __init__(self, T: int = 50, d: int = 1, seed: Optional[int] = None):
        """
        Initialize the time series generator.
        
        Args:
            T: Length of time series (excluding initial condition)
            d: Dimension of observations at each time step
            seed: Random seed for reproducibility
        """
        self.T = T
        self.d = d
        if seed is not None:
            np.random.seed(seed)
    
    def generate_ar_process(self, 
                           n: int,
                           ar_coef: float = 0.7,
                           noise_std: float = 0.1,
                           initial_mean: float = 0.0,
                           initial_std: float = 1.0,
                           trend_coef: float = 0.0) -> np.ndarray:
        """Generate AR(1) time series with optional trend."""
        data = np.zeros((n, self.T + 1, self.d))
        
        # Generate initial conditions Y_0
        data[:, 0, :] = np.random.normal(initial_mean, initial_std, (n, self.d))
        
        # Generate the rest of the time series
        for t in range(1, self.T + 1):
            noise = np.random.normal(0, noise_std, (n, self.d))
            trend = trend_coef * t
            data[:, t, :] = ar_coef * data[:, t-1, :] + trend + noise
            
        return data
    
    def introduce_covariate_shift(self,
                                data: np.ndarray,
                                conditional_model: str = 'ar1',
                                model_params: dict = None,
                                shift_type: str = 'mean_shift',
                                shift_params: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Introduce covariate shift by modifying covariates and regenerating targets.
        """
        if shift_params is None:
            shift_params = {}
        if model_params is None:
            model_params = {'ar_coef': 0.7, 'noise_std': 0.1}
            
        n, T_plus_1, d = data.shape
        shifted_data = data.copy()
        
        # Step 1: Apply covariate shift to historical values (Y_0 to Y_{T-1})
        if shift_type == 'mean_shift':
            shift_amount = shift_params.get('shift_amount', 1.0)
            shifted_data[:, :-1, :] += shift_amount
            
        elif shift_type == 'scale_shift':
            scale_factor = shift_params.get('scale_factor', 1.5)
            mean = np.mean(shifted_data[:, :-1, :], axis=(0, 1), keepdims=True)
            shifted_data[:, :-1, :] = mean + scale_factor * (shifted_data[:, :-1, :] - mean)
            
        elif shift_type == 'selection_bias':
            threshold = shift_params.get('threshold', 0.0)
            selection_prob = shift_params.get('selection_prob', 0.7)
            
            initial_values = shifted_data[:, 0, 0]
            probs = np.where(initial_values > threshold, selection_prob, 1 - selection_prob)
            
            selected_indices = np.random.binomial(1, probs).astype(bool)
            shifted_data = shifted_data[selected_indices]
            n = shifted_data.shape[0]
            
        # Step 2: Regenerate Y_T using the same conditional model
        if conditional_model == 'ar1':
            ar_coef = model_params.get('ar_coef', 0.7)
            noise_std = model_params.get('noise_std', 0.1)
            trend_coef = model_params.get('trend_coef', 0.0)
            
            noise = np.random.normal(0, noise_std, (n, d))
            trend = trend_coef * (T_plus_1 - 1)
            shifted_data[:, -1, :] = ar_coef * shifted_data[:, -2, :] + trend + noise
            
        return data, shifted_data
    
    def compute_likelihood_ratios(self,
                                original_data: np.ndarray,
                                shifted_data: np.ndarray,
                                method: str = 'gaussian_kde') -> np.ndarray:
        """Compute likelihood ratios for weighting."""
        # Use the covariates (all but last time point) for density estimation
        original_covariates = original_data[:, :-1, :].reshape(original_data.shape[0], -1)
        shifted_covariates = shifted_data[:, :-1, :].reshape(shifted_data.shape[0], -1)
        
        # For simplicity, use just the initial conditions
        original_initial = original_data[:, 0, 0]
        shifted_initial = shifted_data[:, 0, 0]
        
        if method == 'gaussian_kde':
            kde_original = stats.gaussian_kde(original_initial)
            kde_shifted = stats.gaussian_kde(shifted_initial)
            
            likelihood_ratios = kde_shifted(shifted_initial) / (kde_original(shifted_initial) + 1e-10)
            
        return likelihood_ratios
    
    def visualize_covariate_shift(self, original_data: np.ndarray, shifted_data: np.ndarray, 
                                shift_time: int = 0, save_plot: bool = False, 
                                filename: str = "covariate_shift.png"):
        """
        Create comprehensive visualization of the covariate shift with timing illustration.
        """
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Sample time series from original data
        plt.subplot(2, 3, 1)
        n_plot = min(10, original_data.shape[0])
        for i in range(n_plot):
            plt.plot(original_data[i, :, 0], alpha=0.6, color='blue', linewidth=1)
        plt.title('Original Time Series (Training Data)', fontsize=12, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Sample time series from shifted data with shift timing highlighted
        plt.subplot(2, 3, 2)
        n_plot = min(10, shifted_data.shape[0])
        T = shifted_data.shape[1]
        
        for i in range(n_plot):
            # Plot pre-shift period in blue (if any)
            if shift_time > 0:
                plt.plot(range(shift_time), shifted_data[i, :shift_time, 0], 
                        alpha=0.6, color='blue', linewidth=1.5)
            
            # Plot post-shift period in red
            plt.plot(range(shift_time, T), shifted_data[i, shift_time:, 0], 
                    alpha=0.6, color='red', linewidth=1.5)
            
        # Add vertical line to show shift timing
        if shift_time > 0:
            plt.axvline(x=shift_time, color='black', linestyle='--', linewidth=2, 
                       label=f'Shift at t={shift_time}')
            plt.legend()
            
        plt.title('Shifted Time Series (Test Data)', fontsize=12, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Overlay comparison with shift timing
        plt.subplot(2, 3, 3)
        for i in range(min(5, original_data.shape[0])):
            plt.plot(original_data[i, :, 0], alpha=0.7, color='blue', linewidth=1.5, 
                    label='Original' if i == 0 else '')
        
        for i in range(min(5, shifted_data.shape[0])):
            # Pre-shift in blue, post-shift in red
            if shift_time > 0:
                plt.plot(range(shift_time), shifted_data[i, :shift_time, 0], 
                        alpha=0.7, color='blue', linewidth=1.5)
            plt.plot(range(shift_time, T), shifted_data[i, shift_time:, 0], 
                    alpha=0.7, color='red', linewidth=1.5,
                    label='Shifted' if i == 0 else '')
        
        # Highlight shift timing
        if shift_time > 0:
            plt.axvline(x=shift_time, color='black', linestyle='--', linewidth=2, alpha=0.8)
            plt.fill_between([shift_time, T-1], plt.ylim()[0], plt.ylim()[1], 
                           alpha=0.1, color='red', label='Shift Period')
        
        plt.title(f'Overlay Comparison (Shift at t={shift_time})', fontsize=12, fontweight='bold')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of initial values (Y_0) or shift-time values
        plt.subplot(2, 3, 4)
        reference_time = max(0, shift_time)
        original_ref = original_data[:, reference_time, 0]
        shifted_ref = shifted_data[:, reference_time, 0]
        
        plt.hist(original_ref, bins=20, alpha=0.7, label=f'Original Y_{reference_time}', 
                color='blue', density=True)
        plt.hist(shifted_ref, bins=20, alpha=0.7, label=f'Shifted Y_{reference_time}', 
                color='red', density=True)
        plt.title(f'Distribution at Shift Time (t={reference_time})', fontsize=12, fontweight='bold')
        plt.xlabel(f'Y_{reference_time} Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 5: Distribution of final values (Y_T)
        plt.subplot(2, 3, 5)
        original_final = original_data[:, -1, 0]
        shifted_final = shifted_data[:, -1, 0]
        
        plt.hist(original_final, bins=20, alpha=0.7, label='Original Y_T', 
                color='blue', density=True)
        plt.hist(shifted_final, bins=20, alpha=0.7, label='Shifted Y_T', 
                color='red', density=True)
        plt.title('Distribution of Final Values', fontsize=12, fontweight='bold')
        plt.xlabel('Y_T Value')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 6: Scatter plot Y_{T-1} vs Y_T to show conditional relationship
        plt.subplot(2, 3, 6)
        original_prev = original_data[:, -2, 0]
        original_final = original_data[:, -1, 0]
        shifted_prev = shifted_data[:, -2, 0]
        shifted_final = shifted_data[:, -1, 0]
        
        plt.scatter(original_prev, original_final, alpha=0.6, color='blue', 
                   label='Original', s=30)
        plt.scatter(shifted_prev, shifted_final, alpha=0.6, color='red', 
                   label='Shifted', s=30)
        
        # Add regression lines to show preserved conditional relationship
        from scipy.stats import linregress
        slope_orig, intercept_orig, r_orig, _, _ = linregress(original_prev, original_final)
        slope_shift, intercept_shift, r_shift, _, _ = linregress(shifted_prev, shifted_final)
        
        x_range = np.linspace(min(np.min(original_prev), np.min(shifted_prev)),
                             max(np.max(original_prev), np.max(shifted_prev)), 100)
        plt.plot(x_range, slope_orig * x_range + intercept_orig, 'b--', 
                label=f'Original slope: {slope_orig:.3f}')
        plt.plot(x_range, slope_shift * x_range + intercept_shift, 'r--', 
                label=f'Shifted slope: {slope_shift:.3f}')
        
        plt.title('Conditional Relationship: Y_{T-1} vs Y_T', fontsize=12, fontweight='bold')
        plt.xlabel('Y_{T-1}')
        plt.ylabel('Y_T')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        
        plt.show()
    
    def print_statistics(self, original_data: np.ndarray, shifted_data: np.ndarray, shift_time: int = 0):
        """Print summary statistics to understand the shift."""
        print("\n" + "="*70)
        print("COVARIATE SHIFT ANALYSIS")
        print("="*70)
        
        # Original data statistics
        orig_initial = original_data[:, 0, 0]
        orig_final = original_data[:, -1, 0]
        
        # Shifted data statistics  
        shift_initial = shifted_data[:, 0, 0]
        shift_final = shifted_data[:, -1, 0]
        
        print(f"\nShift Timing: t = {shift_time}")
        if shift_time == 0:
            print("  → Entire covariate history Y₀...T₋₁ is shifted")
        else:
            print(f"  → Time points Y₀...Y_{shift_time-1} unchanged")
            print(f"  → Time points Y_{shift_time}...T₋₁ shifted")
        
        print(f"\nOriginal Data (n={original_data.shape[0]}):")
        print(f"  Y₀ - Mean: {np.mean(shift_initial):.3f}, Std: {np.std(shift_initial):.3f}")
        print(f"  Y_T - Mean: {np.mean(shift_final):.3f}, Std: {np.std(shift_final):.3f}")
        
        # Compare at shift time
        reference_time = max(0, shift_time)
        orig_ref = original_data[:, reference_time, 0]
        shift_ref = shifted_data[:, reference_time, 0]
        
        print(f"\nShift in Covariates at t={reference_time}:")
        print(f"  Mean difference: {np.mean(shift_ref) - np.mean(orig_ref):.3f}")
        print(f"  Std ratio: {np.std(shift_ref) / (np.std(orig_ref) + 1e-10):.3f}")
        
        # Check if conditional relationship is preserved
        from scipy.stats import linregress
        orig_slope, _, orig_r2, _, _ = linregress(original_data[:, -2, 0], original_data[:, -1, 0])
        shift_slope, _, shift_r2, _, _ = linregress(shifted_data[:, -2, 0], shifted_data[:, -1, 0])
        
        print(f"\nConditional Relationship P(Y_T | Y_{{T-1}}):")
        print(f"  Original slope: {orig_slope:.3f} (R²: {orig_r2:.3f})")
        print(f"  Shifted slope: {shift_slope:.3f} (R²: {shift_r2:.3f})")
        print(f"  Slope preservation: {'✓' if abs(orig_slope - shift_slope) < 0.1 else '✗'}")
        
        if shift_time > 0:
            print(f"\nTiming Analysis:")
            # Check pre-shift period correlation
            if shift_time > 1:
                pre_orig = original_data[:, :shift_time, 0].mean(axis=1)
                pre_shift = shifted_data[:, :shift_time, 0].mean(axis=1)
                pre_corr = np.corrcoef(pre_orig, pre_shift)[0, 1]
                print(f"  Pre-shift correlation: {pre_corr:.3f} (should be ~1.0)")
            
            # Check post-shift period difference
            post_orig = original_data[:, shift_time:, 0].mean(axis=1)
            post_shift = shifted_data[:, shift_time:, 0].mean(axis=1)
            post_diff = np.mean(post_shift - post_orig)
            print(f"  Post-shift mean difference: {post_diff:.3f}")
        print(f"  Y_T - Mean: {np.mean(orig_final):.3f}, Std: {np.std(orig_final):.3f}")
        
        print(f"\nShifted Data (n={shifted_data.shape[0]}):")
        print(f"  Y₀ - Mean: {np.mean(shift_initial):.3f}, Std: {np.std(shift_initial):.3f}")
        print(f"  Y_T - Mean: {np.mean(shift_final):.3f}, Std: {np.std(shift_final):.3f}")
        print(f"  Y_T - Mean: {np.mean(shift_final):.3f}, Std: {np.std(shift_final):.3f}")
        
        print(f"\nShift in Covariates (Y₀):")
        print(f"  Mean difference: {np.mean(shift_initial) - np.mean(orig_initial):.3f}")
        print(f"  Std ratio: {np.std(shift_initial) / np.std(orig_initial):.3f}")
        
        # Check if conditional relationship is preserved
        from scipy.stats import linregress
        orig_slope, _, orig_r2, _, _ = linregress(original_data[:, -2, 0], original_data[:, -1, 0])
        shift_slope, _, shift_r2, _, _ = linregress(shifted_data[:, -2, 0], shifted_data[:, -1, 0])
        
        print(f"\nConditional Relationship P(Y_T | Y_{{T-1}}):")
        print(f"  Original slope: {orig_slope:.3f} (R²: {orig_r2:.3f})")
        print(f"  Shifted slope: {shift_slope:.3f} (R²: {shift_r2:.3f})")
        print(f"  Slope preservation: {'✓' if abs(orig_slope - shift_slope) < 0.1 else '✗'}")


def main():
    """Main function to run the time series generation and visualization."""
    parser = argparse.ArgumentParser(description='Generate time series with covariate shift')
    parser.add_argument('--n_train', type=int, default=100, help='Number of training series')
    parser.add_argument('--n_test', type=int, default=50, help='Number of test series') 
    parser.add_argument('--T', type=int, default=30, help='Length of time series')
    parser.add_argument('--shift_amount', type=float, default=2.0, help='Amount of mean shift')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR coefficient')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shift_time', type=int, default=0, help='When covariate shift occurs (0=from start)')
    parser.add_argument('--save_plot', action='store_true', help='Save plot to file')
    
    args = parser.parse_args()
    
    print("Time Series Generator with Covariate Shift")
    print("="*50)
    print(f"Parameters:")
    print(f"  Training series: {args.n_train}")
    print(f"  Test series: {args.n_test}")
    print(f"  Time series length: {args.T + 1}")
    print(f"  AR coefficient: {args.ar_coef}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Shift amount: {args.shift_amount}")
    print(f"  Shift time: {args.shift_time}")
    print(f"  Random seed: {args.seed}")
    
    # Initialize generator
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)
    
    # Generate original training data
    print(f"\nGenerating {args.n_train} training time series...")
    train_data = generator.generate_ar_process(
        n=args.n_train,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std,
        initial_mean=0.0,
        initial_std=1.0,
        trend_coef=0.0
    )
    
    # Create test data with covariate shift
    print(f"Applying covariate shift to {args.n_test} test series...")
    original_test, shifted_test = generator.introduce_covariate_shift(
        train_data[:args.n_test],
        conditional_model='ar1',
        model_params={
            'ar_coef': args.ar_coef, 
            'noise_std': args.noise_std, 
            'trend_coef': 0.0
        },
        shift_type='mean_shift',
        shift_params={'shift_amount': args.shift_amount},
        shift_time=args.shift_time
    )
    
    # Compute likelihood ratios
    likelihood_ratios = generator.compute_likelihood_ratios(train_data, shifted_test)
    
    # Print statistics
    generator.print_statistics(original_test, shifted_test, shift_time=args.shift_time)
    
    print(f"\nLikelihood Ratios:")
    print(f"  Mean: {np.mean(likelihood_ratios):.3f}")
    print(f"  Std: {np.std(likelihood_ratios):.3f}")
    print(f"  Min: {np.min(likelihood_ratios):.3f}")
    print(f"  Max: {np.max(likelihood_ratios):.3f}")
    
    # Create visualization
    print(f"\nGenerating visualization...")
    generator.visualize_covariate_shift(
        original_test, 
        shifted_test, 
        shift_time=args.shift_time,
        save_plot=args.save_plot,
        filename="covariate_shift_visualization.png"
    )
    
    print("\nAnalysis complete! Check the generated plots to see the covariate shift.")

if __name__ == "__main__":
    main()