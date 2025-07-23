#!/usr/bin/env python3
"""
=============================================================================
MULTIVARIATE TIME SERIES GENERATOR (MINIMAL EXTENSION)
=============================================================================

PURPOSE: Minimal extension of ts_generator.py to support multivariate time series.
         Same AR(1) model, same shift timing, just multiple dimensions.

MODEL: Y_t = α * Y_{t-1} + trend * t + ε_t (same as ts_generator)
       where Y_t is now d-dimensional instead of univariate

COVARIATE SHIFT: Identical to ts_generator - modifies Y_{0...T-1} while preserving 
                 P(Y_T | Y_{0...T-1}) by regenerating with same AR(1) model

MINIMAL CHANGES FROM ts_generator:
- Added d parameter for dimensions  
- Simple diagonal noise covariance (independent dimensions)
- No visualization (use ts_generator for that)
- Same defaults except d=2

DEFAULT VALUES: Same as ts_generator except d=2
- T=30, ar_coef=0.7, noise_std=0.2, shift_amount=2.0, shift_time=0

USAGE:
  python multivariate_ts_generator.py                # 2D version of ts_generator  
  python multivariate_ts_generator.py --d 1          # Same as ts_generator
  python multivariate_ts_generator.py --d 3          # 3D extension
=============================================================================
"""

import numpy as np
from scipy import stats
from typing import Tuple, Optional
import argparse

class MultivariateTimeSeriesGenerator:
    """
    Minimal multivariate extension of TimeSeriesGenerator.
    Same AR(1) model, just d-dimensional.
    """
    
    def __init__(self, T: int = 30, d: int = 2, seed: Optional[int] = None):
        """
        Args:
            T: Length of time series (same as ts_generator default=30)
            d: Dimension of observations (NEW: default=2)  
            seed: Random seed
        """
        self.T = T
        self.d = d
        if seed is not None:
            np.random.seed(seed)
    
    def generate_ar_process(self, 
                           n: int,
                           ar_coef: float = 0.7,
                           noise_std: float = 0.2,
                           initial_mean: float = 0.0,
                           initial_std: float = 1.0,
                           trend_coef: float = 0.0) -> np.ndarray:
        """
        Generate AR(1) time series - identical to ts_generator but d-dimensional.
        
        Model: Y_t = ar_coef * Y_{t-1} + trend * t + ε_t
        where Y_t is d-dimensional, ε_t ~ N(0, noise_std²*I)
        """
        data = np.zeros((n, self.T + 1, self.d))
        
        # Generate initial conditions Y_0 (independent across dimensions)
        for dim in range(self.d):
            data[:, 0, dim] = np.random.normal(initial_mean, initial_std, n)
        
        # Generate the rest of the time series (same AR(1) for each dimension)
        for t in range(1, self.T + 1):
            for dim in range(self.d):
                noise = np.random.normal(0, noise_std, n)
                trend = trend_coef * t
                data[:, t, dim] = ar_coef * data[:, t-1, dim] + trend + noise
        
        return data
    
    def introduce_covariate_shift(self,
                                data: np.ndarray,
                                shift_type: str = 'mean_shift',
                                shift_params: dict = None,
                                shift_time: int = 0,
                                ar_coef: float = 0.7,
                                noise_std: float = 0.2,
                                trend_coef: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identical logic to ts_generator, just handles d dimensions.
        """
        if shift_params is None:
            shift_params = {}
            
        n, T_plus_1, d = data.shape
        shifted_data = data.copy()
        
        # Determine which time points to shift (same logic as ts_generator)
        if shift_time == 0:
            time_mask = slice(None, -1)  # Y_0 to Y_{T-1}
        else:
            if shift_time >= T_plus_1 - 1:
                raise ValueError(f"shift_time ({shift_time}) must be < {T_plus_1 - 1}")
            time_mask = slice(shift_time, -1)  # Y_{shift_time} to Y_{T-1}
        
        # Apply shift to selected time points (same as ts_generator)
        if shift_type == 'mean_shift':
            shift_amount = shift_params.get('shift_amount', 2.0)
            shifted_data[:, time_mask, :] += shift_amount  # Broadcasts across all dimensions
            
        elif shift_type == 'scale_shift':
            scale_factor = shift_params.get('scale_factor', 1.5)
            mean = np.mean(shifted_data[:, time_mask, :], axis=(0, 1), keepdims=True)
            shifted_data[:, time_mask, :] = mean + scale_factor * (shifted_data[:, time_mask, :] - mean)
            
        elif shift_type == 'selection_bias':
            threshold = shift_params.get('threshold', 0.0)
            selection_prob = shift_params.get('selection_prob', 0.7)
            
            reference_time = shift_time if shift_time > 0 else 0
            # Use first dimension for selection (like ts_generator uses the single dimension)
            initial_values = shifted_data[:, reference_time, 0]
            probs = np.where(initial_values > threshold, selection_prob, 1 - selection_prob)
            
            selected_indices = np.random.binomial(1, probs).astype(bool)
            shifted_data = shifted_data[selected_indices]
            n = shifted_data.shape[0]
        
        # Regenerate time points after shift (same AR(1) logic, applied to each dimension)
        if shift_time == 0:
            regen_start = 1
        else:
            regen_start = shift_time + 1
            
        for t in range(regen_start, T_plus_1):  # Including final time point
            for dim in range(d):
                noise = np.random.normal(0, noise_std, n)
                trend = trend_coef * t
                shifted_data[:, t, dim] = ar_coef * shifted_data[:, t-1, dim] + trend + noise
        
        return data, shifted_data
    
    def compute_likelihood_ratios(self,
                                original_data: np.ndarray,
                                shifted_data: np.ndarray) -> np.ndarray:
        """
        Simple extension of ts_generator - use first dimension like ts_generator does.
        """
        # Use first dimension (like ts_generator uses its single dimension)
        original_initial = original_data[:, 0, 0]
        shifted_initial = shifted_data[:, 0, 0]
        
        kde_original = stats.gaussian_kde(original_initial)
        kde_shifted = stats.gaussian_kde(shifted_initial)
        
        likelihood_ratios = kde_shifted(shifted_initial) / (kde_original(shifted_initial) + 1e-10)
        
        return likelihood_ratios
    
    def get_covariates_and_targets(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data for conformal prediction framework.
        
        Returns:
            X: Covariates of shape (n, T, d) - all but last time point
            Y: Targets of shape (n, d) - last time point
        """
        X = data[:, :-1, :]  # All but last time point
        Y = data[:, -1, :]   # Last time point
        return X, Y
    
    def print_summary(self, data: np.ndarray, label: str = "Data"):
        """Simple summary statistics."""
        n, T_plus_1, d = data.shape
        print(f"\n{label} Summary:")
        print(f"  Shape: {data.shape} (n={n}, T+1={T_plus_1}, d={d})")
        for dim in range(d):
            initial_mean = np.mean(data[:, 0, dim])
            final_mean = np.mean(data[:, -1, dim])
            print(f"  Dim {dim}: Y₀ mean={initial_mean:.3f}, Y_T mean={final_mean:.3f}")


def main():
    """Minimal main function - same interface as ts_generator plus --d parameter."""
    parser = argparse.ArgumentParser(description='Multivariate extension of ts_generator')
    
    # Same parameters as ts_generator
    parser.add_argument('--n_train', type=int, default=100, help='Number of training series')
    parser.add_argument('--n_test', type=int, default=50, help='Number of test series') 
    parser.add_argument('--T', type=int, default=30, help='Length of time series')
    parser.add_argument('--shift_amount', type=float, default=2.0, help='Amount of mean shift')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR coefficient')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Noise standard deviation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--shift_time', type=int, default=0, help='When covariate shift occurs')
    
    # NEW: dimension parameter
    parser.add_argument('--d', type=int, default=2, help='Number of dimensions')
    
    args = parser.parse_args()
    
    print("Multivariate Time Series Generator (Minimal Extension)")
    print("="*60)
    print(f"Parameters:")
    print(f"  Dimensions: {args.d} (NEW - ts_generator is d=1)")
    print(f"  Training series: {args.n_train}")
    print(f"  Test series: {args.n_test}")
    print(f"  Time series length: {args.T + 1}")
    print(f"  AR coefficient: {args.ar_coef}")
    print(f"  Noise std: {args.noise_std}")
    print(f"  Shift amount: {args.shift_amount}")
    print(f"  Shift time: {args.shift_time}")
    
    # Initialize generator
    generator = MultivariateTimeSeriesGenerator(T=args.T, d=args.d, seed=args.seed)
    
    # Generate training data (same as ts_generator)
    print(f"\nGenerating {args.n_train} training time series...")
    train_data = generator.generate_ar_process(
        n=args.n_train,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std,
        initial_mean=0.0,
        initial_std=1.0,
        trend_coef=0.0
    )
    
    # Apply covariate shift (same logic as ts_generator)
    print(f"Applying covariate shift to {args.n_test} test series...")
    original_test, shifted_test = generator.introduce_covariate_shift(
        train_data[:args.n_test],
        shift_type='mean_shift',
        shift_params={'shift_amount': args.shift_amount},
        shift_time=args.shift_time,
        ar_coef=args.ar_coef,
        noise_std=args.noise_std,
        trend_coef=0.0
    )
    
    # Compute likelihood ratios (same as ts_generator)
    likelihood_ratios = generator.compute_likelihood_ratios(train_data, shifted_test)
    
    # Print results
    generator.print_summary(train_data, "Training Data")
    generator.print_summary(original_test, "Original Test Data") 
    generator.print_summary(shifted_test, "Shifted Test Data")
    
    print(f"\nLikelihood Ratios:")
    print(f"  Mean: {np.mean(likelihood_ratios):.3f}")
    print(f"  Std: {np.std(likelihood_ratios):.3f}")
    print(f"  Range: [{np.min(likelihood_ratios):.3f}, {np.max(likelihood_ratios):.3f}]")
    
    # Get data ready for conformal prediction
    X_train, Y_train = generator.get_covariates_and_targets(train_data)
    X_test, Y_test = generator.get_covariates_and_targets(shifted_test)
    
    print(f"\nReady for Conformal Prediction:")
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Y_train shape: {Y_train.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Y_test shape: {Y_test.shape}")
    
    print(f"\nFor visualization, run: python ts_generator.py --shift_time {args.shift_time} --T {args.T}")

if __name__ == "__main__":
    main()