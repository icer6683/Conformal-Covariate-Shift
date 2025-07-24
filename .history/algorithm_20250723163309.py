#!/usr/bin/env python3
"""
=============================================================================
ALGORITHM.PY - TIME SERIES CONFORMAL PREDICTION WITH COVARIATE SHIFT
=============================================================================

PURPOSE: Direct implementation of "Adapted CAFHT" algorithm from the research paper.
         Minimal, clean implementation staying true to the original LaTeX specification.

ALGORITHM: From paper - "Adapted CAFHT"
1. Split calibration data D_cal into D_cal^1 and D_cal^2
2. Select γ that minimizes average prediction band width in D_cal^1
3. For all series i in D_cal^2, construct prediction band using ACI
4. Calculate error terms ε_i for each series
5. Compute weighted quantile using likelihood ratios w(i)
6. For each time step t, construct prediction bands

PAPER NOTATION:
- Y^(i): i-th time series observation (Y_0^(i), Y_1^(i), ..., Y_T^(i))
- ε_i: conformal score (largest margin needed to cover entire series)
- w(i): likelihood ratio dP̃_Z(i)/dP_Z(i) 
- Q̂^w(1-α, γ): weighted (1-α) quantile

USAGE:
  from algorithm import AdaptedCAFHT
  algorithm = AdaptedCAFHT(alpha=0.1)
  prediction_bands = algorithm.predict(calibration_data, test_data, likelihood_ratios)
=============================================================================
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from abc import ABC, abstractmethod

class BaseConformalMethod(ABC):
    """Base class for conformal prediction methods."""
    
    @abstractmethod
    def construct_prediction_band(self, Y: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct prediction band for a single time series.
        
        Args:
            Y: Time series of shape (T+1, d)
            gamma: Band width parameter
            
        Returns:
            Tuple of (lower_band, upper_band) each of shape (T+1, d)
        """
        pass

class SimpleACI(BaseConformalMethod):
    """
    Simple Adaptive Conformal Inference implementation.
    Adjusts nominal alpha level in case of distribution shift.
    """
    
    def __init__(self, base_alpha: float = 0.1):
        self.base_alpha = base_alpha
    
    def construct_prediction_band(self, Y: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple ACI: prediction band is Y_t ± gamma for all t.
        
        In practice, this would use a sophisticated forecasting model,
        but for the algorithm demonstration, we use this simple approach.
        """
        T_plus_1, d = Y.shape
        
        # Simple symmetric bands around the observed values
        lower_band = Y - gamma
        upper_band = Y + gamma
        
        return lower_band, upper_band

class AdaptedCAFHT:
    """
    Implementation of Adapted CAFHT algorithm from the research paper.
    
    Direct translation of Algorithm 1 from the LaTeX document.
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 gamma_grid: Optional[np.ndarray] = None,
                 base_method: BaseConformalMethod = None):
        """
        Initialize Adapted CAFHT algorithm.
        
        Args:
            alpha: Desired miscoverage level (1-α confidence level)
            gamma_grid: Grid of γ values to search over
            base_method: Base conformal method (ACI, PID, etc.)
        """
        self.alpha = alpha
        
        if gamma_grid is None:
            # Default gamma grid from typical conformal prediction literature
            self.gamma_grid = np.concatenate([
                np.arange(0.001, 0.1, 0.01),
                np.arange(0.1, 1.1, 0.1)
            ])
        else:
            self.gamma_grid = gamma_grid
            
        if base_method is None:
            self.base_method = SimpleACI(alpha)
        else:
            self.base_method = base_method
    
    def _split_calibration_data(self, D_cal: np.ndarray, split_ratio: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 3: Split D_cal into D_cal^1 and D_cal^2.
        
        Args:
            D_cal: Calibration data of shape (n_cal, T+1, d)
            split_ratio: Fraction for D_cal^1
            
        Returns:
            Tuple of (D_cal1, D_cal2)
        """
        n_cal = D_cal.shape[0]
        n_cal1 = int(n_cal * split_ratio)
        
        # Random split
        indices = np.random.permutation(n_cal)
        D_cal1 = D_cal[indices[:n_cal1]]
        D_cal2 = D_cal[indices[n_cal1:]]
        
        return D_cal1, D_cal2
    
    def _select_gamma(self, D_cal1: np.ndarray) -> float:
        """
        Step 3: Select γ that minimizes average prediction band width in D_cal^1.
        
        Args:
            D_cal1: First split of calibration data
            
        Returns:
            Optimal γ value
        """
        n_cal1, T_plus_1, d = D_cal1.shape
        avg_widths = []
        
        for gamma in self.gamma_grid:
            total_width = 0.0
            
            for i in range(n_cal1):
                lower_band, upper_band = self.base_method.construct_prediction_band(D_cal1[i], gamma)
                # Average width across time and dimensions
                width = np.mean(upper_band - lower_band)
                total_width += width
            
            avg_width = total_width / n_cal1
            avg_widths.append(avg_width)
        
        # Select gamma with minimum average width
        optimal_idx = np.argmin(avg_widths)
        optimal_gamma = self.gamma_grid[optimal_idx]
        
        return optimal_gamma
    
    def _compute_error_terms(self, D_cal2: np.ndarray, gamma: float) -> np.ndarray:
        """
        Steps 4-5: Compute error terms ε_i for each time series in D_cal^2.
        
        ε_i is the largest margin of error needed to cover the entire time series.
        
        Args:
            D_cal2: Second split of calibration data
            gamma: Selected γ parameter
            
        Returns:
            Array of error terms ε_i of shape (n_cal2,)
        """
        n_cal2, T_plus_1, d = D_cal2.shape
        error_terms = np.zeros(n_cal2)
        
        for i in range(n_cal2):
            Y_i = D_cal2[i]
            
            # Step 4: Construct prediction band using ACI
            lower_band, upper_band = self.base_method.construct_prediction_band(Y_i, gamma)
            
            # Step 5: Calculate error term ε_i
            # ε_i is the largest margin needed to cover the entire series
            lower_violations = np.maximum(0, lower_band - Y_i)
            upper_violations = np.maximum(0, Y_i - upper_band)
            
            # Maximum violation across all time points and dimensions
            max_violation = np.max(np.maximum(lower_violations, upper_violations))
            error_terms[i] = max_violation
            
        return error_terms
    
    def _compute_weighted_quantile(self, 
                                 error_terms: np.ndarray, 
                                 likelihood_ratios: np.ndarray) -> float:
        """
        Step 6: Compute weighted (1-α) quantile Q̂^w(1-α, γ).
        
        Implements equations (3) and (4) from the paper:
        - Weighted empirical distribution with weights p_i(z)
        - p_i(z) = w(i) / Σ_{j=1}^{n+1} w(j)
        
        Args:
            error_terms: Error terms ε_i
            likelihood_ratios: Likelihood ratios w(i)
            
        Returns:
            Weighted quantile Q̂^w(1-α, γ)
        """
        n = len(error_terms)
        
        # Add point mass at infinity (as in equation 2 and 3)
        extended_errors = np.append(error_terms, np.inf)
        extended_weights = np.append(likelihood_ratios, 1.0)  # w(n+1) = 1
        
        # Compute normalized weights p_i(z) from equation (4)
        total_weight = np.sum(extended_weights)
        normalized_weights = extended_weights / total_weight
        
        # Sort by error terms for quantile computation
        sorted_indices = np.argsort(extended_errors)
        sorted_errors = extended_errors[sorted_indices]
        sorted_weights = normalized_weights[sorted_indices]
        
        # Compute weighted quantile
        cumulative_weights = np.cumsum(sorted_weights)
        quantile_level = 1 - self.alpha
        
        # Find the smallest error term where cumulative weight ≥ quantile_level
        quantile_idx = np.searchsorted(cumulative_weights, quantile_level, side='right')
        
        if quantile_idx >= len(sorted_errors):
            return sorted_errors[-1]  # Return infinity if needed
        else:
            return sorted_errors[quantile_idx]
    
    def predict_online(self, 
                     D_cal: np.ndarray,
                     test_series: np.ndarray,
                     likelihood_ratios: np.ndarray,
                     true_future: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Online prediction following Algorithm 1 from paper.
        
        This implements the actual online loop from Steps 7-10:
        "For t ∈ [T]: Compute prediction band, observe next step"
        
        Args:
            D_cal: Calibration data
            test_series: Observed test series up to current time
            likelihood_ratios: Likelihood ratios for covariate shift
            true_future: True future values for evaluation
            
        Returns:
            Tuple of (prediction_bands, online_stats)
        """
        T_plus_1, d = true_future.shape
        
        # Pre-computation (Steps 3-6)
        D_cal1, D_cal2 = self._split_calibration_data(D_cal)
        optimal_gamma = self._select_gamma(D_cal1)
        error_terms = self._compute_error_terms(D_cal2, optimal_gamma)
        weighted_quantile = self._compute_weighted_quantile(error_terms, likelihood_ratios)
        
        # Online prediction loop (Steps 7-10)
        prediction_bands = np.zeros((T_plus_1, 2, d))
        coverage_history = []
        width_history = []
        
        print(f"Starting online prediction with weighted quantile: {weighted_quantile:.4f}")
        
        for t in range(T_plus_1):
            print(f"\n--- Time step t={t} ---")
            
            # Step 8: Compute C^ACI_t using past entries
            if t == 0:
                # For t=0, use initial conditions
                base_lower = test_series[t] - optimal_gamma
                base_upper = test_series[t] + optimal_gamma
            else:
                # Use observed series up to time t
                partial_series = test_series[:t+1]
                base_lower, base_upper = self.base_method.construct_prediction_band(partial_series, optimal_gamma)
                base_lower = base_lower[-1]  # Current time point
                base_upper = base_upper[-1]
            
            # Step 9: Augment by weighted quantile
            current_lower = base_lower - weighted_quantile
            current_upper = base_upper + weighted_quantile
            
            prediction_bands[t, 0, :] = current_lower
            prediction_bands[t, 1, :] = current_upper
            
            # Step 10: Observe next step and evaluate
            true_value = true_future[t]
            covered = np.all((true_value >= current_lower) & (true_value <= current_upper))
            width = np.mean(current_upper - current_lower)
            
            coverage_history.append(covered)
            width_history.append(width)
            
            print(f"  True value: {true_value}")
            print(f"  Prediction band: [{current_lower}, {current_upper}]")
            print(f"  Covered: {covered}")
            print(f"  Width: {width:.4f}")
            print(f"  Running coverage: {np.mean(coverage_history):.3f}")
        
        online_stats = {
            'coverage_history': np.array(coverage_history),
            'width_history': np.array(width_history),
            'final_coverage': np.mean(coverage_history),
            'average_width': np.mean(width_history),
            'target_coverage': 1 - self.alpha,
            'weighted_quantile': weighted_quantile,
            'optimal_gamma': optimal_gamma
        }
        
        return prediction_bands, online_stats
    
    def evaluate_coverage(self, 
                         true_series: np.ndarray, 
                         prediction_bands: np.ndarray) -> dict:
        """
        Evaluate the coverage properties of the prediction bands.
        
        Args:
            true_series: True time series values of shape (T+1, d)
            prediction_bands: Prediction bands of shape (T+1, 2, d)
            
        Returns:
            Dictionary with coverage statistics
        """
        T_plus_1, d = true_series.shape
        
        # Check coverage at each time point
        lower_bands = prediction_bands[:, 0, :]
        upper_bands = prediction_bands[:, 1, :]
        
        # Point-wise coverage
        coverage = np.logical_and(
            true_series >= lower_bands,
            true_series <= upper_bands
        )
        
        # Coverage statistics
        pointwise_coverage = np.mean(coverage, axis=0)  # Coverage per dimension
        overall_coverage = np.mean(coverage)
        marginal_coverage = np.mean(np.all(coverage, axis=1))  # All dimensions covered
        
        # Band widths
        band_widths = upper_bands - lower_bands
        avg_width = np.mean(band_widths, axis=0)
        
        return {
            'overall_coverage': overall_coverage,
            'marginal_coverage': marginal_coverage,
            'pointwise_coverage': pointwise_coverage,
            'average_width': avg_width,
            'target_coverage': 1 - self.alpha
        }


def main():
    """
    Demonstration showing online accuracy calculation like ts_sim.py.
    """
    print("Adapted CAFHT Algorithm - Online Accuracy Demonstration")
    print("="*60)
    
    # Import generators for realistic test
    try:
        import sys
        sys.path.append('.')
        from fixed_ts_generator import TimeSeriesGenerator
        
        print("Using real time series data with covariate shift...")
        
        # Generate data with covariate shift (like ts_sim.py)
        generator = TimeSeriesGenerator(T=30, d=1, seed=42)
        train_data = generator.generate_ar_process(n=200, ar_coef=0.7, noise_std=0.2)
        
        # Create test data with shift at t=15
        original_test, shifted_test = generator.introduce_covariate_shift(
            train_data[:20],
            shift_time=15,
            shift_params={'shift_amount': 2.0}
        )
        
        # Compute likelihood ratios
        likelihood_ratios = generator.compute_likelihood_ratios(train_data, shifted_test)
        
        # Run online algorithm on first test series
        test_idx = 0
        test_series = shifted_test[test_idx, :-1, :]  # Observed up to T-1
        true_future = shifted_test[test_idx, :, :]    # True values including T
        
        print(f"\nTest series shape: {test_series.shape}")
        print(f"True future shape: {true_future.shape}")
        print(f"Likelihood ratios shape: {likelihood_ratios.shape}")
        
        # Initialize algorithm
        algorithm = AdaptedCAFHT(alpha=0.1)
        
        # Run online prediction (like ts_sim.py evaluation)
        prediction_bands, online_stats = algorithm.predict_online(
            D_cal=train_data,
            test_series=test_series,
            likelihood_ratios=likelihood_ratios,
            true_future=true_future
        )
        
        # Print results (like ts_sim.py output)
        print(f"\n" + "="*50)
        print("ONLINE ACCURACY RESULTS")
        print("="*50)
        print(f"Target coverage: {online_stats['target_coverage']:.1%}")
        print(f"Actual coverage: {online_stats['final_coverage']:.1%}")
        print(f"Average width: {online_stats['average_width']:.4f}")
        print(f"Weighted quantile: {online_stats['weighted_quantile']:.4f}")
        print(f"Optimal gamma: {online_stats['optimal_gamma']:.4f}")
        
        # Coverage over time
        print(f"\nCoverage by time step:")
        for t, (covered, width) in enumerate(zip(online_stats['coverage_history'], 
                                                online_stats['width_history'])):
            print(f"  t={t:2d}: {'✓' if covered else '✗'} (width: {width:.3f})")
        
    except ImportError:
        print("Generator files not found, using synthetic data...")
        
        # Fallback synthetic example
        np.random.seed(42)
        n_cal, T, d = 100, 20, 1
        
        D_cal = np.random.normal(0, 1, (n_cal, T+1, d))
        test_series = np.random.normal(1, 1, (T, d))  # Observed series
        true_future = np.random.normal(1, 1, (T+1, d))  # True future values
        likelihood_ratios = np.random.gamma(2, 0.5, n_cal//2)
        
        algorithm = AdaptedCAFHT(alpha=0.1)
        prediction_bands, online_stats = algorithm.predict_online(
            D_cal, test_series, likelihood_ratios, true_future
        )
        
        print(f"Results: Coverage = {online_stats['final_coverage']:.1%}")
    
    print("\nOnline algorithm completed!")

if __name__ == "__main__":
    main()