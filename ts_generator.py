#!/usr/bin/env python3
"""
=============================================================================
UNIVARIATE TIME SERIES GENERATOR WITH VISUALIZATION
=============================================================================

PURPOSE: Generate univariate AR(1) time series driven by a time‐invariant
         Poisson covariate, plus comprehensive visualization for understanding
         conformal prediction concepts under covariate shift.

MODEL:
    X ∼ Pois(rate)
    Y₀ ∼ N(initial_mean, initial_std²)
    For t = 1,…,T:
        Y_t = α · Y_{t−1} + β · X + trend · t + ε_t,
        ε_t ∼ N(0, noise_std²)

COVARIATE SHIFT:
    Re-draw X under a new Poisson rate, then re-generate {Y_t} from Y₀ with
    the same recurrence to preserve P(Y_T ∣ Y_{0:T−1}) under the shifted covariate.

KEY FEATURES:
- Univariate AR(1) + time-invariant covariate generation
- Six-panel visualization of original vs. shifted series
- Summary statistics and regression checks
- Likelihood‐ratio weighting based on shift
- Flexible command‐line interface

DEFAULTS:
- T=30, n_train=100, n_test=50
- α=0.7, β=1.0, noise_std=0.2, trend=0.0
- covar_rate_original=1.0, covar_rate_shift=3.0
- seed=42, save_plot=False

USAGE:
  python ts_generator.py                           # basic run
  python ts_generator.py --covar_rate_shift 5.0    # larger Poisson shift

OUTPUT:
  Plots and console summaries comparing original vs. shifted covariate effects.
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, Optional, Callable
import pandas as pd
import argparse

class TimeSeriesGenerator:
    """
    Generate time series data for conformal prediction with covariate shift.
    """
    
    def __init__(self, T: int = 50, d: int = 1, seed: Optional[int] = None):
        self.T = T
        self.d = d
        if seed is not None:
            np.random.seed(seed)
    
    def generate_with_poisson_covariate(
        self,
        n: int,
        ar_coef: float = 0.7,
        beta: float = 1.0,
        covar_rate: float = 1.0,
        noise_std: float = 0.1,
        initial_mean: float = 0.0,
        initial_std: float = 1.0,
        trend_coef: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate n realizations of:
          X ~ Pois(covar_rate)
          Y₀ ~ N(initial_mean, initial_std²)
          Y_t = ar_coef·Y_{t-1} + beta·X + trend_coef·t + ε_t

        Returns:
            data: shape (n, T+1, d)
            X:    shape (n,)
        """
        # draw time‐invariant covariate
        X = np.random.poisson(lam=covar_rate, size=(n,))

        # allocate series: (n, T+1, d)
        data = np.zeros((n, self.T + 1, self.d))

        # initial Y₀
        data[:, 0, :] = np.random.normal(initial_mean, initial_std, (n, self.d))

        # build forward
        for t in range(1, self.T + 1):
            noise = np.random.normal(0, noise_std, (n, self.d))
            trend = trend_coef * t
            # incorporate X (reshape to broadcast over the last dim)
            data[:, t, :] = (
                ar_coef * data[:, t-1, :]
                + beta * X.reshape(n, 1)
                + trend
                + noise
            )

        return data, X


    def introduce_poisson_shift(
        self,
        original_data: np.ndarray,
        original_X: np.ndarray,
        shift_rate: float = 3.0,
        model_params: dict = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Introduce a covariate shift by re-drawing the time-invariant Poisson covariate
        at a new rate, and re-generating each series from the same initial Y₀.

        Args:
            original_data: array of shape (n, T+1, d) containing the original series.
            original_X:    array of shape (n,) containing the original covariate X ~ Pois(rate_orig).
            shift_rate:    new Poisson rate to draw X_shifted ~ Pois(shift_rate).
            model_params:  dict with keys:
                'ar_coef'    (float, default 0.7),
                'beta'       (float, default 1.0),
                'noise_std'  (float, default 0.1),
                'trend_coef' (float, default 0.0).

        Returns:
            shifted_data: array of shape (n, T+1, d) regenerated under the new X.
            shifted_X:    array of shape (n,) of new Poisson covariates.
        """
        if model_params is None:
            model_params = {
                'ar_coef':    0.7,
                'beta':       1.0,
                'noise_std':  0.1,
                'trend_coef': 0.0
            }

        n, T_plus_1, d = original_data.shape

        # Step 1: redraw X under the new Poisson rate
        shifted_X = np.random.poisson(lam=shift_rate, size=(n,))

        # Step 2: allocate and seed the shifted series with the same initial Y₀
        shifted_data = np.zeros_like(original_data)
        shifted_data[:, 0, :] = original_data[:, 0, :]

        # Step 3: regenerate forward using the AR(1) + covariate model
        for t in range(1, T_plus_1):
            noise = np.random.normal(
                loc=0,
                scale=model_params['noise_std'],
                size=(n, d)
            )
            trend = model_params['trend_coef'] * t
            # incorporate X (broadcast to d dimensions)
            X_term = model_params['beta'] * shifted_X.reshape(n, 1)
            shifted_data[:, t, :] = (
                model_params['ar_coef'] * shifted_data[:, t-1, :]
                + X_term
                + trend
                + noise
            )

        return shifted_data, shifted_X

    
    def compute_likelihood_ratios(self,
                                original_data: np.ndarray,
                                shifted_data: np.ndarray,
                                method: str = 'gaussian_kde') -> np.ndarray:
        """Compute likelihood ratios for weighting."""
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
    
    def visualize_covariate_shift(
        self,
        original_data: np.ndarray,
        shifted_data: np.ndarray,
        original_X: np.ndarray,
        shifted_X: np.ndarray,
        save_plot: bool = False,
        filename: str = "covariate_shift.png"
    ):
        """
        Create comprehensive visualization of the covariate shift.
        """
        fig = plt.figure(figsize=(18, 12))

        # 1) Sample original time series
        plt.subplot(2, 3, 1)
        n_plot = min(10, original_data.shape[0])
        for i in range(n_plot):
            plt.plot(original_data[i, :, 0], alpha=0.6, color='blue', linewidth=1)
        plt.title('Original Time Series', fontsize=12, fontweight='bold')
        plt.xlabel('Time'); plt.ylabel('Value')
        plt.grid(True, alpha=0.3)

        # 2) Sample shifted time series
        plt.subplot(2, 3, 2)
        n_plot = min(10, shifted_data.shape[0])
        T = shifted_data.shape[1]
        for i in range(n_plot):
            plt.plot(range(T), shifted_data[i, :, 0], alpha=0.6, color='red', linewidth=1)
        plt.title('Shifted Time Series', fontsize=12, fontweight='bold')
        plt.xlabel('Time'); plt.ylabel('Value')
        plt.grid(True, alpha=0.3)

        # 3) Overlay comparison
        plt.subplot(2, 3, 3)
        for i in range(min(5, original_data.shape[0])):
            plt.plot(original_data[i, :, 0],
                    alpha=0.7,
                    color='blue',
                    linewidth=1.5,
                    label='Original' if i == 0 else '')
        for i in range(min(5, shifted_data.shape[0])):
            plt.plot(range(T),
                    shifted_data[i, :, 0],
                    alpha=0.7,
                    color='red',
                    linewidth=1.5,
                    label='Shifted' if i == 0 else '')
        plt.title('Overlay Comparison', fontsize=12, fontweight='bold')
        plt.xlabel('Time'); plt.ylabel('Value')
        plt.legend(); plt.grid(True, alpha=0.3)

        # 4) Distribution of covariate X
        plt.subplot(2, 3, 4)
        bins = 20
        plt.hist(shifted_X,
                bins=bins,
                density=True,
                alpha=0.7,
                color='red',
                label='Shifted X')
        plt.hist(original_X,
                bins=bins,
                density=True,
                alpha=0.7,
                color='blue',
                label='Original X')
        plt.title('Distribution of Covariate X', fontsize=12, fontweight='bold')
        plt.xlabel('X value'); plt.ylabel('Density')
        plt.legend(); plt.grid(True, alpha=0.3)

        # 5) Distribution of final Yₜ
        plt.subplot(2, 3, 5)
        orig_final = original_data[:, -1, 0]
        shf_final = shifted_data[:, -1, 0]
        plt.hist(orig_final,
                bins=bins,
                density=True,
                alpha=0.7,
                color='blue',
                label='Original Y_T')
        plt.hist(shf_final,
                bins=bins,
                density=True,
                alpha=0.7,
                color='red',
                label='Shifted Y_T')
        plt.title('Distribution of Final Y Values', fontsize=12, fontweight='bold')
        plt.xlabel('Y_T'); plt.ylabel('Density')
        plt.legend(); plt.grid(True, alpha=0.3)

        # (Plot 6 intentionally omitted)

        plt.tight_layout()
        if save_plot:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved as {filename}")
        plt.show()
    
    def print_statistics(
        self,
        original_data: np.ndarray,
        shifted_data: np.ndarray,
        original_X: np.ndarray,
        shifted_X: np.ndarray
    ):
        """Print summary statistics for both Y and the Poisson covariate X."""
        print("\n" + "="*70)
        print("COVARIATE SHIFT ANALYSIS")
        print("="*70)

        # Covariate X statistics
        print(f"\nCovariate X (original): mean={np.mean(original_X):.3f}, std={np.std(original_X):.3f}")
        print(f"Covariate X (shifted) : mean={np.mean(shifted_X):.3f}, std={np.std(shifted_X):.3f}")

        # Y series statistics
        orig_initial = original_data[:, 0, 0]
        orig_final = original_data[:, -1, 0]
        shift_initial = shifted_data[:, 0, 0]
        shift_final = shifted_data[:, -1, 0]

        print(f"\nOriginal Y (n={original_data.shape[0]})")
        print(f"  Y₀: mean={np.mean(orig_initial):.3f}, std={np.std(orig_initial):.3f}")
        print(f"  Y_T: mean={np.mean(orig_final):.3f}, std={np.std(orig_final):.3f}")

        print(f"\nShifted Y (n={shifted_data.shape[0]})")
        print(f"  Y₀: mean={np.mean(shift_initial):.3f}, std={np.std(shift_initial):.3f}")
        print(f"  Y_T: mean={np.mean(shift_final):.3f}, std={np.std(shift_final):.3f}")

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate AR(1) series with Poisson covariate shift'
    )
    parser.add_argument('--n_train', type=int, default=100,
                        help='Number of training series')
    parser.add_argument('--n_test', type=int, default=50,
                        help='Number of test series')
    parser.add_argument('--T', type=int, default=30,
                        help='Length of each series (T + 1 points)')
    parser.add_argument('--ar_coef', type=float, default=0.7,
                        help='AR(1) coefficient α')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Covariate coefficient β')
    parser.add_argument('--noise_std', type=float, default=0.2,
                        help='Noise standard deviation')
    parser.add_argument('--trend_coef', type=float, default=0.0,
                        help='Linear trend coefficient')
    parser.add_argument('--covar_rate', type=float, default=1.0,
                        help='Original Poisson rate for X')
    parser.add_argument('--covar_rate_shift', type=float, default=3.0,
                        help='Shifted Poisson rate for X')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_plot', action='store_true',
                        help='Save the visualization to file')

    args = parser.parse_args()

    print("Time Series Generator w/ Poisson Covariate Shift")
    print("="*50)
    print(f"Training series: {args.n_train}")
    print(f"Test series    : {args.n_test}")
    print(f"Series length  : {args.T + 1}")
    print(f"AR coef (α)    : {args.ar_coef}")
    print(f"Beta (β)       : {args.beta}")
    print(f"Noise std      : {args.noise_std}")
    print(f"Trend coef     : {args.trend_coef}")
    print(f"X rate         : {args.covar_rate} → {args.covar_rate_shift}")
    print(f"Seed           : {args.seed}")

    # set up
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)

    # generate original
    train_data, train_X = generator.generate_with_poisson_covariate(
        n=args.n_train,
        ar_coef=args.ar_coef,
        beta=args.beta,
        covar_rate=args.covar_rate,
        noise_std=args.noise_std,
        initial_mean=0.0,
        initial_std=1.0,
        trend_coef=args.trend_coef
    )
    print(f"\nGenerated {args.n_train} series, shape={train_data.shape}")

    # select test
    test_data = train_data[:args.n_test]
    test_X    = train_X[:args.n_test]

    # apply shift
    shifted_data, shifted_X = generator.introduce_poisson_shift(
        test_data,
        test_X,
        shift_rate=args.covar_rate_shift,
        model_params={
            'ar_coef':    args.ar_coef,
            'beta':       args.beta,
            'noise_std':  args.noise_std,
            'trend_coef': args.trend_coef
        }
    )

    # stats
    generator.print_statistics(test_data,
                                shifted_data,
                                test_X,
                                shifted_X)

    # likelihood ratios remain Y-based
    lr = generator.compute_likelihood_ratios(train_data, shifted_data)
    print(f"\nLikelihood ratios (on Y₀): mean={np.mean(lr):.3f}, std={np.std(lr):.3f}")

    # visualization
    print("\nGenerating visualization...")
    generator.visualize_covariate_shift(
        test_data,
        shifted_data,
        test_X,
        shifted_X,
        save_plot=args.save_plot,
        filename="covariate_shift.png"
    )

    print("\nDone.")

if __name__ == "__main__":
    main()
