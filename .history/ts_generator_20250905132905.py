#!/usr/bin/env python3
"""
=============================================================================
UNIVARIATE TIME SERIES GENERATOR WITH VISUALIZATION
=============================================================================

PURPOSE:
  Generate univariate AR(1) time series with a covariate X and visualize
  covariate shift effects. Supports two covariate modes:
    1) static  : time-invariant X
    2) dynamic : time-varying X_t following its own AR(1)

MODELS:

  Static-X mode (time-invariant):
    X ~ Pois(covar_rate)
    Y₀ ~ N(initial_mean, initial_std²)
    For t = 1,…,T:
        Y_t = α · Y_{t−1} + β · X + trend · t + ε_t,
        ε_t ∼ N(0, noise_std²)

  Dynamic-X mode (time-varying):
    X₀ ~ Pois(x0_lambda)
    For t = 0,…,T−1:
        X_{t+1} = ρ_X · X_t + trend_X · t + η_t,      η_t ∼ N(0, x_noise_std²)
        Y_{t+1} = α · Y_t + β · X_t + trend · t + ε_t, ε_t ∼ N(0, noise_std²)

COVARIATE SHIFT:

  Static-X shift:
    Re-draw X under a new Poisson rate (covar_rate_shift), then re-generate {Y_t}
    from the same Y₀ with unchanged (α, β, trend, noise_std).

  Dynamic-X shift:
    Re-generate the entire covariate path {X_t} using shifted parameters
    (e.g., ρ_X → x_rate_shift, trend_X → x_trend_shift, optionally x_noise_std_shift,
    and optionally re-draw X₀ via x0_lambda_shift), then re-generate {Y_t}
    from the same Y₀ with unchanged (α, β, trend, noise_std).

KEY FEATURES:
- Choice of covariate mode: static (time-invariant) or dynamic (time-varying AR(1))
- Six-panel visualization comparing original vs. shifted series
- Summary statistics for Y and X; likelihood-ratio weighting (default based on Y₀)
- Consistent interfaces for generation, shifting, statistics, and plotting
- Flexible command-line interface

DEFAULTS:
- T=30, n_train=100, n_test=50
- α=0.7, β=1.0, noise_std=0.2, trend=0.0
- covariate_mode="static"
- Static-X:   covar_rate=1.0, covar_rate_shift=3.0
- Dynamic-X:  x_rate=0.7, x_trend=0.0, x_noise_std=0.2, x0_lambda=1.0
               x_rate_shift=x_rate, x_trend_shift=x_trend, x_noise_std_shift=x_noise_std,
               x0_lambda_shift=x0_lambda
- seed=42, save_plot=False

USAGE:
  # Static-X (time-invariant), basic run
  python ts_generator.py

  # Static-X with larger Poisson shift
  python ts_generator.py --covar_rate_shift 5.0

  # Dynamic-X (time-varying) with shifted X dynamics
  python ts_generator.py --covariate_mode dynamic --x_rate 0.6 --x_rate_shift 0.9

  # Dynamic-X with trend shift in X_t
  python ts_generator.py --covariate_mode dynamic --x_trend 0.05 --x_trend_shift 0.15

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
        ar_coef: float = 1.0,
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

    def generate_with_dynamic_covariate(
        self,
        n: int,
        ar_coef: float = 0.7,       # α for Y
        beta: float = 1.0,          # β linking X_t to Y_{t+1}
        noise_std: float = 0.1,     # σ for Y noise ε_t
        initial_mean: float = 0.0,  # mean of Y_0
        initial_std: float = 1.0,   # std of Y_0
        trend_coef: float = 0.0,    # trend for Y (multiplied by t)
        x_rate: float = 0.7,        # ρ_X for X AR(1)
        x_trend: float = 0.0,       # trend_X (multiplied by t)
        x_noise_std: float = 0.2,   # σ′ for X noise η_t
        x0_lambda: float = 1.0      # Poisson rate for X_0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate n series under the dynamic-X model:

            X_0 ~ Poisson(x0_lambda)
            For t = 0,...,T-1:
                X_{t+1} = x_rate * X_t + x_trend * t + η_t,   η_t ~ N(0, x_noise_std^2)
                Y_{t+1} = ar_coef * Y_t + beta * X_t + trend_coef * t + ε_t,
                        ε_t ~ N(0, noise_std^2)

        Returns:
            Y:         array of shape (n, T+1, d)
            X_series:  array of shape (n, T+1)  (full X_t path)
        """
        nT = self.T + 1

        # Allocate outputs
        Y = np.zeros((n, nT, self.d), dtype=float)
        X_series = np.zeros((n, nT), dtype=float)

        # Initial states
        X_series[:, 0] = np.random.poisson(lam=x0_lambda, size=n).astype(float)
        Y[:, 0, :] = np.random.normal(loc=initial_mean, scale=initial_std, size=(n, self.d))

        # Forward simulation
        for t in range(self.T):  # t = 0,...,T-1 to produce t+1
            # X update
            eta = np.random.normal(loc=0.0, scale=x_noise_std, size=n)
            X_series[:, t + 1] = x_rate * X_series[:, t] + x_trend * t + eta

            # Y update (uses X_t)
            eps = np.random.normal(loc=0.0, scale=noise_std, size=(n, self.d))
            trend = trend_coef * t
            Y[:, t + 1, :] = (
                ar_coef * Y[:, t, :]
                + beta * X_series[:, t].reshape(n, 1)
                + trend
                + eps
            )

        return Y, X_series

    def introduce_covariate_shift(
        self,
        original_Y: np.ndarray,                 # (n, T+1, d)
        original_X,                             # static: (n,), dynamic: (n, T+1) — tolerated
        covariate_mode: str = "static",         # "static" or "dynamic"
        model_params: dict | None = None,       # for Y: {'ar_coef','beta','noise_std','trend_coef'}
        shift_params: dict | None = None        # static: {'shift_rate'}
                                                # dynamic: {'x_rate_shift','x_trend_shift',
                                                #           'x_noise_std_shift','x0_lambda_shift',
                                                #           'x0_redraw'(bool, default True)}
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply covariate shift and regenerate Y from the same Y₀.

        Returns:
            shifted_Y         : (n, T+1, d)
            shifted_X_series  : (n, T+1)  (constant in static mode; AR(1) path in dynamic mode)
        """
        import numpy as np

        if model_params is None:
            model_params = {
                'ar_coef':    0.7,
                'beta':       1.0,
                'noise_std':  0.1,
                'trend_coef': 0.0
            }
        if shift_params is None:
            shift_params = {}

        n, T_plus_1, d = original_Y.shape
        T = T_plus_1 - 1

        # Always reuse the same initial Y₀
        shifted_Y = np.zeros_like(original_Y)
        shifted_Y[:, 0, :] = original_Y[:, 0, :]

        # Helper to broadcast X_t into Y-update term
        def y_step(y_prev: np.ndarray, x_t: np.ndarray, t: int) -> np.ndarray:
            noise = np.random.normal(0.0, model_params['noise_std'], size=(n, d))
            trend = model_params['trend_coef'] * t
            return (
                model_params['ar_coef'] * y_prev
                + model_params['beta'] * x_t.reshape(n, 1)
                + trend
                + noise
            )

        mode = covariate_mode.lower()
        if mode == "static":
            # ----- Static-X shift: X ~ Pois(shift_rate), constant across time -----
            shift_rate = shift_params.get('shift_rate', 3.0)
            shifted_X0 = np.random.poisson(lam=shift_rate, size=n).astype(float)

            # Build constant X_t series
            shifted_X_series = np.repeat(shifted_X0[:, None], repeats=T_plus_1, axis=1)

            # Regenerate Y using constant X_t
            for t in range(T):
                shifted_Y[:, t + 1, :] = y_step(shifted_Y[:, t, :], shifted_X_series[:, t], t)

            return shifted_Y, shifted_X_series

        elif mode == "dynamic":
            # ----- Dynamic-X shift: regenerate AR(1) X_t path with shifted params -----
            x_rate_shift       = shift_params.get('x_rate_shift', 0.7)
            x_trend_shift      = shift_params.get('x_trend_shift', 0.0)
            x_noise_std_shift  = shift_params.get('x_noise_std_shift', 0.2)
            x0_lambda_shift    = shift_params.get('x0_lambda_shift', 1.0)
            x0_redraw          = shift_params.get('x0_redraw', True)  # per your assumption

            shifted_X_series = np.zeros((n, T_plus_1), dtype=float)

            # Decide X₀ under shift
            if x0_redraw:
                shifted_X_series[:, 0] = np.random.poisson(lam=x0_lambda_shift, size=n).astype(float)
            else:
                # If not redrawing, reuse original X₀ (support (n,) or (n,T+1))
                if isinstance(original_X, np.ndarray) and original_X.ndim == 2:
                    shifted_X_series[:, 0] = original_X[:, 0].astype(float)
                else:
                    shifted_X_series[:, 0] = np.asarray(original_X, dtype=float)

            # Evolve X and Y
            for t in range(T):
                eta = np.random.normal(0.0, x_noise_std_shift, size=n)
                shifted_X_series[:, t + 1] = x_rate_shift * shifted_X_series[:, t] + x_trend_shift * t + eta
                shifted_Y[:, t + 1, :] = y_step(shifted_Y[:, t, :], shifted_X_series[:, t], t)

            return shifted_Y, shifted_X_series

        else:
            raise ValueError(f"Unknown covariate_mode '{covariate_mode}'. Use 'static' or 'dynamic'.")


    
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
        original_X: np.ndarray,   # static: (n,) or (n,T+1) const; dynamic: (n,T+1)
        shifted_X: np.ndarray,    # static: (n,) or (n,T+1) const; dynamic: (n,T+1)
        covariate_mode: str = "static",
        save_plot: bool = False,
        filename: str = "covariate_shift.png"
    ):
        """
        Create comprehensive visualization of the covariate shift.
        - Plots sample Y paths (original vs shifted)
        - For X:
            * dynamic: overlays X_t sequences
            * static : overlays histograms of X (uses X0 if X-series is provided)
        - Shows histogram of Y_T (original vs shifted)
        """
        fig = plt.figure(figsize=(18, 12))

        # 1) Sample original Y time series
        plt.subplot(2, 3, 1)
        n_plot = min(10, original_data.shape[0])
        for i in range(n_plot):
            plt.plot(original_data[i, :, 0], alpha=0.6, color='blue', linewidth=1)
        plt.title('Original Time Series (Y)', fontsize=12, fontweight='bold')
        plt.xlabel('Time'); plt.ylabel('Y')
        plt.grid(True, alpha=0.3)

        # 2) Sample shifted Y time series
        plt.subplot(2, 3, 2)
        n_plot = min(10, shifted_data.shape[0])
        T = shifted_data.shape[1]
        for i in range(n_plot):
            plt.plot(range(T), shifted_data[i, :, 0], alpha=0.6, color='red', linewidth=1)
        plt.title('Shifted Time Series (Y)', fontsize=12, fontweight='bold')
        plt.xlabel('Time'); plt.ylabel('Y')
        plt.grid(True, alpha=0.3)

        # 3) Overlay comparison for Y
        plt.subplot(2, 3, 3)
        for i in range(min(5, original_data.shape[0])):
            plt.plot(original_data[i, :, 0],
                    alpha=0.7, color='blue', linewidth=1.5,
                    label='Original Y' if i == 0 else '')
        for i in range(min(5, shifted_data.shape[0])):
            plt.plot(range(T),
                    shifted_data[i, :, 0],
                    alpha=0.7, color='red', linewidth=1.5,
                    label='Shifted Y' if i == 0 else '')
        plt.title('Overlay Comparison (Y)', fontsize=12, fontweight='bold')
        plt.xlabel('Time'); plt.ylabel('Y')
        plt.legend(); plt.grid(True, alpha=0.3)

        # 4) X visualization: branch by covariate_mode
        plt.subplot(2, 3, 4)
        mode = covariate_mode.lower()
        if mode == "dynamic":
            # Overlay X_t sequences
            n_plot_x = min(10, original_X.shape[0])
            for i in range(n_plot_x):
                plt.plot(original_X[i, :], alpha=0.6, color='blue', linewidth=1)
            n_plot_x = min(10, shifted_X.shape[0])
            for i in range(n_plot_x):
                plt.plot(shifted_X[i, :], alpha=0.6, color='red', linewidth=1)
            plt.title('Covariate X_t Sequences (Overlay)', fontsize=12, fontweight='bold')
            plt.xlabel('Time'); plt.ylabel('X_t')
            plt.grid(True, alpha=0.3)
        elif mode == "static":
            # Histogram of X (use X0 if a full series is provided)
            bins = 20
            def as_scalar_x(arr: np.ndarray) -> np.ndarray:
                if arr.ndim == 2:
                    return arr[:, 0]  # X0
                return arr
            x_orig = as_scalar_x(original_X)
            x_shift = as_scalar_x(shifted_X)
            # draw shifted first, then original on top
            plt.hist(x_shift, bins=bins, density=True, alpha=0.7, color='red',  label='Shifted X')
            plt.hist(x_orig,  bins=bins, density=True, alpha=0.7, color='blue', label='Original X')
            plt.title('Distribution of Covariate X', fontsize=12, fontweight='bold')
            plt.xlabel('X value'); plt.ylabel('Density')
            plt.legend(); plt.grid(True, alpha=0.3)
        else:
            raise ValueError(f"Unknown covariate_mode '{covariate_mode}'. Use 'static' or 'dynamic'.")

        # 5) Distribution of final Y_T
        plt.subplot(2, 3, 5)
        bins = 20
        orig_final = original_data[:, -1, 0]
        shf_final  = shifted_data[:, -1, 0]
        plt.hist(orig_final, bins=bins, density=True, alpha=0.7, color='blue', label='Original Y_T')
        plt.hist(shf_final,  bins=bins, density=True, alpha=0.7, color='red',  label='Shifted  Y_T')
        plt.title('Distribution of Final Y_T', fontsize=12, fontweight='bold')
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
        shifted_X: np.ndarray,
        covariate_mode: str = "static"
    ):
        """Print summary statistics for Y and X under static or dynamic covariate modes."""
        print("\n" + "=" * 70)
        print("COVARIATE SHIFT ANALYSIS")
        print("=" * 70)

        mode = covariate_mode.lower()

        # ---- Covariate X statistics ----
        if mode == "static":
            # Accept (n,) or (n, T+1) constant series; if series, take X0
            def x_scalar(arr: np.ndarray) -> np.ndarray:
                return arr[:, 0] if arr.ndim == 2 else arr

            x_orig = x_scalar(original_X)
            x_shift = x_scalar(shifted_X)

            print("\nCovariate X (static):")
            print(f"  Original: mean={np.mean(x_orig):.3f}, std={np.std(x_orig):.3f}")
            print(f"  Shifted : mean={np.mean(x_shift):.3f}, std={np.std(x_shift):.3f}")

        elif mode == "dynamic":
            # Expect full paths (n, T+1)
            if original_X.ndim != 2 or shifted_X.ndim != 2:
                raise ValueError("In dynamic mode, original_X and shifted_X must have shape (n, T+1).")

            x0_orig, xT_orig = original_X[:, 0], original_X[:, -1]
            x0_shift, xT_shift = shifted_X[:, 0], shifted_X[:, -1]

            x_all_orig = original_X.reshape(-1)
            x_all_shift = shifted_X.reshape(-1)

            print("\nCovariate X (dynamic):")
            print(f"  X0      - Original: mean={np.mean(x0_orig):.3f}, std={np.std(x0_orig):.3f} | "
                f"Shifted: mean={np.mean(x0_shift):.3f}, std={np.std(x0_shift):.3f}")
            print(f"  XT      - Original: mean={np.mean(xT_orig):.3f}, std={np.std(xT_orig):.3f} | "
                f"Shifted: mean={np.mean(xT_shift):.3f}, std={np.std(xT_shift):.3f}")
            print(f"  All t   - Original: mean={np.mean(x_all_orig):.3f}, std={np.std(x_all_orig):.3f} | "
                f"Shifted: mean={np.mean(x_all_shift):.3f}, std={np.std(x_all_shift):.3f}")

        else:
            raise ValueError(f"Unknown covariate_mode '{covariate_mode}'. Use 'static' or 'dynamic'.")

        # ---- Y series statistics ----
        orig_initial = original_data[:, 0, 0]
        orig_final   = original_data[:, -1, 0]
        shft_initial = shifted_data[:, 0, 0]
        shft_final   = shifted_data[:, -1, 0]

        print(f"\nOriginal Y (n={original_data.shape[0]})")
        print(f"  Y₀ : mean={np.mean(orig_initial):.3f}, std={np.std(orig_initial):.3f}")
        print(f"  Y_T: mean={np.mean(orig_final):.3f},  std={np.std(orig_final):.3f}")

        print(f"\nShifted Y (n={shifted_data.shape[0]})")
        print(f"  Y₀ : mean={np.mean(shft_initial):.3f}, std={np.std(shft_initial):.3f}")
        print(f"  Y_T: mean={np.mean(shft_final):.3f},  std={np.std(shft_final):.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Generate AR(1) series with covariate X (static or dynamic) and apply covariate shift'
    )
    # common
    parser.add_argument('--n_train', type=int, default=100, help='Number of training series')
    parser.add_argument('--n_test',  type=int, default=50,  help='Number of test series')
    parser.add_argument('--T',       type=int, default=30,  help='Length of each series (T + 1 points)')
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR(1) coefficient α for Y')
    parser.add_argument('--beta',    type=float, default=1.0, help='Covariate coefficient β (X → Y)')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Std dev of Y-noise ε_t')
    parser.add_argument('--trend_coef', type=float, default=0.0, help='Linear trend coefficient for Y')
    parser.add_argument('--seed',    type=int, default=42, help='Random seed')
    parser.add_argument('--covariate_mode', choices=['static','dynamic'], default='static',
                        help='Use time-invariant X (static) or time-varying X_t (dynamic)')
    parser.add_argument('--save_plot', action='store_true', help='Save the visualization to file')

    # static-X params
    parser.add_argument('--covar_rate',        type=float, default=1.0, help='Poisson rate for X (static mode)')
    parser.add_argument('--covar_rate_shift',  type=float, default=3.0, help='Shifted Poisson rate for X (static mode)')

    # dynamic-X params (generation)
    parser.add_argument('--x_rate',       type=float, default=0.7, help='ρ_X for dynamic X_t AR(1)')
    parser.add_argument('--x_trend',      type=float, default=0.0, help='trend_X for dynamic X_t')
    parser.add_argument('--x_noise_std',  type=float, default=0.2, help='Std dev of η_t for X_t')
    parser.add_argument('--x0_lambda',    type=float, default=1.0, help='Poisson rate for X₀ in dynamic mode')

    # dynamic-X params (shift)
    parser.add_argument('--x_rate_shift',      type=float, default=None, help='Shifted ρ_X (defaults to x_rate)')
    parser.add_argument('--x_trend_shift',     type=float, default=None, help='Shifted trend_X (defaults to x_trend)')
    parser.add_argument('--x_noise_std_shift', type=float, default=None, help='Shifted X-noise std (defaults to x_noise_std)')
    parser.add_argument('--x0_lambda_shift',   type=float, default=None, help='Shifted Poisson rate for X₀ (defaults to x0_lambda)')
    # per your instruction, assume x0_redraw=True under shift; no CLI needed

    args = parser.parse_args()

    print("Time Series Generator with Covariate Shift")
    print("="*50)
    print(f"Mode           : {args.covariate_mode}")
    print(f"Training series: {args.n_train}")
    print(f"Test series    : {args.n_test}")
    print(f"Series length  : {args.T + 1}")
    print(f"AR coef (α)    : {args.ar_coef}")
    print(f"Beta (β)       : {args.beta}")
    print(f"Y noise std    : {args.noise_std}")
    print(f"Y trend        : {args.trend_coef}")
    if args.covariate_mode == 'static':
        print(f"X rate (static): {args.covar_rate} → {args.covar_rate_shift}")
    else:
        print(f"X params (gen) : ρ_X={args.x_rate}, trend_X={args.x_trend}, σ_η={args.x_noise_std}, λ_X0={args.x0_lambda}")
        print(f"X params (shift): ρ_X→{args.x_rate_shift or args.x_rate}, trend_X→{args.x_trend_shift or args.x_trend}, "
              f"σ_η→{args.x_noise_std_shift or args.x_noise_std}, λ_X0→{args.x0_lambda_shift or args.x0_lambda}")
    print(f"Seed           : {args.seed}")

    # set up
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)

    # generate original (training) data
    if args.covariate_mode == 'static':
        train_Y, train_X = generator.generate_with_poisson_covariate(
            n=args.n_train,
            ar_coef=args.ar_coef,
            beta=args.beta,
            covar_rate=args.covar_rate,
            noise_std=args.noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=args.trend_coef
        )  # train_X: shape (n,)
    else:
        train_Y, train_X = generator.generate_with_dynamic_covariate(
            n=args.n_train,
            ar_coef=args.ar_coef,
            beta=args.beta,
            noise_std=args.noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=args.trend_coef,
            x_rate=args.x_rate,
            x_trend=args.x_trend,
            x_noise_std=args.x_noise_std,
            x0_lambda=args.x0_lambda
        )  # train_X: shape (n, T+1)

    print(f"\nGenerated {args.n_train} series, Y shape={train_Y.shape}")

    # select test subset
    test_Y = train_Y[:args.n_test]
    test_X = train_X[:args.n_test]

    # apply covariate shift (unified)
    if args.covariate_mode == 'static':
        shifted_Y, shifted_X_series = generator.introduce_covariate_shift(
            original_Y=test_Y,
            original_X=test_X,  # (n,)
            covariate_mode='static',
            model_params={
                'ar_coef':    args.ar_coef,
                'beta':       args.beta,
                'noise_std':  args.noise_std,
                'trend_coef': args.trend_coef
            },
            shift_params={
                'shift_rate': args.covar_rate_shift
            }
        )
    else:
        shifted_Y, shifted_X_series = generator.introduce_covariate_shift(
            original_Y=test_Y,
            original_X=test_X,  # (n, T+1)
            covariate_mode='dynamic',
            model_params={
                'ar_coef':    args.ar_coef,
                'beta':       args.beta,
                'noise_std':  args.noise_std,
                'trend_coef': args.trend_coef
            },
            shift_params={
                'x_rate_shift':      args.x_rate_shift if args.x_rate_shift is not None else args.x_rate,
                'x_trend_shift':     args.x_trend_shift if args.x_trend_shift is not None else args.x_trend,
                'x_noise_std_shift': args.x_noise_std_shift if args.x_noise_std_shift is not None else args.x_noise_std,
                'x0_lambda_shift':   args.x0_lambda_shift if args.x0_lambda_shift is not None else args.x0_lambda,
                'x0_redraw': True
            }
        )

    # stats
    generator.print_statistics(
        original_data=test_Y,
        shifted_data=shifted_Y,
        original_X=test_X,
        shifted_X=shifted_X_series,
        covariate_mode=args.covariate_mode
    )

    # likelihood ratios (still based on Y₀ KDE)
    lr = generator.compute_likelihood_ratios(train_Y, shifted_Y)
    print(f"\nLikelihood ratios (on Y₀): mean={np.mean(lr):.3f}, std={np.std(lr):.3f}")

    # visualization
    print("\nGenerating visualization...")
    generator.visualize_covariate_shift(
        original_data=test_Y,
        shifted_data=shifted_Y,
        original_X=test_X,
        shifted_X=shifted_X_series,
        covariate_mode=args.covariate_mode,
        save_plot=args.save_plot,
        filename="covariate_shift.png"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()

