#!/usr/bin/env python3
"""
=============================================================================
TEST CONFORMAL COVERAGE - TIME-BASED ANALYSIS 
=============================================================================

PURPOSE:
  Generate data using ts_generator.py and test conformal predictors with 
  TIME-BASED coverage visualization. Shows how coverage changes as prediction 
  horizon increases (t=1,2,3,...,T).
  
  Supports two predictor modes:
    - adaptive  : OnlineConformalPredictor (AR(1) on Y only, sliding-window split conformal)
    - algorithm : AdaptedCAFHT from the paper (weighted conformal with likelihood ratios)
  
  Supports two covariate modes:
    - static  : time-invariant X (Poisson)
    - dynamic : time-varying X_t following its own AR(1)

WHAT THIS SHOWS:
  Coverage rate and interval width as a function of prediction time step t.
  Compares how adaptive vs algorithm methods handle covariate shift scenarios.

USAGE:
  # Adaptive predictor with default settings
  python test_conformal.py --predictor adaptive

  # Algorithm predictor with default settings
  python test_conformal.py --predictor algorithm

  # Dynamic covariates with test-set shift using algorithm predictor
  python test_conformal.py --predictor algorithm --covariate_mode dynamic \
      --with_shift --x_rate 0.6 --x_rate_shift 0.9 --beta 1.0

EXAMPLE USAGE:

Adaptive predictor examples:
    python test_conformal.py --predictor adaptive --n_series 300
    python test_conformal.py --predictor adaptive --with_shift --n_series 300

Algorithm predictor examples:
    python test_conformal.py --predictor algorithm --n_series 300
    python test_conformal.py --predictor algorithm --with_shift --n_series 300

=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# ── Shared plot style ────────────────────────────────────────────────────────
_C_COV    = "#2166ac"   # coverage / primary  (blue)
_C_TARGET = "#d6604d"   # target line         (red-orange)
_C_WIDTH  = "#4dac26"   # width line          (green)
_C_ALPHA  = "#7b2d8b"   # ACI alpha           (purple)

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core.ts_generator import TimeSeriesGenerator
from core.adaptive_conformal import OnlineConformalPredictor
from core.algorithm import AdaptedCAFHT
# Candidate ACI stepsizes for gamma selection
GAMMA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]

def run_time_based_coverage_experiment(
    generator: TimeSeriesGenerator,
    predictor,  # Can be OnlineConformalPredictor or AdaptedCAFHT
    predictor_type: str,  # 'adaptive' or 'algorithm'
    *,
    n_series: int,
    covariate_mode: str,
    # Y model params
    ar_coef: float,
    beta: float,
    noise_std: float,
    trend_coef: float,
    # Static-X generation params
    covar_rate: float,
    covar_rate_shift: float,
    # Dynamic-X generation params
    x_rate: float,
    x_trend: float,
    x_noise_std: float,
    x0_lambda: float,
    # Dynamic-X shift params (if None, default to generation values)
    x_rate_shift: float,
    x_trend_shift: float,
    x_noise_std_shift: float,
    x0_lambda_shift: float,
    # Experiment options
    with_shift: bool,
    n_train: int,
    n_cal: int,
    aci_stepsize: float,
    use_lr: bool = True,
):
    """
    Run a time-based coverage experiment: evaluate coverage at each time step t.

    Returns:
        results_by_time: dict with keys = time steps, values = coverage stats
    """
    mode = covariate_mode.lower()
    assert mode in {"static", "dynamic"}, "covariate_mode must be 'static' or 'dynamic'"
    assert predictor_type in {"adaptive", "algorithm"}, "predictor_type must be 'adaptive' or 'algorithm'"

    if with_shift:
        print(f"Running TIME-BASED coverage experiment with TEST covariate shift on {n_series} series...")
    else:
        print(f"Running TIME-BASED coverage experiment on {n_series} series (NO TEST SHIFT)...")
    print(f"Predictor type: {predictor_type}")
    print(f"Covariate mode: {mode}")

    # -----------------------
    # Generate training data (NO SHIFT)
    # -----------------------
    print("Generating training data...")
    if mode == "static":
        train_Y, _ = generator.generate_with_poisson_covariate(
            n=n_train,
            ar_coef=ar_coef,
            beta=beta,
            covar_rate=covar_rate,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
        )
    else:
        train_Y, _ = generator.generate_with_dynamic_covariate(
            n=n_train,
            ar_coef=ar_coef,
            beta=beta,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
            x_rate=x_rate,
            x_trend=x_trend,
            x_noise_std=x_noise_std,
            x0_lambda=x0_lambda,
        )

    # -----------------------
    # Generate calibration data (NO SHIFT)
    # -----------------------
    print("Generating calibration data...")
    if mode == "static":
        cal_Y, _ = generator.generate_with_poisson_covariate(
            n=n_cal,
            ar_coef=ar_coef,
            beta=beta,
            covar_rate=covar_rate,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
        )
    else:
        cal_Y, _ = generator.generate_with_dynamic_covariate(
            n=n_cal,
            ar_coef=ar_coef,
            beta=beta,
            noise_std=noise_std,
            initial_mean=0.0,
            initial_std=1.0,
            trend_coef=trend_coef,
            x_rate=x_rate,
            x_trend=x_trend,
            x_noise_std=x_noise_std,
            x0_lambda=x0_lambda,
        )

    # Reset adaptation for algorithm predictor at the start of testing
    if predictor_type == "algorithm" and hasattr(predictor, 'reset_adaptation'):
        predictor.reset_adaptation()

    # -----------------------
    # Generate test data
    # -----------------------
    print("Generating test data...")
    
    # Resolve shift params defaults for dynamic mode
    xr_s = x_rate if x_rate_shift is None else x_rate_shift
    xt_s = x_trend if x_trend_shift is None else x_trend_shift
    xn_s = x_noise_std if x_noise_std_shift is None else x_noise_std_shift
    xl_s = x0_lambda if x0_lambda_shift is None else x0_lambda_shift

    test_series_list = []
    
    for i in range(n_series):
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{n_series} test series...")

        # Generate one original test series (NO SHIFT) in the chosen mode
        if mode == "static":
            Y_orig, X_orig = generator.generate_with_poisson_covariate(
                n=1,
                ar_coef=ar_coef,
                beta=beta,
                covar_rate=covar_rate,
                noise_std=noise_std,
                initial_mean=0.0,
                initial_std=1.0,
                trend_coef=trend_coef,
            )
        else:
            Y_orig, X_orig = generator.generate_with_dynamic_covariate(
                n=1,
                ar_coef=ar_coef,
                beta=beta,
                noise_std=noise_std,
                initial_mean=0.0,
                initial_std=1.0,
                trend_coef=trend_coef,
                x_rate=x_rate,
                x_trend=x_trend,
                x_noise_std=x_noise_std,
                x0_lambda=x0_lambda,
            )

        # Optionally apply TEST covariate shift via unified function
        if with_shift:
            if mode == "static":
                Y_shift, _ = generator.introduce_covariate_shift(
                    original_Y=Y_orig,
                    original_X=X_orig,  # shape (1,)
                    covariate_mode="static",
                    model_params={
                        "ar_coef": ar_coef,
                        "beta": beta,
                        "noise_std": noise_std,
                        "trend_coef": trend_coef,
                    },
                    shift_params={"shift_rate": covar_rate_shift if covar_rate_shift is not None else 3.0},
                )
            else:
                Y_shift, _ = generator.introduce_covariate_shift(
                    original_Y=Y_orig,
                    original_X=X_orig,  # shape (1, T+1)
                    covariate_mode="dynamic",
                    model_params={
                        "ar_coef": ar_coef,
                        "beta": beta,
                        "noise_std": noise_std,
                        "trend_coef": trend_coef,
                    },
                    shift_params={
                        "x_rate_shift": xr_s,
                        "x_trend_shift": xt_s,
                        "x_noise_std_shift": xn_s,
                        "x0_lambda_shift": xl_s,
                        "x0_redraw": True,  # per spec
                    },
                )
            test_series = Y_shift[0]  # (T+1, 1)
        else:
            test_series = Y_orig[0]   # (T+1, 1)
        
        test_series_list.append(test_series)

    # Convert to numpy array
    test_data = np.array(test_series_list)  # Shape: (n_series, T+1, 1)
    
    # ALWAYS USE THE FIRST TEST SERIES
    example_idx = 15  # Always use the first series
    example_series = test_data[example_idx].copy()  # Shape: (T+1, 1)
    print(f"Storing first test series (index 0) for visualization")
    
    # -----------------------
    # TIME-BASED EVALUATION
    # -----------------------
    print("\nEvaluating coverage by time step...")
    
    n_test, T_plus_1, d = test_data.shape
    T = T_plus_1 - 1  # Number of prediction steps possible
    
    results_by_time = {}
    
    # Store the example series index and full data
    results_by_time['example_idx'] = example_idx
    results_by_time['example_full_series'] = example_series
    
    # INITIALIZE THE LISTS HERE - BEFORE THE LOOP
    results_by_time['example_predictions'] = []
    results_by_time['example_lower_bounds'] = []
    results_by_time['example_upper_bounds'] = []
    results_by_time['example_true_values'] = []
    results_by_time['example_alpha_levels'] = []
    
    # -----------------------
    # ACI state (algorithm only): per-series nominal alpha
    # -----------------------
    base_alpha = float(getattr(predictor, 'alpha', 0.1))
    if predictor_type == "algorithm":
        alpha_series = np.full(n_test, base_alpha, dtype=float)  # alpha_1^{(i)} = alpha
    else:
        alpha_series = None
    
    # ACI gamma selection state (algorithm only)
    gamma_opt = float(aci_stepsize)
    if predictor_type == "algorithm":
        results_by_time['gamma_opt_history'] = []

    # For each time step t, predict step t+1
    for t in range(T):  # t = 0, 1, 2, ..., T-1 (predicting t+1)
        print(f"  Evaluating predictions at time step {t+1}...")

        # ACI: alpha used for bands at this time step (only algorithm)
        if predictor_type == "algorithm":
            alpha_used = alpha_series.copy()
            alpha_next = alpha_series.copy()
        else:
            alpha_used = None
            alpha_next = None

        if predictor_type == "algorithm":
            results_by_time['example_alpha_levels'].append(float(alpha_used[example_idx]))

        # -------------------------------------------------
        # Gamma selection: every 10 steps, pick best gamma on a 3-way split of training data
        # -------------------------------------------------
        if predictor_type == "algorithm" and t > 0 and (t % 10 == 0):
            sel_seed = int(getattr(generator, 'seed', 0)) + 10000 + t
            gamma_opt, _gamma_scores = _select_gamma_simple_aci(
                train_Y=train_Y,
                base_alpha=base_alpha,
                t_max=t,
                gamma_grid=GAMMA_GRID,
                seed=sel_seed,
            )

        if predictor_type == "algorithm":
            results_by_time['gamma_opt_history'].append(float(gamma_opt))


        coverage_results = []
        predictions = []
        intervals = []

        # Use increasing amount of data as time progresses
        predictor.fit_ar_model(train_Y[:, :t+2, :])

        if predictor_type == "algorithm" and t >= 1 and use_lr:
            # Split test set into two halves. First half uses second half as "shifted" positives, and vice versa.
            mid = n_test // 2
            idx_half1 = np.arange(0, mid)
            idx_half2 = np.arange(mid, n_test)

            # Prepare prefixes at time t (length t+1)
            train_prefixes = train_Y[:, :t+1, :]

            # First pass: predict on half1 using half2 to train the classifier
            predictor.update_weighting_context(
                train_prefixes=train_prefixes,
                test_prefixes=test_data[idx_half2, :t+1, :],
                is_shifted=True
            )
            predictor.calibrate(cal_Y[:, :t+2, :])
            for i in idx_half1:
                series = test_data[i]
                input_series = series[:t+1]
                true_value = series[t+1, 0]

                pred, lower, upper = predictor.predict_with_interval(input_series, alpha_level=alpha_used[i])

                covered = (lower <= true_value <= upper)
                err = 0 if covered else 1
                alpha_next[i] = alpha_used[i] + gamma_opt * (base_alpha - err)
                coverage_results.append(covered)
                predictions.append(pred)
                intervals.append([lower, upper])

                if i == example_idx:
                    results_by_time['example_predictions'].append(pred)
                    results_by_time['example_lower_bounds'].append(lower)
                    results_by_time['example_upper_bounds'].append(upper)
                    results_by_time['example_true_values'].append(true_value)

            # Second pass: predict on half2 using half1 to train the classifier
            predictor.update_weighting_context(
                train_prefixes=train_prefixes,
                test_prefixes=test_data[idx_half1, :t+1, :],
                is_shifted=True
            )
            predictor.calibrate(cal_Y[:, :t+2, :])
            for i in idx_half2:
                series = test_data[i]
                input_series = series[:t+1]
                true_value = series[t+1, 0]

                pred, lower, upper = predictor.predict_with_interval(input_series, alpha_level=alpha_used[i])

                covered = (lower <= true_value <= upper)
                err = 0 if covered else 1
                alpha_next[i] = alpha_used[i] + gamma_opt * (base_alpha - err)
                coverage_results.append(covered)
                predictions.append(pred)
                intervals.append([lower, upper])

                if i == example_idx:
                    results_by_time['example_predictions'].append(pred)
                    results_by_time['example_lower_bounds'].append(lower)
                    results_by_time['example_upper_bounds'].append(upper)
                    results_by_time['example_true_values'].append(true_value)

            # ACI: apply alpha updates after completing BOTH halves at this time step
            alpha_series = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)

        else:
            # Baseline behavior (adaptive, or algorithm with no shift or t==0):
            predictor.calibrate(cal_Y[:, :t+2, :])

            for i in range(n_test):
                series = test_data[i]

                # Use data up to time t to predict time t+1
                input_series = series[:t+1]
                true_value = series[t+1, 0]

                # Get prediction and interval - handle different predictor types
                if predictor_type == "adaptive":
                    pred, lower, upper = predictor.predict_with_interval(input_series)
                else:  # algorithm (no shift or t==0)
                    pred, lower, upper = predictor.predict_with_interval(input_series, alpha_level=alpha_used[i])

                # Check coverage
                covered = (lower <= true_value <= upper)
                coverage_results.append(covered)

                predictions.append(pred)
                intervals.append([lower, upper])

                # STORE DATA FOR THE FIRST SERIES (i == example_idx)
                if i == example_idx:
                    results_by_time['example_predictions'].append(pred)
                    results_by_time['example_lower_bounds'].append(lower)
                    results_by_time['example_upper_bounds'].append(upper)
                    results_by_time['example_true_values'].append(true_value)

                if predictor_type == "algorithm":
                    err = 0 if covered else 1
                    alpha_next[i] = alpha_used[i] + gamma_opt * (base_alpha - err)

            if predictor_type == "algorithm":
                alpha_series = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)

        coverage_results = np.array(coverage_results)
        predictions = np.array(predictions)
        intervals = np.array(intervals)

        # Store results for this time step
        results_by_time[t+1] = {
            'coverage_rate': np.mean(coverage_results),
            'coverage_std': np.std(coverage_results),
            'interval_width': np.mean(intervals[:, 1] - intervals[:, 0]),
            'width_std': np.std(intervals[:, 1] - intervals[:, 0]),
            'n_predictions': len(coverage_results),
            'predictions': predictions,
            'intervals': intervals,
            'coverage_history': coverage_results
            ,
            'alpha_mean': float(np.mean(alpha_used)) if alpha_used is not None else None,
            'alpha_std': float(np.std(alpha_used)) if alpha_used is not None else None,
            'gamma_opt': float(gamma_opt) if predictor_type == "algorithm" else None
        }

        print(f"    Time {t+1}: Coverage = {np.mean(coverage_results):.1%}, "
              f"Width = {np.mean(intervals[:, 1] - intervals[:, 0]):.3f}")

    return results_by_time



def _select_gamma_simple_aci(train_Y: np.ndarray, base_alpha: float, t_max: int,
                            gamma_grid, seed: int = 0):
    """
    Select gamma by running simple ACI (no LR weighting) on a 3-way split of train_Y up to time t_max.

    Split:
      - D_tr^(1): fit AR model
      - D_tr^(2): calibration
      - D_tr^(3): evaluation

    Metric: average coverage over second half of the horizon (in time-index t space).

    Args:
        train_Y: (n_train, T+1, 1)
        base_alpha: target miscoverage alpha
        t_max: current outer-loop t (0-based). We simulate steps 0..t_max (predicting 1..t_max+1).
        gamma_grid: iterable of candidate gammas
        seed: RNG seed for splitting

    Returns:
        (best_gamma, scores_dict) where scores_dict maps gamma -> metric
    """
    n_train, Tp1, _ = train_Y.shape
    if n_train < 9 or t_max < 2:
        return float(list(gamma_grid)[0]), {float(g): float('nan') for g in gamma_grid}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)

    n1 = n_train // 3
    n2 = n_train // 3
    n3 = n_train - n1 - n2
    if n1 == 0 or n2 == 0 or n3 == 0:
        return float(list(gamma_grid)[0]), {float(g): float('nan') for g in gamma_grid}

    idx1 = perm[:n1]
    idx2 = perm[n1:n1 + n2]
    idx3 = perm[n1 + n2:]

    tr1 = train_Y[idx1]
    tr2 = train_Y[idx2]
    tr3 = train_Y[idx3]

    # Ensure we don't exceed available time length
    horizon = min(t_max, Tp1 - 2)  # t ranges 0..horizon
    start_eval = max(0, horizon // 2)  # second half (in t-index)

    scores = {}
    target = 1.0 - base_alpha

    for gamma in gamma_grid:
        gamma = float(gamma)

        tmp = AdaptedCAFHT(alpha=base_alpha)
        if hasattr(tmp, "reset_adaptation"):
            tmp.reset_adaptation()

        alpha_series = np.full(tr3.shape[0], base_alpha, dtype=float)
        cov_hist = []

        for t in range(horizon + 1):
            tmp.fit_ar_model(tr1[:, :t + 2, :])
            tmp.calibrate(tr2[:, :t + 2, :])

            alpha_used = alpha_series.copy()
            alpha_next = alpha_series.copy()

            step_cov = []
            for i in range(tr3.shape[0]):
                series = tr3[i]
                input_series = series[:t + 1]
                true_value = series[t + 1, 0]

                pred, lower, upper = tmp.predict_with_interval(input_series, alpha_level=alpha_used[i])
                covered = (lower <= true_value <= upper)
                step_cov.append(1 if covered else 0)

                err = 0 if covered else 1
                alpha_next[i] = alpha_used[i] + gamma * (base_alpha - err)

            alpha_series = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)
            cov_hist.append(float(np.mean(step_cov)) if step_cov else float('nan'))

        tail = cov_hist[start_eval:]
        metric = float(np.mean(tail)) if len(tail) > 0 else float('nan')
        scores[gamma] = metric

    # Choose gamma whose metric is closest to target coverage
    best_gamma = float(list(gamma_grid)[0])
    best_obj = float('inf')
    for gamma, metric in scores.items():
        if not np.isfinite(metric):
            continue
        obj = abs(metric - target)
        if obj < best_obj:
            best_obj = obj
            best_gamma = float(gamma)

    return best_gamma, scores





def plot_time_based_results(results_by_time, target_coverage, predictor_type="adaptive",
                            covariate_mode="static", with_shift=False, save_path=None):
    """Plot coverage results by time step."""
    time_steps      = sorted([k for k in results_by_time.keys() if isinstance(k, int)])
    coverage_rates  = [results_by_time[t]['coverage_rate']  for t in time_steps]
    interval_widths = [results_by_time[t]['interval_width'] for t in time_steps]

    predictor_str = "Algorithm (AdaptedCAFHT)" if predictor_type == "algorithm" \
                    else predictor_type.capitalize()
    mode_str  = "Dynamic $X_t$" if str(covariate_mode).lower() == "dynamic" else "Static X"
    shift_str = "with Shift" if with_shift else "no Shift"

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Time-Based Coverage  —  {predictor_str},  {mode_str},  {shift_str}",
        fontsize=12, fontweight="bold",
    )

    # ── Plot 1: Coverage rate over all series ─────────────────────────────────
    ax = axes[0, 0]
    ax.plot(time_steps, coverage_rates, color=_C_COV, linewidth=2)
    ax.set_ylim(0.8, 1.0)
    ax.axhline(target_coverage, color=_C_TARGET, linestyle="--", linewidth=1.8,
               label=f"Target ({target_coverage:.1%})")
    ax.set_xlabel("Time step $t$", fontsize=10)
    ax.set_ylabel("Coverage rate", fontsize=10)
    ax.set_title("Coverage Rate vs. Time (All Series)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── Plot 2: Interval width over all series ────────────────────────────────
    ax = axes[0, 1]
    ax.plot(time_steps, interval_widths, color=_C_WIDTH, linewidth=2)
    ax.set_xlabel("Time step $t$", fontsize=10)
    ax.set_ylabel("Average interval width", fontsize=10)
    ax.set_title("Prediction Interval Width vs. Time (All Series)", fontsize=10)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── Plot 3: First test series — true values + prediction interval ─────────
    ax = axes[0, 2]
    if 'example_true_values' in results_by_time:
        true_values  = results_by_time['example_true_values']
        lower_bounds = results_by_time['example_lower_bounds']
        upper_bounds = results_by_time['example_upper_bounds']
        plot_steps   = time_steps[:len(true_values)]

        ax.fill_between(plot_steps, lower_bounds, upper_bounds,
                        alpha=0.25, color=_C_COV)
        ax.plot(plot_steps, lower_bounds, color=_C_COV, linewidth=1.2,
                linestyle="--", alpha=0.7, label="Bounds")
        ax.plot(plot_steps, upper_bounds, color=_C_COV, linewidth=1.2,
                linestyle="--", alpha=0.7)
        ax.plot(plot_steps, true_values, color=_C_TARGET, linewidth=1.8,
                label="True values")
        ax.set_xlabel("Time step $t$", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.set_title("First Test Series: True vs. Interval", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
    else:
        ax.axis("off")
    _style_ax(ax)

    # ── Plot 4: Running average coverage for first test series ────────────────
    ax = axes[1, 0]
    if 'example_true_values' in results_by_time:
        first_cov = [
            1 if lower_bounds[i] <= true_values[i] <= upper_bounds[i] else 0
            for i in range(len(plot_steps))
        ]
        run_avg = [np.mean(first_cov[:i + 1]) for i in range(len(first_cov))]

        ax.plot(plot_steps, run_avg, color=_C_COV, linewidth=2)
        ax.set_ylim(0.7, 1.05)
        ax.axhline(target_coverage, color=_C_TARGET, linestyle="--", linewidth=1.8,
                   label=f"Target ({target_coverage:.1%})")
        ax.set_xlabel("Time step $t$", fontsize=10)
        ax.set_ylabel("Running avg. coverage", fontsize=10)
        ax.set_title("First Test Series: Running Coverage", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.05, f"Final: {run_avg[-1]:.1%}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    else:
        ax.axis("off")
    _style_ax(ax)

    # ── Plot 5: Interval width for first test series ──────────────────────────
    ax = axes[1, 1]
    if 'example_true_values' in results_by_time:
        first_widths = [upper_bounds[i] - lower_bounds[i]
                        for i in range(len(plot_steps))]
        ax.plot(plot_steps, first_widths, color=_C_WIDTH, linewidth=2)
        ax.set_xlabel("Time step $t$", fontsize=10)
        ax.set_ylabel("Interval width", fontsize=10)
        ax.set_title("First Test Series: Interval Width", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.text(0.02, 0.95, f"Mean: {np.mean(first_widths):.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    else:
        ax.axis("off")
    _style_ax(ax)

    # ── Plot 6: ACI alpha for first series (algorithm only) ───────────────────
    ax = axes[1, 2]
    if predictor_type == "algorithm" and 'example_alpha_levels' in results_by_time:
        alpha_vals       = results_by_time['example_alpha_levels']
        plot_steps_alpha = time_steps[:len(alpha_vals)]
        ax.plot(plot_steps_alpha, alpha_vals, color=_C_ALPHA, linewidth=2)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Time step $t$", fontsize=10)
        ax.set_ylabel("Alpha level", fontsize=10)
        ax.set_title("First Test Series: ACI Alpha", fontsize=10)
        ax.grid(True, alpha=0.25)
        _style_ax(ax)
    else:
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[Plot] Saved to {out}")


def main():
    parser = argparse.ArgumentParser(description='Test conformal prediction with time-based coverage analysis')
    
    # MAIN TOGGLE: Predictor type
    parser.add_argument('--predictor', choices=['adaptive', 'algorithm'], default='algorithm',
                        help='Choose predictor type: adaptive (sliding-window split conformal) or algorithm (AdaptedCAFHT)')
    
    # Experiment sizes & basics
    parser.add_argument('--n_series', type=int, default=None, 
                        help='Number of test series (default varies by predictor)')
    parser.add_argument('--n_train', type=int, default=None,
                        help='Number of training series (default varies by predictor)')
    parser.add_argument('--n_cal', type=int, default=None,
                        help='Number of calibration series (default varies by predictor)')
    parser.add_argument('--alpha', type=float, default=0.1, help='Miscoverage level')
    parser.add_argument('--aci_stepsize', type=float, default=0.005,
                        help='ACI stepsize γ for algorithm predictor (default 0.005)')
    parser.add_argument('--T', type=int, default=None,
                        help='Time series length (default varies by predictor)')
    parser.add_argument('--seed', type=int, default=100, help='Random seed')

    # Adaptive-specific parameters
    parser.add_argument('--window_size', type=int, default=800,
                        help='Window size for adaptive predictor (only used with --predictor adaptive)')

    # Y model params
    parser.add_argument('--ar_coef', type=float, default=0.7, help='AR(1) coefficient for Y')
    parser.add_argument('--beta', type=float, default=1.0, help='Covariate effect β on Y')
    parser.add_argument('--noise_std', type=float, default=0.2, help='Std dev of Y noise')
    parser.add_argument('--trend_coef', type=float, default=0.0, help='Linear trend for Y')

    # Covariate mode
    parser.add_argument('--covariate_mode', choices=['static', 'dynamic'], default='static',
                        help='Use time-invariant X (static) or time-varying X_t (dynamic)')
    parser.add_argument('--with_shift', action='store_true',
                        help='Apply covariate shift on the TEST set')

    # Static-X params (generation + shift)
    parser.add_argument('--covar_rate', type=float, default=1.0, help='Poisson rate for X (static mode)')
    parser.add_argument('--covar_rate_shift', type=float, default=None,
                        help='Shifted Poisson rate for TEST X (default varies by predictor)')

    # Dynamic-X params (generation)
    parser.add_argument('--x_rate', type=float, default=0.7, help='ρ_X for dynamic X_t')
    parser.add_argument('--x_trend', type=float, default=0.0, help='trend_X for dynamic X_t')
    parser.add_argument('--x_noise_std', type=float, default=0.2, help='Std dev of η_t for X_t')
    parser.add_argument('--x0_lambda', type=float, default=1.0, help='Poisson rate for X₀ (dynamic)')

    # Dynamic-X params (shift) — default to generation values if not provided
    parser.add_argument('--x_rate_shift', type=float, default=None, help='Shifted ρ_X for TEST X_t')
    parser.add_argument('--x_trend_shift', type=float, default=None, help='Shifted trend_X for TEST X_t')
    parser.add_argument('--x_noise_std_shift', type=float, default=None, help='Shifted X noise std for TEST')
    parser.add_argument('--x0_lambda_shift', type=float, default=None, help='Shifted Poisson rate for X₀ (TEST)')
    parser.add_argument('--save_plot', default=None, help='Save figure to this path (PNG or PDF)')

    args = parser.parse_args()

    # Set predictor-specific defaults
    if args.predictor == "adaptive":
        defaults = {
            'n_series': 500,
            'n_train': 1000,
            'n_cal': 1000,
            'T': 40,
            'covar_rate_shift': 2.0
        }
    else:  # algorithm
        defaults = {
            'n_series': 500,
            'n_train': 200,
            'n_cal': 200,  # Slightly more for better gamma selection
            'T': 20,
            'covar_rate_shift': 2.0
        }

    # Apply defaults where arguments weren't provided
    for key, default_val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, default_val)

    print(f"{args.predictor.upper()} CONFORMAL PREDICTION - TIME-BASED COVERAGE ANALYSIS")
    print("="*65)
    print("Analyzing how coverage changes as prediction horizon increases...")
    print("Parameters:")
    print(f"  Predictor type   : {args.predictor}")
    if args.predictor == "algorithm":
        print(f"                     (AdaptedCAFHT with weighted quantiles)")
    print(f"  Target coverage  : {1-args.alpha:.1%} (α={args.alpha})")
    print(f"  Series (train/cal/test): {args.n_train}/{args.n_cal}/{args.n_series}")
    print(f"  Series length    : {args.T + 1}")
    if args.predictor == "adaptive":
        print(f"  Window size      : {args.window_size}")
    print(f"  AR coef (Y)      : {args.ar_coef}")
    print(f"  β (X→Y)          : {args.beta}")
    print(f"  Y noise std      : {args.noise_std}")
    print(f"  Y trend          : {args.trend_coef}")
    print(f"  Covariate mode   : {args.covariate_mode}")
    if args.covariate_mode == 'static':
        shift_info = f" -> {args.covar_rate_shift}" if args.with_shift else ""
        print(f"  X (static) rate  : {args.covar_rate}{shift_info}")
    else:
        xr_s = args.x_rate if args.x_rate_shift is None else args.x_rate_shift
        xt_s = args.x_trend if args.x_trend_shift is None else args.x_trend_shift
        xn_s = args.x_noise_std if args.x_noise_std_shift is None else args.x_noise_std_shift
        xl_s = args.x0_lambda if args.x0_lambda_shift is None else args.x0_lambda_shift
        print(f"  X (dynamic) gen  : ρ_X={args.x_rate}, trend_X={args.x_trend}, σ_η={args.x_noise_std}, λ_X0={args.x0_lambda}")
        if args.with_shift:
            print(f"  X (dynamic) TEST : ρ_X→{xr_s}, trend_X→{xt_s}, σ_η→{xn_s}, λ_X0→{xl_s} (X₀ re-drawn)")
    print(f"  With TEST shift  : {args.with_shift}")
    print(f"  Seed             : {args.seed}")

    # Initialize generator
    generator = TimeSeriesGenerator(T=args.T, d=1, seed=args.seed)

    # Initialize appropriate predictor
    if args.predictor == "adaptive":
        predictor = OnlineConformalPredictor(alpha=args.alpha, window_size=args.window_size)
    else:  # algorithm
        predictor = AdaptedCAFHT(alpha=args.alpha)

    # Run time-based coverage experiment
    results_by_time = run_time_based_coverage_experiment(
        generator=generator,
        predictor=predictor,
        predictor_type=args.predictor,
        n_series=args.n_series,
        covariate_mode=args.covariate_mode,
        ar_coef=args.ar_coef,
        beta=args.beta,
        noise_std=args.noise_std,
        trend_coef=args.trend_coef,
        covar_rate=args.covar_rate,
        covar_rate_shift=args.covar_rate_shift,
        x_rate=args.x_rate,
        x_trend=args.x_trend,
        x_noise_std=args.x_noise_std,
        x0_lambda=args.x0_lambda,
        x_rate_shift=args.x_rate_shift,
        x_trend_shift=args.x_trend_shift,
        x_noise_std_shift=args.x_noise_std_shift,
        x0_lambda_shift=args.x0_lambda_shift,
        with_shift=args.with_shift,
        n_train=args.n_train,
        n_cal=args.n_cal,
        aci_stepsize=args.aci_stepsize,
    )

    # Compute overall statistics
    all_coverage = []
    all_widths = []
    for key, value in results_by_time.items():
        # Skip non-integer keys (metadata keys)
        if not isinstance(key, int):
            continue
        
        time_results = value
        all_coverage.extend(time_results['coverage_history'])
        all_widths.extend([time_results['interval_width']] * time_results['n_predictions'])
    
    overall_coverage = float(np.mean(all_coverage)) if all_coverage else float('nan')
    target_coverage = 1 - args.alpha

    print(f"\n" + "="*50)
    print("OVERALL COVERAGE RESULTS")
    print("="*50)
    print(f"Target coverage     : {target_coverage:.1%}")
    print(f"Overall coverage    : {overall_coverage:.1%}")
    print(f"Coverage error      : {overall_coverage - target_coverage:+.1%}")
    print(f"Coverage std        : {np.std(all_coverage):.3f}")
    print(f"Mean interval width : {np.mean(all_widths):.4f}")

    print(f"\nCOVERAGE BY TIME STEP:")
    print("-" * 40)
    # Filter to only get integer keys (time steps)
    time_steps = sorted([k for k in results_by_time.keys() if isinstance(k, int)])
    for time_step in time_steps:
        time_results = results_by_time[time_step]
        print(f"Time {time_step:2d}: {time_results['coverage_rate']:.1%} "
              f"(width: {time_results['interval_width']:.3f}, "
              f"n={time_results['n_predictions']})")

    # Create time-based coverage plots
    print("\nCreating time-based coverage plots...")
    plot_time_based_results(
        results_by_time,
        target_coverage,
        predictor_type=args.predictor,
        covariate_mode=args.covariate_mode,
        with_shift=args.with_shift,
        save_path=args.save_plot,
    )

    # Final assessment
    coverage_rates = [results_by_time[t]['coverage_rate'] for t in time_steps]
    
    early_coverage = np.mean(coverage_rates[:len(coverage_rates)//3]) if coverage_rates else 0
    late_coverage = np.mean(coverage_rates[-len(coverage_rates)//3:]) if coverage_rates else 0
    coverage_degradation = early_coverage - late_coverage
    
    print(f"\nTIME-BASED ANALYSIS:")
    print(f"  Early coverage (first 1/3): {early_coverage:.1%}")
    print(f"  Late coverage (last 1/3):   {late_coverage:.1%}")
    print(f"  Coverage degradation:        {coverage_degradation:+.1%}")
    
    coverage_near_target = abs(overall_coverage - target_coverage) < 0.05
    print(f"\nFinal Assessment:")
    print(f"  Coverage near target: {'✓' if coverage_near_target else '✗'}")
    print(f"  Degradation < 10%:    {'✓' if abs(coverage_degradation) < 0.10 else '✗'}")


if __name__ == "__main__":
    main()