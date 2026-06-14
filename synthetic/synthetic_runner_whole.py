"""
synthetic/synthetic_runner_whole.py — single-seed runner for Algorithm 1
(whole-trajectory coverage) on the synthetic AR(1) DGPs.

Self-contained (§ 0): defines its own data split, predictor, featurizer, main
loop, and CLI; the only cross-module imports are core.ts_generator (STAY) and
core.weighted_cafht_whole (the algorithm).

Mapping to WEIGHTED_CAFHT_PLAN.md § 3:
  - Predictor (§ 3.1): a GLOBAL AR(1) fit by OLS on all training (Y_{t-1}, Y_t)
    pairs, applied ONE-STEP-AHEAD (Ŷ_t = â·Y_{t-1} + ĉ). The algorithm box has
    f̂_t regress Y_t on the OBSERVED prefix, so we use the observed Y_{t-1}
    (not a free-running forecast). The predictor deliberately ignores the
    Poisson covariate X, so the omitted β·X term is exactly the confounder the
    LR reweighting has to correct under shift.
  - LR featurizer (§ 3.3): the ACTUAL covariate X (what shifts in the DGP) —
    NOT Y_0. The plan's "Y_0 (== X_1 under X=Y_lag)" would be unshifted under
    this DGP (Y_0 ~ N(0,1) for both source and target), so the LR weighting
    would be a no-op; we feed X_1 = the Poisson covariate instead (static: the
    scalar X; dynamic: X_0, the first path value — per Q2(a), X_1 only).
  - Four-way data split (§ 2.0): D_ACI is peeled off the SOURCE pool before the
    D_tr / D_cal split; D_test is an independent draw from the target law.

Comparison (§ 5.1): two conditions only.
  - full    : "our version" — LR weights (real featurizer) + γ-selected ACI.
  - uniform : "no-LR version" — uniform weights (constant featurizer), ACI on.
  (The LR-only γ=0 ablation is intentionally dropped; we compare LR vs no-LR.)

Performance metrics: whole-trajectory (joint) coverage and mean band width.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

# core.ts_generator imports pandas/matplotlib, which emit env UserWarnings.
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.ts_generator import TimeSeriesGenerator              # noqa: E402
from core.weighted_cafht_whole import WeightedCAFHTWholeTrajectory  # noqa: E402
from core.weighted_cafht_whole_revised import (                # noqa: E402
    WeightedCAFHTWholeTrajectoryRevised)                       # experimental

# Synthetic γ-grid (WEIGHTED_CAFHT_PLAN.md § "Gamma selection").
GAMMA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]


# =============================================================================
# Data generation: independent source (P) and target (P̃) pools
# =============================================================================
def _gen_pool(gen, n, covariate_mode, params, shifted):
    """Draw n series from the source (shifted=False) or target (shifted=True)
    law. Only the covariate distribution differs between the two; the Y|Z
    kernel (ar_coef, beta, noise) is identical, matching the DGP's
    covariate-shift definition. Returns (Y (n, T+1, 1), X covariate)."""
    if covariate_mode == "static":
        rate = params["covar_rate_shift"] if shifted else params["covar_rate"]
        Y, X = gen.generate_with_poisson_covariate(
            n=n, ar_coef=params["ar_coef"], beta=params["beta"],
            covar_rate=rate, noise_std=params["noise_std"],
            initial_mean=0.0, initial_std=1.0, trend_coef=params["trend_coef"])
        return Y, X                                   # X: (n,)
    else:  # dynamic: shift the X-path parameters
        kw = dict(n=n, ar_coef=params["ar_coef"], beta=params["beta"],
                  noise_std=params["noise_std"], initial_mean=0.0,
                  initial_std=1.0, trend_coef=params["trend_coef"])
        if shifted:
            kw.update(x_rate=params["x_rate_shift"], x_trend=params["x_trend_shift"],
                      x_noise_std=params["x_noise_std_shift"],
                      x0_lambda=params["x0_lambda_shift"])
        else:
            kw.update(x_rate=params["x_rate"], x_trend=params["x_trend"],
                      x_noise_std=params["x_noise_std"], x0_lambda=params["x0_lambda"])
        Y, X = gen.generate_with_dynamic_covariate(**kw)
        return Y, X                                   # X: (n, T+1)


# =============================================================================
# Predictor (§ 3.1): global AR(1), applied one-step-ahead
# =============================================================================
def build_whole_trajectory_predictor(Y_train):
    """Fit Y_t = a + b·Y_{t-1} by OLS over all training (t-1, t) pairs and
    return a one-step-ahead predictor producing (n, T, 1) = [Ŷ_1, …, Ŷ_T]."""
    Y = np.asarray(Y_train, float)[..., 0]            # (n_tr, T+1)
    y_prev = Y[:, :-1].ravel()
    y_next = Y[:, 1:].ravel()
    A = np.column_stack([np.ones_like(y_prev), y_prev])
    coef, *_ = np.linalg.lstsq(A, y_next, rcond=None)
    a, b = float(coef[0]), float(coef[1])

    def predict(Y_series):
        Yp = np.asarray(Y_series, float)[..., 0]      # (n, T+1)
        yhat = a + b * Yp[:, :-1]                      # (n, T) one-step-ahead
        return yhat[..., None]                         # (n, T, 1)

    return predict


# =============================================================================
# LR featurizer (§ 3.3): X_1 = the actual covariate (what shifts)
# =============================================================================
def featurize_x1(X, covariate_mode):
    """Whole-trajectory classifier feature = X_1 (the first covariate). Static-X
    has a single time-invariant covariate; dynamic-X uses X_0 (the first path
    value)."""
    X = np.asarray(X, float)
    if covariate_mode == "static":
        return X.reshape(-1, 1)                        # the scalar Poisson X
    return X[:, 0:1]                                   # dynamic: X_0


def featurize_prefix(Y, X, covariate_mode, horizon):
    """REVISED algo (REVISED_WHOLE_TRAJECTORY.md): per-step classifier features
    (n, horizon, d). Column h summarizes the OBSERVED prefix up to step h — the
    TARGET prefix Y_{0:h} (never Y_{h+1}, the value being predicted) plus the
    covariate prefix — with per-channel {mean, std, last}.

    Including the Y prefix gives the classifier the shift signal that propagates
    into the response (higher X ⇒ higher Y level); under static-X the constant
    covariate alone is time-invariant, so the Y prefix is what makes p̂_t vary
    with t. Channels: static = Y-prefix (3) + X scalar (1) = 4; dynamic =
    Y-prefix (3) + X-prefix (3) = 6."""
    Yc = np.asarray(Y, float)[:, :, 0]                 # (n, L)
    X = np.asarray(X, float)
    n = Yc.shape[0]
    if covariate_mode == "static":
        x_scalar = X.reshape(n)                        # constant covariate
        feats = np.zeros((n, horizon, 4))
        for h in range(horizon):
            yp = Yc[:, :h + 1]                         # Y_{0:h} (excludes Y_{h+1})
            feats[:, h, 0] = yp.mean(axis=1)
            feats[:, h, 1] = yp.std(axis=1)
            feats[:, h, 2] = yp[:, -1]
            feats[:, h, 3] = x_scalar
        return feats
    feats = np.zeros((n, horizon, 6))
    for h in range(horizon):
        yp = Yc[:, :h + 1]                             # Y_{0:h}
        xp = X[:, :h + 1]                              # X_{0:h}
        feats[:, h, 0] = yp.mean(axis=1)
        feats[:, h, 1] = yp.std(axis=1)
        feats[:, h, 2] = yp[:, -1]
        feats[:, h, 3] = xp.mean(axis=1)
        feats[:, h, 4] = xp.std(axis=1)
        feats[:, h, 5] = xp[:, -1]
    return feats


# =============================================================================
# Single-seed experiment
# =============================================================================
def run_single(seed, covariate_mode="static", with_shift=True, mode="full",
               T=20, n_tr=300, n_aci=150, n_cal=300, n_test=200,
               alpha=0.1, params=None, gamma_grid=None, verbose=False,
               revised=False):
    """Run one seed and return a results dict. `mode` ∈ {full, uniform}.
    `revised=True` uses the experimental per-step-classifier Algorithm 1."""
    params = params or _default_params(covariate_mode)
    gamma_grid = gamma_grid if gamma_grid is not None else GAMMA_GRID

    gen = TimeSeriesGenerator(T=T, d=1, seed=seed)

    # Source pool (P) -> split into D_tr / D_ACI / D_cal; target pool (P̃) -> D_test.
    n_src = n_tr + n_aci + n_cal
    Y_src, X_src = _gen_pool(gen, n_src, covariate_mode, params, shifted=False)
    Y_test, X_test = _gen_pool(gen, n_test, covariate_mode, params, shifted=with_shift)

    Y_tr, Y_aci, Y_cal = Y_src[:n_tr], Y_src[n_tr:n_tr + n_aci], Y_src[n_tr + n_aci:]
    X_tr, X_cal = X_src[:n_tr], X_src[n_tr + n_aci:]   # D_ACI needs no covariates

    # Predictor fit on the FULL D_tr, applied one-step-ahead to every subset.
    predict = build_whole_trajectory_predictor(Y_tr)
    tr_pred, tr_true = predict(Y_tr), Y_tr[:, 1:, :]
    cal_pred, cal_true = predict(Y_cal), Y_cal[:, 1:, :]
    test_pred, test_true = predict(Y_test), Y_test[:, 1:, :]
    aci_pred, aci_true = predict(Y_aci), Y_aci[:, 1:, :]

    horizon = tr_true.shape[1]
    if revised:
        # Per-step classifiers on prefix features (z-scored on D_tr stats).
        # uniform => constant (zero) features => uniform per-step weights.
        if mode == "uniform":
            z = lambda Xr: np.zeros((len(np.asarray(Xr)), horizon, 1))
            Xc_tr, Xc_cal, Xc_te = z(X_tr), z(X_cal), z(X_test)
        else:
            Pt = featurize_prefix(Y_tr, X_tr, covariate_mode, horizon)
            mu, sd = Pt.mean(axis=0), Pt.std(axis=0) + 1e-8
            Xc_tr = (Pt - mu) / sd
            Xc_cal = (featurize_prefix(Y_cal, X_cal, covariate_mode, horizon) - mu) / sd
            Xc_te = (featurize_prefix(Y_test, X_test, covariate_mode, horizon) - mu) / sd
        algo = WeightedCAFHTWholeTrajectoryRevised(
            alpha=alpha, gamma_grid=gamma_grid, featurize_fn=None, verbose=verbose)
        bands = algo.predict_bands(
            (tr_pred, tr_true), (cal_pred, cal_true), (test_pred, test_true),
            (aci_pred, aci_true), Xc_tr, Xc_cal, Xc_te, seed=seed)
    else:
        # Original: one classifier on X_1. uniform => constant features.
        if mode == "uniform":
            feat = lambda X: np.zeros((len(np.asarray(X)), 1))
        else:  # full
            feat = lambda X: featurize_x1(X, covariate_mode)
        algo = WeightedCAFHTWholeTrajectory(
            alpha=alpha, gamma_grid=gamma_grid, featurize_fn=feat, verbose=verbose)
        bands = algo.predict_bands(
            (tr_pred, tr_true), (cal_pred, cal_true), (test_pred, test_true),
            (aci_pred, aci_true), X_tr=X_tr, X_cal=X_cal, X_test=X_test, seed=seed)

    metrics = _coverage_metrics(bands, test_true, alpha)
    return {
        "regime": "whole_trajectory", "domain": "synthetic", "mode": mode,
        "covariate_mode": covariate_mode, "with_shift": bool(with_shift),
        "revised": bool(revised),
        "alpha": alpha, "seed": int(seed), "T": int(T),
        "n_tr": n_tr, "n_aci": n_aci, "n_cal": n_cal, "n_test": n_test,
        "gamma_opt": algo.gamma_opt_, "n_inf": int(algo.n_inf_),
        "score_bank_shape": list(algo.score_bank_shape_),
        **metrics,
    }


def _coverage_metrics(bands, truth, alpha):
    """bands (n, T, 2, 1); truth (n, T, 1). Whole-trajectory coverage is the
    JOINT rate, but we OMIT the first ceil(T/10) steps: a trajectory counts as
    covered iff every step from the first non-omitted one to the end is covered
    (skips the ACI cold-start warm-up). We also report the per-step profile,
    widths (over finite bands; δ_∞ bands counted as covered but excluded from the
    width mean), and the un-omitted joint rate for reference."""
    low = bands[:, :, 0, 0]
    high = bands[:, :, 1, 0]
    y = truth[:, :, 0]
    covered = (y >= low) & (y <= high)                # (n, T); ±inf => covered
    widths = high - low
    finite = np.isfinite(widths)

    n_omit = int(np.ceil(covered.shape[1] / 10))      # first 1/10 (round up)
    joint = float(covered[:, n_omit:].all(axis=1).mean())

    width_by_time = [
        float(widths[finite[:, t], t].mean()) if finite[:, t].any() else float("inf")
        for t in range(widths.shape[1])
    ]
    return {
        "coverage_by_time": covered.mean(axis=0).tolist(),  # per-step pooled
        "joint_coverage": joint,                      # whole-traj target (omit first 1/10)
        "overall_coverage": joint,
        "joint_coverage_full": float(covered.all(axis=1).mean()),  # all steps, reference
        "n_omit": n_omit,
        "pooled_coverage": float(covered.mean()),
        "width_by_time": width_by_time,
        "mean_width": float(widths[finite].mean()) if finite.any() else float("inf"),
        "target_coverage": 1.0 - alpha,
    }


def _default_params(covariate_mode):
    if covariate_mode == "static":
        return dict(ar_coef=0.7, beta=1.0, noise_std=0.2, trend_coef=0.0,
                    covar_rate=1.0, covar_rate_shift=2.0)
    return dict(ar_coef=0.7, beta=1.0, noise_std=0.2, trend_coef=0.0,
                x_rate=0.6, x_trend=0.0, x_noise_std=0.2, x0_lambda=1.0,
                x_rate_shift=0.9, x_trend_shift=0.0, x_noise_std_shift=0.2,
                x0_lambda_shift=1.0)


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="Synthetic whole-trajectory runner (Alg. 1)")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--covariate_mode", choices=["static", "dynamic"], default="static")
    p.add_argument("--with_shift", action="store_true")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_tr", type=int, default=300)
    p.add_argument("--n_aci", type=int, default=150)
    p.add_argument("--n_cal", type=int, default=300)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--covar_rate", type=float, default=1.0)
    p.add_argument("--covar_rate_shift", type=float, default=2.0)
    p.add_argument("--x_rate", type=float, default=0.6)
    p.add_argument("--x_rate_shift", type=float, default=0.9)
    p.add_argument("--save_json", type=str, default=None)
    p.add_argument("--revised", action="store_true",
                   help="experimental per-step-classifier Algorithm 1")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    params = _default_params(args.covariate_mode)
    params["covar_rate"], params["covar_rate_shift"] = args.covar_rate, args.covar_rate_shift
    if args.covariate_mode == "dynamic":
        params["x_rate"], params["x_rate_shift"] = args.x_rate, args.x_rate_shift

    res = run_single(seed=args.seed, covariate_mode=args.covariate_mode,
                     with_shift=args.with_shift, mode=args.mode, T=args.T,
                     n_tr=args.n_tr, n_aci=args.n_aci, n_cal=args.n_cal,
                     n_test=args.n_test, alpha=args.alpha, params=params,
                     verbose=args.verbose, revised=args.revised)
    print(f"[{args.mode}/{args.covariate_mode}/"
          f"{'shift' if args.with_shift else 'noshift'}"
          f"{'/REVISED' if args.revised else ''}] "
          f"joint_cov={res['joint_coverage']:.3f} "
          f"pooled_cov={res['pooled_coverage']:.3f} "
          f"mean_width={res['mean_width']:.3f} "
          f"gamma_opt={res['gamma_opt']} n_inf={res['n_inf']}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
