"""
synthetic/synthetic_runner_last.py — single-seed runner for Algorithm 2
(last-step coverage) on the synthetic AR(1) DGPs.

Self-contained (§ 0). Mapping to WEIGHTED_CAFHT_PLAN.md § 3:
  - Predictor (§ 3.2): OLS of the LAST value Y_T on the full history Y_{0:T-1}
    (== Y_{1:T} in 1-indexed notation). Plain OLS (the synthetic feature dim is
    low); finance/medical use ridge.
  - LR featurizer (§ 3.3): the ACTUAL covariate X_{1:T} (what shifts), kept
    consistent with the whole-trajectory runner. Static-X has a single
    time-invariant covariate; dynamic-X uses the full X path.
  - Data split: source pool (P) -> D_tr / D_cal; target pool (P̃) -> D_test.
    No D_ACI / no γ / no ACI (last-step regime).

Ablation modes (§ 5.1): full (LR weights) and uniform (uniform weights). There
is no `zerog` for last-step — the regime has no γ.

NB: the predictor regresses on the Y history, which already encodes much of the
covariate, so the per-series score is closer to homoscedastic than in the
whole-trajectory runner; the LR reweighting is therefore a smaller (but still
valid) correction here.
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
from core.weighted_cafht_last import WeightedCAFHTLastStep     # noqa: E402


# =============================================================================
# Data generation (identical source/target laws as the whole-traj runner)
# =============================================================================
def _gen_pool(gen, n, covariate_mode, params, shifted):
    """Draw n series from the source (shifted=False) or target (shifted=True)
    law; only the covariate distribution differs. Returns (Y (n,T+1,1), X)."""
    if covariate_mode == "static":
        rate = params["covar_rate_shift"] if shifted else params["covar_rate"]
        Y, X = gen.generate_with_poisson_covariate(
            n=n, ar_coef=params["ar_coef"], beta=params["beta"],
            covar_rate=rate, noise_std=params["noise_std"],
            initial_mean=0.0, initial_std=1.0, trend_coef=params["trend_coef"])
        return Y, X
    else:
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
        return Y, X


# =============================================================================
# Predictor (§ 3.2): OLS of Y_T on the full history Y_{0:T-1}
# =============================================================================
def build_last_step_predictor(Y_train):
    """Fit Y_T = c0 + Σ_s c_s·Y_s (s = 0..T-1) by OLS; return a predictor that
    maps a batch of series to scalar last-step predictions Ŷ_T (n,)."""
    Y = np.asarray(Y_train, float)[..., 0]            # (n_tr, T+1)
    Xh = Y[:, :-1]                                     # (n_tr, T) = Y_0..Y_{T-1}
    yT = Y[:, -1]                                      # (n_tr,)   = Y_T
    A = np.column_stack([np.ones(len(Xh)), Xh])
    coef, *_ = np.linalg.lstsq(A, yT, rcond=None)

    def predict(Y_series):
        Yh = np.asarray(Y_series, float)[:, :-1, 0]   # (n, T)
        return coef[0] + Yh @ coef[1:]                 # (n,)

    return predict


# =============================================================================
# LR featurizer (§ 3.3): X_{1:T} = the actual covariate path
# =============================================================================
def featurize_xall(X, covariate_mode):
    """Last-step classifier feature = the whole covariate X_{1:T}. Static-X is a
    single time-invariant value; dynamic-X is the flattened path."""
    X = np.asarray(X, float)
    if covariate_mode == "static":
        return X.reshape(-1, 1)
    return X.reshape(len(X), -1)                        # dynamic: flatten (n, T+1)


# =============================================================================
# Single-seed experiment
# =============================================================================
def run_single(seed, covariate_mode="static", with_shift=True, mode="full",
               T=20, n_tr=300, n_cal=300, n_test=200, alpha=0.1,
               params=None, verbose=False):
    """Run one seed and return a results dict. `mode` ∈ {full, uniform}."""
    params = params or _default_params(covariate_mode)
    gen = TimeSeriesGenerator(T=T, d=1, seed=seed)

    n_src = n_tr + n_cal
    Y_src, X_src = _gen_pool(gen, n_src, covariate_mode, params, shifted=False)
    Y_test, X_test = _gen_pool(gen, n_test, covariate_mode, params, shifted=with_shift)

    Y_tr, Y_cal = Y_src[:n_tr], Y_src[n_tr:]
    X_tr, X_cal = X_src[:n_tr], X_src[n_tr:]

    predict = build_last_step_predictor(Y_tr)
    cal_pred, cal_true = predict(Y_cal), Y_cal[:, -1, 0]
    test_pred, test_true = predict(Y_test), Y_test[:, -1, 0]

    feat = ((lambda X: np.zeros((len(np.asarray(X)), 1))) if mode == "uniform"
            else (lambda X: featurize_xall(X, covariate_mode)))

    algo = WeightedCAFHTLastStep(alpha=alpha, featurize_fn=feat, verbose=verbose)
    bands = algo.predict_bands((cal_pred, cal_true), (test_pred, test_true),
                               X_tr=X_tr, X_cal=X_cal, X_test=X_test, seed=seed)

    metrics = _coverage_metrics(bands, test_true, alpha)
    return {
        "regime": "last_step", "domain": "synthetic", "mode": mode,
        "covariate_mode": covariate_mode, "with_shift": bool(with_shift),
        "alpha": alpha, "seed": int(seed), "T": int(T),
        "n_tr": n_tr, "n_cal": n_cal, "n_test": n_test,
        "n_inf": int(algo.n_inf_), **metrics,
    }


def _coverage_metrics(bands, truth, alpha):
    """bands (n, 1, 2, 1); truth (n,). Last-step coverage = fraction of test
    series whose Y_T lands in the single interval."""
    low = bands[:, 0, 0, 0]
    high = bands[:, 0, 1, 0]
    y = np.asarray(truth, float)
    covered = (y >= low) & (y <= high)
    widths = high - low
    finite = np.isfinite(widths)
    return {
        "coverage": float(covered.mean()),
        "overall_coverage": float(covered.mean()),
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
    p = argparse.ArgumentParser(description="Synthetic last-step runner (Alg. 2)")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--covariate_mode", choices=["static", "dynamic"], default="static")
    p.add_argument("--with_shift", action="store_true")
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_tr", type=int, default=300)
    p.add_argument("--n_cal", type=int, default=300)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--covar_rate", type=float, default=1.0)
    p.add_argument("--covar_rate_shift", type=float, default=2.0)
    p.add_argument("--x_rate", type=float, default=0.6)
    p.add_argument("--x_rate_shift", type=float, default=0.9)
    p.add_argument("--save_json", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    params = _default_params(args.covariate_mode)
    params["covar_rate"], params["covar_rate_shift"] = args.covar_rate, args.covar_rate_shift
    if args.covariate_mode == "dynamic":
        params["x_rate"], params["x_rate_shift"] = args.x_rate, args.x_rate_shift

    res = run_single(seed=args.seed, covariate_mode=args.covariate_mode,
                     with_shift=args.with_shift, mode=args.mode, T=args.T,
                     n_tr=args.n_tr, n_cal=args.n_cal, n_test=args.n_test,
                     alpha=args.alpha, params=params, verbose=args.verbose)
    print(f"[{args.mode}/{args.covariate_mode}/"
          f"{'shift' if args.with_shift else 'noshift'}] "
          f"coverage={res['coverage']:.3f} mean_width={res['mean_width']:.3f} "
          f"n_inf={res['n_inf']}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
