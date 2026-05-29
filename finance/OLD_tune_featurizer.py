#!/usr/bin/env python3
"""
tune_featurizer.py
==================
Systematically compare featurizer variants for the LR-based covariate shift
correction in AdaptedCAFHT.

For each variant this script runs a single-seed finance experiment
(with_shift=True) on one data window and reports four diagnostics:

  clf_acc   — logistic classifier balanced accuracy at the final time step
               (proxy for how well features discriminate test vs. train)
  ess_pct   — effective sample size of LR weights at the final time step,
               as a percentage of calibration set size
               (lower = more extreme correction; higher = near-uniform = weak)
  cov       — overall empirical coverage with LR weighting
  gap       — cov(with_shift) - cov(no_shift)   (positive = weighting helps)
  width_ratio — mean_width(with_shift) / mean_width(no_shift)
                (< 1 = weighting narrowed bands; > 1 = widened)

The no-shift baseline (uniform weights) is run once and compared against
every variant.

Usage
-----
  python finance/tune_featurizer.py --npz finance/data/sp500_20240201_20240328.npz

  # Test only specific variants:
  python finance/tune_featurizer.py --npz finance/data/sp500_20240201_20240328.npz \\
      --variants baseline x_full x_std normalize

  # Different sector or alpha:
  python finance/tune_featurizer.py --npz finance/data/sp500_20240701_20240830.npz \\
      --test_sector Technology --alpha 0.1
"""

import argparse
import json
import types
from pathlib import Path

import numpy as np

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from finance.finance_data import load_stored
from core.algorithm import AdaptedCAFHT
from finance.finance_conformal import (
    LinearCovariateModel,
    _select_gamma,
    GAMMA_GRID,
    Y_WINDOW,
    AR1_MIN_STEPS,
    _print_weight_diagnostic,
    _print_feature_diagnostic,
    _print_classifier_diagnostic,
)

# ── Featurizer variant registry ───────────────────────────────────────────────
# Each entry is a dict of kwargs passed to make_featurizer().
# Keys: y_window, x_window, x_std, x_ar1, normalize
VARIANTS = {
    # ── Current default ────────────────────────────────────────────────────────
    "baseline":        {"y_window": 30, "x_window":  5, "x_std": False, "x_ar1": False, "normalize": False},

    # ── X window length ────────────────────────────────────────────────────────
    "x_full":          {"y_window": 30, "x_window":  0, "x_std": False, "x_ar1": False, "normalize": False},
    "x_w10":           {"y_window": 30, "x_window": 10, "x_std": False, "x_ar1": False, "normalize": False},
    "x_w20":           {"y_window": 30, "x_window": 20, "x_std": False, "x_ar1": False, "normalize": False},

    # ── Add X standard deviation ───────────────────────────────────────────────
    "x_std":           {"y_window": 30, "x_window":  5, "x_std": True,  "x_ar1": False, "normalize": False},
    "x_full_std":      {"y_window": 30, "x_window":  0, "x_std": True,  "x_ar1": False, "normalize": False},

    # ── Add X AR(1) coefficient ────────────────────────────────────────────────
    "x_ar1":           {"y_window": 30, "x_window":  5, "x_std": False, "x_ar1": True,  "normalize": False},
    "x_std_ar1":       {"y_window": 30, "x_window":  5, "x_std": True,  "x_ar1": True,  "normalize": False},

    # ── Shorter Y window ───────────────────────────────────────────────────────
    "y_w10":               {"y_window": 10, "x_window":  5, "x_std": False, "x_ar1": False, "normalize": False},
    "y_w10_x_std":         {"y_window": 10, "x_window":  5, "x_std": True,  "x_ar1": False, "normalize": False},
    "y_w10_x_full_std":    {"y_window": 10, "x_window":  0, "x_std": True,  "x_ar1": False, "normalize": False},

    # ── Z-score normalization ──────────────────────────────────────────────────
    "normalize":       {"y_window": 30, "x_window":  5, "x_std": False, "x_ar1": False, "normalize": True},
    "normalize_x_std": {"y_window": 30, "x_window":  5, "x_std": True,  "x_ar1": False, "normalize": True},

    # ── Kitchen sink ──────────────────────────────────────────────────────────
    "full_stats":      {"y_window": 30, "x_window":  0, "x_std": True,  "x_ar1": True,  "normalize": True},
}


def make_featurizer(y_window=30, x_window=5, x_std=False, x_ar1=False, normalize=False):
    """
    Returns a featurizer function compatible with AdaptedCAFHT._featurize_prefixes.

    Parameters
    ----------
    y_window   : int   Rolling window for Y summary statistics.
    x_window   : int   Rolling window for X summaries. 0 = full prefix.
    x_std      : bool  Include per-covariate std over the X window.
    x_ar1      : bool  Include per-covariate AR(1) coefficient over the X window.
    normalize  : bool  Z-score each feature column (across the n series at this t).
    """
    def _featurize(self, Y_prefixes, X_prefixes=None):
        Y = Y_prefixes[..., 0]          # (n, t+1)
        n, T = Y.shape

        # ── Y features ────────────────────────────────────────────────────────
        w   = min(y_window, T)
        Y_w = Y[:, -w:]                 # (n, w)

        y_mean = Y_w.mean(axis=1)
        y_std  = Y_w.std(axis=1) + 1e-8

        if w >= AR1_MIN_STEPS:
            xc  = Y_w[:, :-1];  xc  = xc  - xc.mean(axis=1, keepdims=True)
            yc  = Y_w[:, 1:];   yc  = yc  - yc.mean(axis=1, keepdims=True)
            num = (xc * yc).sum(axis=1)
            den = (xc ** 2).sum(axis=1) + 1e-8
            y_ar1 = np.clip(num / den, -5.0, 5.0)
        else:
            y_ar1 = np.zeros(n)

        feats = [y_mean, y_std, y_ar1]

        # ── X features ────────────────────────────────────────────────────────
        if X_prefixes is not None and X_prefixes.shape[-1] > 0:
            w_x = min(x_window, T) if x_window > 0 else T
            X_w = X_prefixes[:, -w_x:, :]          # (n, w_x, n_cov)

            feats.append(X_w.mean(axis=1))          # always include X mean

            if x_std:
                feats.append(X_w.std(axis=1) + 1e-8)

            if x_ar1:
                n_cov = X_w.shape[-1]
                if w_x >= AR1_MIN_STEPS:
                    Xc   = X_w[:, :-1, :]          # (n, w_x-1, n_cov)
                    Xn   = X_w[:, 1:,  :]
                    Xc   = Xc - Xc.mean(axis=1, keepdims=True)
                    Xn   = Xn - Xn.mean(axis=1, keepdims=True)
                    num  = (Xc * Xn).sum(axis=1)   # (n, n_cov)
                    den  = (Xc ** 2).sum(axis=1) + 1e-8
                    x_ar1_feats = np.clip(num / den, -5.0, 5.0)
                else:
                    x_ar1_feats = np.zeros((n, n_cov))
                feats.append(x_ar1_feats)

        out = np.column_stack(feats)    # (n, d)

        # ── Optional z-score normalization (across series at this t) ──────────
        if normalize:
            mu  = out.mean(axis=0, keepdims=True)
            sig = out.std(axis=0,  keepdims=True) + 1e-8
            out = (out - mu) / sig

        return out

    return _featurize


def _ess_pct(weights):
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    ess = 1.0 / (w ** 2).sum()
    return 100.0 * ess / len(w)


def run_one(result, test_sector, alpha, seed, gamma_grid, featurizer_fn=None):
    """
    Run a single-seed finance experiment and return summary metrics.

    Returns dict with keys:
      overall_coverage, mean_width,
      final_clf_acc, final_ess_pct
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    Y         = result["Y"]
    X         = result["X"]
    dates     = result["dates"]
    cov_names = result["cov_names"]
    meta      = result["meta"]
    tickers   = result["tickers"]

    n_series, L, n_cov = X.shape
    T = L - 1

    rng       = np.random.default_rng(seed)
    test_mask = np.array([m["sector"].lower() == test_sector.lower() for m in meta])
    n_test    = int(test_mask.sum())
    other_idx = rng.permutation(np.where(~test_mask)[0])
    n_cal     = len(other_idx) // 2
    n_train   = len(other_idx) - n_cal
    train_idx = other_idx[:n_train]
    cal_idx   = other_idx[n_train:]
    test_idx  = np.where(test_mask)[0]

    Y_train, X_train = Y[train_idx], X[train_idx]
    Y_cal,   X_cal   = Y[cal_idx],   X[cal_idx]
    Y_test,  X_test  = Y[test_idx],  X[test_idx]

    with_shift = featurizer_fn is not None

    predictor    = AdaptedCAFHT(alpha=alpha)
    linear_model = LinearCovariateModel(cov_names)

    if with_shift:
        predictor._featurize_prefixes = types.MethodType(featurizer_fn, predictor)

    alpha_t   = np.full(n_test, alpha, dtype=float)
    gamma_opt = float(gamma_grid[0])

    coverage_by_time = []
    width_by_time    = []
    all_covered      = []

    # For final-step diagnostics
    final_clf_acc = float("nan")
    final_ess_pct = float("nan")

    for t in range(T):
        linear_model.fit(Y_train[:, :t+2, :], X_train[:, :t+2, :])
        predictor.noise_std = linear_model.noise_std

        cal_scores = []
        for i in range(n_cal):
            for s in range(t + 2):
                cal_scores.append(abs(
                    float(Y_cal[i, s, 0]) -
                    linear_model.predict(X_cal[i, s, :])
                ))
        cal_scores_arr = np.array(cal_scores, dtype=float)

        predictor._scores  = cal_scores_arr
        predictor._weights = np.ones(len(cal_scores_arr), dtype=float)
        predictor._q       = None

        if t > 0 and (t % 10 == 0):
            gamma_opt, _ = _select_gamma(
                Y_train=Y_train, X_train=X_train, cov_names=cov_names,
                base_alpha=alpha, t_max=t, gamma_grid=gamma_grid,
                seed=seed + 10000 + t,
            )

        alpha_used = alpha_t.copy()
        alpha_next = alpha_t.copy()
        covered_t  = []
        width_t    = []

        if with_shift and t >= 1:
            mid    = n_test // 2
            half1  = np.arange(0, mid)
            half2  = np.arange(mid, n_test)

            train_Y_pre = Y_train[:, :t+1, :]
            train_X_pre = X_train[:, :t+1, :]

            for pred_idx, ctx_idx in [(half1, half2), (half2, half1)]:
                predictor.update_weighting_context(
                    train_prefixes=train_Y_pre,
                    test_prefixes=Y_test[ctx_idx, :t+1, :],
                    is_shifted=True,
                    train_X_prefixes=train_X_pre,
                    test_X_prefixes=X_test[ctx_idx, :t+1, :],
                )

                cal_feat = predictor._featurize_prefixes(
                    Y_cal[:, :t+1, :],
                    X_cal[:, :t+1, :],
                )
                per_w = predictor._compute_density_ratio_weights(
                    predictor._train_feat_t,
                    predictor._test_feat_t,
                    cal_feat,
                )

                n_steps = t + 2
                tiled_w = np.repeat(per_w, n_steps)
                predictor._scores  = cal_scores_arr
                predictor._weights = tiled_w
                predictor._q       = None

                # Capture final-step diagnostics
                if t == T - 1 and pred_idx is half1:
                    final_ess_pct = _ess_pct(per_w)
                    clf = predictor._clf
                    if clf is not None:
                        X_all = np.vstack([predictor._train_feat_t,
                                           predictor._test_feat_t])
                        y_all = np.concatenate([
                            np.zeros(predictor._train_feat_t.shape[0], int),
                            np.ones( predictor._test_feat_t.shape[0],  int),
                        ])
                        y_pred = clf.predict(X_all)
                        final_clf_acc = float(balanced_accuracy_score(y_all, y_pred))

                for i in pred_idx:
                    y_true = float(Y_test[i, t+1, 0])
                    y_pred_v = linear_model.predict(X_test[i, t+1, :])
                    a = float(np.clip(alpha_used[i], 1e-6, 1-1e-6))
                    q = predictor._weighted_quantile(
                        predictor._scores, predictor._weights, 1.0 - a)
                    lo, hi  = y_pred_v - q, y_pred_v + q
                    covered = int(lo <= y_true <= hi)
                    covered_t.append(covered)
                    width_t.append(hi - lo)
                    err = 0 if covered else 1
                    alpha_next[i] = alpha_used[i] + gamma_opt * (alpha - err)

        else:
            for i in range(n_test):
                y_true   = float(Y_test[i, t+1, 0])
                y_pred_v = linear_model.predict(X_test[i, t+1, :])
                a = float(np.clip(alpha_used[i], 1e-6, 1-1e-6))
                q = predictor._weighted_quantile(
                    predictor._scores, predictor._weights, 1.0 - a)
                lo, hi   = y_pred_v - q, y_pred_v + q
                covered  = int(lo <= y_true <= hi)
                covered_t.append(covered)
                width_t.append(hi - lo)
                err = 0 if covered else 1
                alpha_next[i] = alpha_used[i] + gamma_opt * (alpha - err)

        alpha_t = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)
        coverage_by_time.append(float(np.mean(covered_t)))
        width_by_time.append(float(np.mean(width_t)))
        all_covered.extend(covered_t)

    return {
        "overall_coverage": float(np.mean(all_covered)),
        "mean_width":        float(np.mean(width_by_time)),
        "final_clf_acc":     final_clf_acc,
        "final_ess_pct":     final_ess_pct,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare featurizer variants for LR-based shift correction."
    )
    parser.add_argument("--npz",         required=True)
    parser.add_argument("--test_sector", default="Technology")
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--gamma_grid",  type=float, nargs="+", default=GAMMA_GRID)
    parser.add_argument("--variants",    nargs="+",  default=None,
                        help="Names of variants to run (default: all). "
                             f"Available: {list(VARIANTS.keys())}")
    parser.add_argument("--save_json",   default=None,
                        help="Save comparison table to JSON.")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    result   = load_stored(npz_path, npz_path.with_suffix(".json"))

    to_run = args.variants if args.variants else list(VARIANTS.keys())
    invalid = [v for v in to_run if v not in VARIANTS]
    if invalid:
        parser.error(f"Unknown variants: {invalid}. "
                     f"Available: {list(VARIANTS.keys())}")

    print(f"\nFeaturizer tuning: {args.test_sector} test sector")
    print(f"Window: {npz_path.stem}")
    print(f"alpha={args.alpha}  seed={args.seed}")
    print(f"Variants to test: {to_run}\n")

    # ── Run no-shift baseline once ────────────────────────────────────────────
    print("Running no-shift baseline (uniform weights)...")
    noshift = run_one(
        result, args.test_sector, args.alpha, args.seed,
        args.gamma_grid, featurizer_fn=None,
    )
    cov_noshift   = noshift["overall_coverage"]
    width_noshift = noshift["mean_width"]
    print(f"  no-shift:  coverage={cov_noshift:.4f}  width={width_noshift:.5f}\n")

    # ── Run each featurizer variant ───────────────────────────────────────────
    rows = []
    for name in to_run:
        kwargs = VARIANTS[name]
        print(f"Running variant '{name}': {kwargs}")
        fn = make_featurizer(**kwargs)
        res = run_one(
            result, args.test_sector, args.alpha, args.seed,
            args.gamma_grid, featurizer_fn=fn,
        )
        gap         = res["overall_coverage"] - cov_noshift
        width_ratio = res["mean_width"] / width_noshift

        rows.append({
            "variant":     name,
            "clf_acc":     res["final_clf_acc"],
            "ess_pct":     res["final_ess_pct"],
            "cov_shift":   res["overall_coverage"],
            "gap":         gap,
            "width_ratio": width_ratio,
            **kwargs,
        })

        print(f"  clf_acc={res['final_clf_acc']:.3f}  "
              f"ess%={res['final_ess_pct']:.1f}  "
              f"cov={res['overall_coverage']:.4f}  "
              f"gap={gap:+.4f}  "
              f"width_ratio={width_ratio:.4f}")
        print()

    # ── Print comparison table ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Variant':<20} {'clf_acc':>8} {'ess%':>7} {'cov_shift':>10} "
          f"{'gap':>8} {'width_ratio':>12}")
    print("-" * 90)

    # Sort by gap descending (best coverage improvement first)
    rows_sorted = sorted(rows, key=lambda r: r["gap"], reverse=True)
    for r in rows_sorted:
        print(f"{r['variant']:<20} {r['clf_acc']:>8.3f} {r['ess_pct']:>7.1f} "
              f"{r['cov_shift']:>10.4f} {r['gap']:>+8.4f} {r['width_ratio']:>12.4f}")

    print("-" * 90)
    print(f"{'no-shift (baseline)':<20} {'---':>8} {'100.0':>7} "
          f"{cov_noshift:>10.4f} {'---':>8} {'1.0000':>12}")
    print("=" * 90)
    print("\nMetrics:")
    print("  clf_acc     — balanced accuracy of LR classifier at final time step")
    print("                (higher = better discrimination; random = 0.50)")
    print("  ess%        — effective sample size of LR weights (% of cal set)")
    print("                (lower = more extreme correction; 100% = uniform)")
    print("  cov_shift   — overall empirical coverage with LR weighting")
    print("  gap         — cov_shift - cov_noshift (positive = weighting helps)")
    print("  width_ratio — mean_width(shift) / mean_width(noshift)")
    print("                (< 1 = narrower bands; > 1 = wider)")

    # ── Optionally save ───────────────────────────────────────────────────────
    output = {
        "npz": str(npz_path),
        "test_sector": args.test_sector,
        "alpha": args.alpha,
        "seed": args.seed,
        "no_shift": {"coverage": cov_noshift, "mean_width": width_noshift},
        "variants": rows_sorted,
    }

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {out}")

    return output


if __name__ == "__main__":
    main()
