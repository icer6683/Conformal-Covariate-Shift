#!/usr/bin/env python3
"""
=============================================================================
FINANCE CONFORMAL PREDICTION  —  AdaptedCAFHT on S&P 500 data
=============================================================================

QUICK START
-----------
Step 1 — pull data (once):
  python finance_data.py --pull --start 2024-01-01 --end 2024-04-01

Step 2 — run experiment:

  # Technology as test sector, no shift correction (default):
  python finance_conformal.py --npz sp500_20231004_20240328.npz --test_sector Technology

  # Technology as test sector, WITH shift correction:
  python finance_conformal.py --npz sp500_20231004_20240328.npz  --test_sector Technology --with_shift

  # Other sectors:
  python finance_conformal.py --npz sp500_20240102_20240229.npz --test_sector Healthcare --with_shift
  python finance_conformal.py --npz sp500_20240102_20240229.npz --test_sector Financials --with_shift

FULL OPTIONS
------------
  --npz          Path to .npz data file (required)
  --test_sector  Sector held out as test set (required)
  --with_shift   Enable likelihood-ratio covariate-shift weighting.
  --alpha        Miscoverage level. Default: 0.1  (targets 90% coverage)
  --cal_frac     Fraction of non-test tickers used for calibration. Default: 0.5
  --gamma_grid   Space-separated ACI step-size candidates.
  --seed         Random seed. Default: 42
  --save_plot    Path to save the output figure.
  --save_json    Path to save results as JSON.

FEATURIZER
----------
  The logistic classifier receives one feature vector per ticker.

  Y features — computed over the most recent Y_WINDOW=30 steps of Y
  (or fewer if t < Y_WINDOW, in which case all available steps are used):
    Y_mean   mean return over the rolling window
    Y_std    standard deviation of returns
    Y_ar1    AR(1) slope (momentum) — only included when window >= 5 steps,
             otherwise set to 0 to avoid noise from too-short prefixes

  X features — computed over the FULL prefix (not windowed):
    X_mean_k  mean of covariate k over all available steps, for each k
             X features use the full prefix because covariates like
             TurnoverRatio reflect structural sector properties that
             are better estimated over longer histories.
             X_mean_Above52wLowReturn is excluded — it has near-zero
             separation and high variance that swamps the classifier.

  Features are standardised (z-scored) before being passed to the logistic
  classifier so that low-variance but informative features like
  TurnoverRatio_lag1 are not dominated by high-variance ones.

  Total feature vector length: 3 (Y) + n_cov - 1 (X, minus Above52wLow)
=============================================================================
"""
import argparse
import json
import types
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from finance_data import load_stored
from algorithm import AdaptedCAFHT

GAMMA_GRID   = [0.001, 0.005, 0.01, 0.05]
Y_WINDOW     = 30   # rolling window for Y features
AR1_MIN_STEPS = 5   # minimum steps needed before including Y_ar1


# =============================================================================
# Featurizer
# =============================================================================
def _featurize_YX_summaries(self, Y_prefixes, X_prefixes=None):
    """
    Y_prefixes: (n, t+1, 1)
    X_prefixes: (n, t+1, n_cov)  optional

    Y features (3) — computed over the most recent Y_WINDOW steps:
      Y_mean, Y_std, Y_ar1

    X features (n_cov - 1) — computed over the full prefix:
      mean of each covariate EXCEPT Above52wLowReturn (index 1), which
      is excluded due to near-zero separation and high variance.

    All features are z-scored before return so that the logistic
    classifier treats each dimension on equal footing.
    """
    Y = Y_prefixes[..., 0]      # (n, t+1)
    n, T = Y.shape

    # ── Y features: rolling window ────────────────────────────────────────────
    # Use the last Y_WINDOW steps, or all steps if T < Y_WINDOW.
    w    = min(Y_WINDOW, T)
    Y_w  = Y[:, -w:]            # (n, w)

    y_mean = Y_w.mean(axis=1)
    y_std  = Y_w.std(axis=1) + 1e-8

    if w >= AR1_MIN_STEPS:
        x_ar = Y_w[:, :-1]
        y_ar = Y_w[:, 1:]
        x_c  = x_ar - x_ar.mean(axis=1, keepdims=True)
        y_c  = y_ar - y_ar.mean(axis=1, keepdims=True)
        num  = (x_c * y_c).sum(axis=1)
        den  = (x_c ** 2).sum(axis=1) + 1e-8
        y_ar1 = np.clip(num / den, -5.0, 5.0)
    else:
        y_ar1 = np.zeros(n)

    y_feats = np.column_stack([y_mean, y_std, y_ar1])   # (n, 3)

    # ── X features: full prefix, exclude Above52wLowReturn (index 1) ─────────
    if X_prefixes is not None and X_prefixes.shape[-1] > 0:
        # Drop column index 1 (Above52wLowReturn) — near-zero |sep| throughout
        # and high variance that swamps the logistic regression coefficients.
        keep_cols = [c for c in range(X_prefixes.shape[-1]) if c != 1]
        X_keep    = X_prefixes[:, :, keep_cols]         # (n, T, n_cov-1)
        x_means   = X_keep.mean(axis=1)                 # (n, n_cov-1)
        feats     = np.concatenate([y_feats, x_means], axis=1)
    else:
        feats = y_feats

    # ── z-score normalisation ─────────────────────────────────────────────────
    # Standardise across the n tickers so every feature has mean=0, std=1.
    # This prevents high-variance features from dominating the LR gradient.
    mu  = feats.mean(axis=0, keepdims=True)
    sig = feats.std(axis=0, keepdims=True) + 1e-8
    return (feats - mu) / sig


# =============================================================================
# Diagnostic helpers
# =============================================================================
def _print_weight_diagnostic(weights, t, label=""):
    w = np.asarray(weights, dtype=float)
    is_uniform = bool(np.allclose(w, w[0], rtol=1e-4, atol=1e-6))
    lo, hi = w.min(), w.max()
    if hi > lo:
        edges     = np.linspace(lo, hi, 6)
        counts, _ = np.histogram(w, bins=edges)
        hist_str  = "  ".join(
            f"[{edges[i]:.4f},{edges[i+1]:.4f}): {counts[i]}" for i in range(5)
        )
    else:
        hist_str = f"all values = {lo:.6f}"
    ess = float(1.0 / ((w / w.sum()) ** 2).sum()) if w.sum() > 0 else 0.0
    tag = f" {label}" if label else ""
    print(f"  [WeightDiag t={t:3d}{tag}]"
          f"  n={len(w)}  min={lo:.5f}  max={hi:.5f}"
          f"  mean={w.mean():.5f}  std={w.std():.5f}"
          f"  ESS={ess:.1f} ({100*ess/len(w):.0f}%)"
          f"  uniform={is_uniform}")
    print(f"    histogram: {hist_str}")


def _print_feature_diagnostic(predictor, cal_feat, t, feat_names):
    train_feat = predictor._train_feat_t
    test_feat  = predictor._test_feat_t
    if train_feat is None or test_feat is None:
        print(f"  [FeatDiag t={t:3d}] features not set")
        return
    print(f"  [FeatDiag t={t:3d}]  train={train_feat.shape}  "
          f"test={test_feat.shape}  cal={cal_feat.shape}  "
          f"(Y window=min({Y_WINDOW},t))")
    for col in range(train_feat.shape[1]):
        tr   = train_feat[:, col]
        te   = test_feat[:, col]
        ca   = cal_feat[:, col]
        sep  = abs(tr.mean() - te.mean()) / (tr.std() + te.std() + 1e-8)
        name = feat_names[col] if col < len(feat_names) else f"feat{col}"
        print(f"    {name:30s}  "
              f"train: mean={tr.mean():+.3f} std={tr.std():.3f}  "
              f"test:  mean={te.mean():+.3f} std={te.std():.3f}  "
              f"|sep|={sep:.3f}")


def _print_classifier_diagnostic(predictor, cal_feat, t):
    clf = predictor._clf
    if clf is None:
        print(f"  [CLFDiag t={t:3d}] classifier is None (sklearn unavailable)")
        return
    train_feat = predictor._train_feat_t
    test_feat  = predictor._test_feat_t
    if train_feat is None or test_feat is None:
        return
    N0, N1   = train_feat.shape[0], test_feat.shape[0]
    X_all    = np.vstack([train_feat, test_feat])
    y_all    = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
    acc       = clf.score(X_all, y_all)
    coef_norm = float(np.linalg.norm(clf.coef_))
    prob_cal  = clf.predict_proba(cal_feat)[:, 1]
    print(f"  [CLFDiag t={t:3d}]  train_acc={acc:.3f}  coef_norm={coef_norm:.4f}  "
          f"prob1_cal: min={prob_cal.min():.3f}  max={prob_cal.max():.3f}  "
          f"mean={prob_cal.mean():.3f}  std={prob_cal.std():.4f}")


# =============================================================================
# Linear prediction model
# =============================================================================
class LinearCovariateModel:
    def __init__(self, cov_names):
        self.cov_names = cov_names
        self.beta      = None
        self.noise_std = 1.0

    def fit(self, Y_train, X_train, verbose=False):
        n, L, n_cov = X_train.shape
        if L < 2:
            return
        y        = Y_train[:, :, 0].reshape(-1)
        X        = X_train.reshape(-1, n_cov)
        X_design = np.hstack([np.ones((len(y), 1)), X])
        self.beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        resid          = y - X_design @ self.beta
        self.noise_std = float(np.std(resid, ddof=X_design.shape[1]))
        if verbose:
            print(f"  [Model] Fitted on {n} series x {L} steps")
            print(f"  [Model] Residual std : {self.noise_std:.4f}")
            print(f"  [Model] Coefficients :")
            print(f"            intercept = {self.beta[0]:.4f}")
            for name, coef in zip(self.cov_names, self.beta[1:]):
                print(f"            {name:22s} = {coef:.4f}")

    def predict(self, x_t):
        if self.beta is None:
            return 0.0
        return float(self.beta[0] + x_t @ self.beta[1:])


# =============================================================================
# Gamma selection
# =============================================================================
def _select_gamma(Y_train, X_train, cov_names, base_alpha, t_max, gamma_grid, seed=0):
    n_train = Y_train.shape[0]
    if n_train < 9 or t_max < 2:
        return float(gamma_grid[0]), {float(g): float('nan') for g in gamma_grid}
    rng  = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    n1 = n_train // 3;  n2 = n_train // 3;  n3 = n_train - n1 - n2
    if n1 == 0 or n2 == 0 or n3 == 0:
        return float(gamma_grid[0]), {float(g): float('nan') for g in gamma_grid}
    idx1 = perm[:n1];  idx2 = perm[n1:n1+n2];  idx3 = perm[n1+n2:]
    Y_fit_sel = Y_train[idx1];  X_fit_sel = X_train[idx1]
    Y_cal_sel = Y_train[idx2];  X_cal_sel = X_train[idx2]
    Y_eval    = Y_train[idx3];  X_eval    = X_train[idx3]
    n_eval    = Y_eval.shape[0]
    horizon   = min(t_max, Y_train.shape[1] - 1)
    start_eval = max(0, horizon // 2)
    target    = 1.0 - base_alpha
    scores    = {}
    for gamma in gamma_grid:
        gamma        = float(gamma)
        predictor    = AdaptedCAFHT(alpha=base_alpha)
        alpha_series = np.full(n_eval, base_alpha, dtype=float)
        cov_hist     = []
        for t in range(horizon + 1):
            sel_model = LinearCovariateModel(cov_names)
            sel_model.fit(Y_fit_sel[:, :t+1, :], X_fit_sel[:, :t+1, :])
            predictor.noise_std = sel_model.noise_std
            cal_scores = [
                abs(float(Y_cal_sel[i, s, 0]) - sel_model.predict(X_cal_sel[i, s, :]))
                for i in range(len(idx2)) for s in range(t + 1)
            ]
            predictor._scores  = np.array(cal_scores, dtype=float)
            predictor._weights = np.ones(len(cal_scores), dtype=float)
            predictor._q       = None
            alpha_used = alpha_series.copy()
            alpha_next = alpha_series.copy()
            step_cov   = []
            for i in range(n_eval):
                y_true  = float(Y_eval[i, t, 0])
                y_pred  = sel_model.predict(X_eval[i, t, :])
                a       = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
                q       = predictor._weighted_quantile(
                    predictor._scores, predictor._weights, 1.0 - a)
                covered = int(y_pred - q <= y_true <= y_pred + q)
                step_cov.append(covered)
                alpha_next[i] = alpha_used[i] + gamma * (base_alpha - (0 if covered else 1))
            alpha_series = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)
            cov_hist.append(float(np.mean(step_cov)) if step_cov else float('nan'))
        tail = cov_hist[start_eval:]
        scores[gamma] = float(np.mean(tail)) if tail else float('nan')
    best_gamma = float(gamma_grid[0])
    best_obj   = float('inf')
    for gamma, metric in scores.items():
        if np.isfinite(metric) and abs(metric - target) < best_obj:
            best_obj   = abs(metric - target)
            best_gamma = float(gamma)
    return best_gamma, scores


# =============================================================================
# Main experiment
# =============================================================================
def run_finance_experiment(result, test_sector, cal_frac=0.5, alpha=0.1, seed=42,
                           gamma_grid=None, with_shift=False):
    if gamma_grid is None:
        gamma_grid = GAMMA_GRID

    Y         = result["Y"]
    X         = result["X"]
    dates     = result["dates"]
    cov_names = result["cov_names"]
    meta      = result["meta"]
    tickers   = result["tickers"]

    n_series, L, n_cov = X.shape
    T = L - 1

    # Feature names for diagnostics.
    # X col 1 (Above52wLowReturn) is excluded from the featurizer.
    x_feat_names = [f"X_mean_{cov_names[c]}" for c in range(n_cov) if c != 1]
    feat_names   = ["Y_mean(w30)", "Y_std(w30)", "Y_ar1(w30)"] + x_feat_names

    test_mask = np.array([m["sector"].lower() == test_sector.lower() for m in meta])
    n_test    = int(test_mask.sum())
    n_other   = int((~test_mask).sum())

    if n_test == 0:
        available = sorted({m["sector"] for m in meta})
        raise ValueError(
            f"No tickers found for sector '{test_sector}'.\nAvailable: {available}"
        )
    if n_other == 0:
        raise ValueError("All tickers belong to the test sector.")

    rng       = np.random.default_rng(seed)
    other_idx = rng.permutation(np.where(~test_mask)[0])
    n_cal     = int(n_other * cal_frac)
    n_train   = n_other - n_cal
    if n_train == 0:
        raise ValueError(f"cal_frac={cal_frac} leaves no training tickers.")

    train_idx = other_idx[:n_train]
    cal_idx   = other_idx[n_train:]
    test_idx  = np.where(test_mask)[0]

    Y_train, X_train = Y[train_idx], X[train_idx]
    Y_cal,   X_cal   = Y[cal_idx],   X[cal_idx]
    Y_test,  X_test  = Y[test_idx],  X[test_idx]
    test_tickers = [tickers[i] for i in test_idx]

    print(f"\n{'='*62}")
    print(f"  Finance Conformal Experiment  (AdaptedCAFHT)")
    print(f"{'='*62}")
    print(f"  Total tickers   : {n_series}")
    print(f"  Test sector     : {test_sector}  ({n_test} tickers)")
    print(f"  Train           : {n_train} non-{test_sector} tickers")
    print(f"  Cal             : {n_cal} non-{test_sector} tickers")
    print(f"  Time steps      : {L}  [{dates[0]} -> {dates[-1]}]")
    print(f"  Covariates      : {cov_names}")
    print(f"  Alpha           : {alpha}  (target = {1-alpha:.0%})")
    print(f"  Gamma grid      : {gamma_grid}")
    print(f"  With shift      : {with_shift}")
    if with_shift:
        print(f"  Y featurizer    : mean/std/ar1 over rolling window={Y_WINDOW} steps")
        print(f"  X featurizer    : mean over full prefix  "
              f"(Above52wLowReturn excluded)")
        print(f"  Standardisation : z-score per feature across tickers")
        print(f"  Feature names   : {feat_names}")
    print()

    predictor    = AdaptedCAFHT(alpha=alpha, logistic_kwargs={"C": 0.1})
    linear_model = LinearCovariateModel(cov_names)

    if with_shift:
        predictor._featurize_prefixes = types.MethodType(
            _featurize_YX_summaries, predictor
        )

    alpha_t   = np.full(n_test, alpha, dtype=float)
    gamma_opt = float(gamma_grid[0])

    coverage_by_time  = []
    width_by_time     = []
    all_covered       = []
    gamma_opt_history = []
    first_true  = []
    first_lower = []
    first_upper = []

    for t in range(T):

        # ── fit linear model ──────────────────────────────────────────────────
        linear_model.fit(Y_train[:, :t+2, :], X_train[:, :t+2, :],
                         verbose=(t == T - 1))
        predictor.noise_std = linear_model.noise_std

        # ── build calibration scores ──────────────────────────────────────────
        cal_scores = []
        for i in range(n_cal):
            for s in range(t + 2):
                y_true = float(Y_cal[i, s, 0])
                y_pred = linear_model.predict(X_cal[i, s, :])
                cal_scores.append(abs(y_true - y_pred))
        cal_scores_arr = np.array(cal_scores, dtype=float)

        predictor._scores  = cal_scores_arr
        predictor._weights = np.ones(len(cal_scores_arr), dtype=float)
        predictor._q       = None

        # ── gamma selection every 10 steps ────────────────────────────────────
        if t > 0 and (t % 10 == 0):
            sel_seed = seed + 10000 + t
            gamma_opt, gamma_scores = _select_gamma(
                Y_train=Y_train, X_train=X_train, cov_names=cov_names,
                base_alpha=alpha, t_max=t, gamma_grid=gamma_grid, seed=sel_seed,
            )
            scores_str = "  ".join(
                f"gamma={g:.3f}->{v:.3f}" for g, v in gamma_scores.items()
                if np.isfinite(v)
            )
            print(f"  [gamma sel t={t:3d}]  best gamma = {gamma_opt}   ({scores_str})")

        gamma_opt_history.append(float(gamma_opt))

        # ── prediction loop ───────────────────────────────────────────────────
        if with_shift and t >= 1:
            train_Y_pre = Y_train[:, :t+1, :]
            train_X_pre = X_train[:, :t+1, :]

            mid   = n_test // 2
            half1 = np.arange(0, mid)
            half2 = np.arange(mid, n_test)

            alpha_used = alpha_t.copy()
            alpha_next = alpha_t.copy()
            covered_t  = []
            width_t    = []

            for pred_idx, ctx_idx in [(half1, half2), (half2, half1)]:

                test_Y_pre = Y_test[ctx_idx, :t+1, :]
                test_X_pre = X_test[ctx_idx, :t+1, :]

                predictor.update_weighting_context(
                    train_prefixes=train_Y_pre,
                    test_prefixes=test_Y_pre,
                    is_shifted=True,
                    train_X_prefixes=train_X_pre,
                    test_X_prefixes=test_X_pre,
                )

                cal_feat = predictor._featurize_prefixes(
                    Y_cal[:, :t+1, :],
                    X_cal[:, :t+1, :],
                )

                per_series_w = predictor._compute_density_ratio_weights(
                    trainX=predictor._train_feat_t,
                    testX=predictor._test_feat_t,
                    evalX=cal_feat,
                )

                # ── diagnostics (swap 0, key time steps) ─────────────────────
                if pred_idx is half1 and (t == 1 or t % 10 == 0 or t == T - 1):
                    _print_feature_diagnostic(predictor, cal_feat, t, feat_names)
                    _print_classifier_diagnostic(predictor, cal_feat, t)

                # Shape guards
                n_steps = t + 2
                assert len(per_series_w) == n_cal, (
                    f"per_series_w {len(per_series_w)} != n_cal {n_cal}")
                tiled_w = np.repeat(per_series_w, n_steps)
                assert len(tiled_w) == len(cal_scores_arr), (
                    f"tiled_w {len(tiled_w)} != cal_scores {len(cal_scores_arr)}")

                predictor._scores  = cal_scores_arr
                predictor._weights = tiled_w
                predictor._q       = None

                if t == 1 or t % 10 == 0 or t == T - 1:
                    label = "half1-ctx" if pred_idx is half1 else "half2-ctx"
                    _print_weight_diagnostic(per_series_w, t, label=label)

                for i in pred_idx:
                    x_t    = X_test[i, t+1, :]
                    y_true = float(Y_test[i, t+1, 0])
                    y_pred = linear_model.predict(x_t)
                    a      = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
                    q      = predictor._weighted_quantile(
                        predictor._scores, predictor._weights, 1.0 - a)
                    lo, hi  = y_pred - q, y_pred + q
                    covered = int(lo <= y_true <= hi)
                    covered_t.append(covered)
                    width_t.append(hi - lo)
                    if i == 0:
                        first_true.append(y_true)
                        first_lower.append(lo)
                        first_upper.append(hi)
                    err = 0 if covered else 1
                    alpha_next[i] = alpha_used[i] + gamma_opt * (alpha - err)

            alpha_t = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)

        else:
            alpha_used = alpha_t.copy()
            alpha_next = alpha_t.copy()
            covered_t  = []
            width_t    = []

            for i in range(n_test):
                x_t    = X_test[i, t+1, :]
                y_true = float(Y_test[i, t+1, 0])
                y_pred = linear_model.predict(x_t)
                a      = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
                q      = predictor._weighted_quantile(
                    predictor._scores, predictor._weights, 1.0 - a)
                lo, hi  = y_pred - q, y_pred + q
                covered = int(lo <= y_true <= hi)
                covered_t.append(covered)
                width_t.append(hi - lo)
                if i == 0:
                    first_true.append(y_true)
                    first_lower.append(lo)
                    first_upper.append(hi)
                err = 0 if covered else 1
                alpha_next[i] = alpha_used[i] + gamma_opt * (alpha - err)

            alpha_t = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)

        coverage_by_time.append(float(np.mean(covered_t)))
        width_by_time.append(float(np.mean(width_t)))
        all_covered.extend(covered_t)

        if (t + 1) % 10 == 0 or t == T - 1:
            print(f"  [t={t+1:3d}/{T}]  coverage={np.mean(covered_t):.3f}  "
                  f"width={np.mean(width_t):.4f}  gamma={gamma_opt}")

    overall_coverage = float(np.mean(all_covered))
    target = 1.0 - alpha
    print(f"\n  Overall coverage : {overall_coverage:.4f}  "
          f"(target = {target:.4f},  error = {overall_coverage - target:+.4f})")
    print(f"  Mean width       : {np.mean(width_by_time):.4f}")
    print(f"  Final gamma_opt  : {gamma_opt}")

    return {
        "coverage_by_time":  coverage_by_time,
        "width_by_time":     width_by_time,
        "overall_coverage":  overall_coverage,
        "target_coverage":   target,
        "dates":             [str(dates[t+1]) for t in range(T)],
        "gamma_opt_history": gamma_opt_history,
        "first_test_ticker": test_tickers[0],
        "first_test_series": {
            "true":  first_true,
            "lower": first_lower,
            "upper": first_upper,
        },
        "config": {
            "test_sector": test_sector,
            "n_train":     int(n_train),
            "n_cal":       int(n_cal),
            "n_test":      int(n_test),
            "L":           int(L),
            "alpha":       alpha,
            "gamma_grid":  [float(g) for g in gamma_grid],
            "seed":        seed,
            "with_shift":  with_shift,
            "Y_window":    Y_WINDOW,
            "cov_names":   cov_names,
            "feat_names":  feat_names,
        },
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(results, save_path=None):
    dates   = results["dates"]
    cov_t   = results["coverage_by_time"]
    width_t = results["width_by_time"]
    target  = results["target_coverage"]
    cfg     = results["config"]
    first   = results["first_test_series"]
    ticker  = results["first_test_ticker"]
    gammas  = results["gamma_opt_history"]

    x = np.arange(len(dates))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(
        f"Finance Conformal Prediction (AdaptedCAFHT)  |  "
        f"Test sector: {cfg['test_sector']}  |  "
        f"alpha={cfg['alpha']}  |  "
        f"train/cal/test = {cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} tickers",
        fontsize=12, fontweight='bold'
    )

    axes[0, 0].plot(x, cov_t, 'b-', linewidth=1.5, label='Empirical coverage')
    axes[0, 0].axhline(target, color='red', linestyle='--', linewidth=2,
                       label=f'Target ({target:.0%})')
    axes[0, 0].set_ylim(0.5, 1.05)
    axes[0, 0].set_ylabel('Coverage rate')
    axes[0, 0].set_title(f'Coverage over Time  ({cfg["test_sector"]} sector)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(
        0.02, 0.05,
        f"Overall: {results['overall_coverage']:.3f}  "
        f"(error {results['overall_coverage'] - target:+.3f})",
        transform=axes[0, 0].transAxes, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    axes[0, 1].plot(x, width_t, 'g-', linewidth=1.5)
    axes[0, 1].set_ylabel('Mean interval width')
    axes[0, 1].set_title('Prediction Interval Width over Time')
    axes[0, 1].grid(True, alpha=0.3)

    true_arr  = np.array(first["true"])
    lower_arr = np.array(first["lower"])
    upper_arr = np.array(first["upper"])
    y_min     = float(np.percentile(true_arr, 1))
    y_max     = float(np.percentile(true_arr, 99))
    margin    = (y_max - y_min) * 0.3
    axes[1, 0].fill_between(x, lower_arr, upper_arr, alpha=0.25, color='steelblue',
                             label='Prediction interval')
    axes[1, 0].plot(x, true_arr, 'k-', linewidth=1.5, label='Actual close')
    axes[1, 0].set_ylim(y_min - margin, y_max + margin)
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].set_title(f'{ticker} - Actual Close vs. Prediction Interval')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x, gammas, drawstyle='steps-post', color='purple', linewidth=1.5)
    axes[1, 1].set_ylabel('Selected gamma (log scale)')
    axes[1, 1].set_title('ACI Gamma Selected over Time')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    tick_every = max(1, len(dates) // 10)
    tick_pos   = x[::tick_every]
    tick_lbl   = [dates[i] for i in tick_pos]
    for row in axes:
        for ax in row:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=8)
            ax.set_xlabel('Date')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved to {save_path}")
    plt.show()


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run AdaptedCAFHT conformal prediction on S&P 500 finance data."
    )
    parser.add_argument("--npz",         required=True)
    parser.add_argument("--json",        default=None)
    parser.add_argument("--test_sector", required=True)
    parser.add_argument("--cal_frac",    type=float, default=0.5)
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--gamma_grid",  type=float, nargs='+', default=GAMMA_GRID)
    parser.add_argument("--with_shift",  action="store_true", default=False)
    parser.add_argument("--save_plot",   default=None)
    parser.add_argument("--save_json",   default=None)
    args = parser.parse_args()

    npz_path  = Path(args.npz)
    json_path = Path(args.json) if args.json else npz_path.with_suffix(".json")

    print(f"Loading {npz_path} ...")
    result = load_stored(npz_path, json_path)

    results = run_finance_experiment(
        result=result,
        test_sector=args.test_sector,
        cal_frac=args.cal_frac,
        alpha=args.alpha,
        seed=args.seed,
        gamma_grid=args.gamma_grid,
        with_shift=args.with_shift,
    )

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Results] Saved to {out}")

    plot_results(results, save_path=args.save_plot)


if __name__ == "__main__":
    main()