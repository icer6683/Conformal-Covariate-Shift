#!/usr/bin/env python3
"""
=============================================================================
FINANCE CONFORMAL PREDICTION  —  AdaptedCAFHT on S&P 500 data
=============================================================================

QUICK START
-----------
Step 1 — pull data (once):
  python finance_data.py --pull --start 2024-02-01 --end 2024-03-28

Step 2 — run experiment:

  # Technology as test sector, no shift correction:
  python finance_conformal.py --npz sp500_20240201_20240328.npz

  # Technology as test sector, WITH shift correction (primary result):
  python finance_conformal.py --npz sp500_20240201_20240328.npz --with_shift

  # Other sectors as test (with shift correction):
  python finance_conformal.py --npz sp500_20240201_20240328.npz --test_sector Healthcare --with_shift
  python finance_conformal.py --npz sp500_20240201_20240328.npz --test_sector Energy --with_shift

  # Mixed test set — null/no-shift baseline (randomly drawn from all sectors):
  python finance_conformal.py --npz sp500_20240201_20240328.npz --mixed
  python finance_conformal.py --npz sp500_20240201_20240328.npz --mixed --with_shift

  # Save outputs:
  python finance_conformal.py --npz sp500_20240201_20240328.npz --save_plot results/v1_tech_noshift_20240201.png

FULL OPTIONS
------------
  --npz              Path to .npz data file (required)
  --test_sector      Sector held out as test set. Default: Technology
                     Ignored when --mixed is set.
  --with_shift       Enable likelihood-ratio covariate-shift weighting.
                     Without this flag, calibration weights are uniform.
  --mixed            Mixed-sector test set: randomly draw --mixed_test_frac of
                     ALL tickers as test (no sector bias). Null/no-shift baseline
                     that validates shift correction is harmless when shift is absent.
  --mixed_test_frac  Fraction of tickers used as test in --mixed mode. Default: 0.15
  --alpha            Miscoverage level. Default: 0.1  (targets 90% coverage)
  --cal_frac         Fraction of non-test tickers used for calibration. Default: 0.5
  --gamma_grid       Space-separated ACI step-size candidates.
  --seed             Random seed. Default: 42
  --save_plot        Path to save the output figure (PNG or PDF).
  --save_json        Path to save results as JSON.

FEATURIZER  (used only with --with_shift)
-----------------------------------------
  Y features — computed over the most recent Y_WINDOW=30 steps of Y
  (or fewer if t < Y_WINDOW):
    Y_mean, Y_std, Y_ar1

  X features — mean of each covariate over the full prefix:
    X_mean_k for each covariate k

  No z-score standardisation. No regularisation override (uses algorithm.py
  default C=1.0).
=============================================================================
"""
import argparse
import json
import types
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from finance.finance_data import load_stored
from core.algorithm import AdaptedCAFHT

# ── Shared plot style ────────────────────────────────────────────────────────
_C_COV    = "#2166ac"   # coverage line  (blue)
_C_TARGET = "#d6604d"   # target line    (red-orange)
_C_WIDTH  = "#4dac26"   # width line     (green)
_C_GAMMA  = "#7b2d8b"   # gamma line     (purple)

def _style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

GAMMA_GRID    = [0.001, 0.005, 0.01, 0.05]
Y_WINDOW      = 10
X_WINDOW      = 0    # rolling window for X features (0 = full prefix mean)
AR1_MIN_STEPS = 5


# =============================================================================
# Featurizer factory: rolling window Y summaries + rolling/full-prefix X means
# =============================================================================
def _make_featurizer(x_window=X_WINDOW):
    """
    Returns a bound-method-compatible featurizer.

    x_window : int
        Number of most-recent time steps used to compute X feature means.
        0 (or negative) → use the full prefix mean (original behaviour).
        Default: X_WINDOW (5 days).

    To revert to full-prefix means, pass x_window=0 (or --x_window 0 on the CLI).
    """
    def _featurize_YX_summaries(self, Y_prefixes, X_prefixes=None):
        """
        Y_prefixes: (n, t+1, 1)
        X_prefixes: (n, t+1, n_cov)  optional

        Y features (3) over last Y_WINDOW steps:
          Y_mean, Y_std, Y_ar1

        X features (n_cov):
          mean of each covariate over the most recent x_window steps
          (or full prefix when x_window <= 0)
        """
        Y = Y_prefixes[..., 0]      # (n, t+1)
        n, T = Y.shape

        # ── Y rolling-window features ─────────────────────────────────────────
        w   = min(Y_WINDOW, T)
        Y_w = Y[:, -w:]             # (n, w)

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

        y_feats = np.column_stack([y_mean, y_std, y_ar1])  # (n, 3)

        if X_prefixes is not None and X_prefixes.shape[-1] > 0:
            # ── X rolling-window features ─────────────────────────────────────
            if x_window > 0:
                w_x = min(x_window, T)
                X_w = X_prefixes[:, -w_x:, :]   # (n, w_x, n_cov)
            else:
                X_w = X_prefixes                 # full prefix
            x_means = X_w.mean(axis=1)           # (n, n_cov)
            x_stds  = X_w.std(axis=1) + 1e-8    # (n, n_cov)
            return np.concatenate([y_feats, x_means, x_stds], axis=1)

        return y_feats

    return _featurize_YX_summaries


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
              f"train: mean={tr.mean():+.4f} std={tr.std():.4f}  "
              f"test:  mean={te.mean():+.4f} std={te.std():.4f}  "
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
    N0, N1    = train_feat.shape[0], test_feat.shape[0]
    X_all     = np.vstack([train_feat, test_feat])
    y_all     = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
    acc        = clf.score(X_all, y_all)
    coef_norm  = float(np.linalg.norm(clf.coef_))
    prob_cal   = clf.predict_proba(cal_feat)[:, 1]
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
                           gamma_grid=None, with_shift=False, mixed=False,
                           mixed_test_frac=0.15, x_window=X_WINDOW):
    """
    mixed=True  : ignore test_sector; randomly draw mixed_test_frac of all tickers
                  as test (no sector bias → null/no-shift baseline).
    mixed=False : hold out test_sector as test (sector covariate shift experiment).
    """
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

    feat_names = ["Y_mean(w30)", "Y_std(w30)", "Y_ar1(w30)"] + \
                 [f"X_mean_{c}" for c in cov_names]

    rng = np.random.default_rng(seed)

    if mixed:
        # ── Mixed mode: random test split, no sector filtering ────────────────
        n_test  = max(10, int(n_series * mixed_test_frac))
        all_idx = rng.permutation(n_series)
        test_mask           = np.zeros(n_series, dtype=bool)
        test_mask[all_idx[:n_test]] = True
        display_sector = "Mixed (all sectors)"
    else:
        test_mask = np.array([m["sector"].lower() == test_sector.lower() for m in meta])
        n_test    = int(test_mask.sum())
        display_sector = test_sector
        if n_test == 0:
            available = sorted({m["sector"] for m in meta})
            raise ValueError(
                f"No tickers found for sector '{test_sector}'.\nAvailable: {available}"
            )

    n_other = int((~test_mask).sum())
    if n_other == 0:
        raise ValueError("All tickers belong to the test set.")

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
    print(f"  Test set        : {display_sector}  ({n_test} tickers)")
    print(f"  Train           : {n_train} tickers")
    print(f"  Cal             : {n_cal} tickers")
    print(f"  Time steps      : {L}  [{dates[0]} -> {dates[-1]}]")
    print(f"  Covariates      : {cov_names}")
    print(f"  Alpha           : {alpha}  (target = {1-alpha:.0%})")
    print(f"  Gamma grid      : {gamma_grid}")
    print(f"  With shift      : {with_shift}")
    if with_shift:
        print(f"  Y featurizer    : mean/std/ar1 over rolling window={Y_WINDOW} steps")
        print(f"  X featurizer    : mean over full prefix")
        print(f"  Feature names   : {feat_names}")
    print()

    predictor    = AdaptedCAFHT(alpha=alpha)
    linear_model = LinearCovariateModel(cov_names)

    if with_shift:
        predictor._featurize_prefixes = types.MethodType(
            _make_featurizer(x_window=x_window), predictor
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
    # ── shift-correction diagnostics (only populated when with_shift=True) ────
    clf_prob1_mean_by_time = []   # mean P(test class) on cal set per step
    clf_prob1_std_by_time  = []   # std  P(test class) on cal set per step
    weight_ess_frac_by_time = []  # effective sample size / n_cal per step

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

            alpha_used  = alpha_t.copy()
            alpha_next  = alpha_t.copy()
            covered_t   = []
            width_t     = []
            _prob1_acc  = []   # collect prob1 arrays across both halves for diagnostics

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

                # ── collect classifier prob1 for this half ────────────────────
                if predictor._last_cal_prob1 is not None:
                    _prob1_acc.append(predictor._last_cal_prob1.copy())

                if pred_idx is half1 and (t == 1 or t % 10 == 0 or t == T - 1):
                    _print_feature_diagnostic(predictor, cal_feat, t, feat_names)
                    _print_classifier_diagnostic(predictor, cal_feat, t)

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

            # ── summarise classifier diagnostics for this step ────────────────
            if _prob1_acc:
                p1_all = np.concatenate(_prob1_acc)
                w_norm = per_series_w / (per_series_w.sum() + 1e-30)
                ess    = float(1.0 / ((w_norm ** 2).sum() + 1e-30)) / n_cal
                clf_prob1_mean_by_time.append(float(p1_all.mean()))
                clf_prob1_std_by_time.append(float(p1_all.std()))
                weight_ess_frac_by_time.append(float(ess))
            else:
                clf_prob1_mean_by_time.append(float("nan"))
                clf_prob1_std_by_time.append(float("nan"))
                weight_ess_frac_by_time.append(float("nan"))

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

            # no shift — no classifier diagnostics
            clf_prob1_mean_by_time.append(float("nan"))
            clf_prob1_std_by_time.append(float("nan"))
            weight_ess_frac_by_time.append(float("nan"))

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
        "coverage_by_time":       coverage_by_time,
        "width_by_time":          width_by_time,
        "overall_coverage":       overall_coverage,
        "target_coverage":        target,
        "dates":                  [str(dates[t+1]) for t in range(T)],
        "gamma_opt_history":      gamma_opt_history,
        "first_test_ticker":      test_tickers[0],
        "first_test_series": {
            "true":  first_true,
            "lower": first_lower,
            "upper": first_upper,
        },
        # shift-correction diagnostics (NaN entries when with_shift=False or t==0)
        "clf_prob1_mean_by_time":  clf_prob1_mean_by_time,
        "clf_prob1_std_by_time":   clf_prob1_std_by_time,
        "weight_ess_frac_by_time": weight_ess_frac_by_time,
        "config": {
            "test_sector": display_sector,
            "n_train":     int(n_train),
            "n_cal":       int(n_cal),
            "n_test":      int(n_test),
            "L":           int(L),
            "alpha":       alpha,
            "gamma_grid":  [float(g) for g in gamma_grid],
            "seed":        seed,
            "with_shift":  with_shift,
            "mixed":       mixed,
            "Y_window":    Y_WINDOW,
            "x_window":    x_window,
            "cov_names":   cov_names,
            "feat_names":  feat_names,
        },
    }


# =============================================================================
# Plotting helpers
# =============================================================================
def _set_date_ticks(axes_flat, dates, x):
    tick_every = max(1, len(dates) // 10)
    tick_pos   = x[::tick_every]
    tick_lbl   = [dates[i] for i in tick_pos]
    for ax in axes_flat:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("Date", fontsize=10)


def _save_fig(fig, save_path):
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  [Plot] Saved to {out}")


# =============================================================================
# Single-run plot
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

    x          = np.arange(len(dates))
    shift_tag  = "with shift correction" if cfg["with_shift"] else "no shift correction"
    mixed_tag  = "  |  mixed test set" if cfg.get("mixed") else ""
    fig, axes  = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"AdaptedCAFHT  |  Test: {cfg['test_sector']}  |  {shift_tag}{mixed_tag}\n"
        f"alpha={cfg['alpha']}  |  "
        f"train/cal/test = {cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} tickers",
        fontsize=11, fontweight="bold",
    )

    # ── Coverage over time ────────────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, cov_t, color=_C_COV, linewidth=1.8, label="Empirical coverage")
    ax.axhline(target, color=_C_TARGET, linestyle="--", linewidth=1.8,
               label=f"Target ({target:.0%})")
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Coverage rate", fontsize=10)
    ax.set_title(f"Coverage over Time  ({cfg['test_sector']})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.text(0.02, 0.05,
            f"Overall: {results['overall_coverage']:.3f}  "
            f"(error {results['overall_coverage'] - target:+.3f})",
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
    _style_ax(ax)

    # ── Width over time ───────────────────────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, width_t, color=_C_WIDTH, linewidth=1.8)
    ax.set_ylabel("Mean interval width", fontsize=10)
    ax.set_title("Prediction Interval Width over Time", fontsize=10)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── First ticker prediction interval ─────────────────────────────────────
    ax        = axes[1, 0]
    true_arr  = np.array(first["true"])
    lower_arr = np.array(first["lower"])
    upper_arr = np.array(first["upper"])
    y_min     = float(np.percentile(true_arr, 1))
    y_max     = float(np.percentile(true_arr, 99))
    margin    = (y_max - y_min) * 0.3
    ax.fill_between(x, lower_arr, upper_arr, alpha=0.25, color=_C_COV,
                    label="Prediction interval")
    ax.plot(x, true_arr, color="black", linewidth=1.5, label="Actual return")
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_ylabel("Intraday return", fontsize=10)
    ax.set_title(f"{ticker}  —  Return vs. Prediction Interval", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── ACI gamma over time ───────────────────────────────────────────────────
    ax = axes[1, 1]
    ax.plot(x, gammas, drawstyle="steps-post", color=_C_GAMMA, linewidth=1.8)
    ax.set_ylabel("Selected gamma", fontsize=10)
    ax.set_title("ACI Gamma Selected over Time", fontsize=10)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    _set_date_ticks([ax for row in axes for ax in row], dates, x)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)


# =============================================================================
# Comparison plot  (with_shift vs no_shift overlaid)
# =============================================================================
def plot_comparison(r_shift, r_noshift, save_path=None):
    """
    Overlay with-shift and no-shift results for direct comparison.

    Layout (3 × 2):
      [0,0] Coverage over time            [0,1] Interval width over time
      [1,0] Pointwise coverage difference [1,1] Cumulative mean coverage
      [2,0] Classifier mean P(test)       [2,1] Weight ESS fraction
    """
    dates  = r_shift["dates"]
    T      = min(len(dates), len(r_noshift["dates"]))
    dates  = dates[:T]
    x      = np.arange(T)

    cov_s  = np.array(r_shift["coverage_by_time"][:T])
    cov_n  = np.array(r_noshift["coverage_by_time"][:T])
    wid_s  = np.array(r_shift["width_by_time"][:T])
    wid_n  = np.array(r_noshift["width_by_time"][:T])
    target = r_shift["target_coverage"]
    cfg    = r_shift["config"]

    # classifier diagnostics — present only in shift results
    p1_mean = np.array(r_shift.get("clf_prob1_mean_by_time", [float("nan")] * T)[:T],
                       dtype=float)
    p1_std  = np.array(r_shift.get("clf_prob1_std_by_time",  [float("nan")] * T)[:T],
                       dtype=float)
    ess     = np.array(r_shift.get("weight_ess_frac_by_time", [float("nan")] * T)[:T],
                       dtype=float)

    C_SHIFT   = "#2166ac"   # blue        — with shift
    C_NOSHIFT = "#d6604d"   # red-orange  — no shift
    C_TARGET  = "#888888"   # grey        — target

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    x_window_str = (f"{cfg.get('x_window', X_WINDOW)}-day"
                    if cfg.get('x_window', X_WINDOW) > 0 else "full-prefix")
    fig.suptitle(
        f"AdaptedCAFHT  |  Test: {cfg['test_sector']}  |  Shift correction comparison\n"
        f"alpha={cfg['alpha']}  |  "
        f"train/cal/test = {cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']} tickers  |  "
        f"X window: {x_window_str}",
        fontsize=11, fontweight="bold",
    )

    # ── [0,0] Coverage over time ──────────────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(x, cov_s, color=C_SHIFT,   linewidth=1.8,
            label=f"With shift  (overall={r_shift['overall_coverage']:.3f})")
    ax.plot(x, cov_n, color=C_NOSHIFT, linewidth=1.8, linestyle="--",
            label=f"No shift    (overall={r_noshift['overall_coverage']:.3f})")
    ax.axhline(target, color=C_TARGET, linestyle=":", linewidth=1.5,
               label=f"Target ({target:.0%})")
    ax.set_ylim(max(0.4, min(cov_s.min(), cov_n.min()) - 0.05), 1.05)
    ax.set_ylabel("Coverage rate", fontsize=10)
    ax.set_title("Coverage over Time", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── [0,1] Interval width over time ───────────────────────────────────────
    ax = axes[0, 1]
    ax.plot(x, wid_s, color=C_SHIFT,   linewidth=1.8, label="With shift")
    ax.plot(x, wid_n, color=C_NOSHIFT, linewidth=1.8, linestyle="--", label="No shift")
    ax.set_ylabel("Mean interval width", fontsize=10)
    ax.set_title("Prediction Interval Width over Time", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── [1,0] Pointwise coverage difference ──────────────────────────────────
    ax   = axes[1, 0]
    diff = cov_s - cov_n
    pos  = np.where(diff >= 0, diff, 0.0)
    neg  = np.where(diff <  0, diff, 0.0)
    ax.bar(x, pos, color=C_SHIFT,   alpha=0.75, label="Shift better")
    ax.bar(x, neg, color=C_NOSHIFT, alpha=0.75, label="No-shift better")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Coverage diff  (shift − no_shift)", fontsize=10)
    ax.set_title("Pointwise Coverage Difference", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    _style_ax(ax)

    # ── [1,1] Cumulative mean coverage ───────────────────────────────────────
    ax    = axes[1, 1]
    cum_s = np.cumsum(cov_s) / (x + 1)
    cum_n = np.cumsum(cov_n) / (x + 1)
    ax.plot(x, cum_s, color=C_SHIFT,   linewidth=1.8, label="With shift")
    ax.plot(x, cum_n, color=C_NOSHIFT, linewidth=1.8, linestyle="--", label="No shift")
    ax.axhline(target, color=C_TARGET, linestyle=":", linewidth=1.5,
               label=f"Target ({target:.0%})")
    ax.set_ylabel("Cumulative mean coverage", fontsize=10)
    ax.set_title("Cumulative Coverage (Running Average)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── [2,0] Classifier mean P(test class) on calibration set ───────────────
    ax = axes[2, 0]
    valid = np.isfinite(p1_mean)
    if valid.any():
        ax.plot(x[valid], p1_mean[valid], color=C_SHIFT, linewidth=1.8,
                label="Mean P(test class)")
        ax.fill_between(x[valid],
                        (p1_mean - p1_std)[valid],
                        (p1_mean + p1_std)[valid],
                        alpha=0.2, color=C_SHIFT, label="±1 std")
        ax.axhline(0.5, color=C_TARGET, linestyle=":", linewidth=1.5,
                   label="0.5  (uniform — no shift detected)")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("P(test class)", fontsize=10)
    ax.set_title("Classifier: P(test class) on Calibration Set\n"
                 "→ near 0.5 = uniform weights (no shift detected)", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    # ── [2,1] Weight effective sample size (ESS fraction) ────────────────────
    ax = axes[2, 1]
    valid = np.isfinite(ess)
    if valid.any():
        ax.plot(x[valid], ess[valid], color=C_SHIFT, linewidth=1.8)
        ax.axhline(1.0, color=C_TARGET, linestyle=":", linewidth=1.5,
                   label="1.0  (uniform weights)")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("ESS / n_cal", fontsize=10)
    ax.set_title("Weight Effective Sample Size Fraction\n"
                 "→ near 1.0 = weights collapsed to uniform", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    _style_ax(ax)

    _set_date_ticks([ax for row in axes for ax in row], dates, x)
    fig.tight_layout()
    if save_path:
        _save_fig(fig, save_path)


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run AdaptedCAFHT conformal prediction on S&P 500 finance data."
    )
    parser.add_argument("--npz",             required=True)
    parser.add_argument("--json",            default=None)
    parser.add_argument("--test_sector",     default="Technology",
                        help="Sector held out as test set (ignored when --mixed is set). "
                             "Default: Technology")
    parser.add_argument("--cal_frac",        type=float, default=0.5)
    parser.add_argument("--alpha",           type=float, default=0.1)
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--gamma_grid",      type=float, nargs="+", default=GAMMA_GRID)
    parser.add_argument("--with_shift",      action="store_true", default=False)
    parser.add_argument("--mixed",           action="store_true", default=False,
                        help="Mixed-sector test set: randomly draw tickers from ALL "
                             "sectors as test (null/no-shift baseline). "
                             "Overrides --test_sector.")
    parser.add_argument("--mixed_test_frac", type=float, default=0.15,
                        help="Fraction of all tickers used as test in --mixed mode. "
                             "Default: 0.15  (~15%% of tickers)")
    parser.add_argument("--x_window",       type=int,   default=X_WINDOW,
                        help=f"Rolling window (days) for X covariate features used in "
                             f"the shift classifier. Default: {X_WINDOW}. "
                             f"Set 0 to revert to full-prefix mean (original behaviour).")
    parser.add_argument("--save_plot",       default=None)
    parser.add_argument("--save_json",       default=None)
    # ── Comparison mode: overlay a second (no-shift) JSON result ─────────────
    parser.add_argument("--compare_json",    default=None,
                        help="Path to a second results JSON to overlay in a comparison "
                             "plot. The current run is treated as 'with shift'; the "
                             "loaded JSON as 'no shift'. Requires --save_compare.")
    parser.add_argument("--save_compare",    default=None,
                        help="Output path for the comparison figure (PNG or PDF). "
                             "Used together with --compare_json.")
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
        mixed=args.mixed,
        mixed_test_frac=args.mixed_test_frac,
        x_window=args.x_window,
    )

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Results] Saved to {out}")

    plot_results(results, save_path=args.save_plot)

    # ── Optional comparison plot ──────────────────────────────────────────────
    if args.compare_json:
        cmp_path = Path(args.compare_json)
        if not cmp_path.exists():
            raise FileNotFoundError(
                f"--compare_json: file not found: {cmp_path}\n"
                f"  Run the no-shift experiment first and save with --save_json, e.g.:\n"
                f"    python finance_conformal.py --npz {args.npz} "
                f"--save_json results/noshift.json"
            )
        with open(cmp_path) as f:
            r_other = json.load(f)
        # Determine which is shift / no_shift by the config flag
        if results["config"]["with_shift"]:
            r_shift, r_noshift = results, r_other
        else:
            r_shift, r_noshift = r_other, results
        plot_comparison(r_shift, r_noshift, save_path=args.save_compare)


if __name__ == "__main__":
    main()