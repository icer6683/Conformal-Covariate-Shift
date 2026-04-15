#!/usr/bin/env python3
"""
=============================================================================
MEDICAL CONFORMAL PREDICTION  —  AdaptedCAFHT on sepsis ICU data
=============================================================================

QUICK START
-----------
Step 1 — make sure ``sepsis_experiment_data.pkl`` is in the working directory
  (the pickle is produced by the upstream MIMIC-III extraction pipeline and
  contains pre-filtered patients; see medical_data.py for format details).

Step 2 — run experiment:

  # Default run (no shift correction):
  python medical_conformal.py --pkl sepsis_experiment_data.pkl

  # WITH likelihood-ratio shift correction:
  python medical_conformal.py --pkl sepsis_experiment_data.pkl --with_shift

  # Custom alpha, cal fraction, and seed:
  python medical_conformal.py --pkl sepsis_experiment_data.pkl --with_shift \
      --alpha 0.1 --cal_frac 0.5 --seed 42

FULL OPTIONS
------------
  --pkl          Path to sepsis_experiment_data.pkl (required)
  --with_shift   Toggle ON likelihood-ratio covariate-shift weighting.
                 Uses cross-split update_weighting_context() to reweight
                 calibration scores for the Norepinephrine distribution shift.
                 Default: off (uniform weights, plain conformal).
  --alpha        Miscoverage level. Default: 0.1  (targets 90% coverage)
  --cal_frac     Fraction of TrainCal patients used for calibration.
                 Default: 0.5  (remaining used for training)
  --gamma_grid   Space-separated ACI step-size candidates.
                 Default: [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
  --seed         Random seed. Default: 42
  --save_plot    Path to save the output figure, e.g. results.png
  --save_json    Path to save results as JSON, e.g. results.json

HOW IT WORKS
------------
  The data has already been split upstream:
    - TrainCal : sepsis patients who did NOT receive Norepinephrine
    - Test     : sepsis patients who DID receive Norepinephrine
  This script further splits TrainCal into Train and Cal (randomly, via
  --cal_frac).

  Target variable (Y):
    Heart Rate at each hour (24 hourly values per patient).

  Covariates (X):
    Six other clinical trajectories, EXCLUDING Norepinephrine (which is
    identically zero for all TrainCal patients, so its coefficient is
    unidentifiable from training data):
      - Respiratory Rate
      - O2 saturation pulseoxymetry
      - Non Invasive Blood Pressure systolic
      - Non Invasive Blood Pressure diastolic
      - Non Invasive Blood Pressure mean
      - NaCl 0.9%

  Loop structure is identical to finance_conformal.py and test_conformal.py:
      for t in range(T):                          # T = 23 (hours 0..22)
          fit   linear model on train[:, :t+2, :]
          build calibration scores on cal[:, :t+2, :]
          (gamma selection every 10 steps)
          if with_shift and t >= 1:
              cross-split test into two halves
              for each (predict_half, context_half):
                  train LR classifier: train (label=0) vs context_half (label=1)
                  reweight cal scores with density-ratio weights
                  predict step t+1 for predict_half
          else:
              predict step t+1 for all test patients (uniform weights)
          ACI alpha update: alpha_{t+1} = alpha_t + gamma*(alpha - err_t)

  Prediction model:
    Cross-sectional linear regression at each hour:
      HR_t ≈ β₀ + β₁·RespRate_t + β₂·O2Sat_t + β₃·BPsys_t
                 + β₄·BPdia_t + β₅·BPmean_t + β₆·NaCl_t
    fitted by OLS across all (patient, timestep) pairs in the training prefix.

  Conformity score:
    |HR_true - HR_pred|   (absolute residual)

  Likelihood-ratio weighting (--with_shift):
    A logistic classifier distinguishes Train from Test featurized prefixes.
    The monkey-patched featurizer computes summary statistics over BOTH the
    Heart Rate (Y) prefix AND the 6 covariate (X) prefixes:
      Per variable: [mean, std, min, max, last]
      Total: 7 variables x 5 stats = 35 features per patient
    The classifier's predicted probabilities give density-ratio weights
    for each calibration series.

    The covariate shift in this dataset is driven by Norepinephrine usage,
    which correlates strongly with lower blood pressure and different fluid
    management (NaCl).  Using only Heart Rate for the classifier yields
    ~55% accuracy (near-random); adding covariates raises it to ~76%.

FEATURE OVERRIDE
----------------
  algorithm.py's default _featurize_prefixes uses only the last value as
  the single classifier feature.  This file monkey-patches a richer
  featurizer that uses both Y and X data.

  Because algorithm.py's _featurize_prefixes signature only accepts Y
  prefixes (n, t+1, 1), we store auxiliary X arrays on the predictor
  instance (predictor._X_ctx) and look them up inside the featurizer.
  The caller sets predictor._X_ctx to the matching X prefix array before
  each call to _featurize_prefixes or update_weighting_context.

    features per patient = 5 stats x 7 variables = 35 features
    stats   : [mean, std, min, max, last]
    variables: Heart Rate + 6 covariates
=============================================================================
"""
import argparse
import json
import types
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from medical_data import load_data
from algorithm import AdaptedCAFHT


# ── Constants ────────────────────────────────────────────────────────────────

TARGET_VAR = "Heart Rate"

COVARIATE_VARS = [
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
    "Non Invasive Blood Pressure systolic",
    "Non Invasive Blood Pressure diastolic",
    "Non Invasive Blood Pressure mean",
    "NaCl 0.9%",
]

GAMMA_GRID = [0.000001, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]


# =============================================================================
# Data conversion: nested dict-of-DataFrames → arrays
# =============================================================================

def _convert_to_arrays(patient_trajectory_list):
    """
    Convert the list-of-dicts representation into (Y, X) arrays.

    Parameters
    ----------
    patient_trajectory_list : list[dict[str, DataFrame]]
        Each dict maps trajectory names to DataFrames with columns
        (hour, value).

    Returns
    -------
    Y : ndarray of shape (n_patients, 24, 1)
        Heart Rate trajectories.
    X : ndarray of shape (n_patients, 24, n_cov)
        Covariate trajectories (6 variables), in COVARIATE_VARS order.
    """
    n = len(patient_trajectory_list)
    L = 24
    n_cov = len(COVARIATE_VARS)

    Y = np.zeros((n, L, 1), dtype=np.float64)
    X = np.zeros((n, L, n_cov), dtype=np.float64)

    for i, patient_dict in enumerate(patient_trajectory_list):
        # Heart Rate → Y
        hr_df = patient_dict[TARGET_VAR]
        Y[i, :, 0] = hr_df["value"].to_numpy(dtype=np.float64)

        # Covariates → X
        for j, var_name in enumerate(COVARIATE_VARS):
            cov_df = patient_dict[var_name]
            X[i, :, j] = cov_df["value"].to_numpy(dtype=np.float64)

    return Y, X


# =============================================================================
# Richer featurizer — monkey-patched onto the predictor instance.
# =============================================================================

def _summarize_series(series_2d):
    """
    Compute 5 summary statistics per variable.

    Parameters
    ----------
    series_2d : (n, T) array — one variable's prefix across n patients.

    Returns
    -------
    (n, 5) array: [mean, std, min, max, last]
    """
    return np.column_stack([
        series_2d.mean(axis=1),
        series_2d.std(axis=1) + 1e-8,
        series_2d.min(axis=1),
        series_2d.max(axis=1),
        series_2d[:, -1],
    ])


def _richer_featurize_prefixes(self, prefixes):
    """
    Build a rich feature vector from both Y (Heart Rate) and X (covariates).

    prefixes : (n, t+1, 1) — Heart Rate prefix (passed by algorithm.py)

    The covariate prefix is read from self._X_ctx, which the caller must
    set to (n, t+1, n_cov) before invoking this method.  If _X_ctx is
    None or has mismatched length, we fall back to Y-only features.

    Returns
    -------
    (n, 35) feature matrix when X is available:
      5 stats (mean, std, min, max, last) x 7 variables
      = 5 for Heart Rate + 5 x 6 covariates = 35 features

    (n, 5) feature matrix when X is unavailable (fallback).
    """
    Y = prefixes[..., 0]          # (n, t+1)
    n, T = Y.shape

    # Heart Rate summary (always available)
    feat_y = _summarize_series(Y)  # (n, 5)

    # Covariate summaries (from auxiliary storage)
    X_ctx = getattr(self, '_X_ctx', None)
    if X_ctx is not None and X_ctx.shape[0] == n and X_ctx.shape[1] == T:
        n_cov = X_ctx.shape[2]
        cov_feats = [_summarize_series(X_ctx[:, :, j]) for j in range(n_cov)]
        raw = np.column_stack([feat_y] + cov_feats)  # (n, 5 + 5*n_cov)
    else:
        raw = feat_y  # (n, 5) — fallback

    # Standardize using training-set statistics so that all groups (train,
    # test, cal) are on the same scale.  self._feat_mu / self._feat_std are
    # set once from the training features and reused for test and cal.
    mu  = getattr(self, '_feat_mu', None)
    std = getattr(self, '_feat_std', None)
    if mu is not None and std is not None and mu.shape[-1] == raw.shape[1]:
        return (raw - mu) / std
    else:
        # No reference stats yet — return raw (will be overwritten once
        # training features are computed and stats are stored).
        return raw


# =============================================================================
# Diagnostic helpers (identical pattern to finance_conformal.py)
# =============================================================================

def _print_weight_diagnostic(weights, t, label=""):
    """Print a concise summary of LR calibration weights at time step t."""
    w = np.asarray(weights, dtype=float)
    is_uniform = bool(np.allclose(w, w[0], rtol=1e-4, atol=1e-6))

    lo, hi = w.min(), w.max()
    if hi > lo:
        edges = np.linspace(lo, hi, 6)
        counts, _ = np.histogram(w, bins=edges)
        hist_str = "  ".join(
            f"[{edges[i]:.4f},{edges[i+1]:.4f}): {counts[i]}" for i in range(5)
        )
    else:
        hist_str = f"all values = {lo:.6f}"

    tag = f" {label}" if label else ""
    print(f"  [WeightDiag t={t:3d}{tag}]"
          f"  n={len(w)}  min={lo:.5f}  max={hi:.5f}"
          f"  mean={w.mean():.5f}  std={w.std():.5f}"
          f"  uniform={is_uniform}")
    print(f"    histogram: {hist_str}")


def _build_feat_names():
    """Build human-readable names for the 35-feature vector."""
    stats = ["mean", "std", "min", "max", "last"]
    var_names = [TARGET_VAR] + COVARIATE_VARS
    return [f"{v}|{s}" for v in var_names for s in stats]


def _print_feature_diagnostic(predictor, cal_feat, t):
    """Compare per-feature mean/std between train, test, and cal prefixes."""
    train_feat = predictor._train_feat_t
    test_feat  = predictor._test_feat_t
    if train_feat is None or test_feat is None:
        print(f"  [FeatDiag t={t:3d}] train/test features not set")
        return
    n_feat = train_feat.shape[1]
    feat_names = _build_feat_names()
    print(f"  [FeatDiag t={t:3d}]  train={train_feat.shape}  "
          f"test={test_feat.shape}  cal={cal_feat.shape}")
    for col in range(n_feat):
        tr = train_feat[:, col]
        te = test_feat[:, col]
        ca = cal_feat[:, col]
        sep = abs(tr.mean() - te.mean()) / (tr.std() + te.std() + 1e-8)
        name = feat_names[col] if col < len(feat_names) else str(col)
        print(f"    {name:48s}  "
              f"train: mean={tr.mean():+9.3f} std={tr.std():8.3f}  "
              f"test:  mean={te.mean():+9.3f} std={te.std():8.3f}  "
              f"cal:   mean={ca.mean():+9.3f} std={ca.std():8.3f}  "
              f"|sep|={sep:.3f}")


def _print_classifier_diagnostic(predictor, cal_feat, t):
    """Report the fitted logistic classifier's training accuracy and coef norm."""
    clf = predictor._clf
    if clf is None:
        print(f"  [CLFDiag t={t:3d}] classifier is None (sklearn unavailable or not fitted)")
        return
    train_feat = predictor._train_feat_t
    test_feat  = predictor._test_feat_t
    if train_feat is None or test_feat is None:
        return
    N0, N1 = train_feat.shape[0], test_feat.shape[0]
    X_all = np.vstack([train_feat, test_feat])
    y_all = np.concatenate([np.zeros(N0, dtype=int), np.ones(N1, dtype=int)])
    acc       = clf.score(X_all, y_all)
    coef_norm = float(np.linalg.norm(clf.coef_))
    prob_cal  = clf.predict_proba(cal_feat)[:, 1]
    print(f"  [CLFDiag t={t:3d}]  train_acc={acc:.3f}  coef_norm={coef_norm:.4f}  "
          f"prob1_cal: min={prob_cal.min():.3f}  max={prob_cal.max():.3f}  "
          f"mean={prob_cal.mean():.3f}  std={prob_cal.std():.4f}")


# =============================================================================
# Prediction model: cross-sectional linear regression on clinical covariates
# =============================================================================

class LinearCovariateModel:
    """
    Cross-sectional OLS model:
      HR_t ≈ β₀ + β₁·RespRate_t + β₂·O2Sat_t + β₃·BPsys_t
                 + β₄·BPdia_t + β₅·BPmean_t + β₆·NaCl_t

    Fitted across all (patient, timestep) pairs in the training prefix.
    """

    def __init__(self, cov_names):
        self.cov_names = cov_names
        self.beta = None
        self.noise_std = 1.0

    def fit(self, Y_train, X_train, verbose=False):
        """
        Parameters
        ----------
        Y_train : (n, L_prefix, 1)   Heart Rate prefix
        X_train : (n, L_prefix, n_cov)   covariate prefix
        """
        n, L, n_cov = X_train.shape
        if L < 2:
            return
        y = Y_train[:, :, 0].reshape(-1)
        X = X_train.reshape(-1, n_cov)
        X_design = np.hstack([np.ones((len(y), 1)), X])
        self.beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ self.beta
        self.noise_std = float(np.std(resid, ddof=X_design.shape[1]))
        if verbose:
            print(f"  [Model] Fitted on {n} patients x {L} steps")
            print(f"  [Model] Residual std : {self.noise_std:.4f}")
            print(f"  [Model] Coefficients :")
            print(f"            intercept = {self.beta[0]:.4f}")
            for name, coef in zip(self.cov_names, self.beta[1:]):
                print(f"            {name:42s} = {coef:.4f}")

    def predict(self, x_t):
        """Predict Heart Rate given covariate vector x_t (shape (n_cov,))."""
        if self.beta is None:
            return 0.0
        return float(self.beta[0] + x_t @ self.beta[1:])


# =============================================================================
# Gamma selection (mirrors _select_gamma in finance_conformal.py)
# =============================================================================

def _select_gamma(Y_train, X_train, cov_names, base_alpha, t_max, gamma_grid,
                  seed=0):
    """
    Select gamma by running simple ACI on a 3-way split of training data up
    to t_max.  Uses the linear covariate model (no LR weighting).
    """
    n_train = Y_train.shape[0]
    if n_train < 9 or t_max < 2:
        return float(gamma_grid[0]), {float(g): float('nan') for g in gamma_grid}

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    n1 = n_train // 3
    n2 = n_train // 3
    n3 = n_train - n1 - n2
    if n1 == 0 or n2 == 0 or n3 == 0:
        return float(gamma_grid[0]), {float(g): float('nan') for g in gamma_grid}

    idx1 = perm[:n1]
    idx2 = perm[n1 : n1 + n2]
    idx3 = perm[n1 + n2 :]
    Y_fit_sel = Y_train[idx1];  X_fit_sel = X_train[idx1]
    Y_cal_sel = Y_train[idx2];  X_cal_sel = X_train[idx2]
    Y_eval    = Y_train[idx3];  X_eval    = X_train[idx3]
    n_eval    = Y_eval.shape[0]
    L         = Y_train.shape[1]
    horizon   = min(t_max, L - 1)
    start_eval = max(0, horizon // 2)
    target    = 1.0 - base_alpha

    scores = {}
    for gamma in gamma_grid:
        gamma = float(gamma)
        predictor = AdaptedCAFHT(alpha=base_alpha)
        alpha_series = np.full(n_eval, base_alpha, dtype=float)
        cov_hist = []
        for t in range(horizon + 1):
            sel_model = LinearCovariateModel(cov_names)
            sel_model.fit(Y_fit_sel[:, :t+1, :], X_fit_sel[:, :t+1, :])
            predictor.noise_std = sel_model.noise_std
            cal_scores = []
            for i in range(len(idx2)):
                for s in range(t + 1):
                    y_true = float(Y_cal_sel[i, s, 0])
                    y_pred = sel_model.predict(X_cal_sel[i, s, :])
                    cal_scores.append(abs(y_true - y_pred))
            predictor._scores  = np.array(cal_scores, dtype=float)
            predictor._weights = np.ones(len(cal_scores), dtype=float)
            predictor._q       = None
            alpha_used = alpha_series.copy()
            alpha_next = alpha_series.copy()
            step_cov = []
            for i in range(n_eval):
                x_t    = X_eval[i, t, :]
                y_true = float(Y_eval[i, t, 0])
                y_pred = sel_model.predict(x_t)
                a = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
                q = predictor._weighted_quantile(
                    predictor._scores, predictor._weights, 1.0 - a)
                covered = int(y_pred - q <= y_true <= y_pred + q)
                step_cov.append(covered)
                err = 0 if covered else 1
                alpha_next[i] = alpha_used[i] + gamma * (base_alpha - err)
            alpha_series = np.clip(alpha_next, 1e-6, 1.0 - 1e-6)
            cov_hist.append(float(np.mean(step_cov)) if step_cov else float('nan'))
        tail   = cov_hist[start_eval:]
        metric = float(np.mean(tail)) if len(tail) > 0 else float('nan')
        scores[gamma] = metric

    best_gamma = float(gamma_grid[0])
    best_obj   = float('inf')
    for gamma, metric in scores.items():
        if not np.isfinite(metric):
            continue
        obj = abs(metric - target)
        if obj < best_obj:
            best_obj   = obj
            best_gamma = float(gamma)
    return best_gamma, scores


# =============================================================================
# Main experiment
# =============================================================================

def run_medical_experiment(data, cal_frac=0.5, alpha=0.1, seed=42,
                           gamma_grid=None, with_shift=False):
    """
    Run AdaptedCAFHT conformal prediction on sepsis ICU data.

    Parameters
    ----------
    data : dict
        Output of medical_data.load_data().
    cal_frac : float
        Fraction of TrainCal patients used for calibration.
    alpha : float
        Miscoverage level.
    seed : int
        Random seed for Train/Cal split and gamma selection.
    gamma_grid : list[float]
        ACI step-size candidates.
    with_shift : bool
        Whether to use likelihood-ratio covariate-shift weighting.

    Returns
    -------
    dict with coverage_by_time, width_by_time, overall_coverage, etc.
    """
    if gamma_grid is None:
        gamma_grid = GAMMA_GRID

    # ── Convert nested dicts to arrays ────────────────────────────────────
    Y_traincal, X_traincal = _convert_to_arrays(
        data["patient_trajectory_list_traincal"])
    Y_test_all, X_test_all = _convert_to_arrays(
        data["patient_trajectory_list_test"])

    n_traincal, L, _ = Y_traincal.shape
    n_test = Y_test_all.shape[0]
    T = L - 1  # 23 prediction steps (hours 0..22, predicting 1..23)

    # ── Split TrainCal → Train + Cal ──────────────────────────────────────
    rng       = np.random.default_rng(seed)
    perm      = rng.permutation(n_traincal)
    n_cal     = int(n_traincal * cal_frac)
    n_train   = n_traincal - n_cal
    if n_train == 0:
        raise ValueError(f"cal_frac={cal_frac} leaves no training patients.")

    train_idx = perm[:n_train]
    cal_idx   = perm[n_train:]

    Y_train, X_train = Y_traincal[train_idx], X_traincal[train_idx]
    Y_cal,   X_cal   = Y_traincal[cal_idx],   X_traincal[cal_idx]
    Y_test,  X_test  = Y_test_all, X_test_all

    cov_names = COVARIATE_VARS

    print(f"\n{'='*62}")
    print(f"  Medical Conformal Experiment  (AdaptedCAFHT)")
    print(f"{'='*62}")
    print(f"  TrainCal total  : {n_traincal} patients (no Norepinephrine)")
    print(f"    Train         : {n_train}")
    print(f"    Cal           : {n_cal}")
    print(f"  Test            : {n_test} patients (received Norepinephrine)")
    print(f"  Time steps      : {L} hours  [0 .. {T}]")
    print(f"  Target (Y)      : {TARGET_VAR}")
    print(f"  Covariates (X)  : {cov_names}")
    print(f"  Alpha           : {alpha}  (target = {1-alpha:.0%})")
    print(f"  Gamma grid      : {gamma_grid}")
    print(f"  With shift      : {with_shift}")
    print()

    # ── Set up predictor ──────────────────────────────────────────────────
    predictor    = AdaptedCAFHT(alpha=alpha)
    linear_model = LinearCovariateModel(cov_names)

    # Monkey-patch richer featurizer (7 summary stats over Heart Rate prefix)
    predictor._featurize_prefixes = types.MethodType(
        _richer_featurize_prefixes, predictor
    )
    print(f"  Featurizer      : richer (mean/std/min/max/last over "
          f"{TARGET_VAR} + 6 covariates = 35 features)")
    print()

    # ── ACI state ─────────────────────────────────────────────────────────
    alpha_t   = np.full(n_test, alpha, dtype=float)
    gamma_opt = float(gamma_grid[0])

    coverage_by_time  = []
    width_by_time     = []
    all_covered       = []
    gamma_opt_history = []
    first_true  = []
    first_lower = []
    first_upper = []

    # ── Main loop: for each hour t, predict hour t+1 ─────────────────────
    for t in range(T):

        # ── Fit linear model on training prefix ──────────────────────────
        linear_model.fit(Y_train[:, :t+2, :], X_train[:, :t+2, :],
                         verbose=(t == T - 1))
        predictor.noise_std = linear_model.noise_std

        # ── Build calibration scores ─────────────────────────────────────
        cal_scores = []
        for i in range(n_cal):
            for s in range(t + 2):
                y_true = float(Y_cal[i, s, 0])
                y_pred = linear_model.predict(X_cal[i, s, :])
                cal_scores.append(abs(y_true - y_pred))
        cal_scores_arr = np.array(cal_scores, dtype=float)

        # Default: uniform weights
        predictor._scores  = cal_scores_arr
        predictor._weights = np.ones(len(cal_scores_arr), dtype=float)
        predictor._q       = None

        # ── Gamma selection every 5 steps ───────────────────────────────
        if t > 0 and (t % 5 == 0):
            sel_seed = seed + 10000 + t
            gamma_opt, gamma_scores = _select_gamma(
                Y_train=Y_train, X_train=X_train, cov_names=cov_names,
                base_alpha=alpha, t_max=t, gamma_grid=gamma_grid,
                seed=sel_seed,
            )
            scores_str = "  ".join(
                f"gamma={g:.3f}->{v:.3f}" for g, v in gamma_scores.items()
                if np.isfinite(v)
            )
            print(f"  [gamma sel t={t:3d}]  best gamma = {gamma_opt}"
                  f"   ({scores_str})")

        gamma_opt_history.append(float(gamma_opt))

        # ── Prediction loop ──────────────────────────────────────────────
        if with_shift and t >= 1:
            train_prefixes = Y_train[:, :t+1, :]
            mid   = n_test // 2
            half1 = np.arange(0, mid)
            half2 = np.arange(mid, n_test)

            alpha_used = alpha_t.copy()
            alpha_next = alpha_t.copy()
            covered_t  = []
            width_t    = []

            for pred_idx, ctx_idx in [(half1, half2), (half2, half1)]:

                # Featurize train prefixes (set X context for train).
                # Compute raw features first, then derive standardization
                # stats from training data and apply to all groups.
                predictor._X_ctx = X_train[:, :t+1, :]
                predictor._feat_mu = None   # reset so train gets raw
                predictor._feat_std = None
                train_raw = predictor._featurize_prefixes(train_prefixes)
                # Now set standardization stats from training features
                predictor._feat_mu  = train_raw.mean(axis=0, keepdims=True)
                predictor._feat_std = train_raw.std(axis=0, keepdims=True) + 1e-8
                # Re-featurize train with standardization applied
                predictor._X_ctx = X_train[:, :t+1, :]
                predictor._train_feat_t = predictor._featurize_prefixes(
                    train_prefixes)

                # Featurize test-half prefixes (uses train stats)
                predictor._X_ctx = X_test[ctx_idx, :t+1, :]
                predictor._test_feat_t = predictor._featurize_prefixes(
                    Y_test[ctx_idx, :t+1, :])

                # Mark context as shifted
                predictor._is_shifted_ctx = True
                predictor._t_ctx = t
                predictor._clf = None

                # Featurize cal prefixes (uses train stats)
                predictor._X_ctx = X_cal[:, :t+1, :]
                cal_feat = predictor._featurize_prefixes(Y_cal[:, :t+1, :])

                # Compute per-series LR weights for cal data
                per_series_w = predictor._compute_density_ratio_weights(
                    trainX=predictor._train_feat_t,
                    testX=predictor._test_feat_t,
                    evalX=cal_feat,
                )

                # Diagnostics (first swap only, at selected steps)
                if pred_idx is half1 and (t == 1 or t % 10 == 0 or t == T - 1):
                    _print_feature_diagnostic(predictor, cal_feat, t)
                if pred_idx is half1 and (t == 1 or t % 10 == 0 or t == T - 1):
                    _print_classifier_diagnostic(predictor, cal_feat, t)

                # Tile per-series weights to per-score weights
                n_steps = t + 2
                assert len(per_series_w) == n_cal, (
                    f"per_series_w length {len(per_series_w)} != n_cal {n_cal}"
                )
                tiled_w = np.repeat(per_series_w, n_steps)
                assert len(tiled_w) == len(cal_scores_arr), (
                    f"tiled_w length {len(tiled_w)} != cal_scores "
                    f"length {len(cal_scores_arr)}"
                )

                predictor._scores  = cal_scores_arr
                predictor._weights = tiled_w
                predictor._q       = None

                # Weight summary
                if t == 1 or t % 10 == 0 or t == T - 1:
                    label = "half1-ctx" if pred_idx is half1 else "half2-ctx"
                    _print_weight_diagnostic(per_series_w, t, label=label)

                for i in pred_idx:
                    x_t    = X_test[i, t+1, :]
                    y_true = float(Y_test[i, t+1, 0])
                    y_pred = linear_model.predict(x_t)
                    a = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
                    q = predictor._weighted_quantile(
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
            # Unweighted path: with_shift=False, or t==0
            alpha_used = alpha_t.copy()
            alpha_next = alpha_t.copy()
            covered_t  = []
            width_t    = []

            for i in range(n_test):
                x_t    = X_test[i, t+1, :]
                y_true = float(Y_test[i, t+1, 0])
                y_pred = linear_model.predict(x_t)
                a = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
                q = predictor._weighted_quantile(
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

        if (t + 1) % 5 == 0 or t == T - 1:
            print(f"  [t={t+1:3d}/{T}]  coverage={np.mean(covered_t):.3f}  "
                  f"width={np.mean(width_t):.4f}  gamma={gamma_opt}")

    overall_coverage = float(np.mean(all_covered))
    target = 1.0 - alpha
    print(f"\n  Overall coverage : {overall_coverage:.4f}  "
          f"(target = {target:.4f},  error = {overall_coverage - target:+.4f})")
    print(f"  Mean width       : {np.mean(width_by_time):.4f}")
    print(f"  Final gamma_opt  : {gamma_opt}")

    # Pick patient IDs for the first test patient (for plot title)
    test_ids = data["patient_ids_test"]
    first_test_id = test_ids[0] if test_ids else "unknown"

    return {
        "coverage_by_time":  coverage_by_time,
        "width_by_time":     width_by_time,
        "overall_coverage":  overall_coverage,
        "target_coverage":   target,
        "hours": list(range(1, L)),  # prediction hours 1..23
        "gamma_opt_history": gamma_opt_history,
        "first_test_patient": first_test_id,
        "first_test_series": {
            "true":  first_true,
            "lower": first_lower,
            "upper": first_upper,
        },
        "config": {
            "n_train":     int(n_train),
            "n_cal":       int(n_cal),
            "n_test":      int(n_test),
            "L":           int(L),
            "alpha":       alpha,
            "gamma_grid":  [float(g) for g in gamma_grid],
            "seed":        seed,
            "with_shift":  with_shift,
            "target_var":  TARGET_VAR,
            "cov_names":   cov_names,
        },
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results, save_path=None):
    """Plot coverage, width, first-patient bands, and gamma history."""
    hours   = results["hours"]
    cov_t   = results["coverage_by_time"]
    width_t = results["width_by_time"]
    target  = results["target_coverage"]
    cfg     = results["config"]
    first   = results["first_test_series"]
    patient = results["first_test_patient"]
    gammas  = results["gamma_opt_history"]

    x = np.arange(len(hours))
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    shift_str = "with LR shift" if cfg["with_shift"] else "no shift"
    fig.suptitle(
        f"Medical Conformal Prediction (AdaptedCAFHT)  |  "
        f"Sepsis ICU  |  alpha={cfg['alpha']}  |  "
        f"train/cal/test = {cfg['n_train']}/{cfg['n_cal']}/{cfg['n_test']}  |  "
        f"{shift_str}",
        fontsize=12, fontweight='bold'
    )

    # Plot 1: Coverage over time
    axes[0, 0].plot(x, cov_t, 'b-', linewidth=1.5, label='Empirical coverage')
    axes[0, 0].axhline(target, color='red', linestyle='--', linewidth=2,
                       label=f'Target ({target:.0%})')
    axes[0, 0].set_ylim(0.5, 1.05)
    axes[0, 0].set_ylabel('Coverage rate')
    axes[0, 0].set_title('Coverage over Time (all test patients)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(
        0.02, 0.05,
        f"Overall: {results['overall_coverage']:.3f}  "
        f"(error {results['overall_coverage'] - target:+.3f})",
        transform=axes[0, 0].transAxes, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.7)
    )

    # Plot 2: Interval width over time
    axes[0, 1].plot(x, width_t, 'g-', linewidth=1.5)
    axes[0, 1].set_ylabel('Mean interval width (bpm)')
    axes[0, 1].set_title('Prediction Interval Width over Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: First test patient — actual HR vs prediction interval
    true_arr  = np.array(first["true"])
    lower_arr = np.array(first["lower"])
    upper_arr = np.array(first["upper"])
    y_min  = float(np.percentile(true_arr, 1))
    y_max  = float(np.percentile(true_arr, 99))
    margin = (y_max - y_min) * 0.3
    axes[1, 0].fill_between(x, lower_arr, upper_arr, alpha=0.25,
                             color='steelblue', label='Prediction interval')
    axes[1, 0].plot(x, true_arr, 'k-', linewidth=1.5,
                     label='Actual Heart Rate')
    axes[1, 0].set_ylim(y_min - margin, y_max + margin)
    axes[1, 0].set_ylabel('Heart Rate (bpm)')
    axes[1, 0].set_title(f'Patient {patient} — Actual HR vs. Prediction Interval')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: ACI gamma selected over time
    axes[1, 1].plot(x, gammas, drawstyle='steps-post', color='purple',
                     linewidth=1.5)
    axes[1, 1].set_ylabel('Selected gamma (log scale)')
    axes[1, 1].set_title('ACI Gamma Selected over Time')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    # Shared x-axis labels (hours)
    tick_every = max(1, len(hours) // 12)
    tick_pos   = x[::tick_every]
    tick_lbl   = [f"h{hours[i]}" for i in tick_pos]
    for row in axes:
        for ax in row:
            ax.set_xticks(tick_pos)
            ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=8)
            ax.set_xlabel('Hour')

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
        description="Run AdaptedCAFHT conformal prediction on sepsis ICU data."
    )
    parser.add_argument("--pkl",         required=True,
                        help="Path to sepsis_experiment_data.pkl")
    parser.add_argument("--cal_frac",    type=float, default=0.5,
                        help="Fraction of TrainCal used for calibration "
                             "(default: 0.5)")
    parser.add_argument("--alpha",       type=float, default=0.1,
                        help="Miscoverage level (default: 0.1)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--gamma_grid",  type=float, nargs='+',
                        default=GAMMA_GRID,
                        help="ACI step-size candidates "
                             "(default: 0.001 0.005 0.01 0.05)")
    parser.add_argument("--with_shift",  action="store_true", default=False,
                        help="Enable likelihood-ratio covariate-shift weighting")
    parser.add_argument("--save_plot",   default=None,
                        help="Path to save the output figure")
    parser.add_argument("--save_json",   default=None,
                        help="Path to save results as JSON")
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    data = load_data(args.pkl)

    results = run_medical_experiment(
        data=data,
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
