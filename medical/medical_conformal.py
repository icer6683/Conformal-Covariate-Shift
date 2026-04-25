#!/usr/bin/env python3
"""
=============================================================================
MEDICAL CONFORMAL PREDICTION  ---  AdaptedCAFHT on sepsis ICU data
=============================================================================

QUICK START
-----------
Step 1 --- make sure ``sepsis_experiment_data_nacl_target.pkl`` is in the
  working directory (the pickle is produced by the upstream MIMIC-III
  extraction pipeline; see medical_data.md for format details).

Step 2 --- run experiment:

  # Default run (no shift correction):
  python medical_conformal.py --pkl sepsis_experiment_data_nacl_target.pkl

  # WITH likelihood-ratio shift correction:
  python medical_conformal.py --pkl sepsis_experiment_data_nacl_target.pkl --with_shift

  # Custom alpha, cal fraction, and seed:
  python medical_conformal.py --pkl sepsis_experiment_data_nacl_target.pkl --with_shift \
      --alpha 0.1 --cal_frac 0.5 --seed 42

  # Quick run with subsampled data (500 TrainCal, 300 Test):
  python medical_conformal.py --pkl sepsis_experiment_data_nacl_target.pkl --with_shift \
      --n_traincal 500 --n_test 300

FULL OPTIONS
------------
  --pkl          Path to sepsis_experiment_data_nacl_target.pkl (required)
  --with_shift   Toggle ON likelihood-ratio covariate-shift weighting.
                 Uses cross-split update_weighting_context() to reweight
                 calibration scores for the Norepinephrine distribution shift.
                 Default: off (uniform weights, plain conformal).
  --alpha        Miscoverage level. Default: 0.1  (targets 90% coverage)
  --cal_frac     Fraction of TrainCal patients used for calibration.
                 Default: 0.5  (remaining used for training)
  --n_traincal   Number of TrainCal patients to randomly subsample.
                 Default: None (use all available TrainCal patients)
  --n_test       Number of Test patients to randomly subsample.
                 Default: None (use all available Test patients)
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
    NaCl 0.9% dosage at each hour (24 hourly values per patient).

  Dynamic covariates (X):
    Three physiologic CHART trajectories:
      - Heart Rate
      - Respiratory Rate
      - O2 saturation pulseoxymetry

  Static covariates (S):
    Three patient-level variables included in both the prediction model
    and the likelihood-ratio classifier:
      - Age (numeric, standardized)
      - gender (binary: M->1, F->0)
      - ethnicity (grouped into 5 major categories, one-hot encoded
        with 4 dummy columns; WHITE is the reference category)

  Loop structure is identical to finance_conformal.py and test_conformal.py:
      for t in range(T):                          # T = 23 (hours 0..22)
          fit   linear model on train[:, :t+2, :]
          build calibration scores on cal[:, :t+2, :]
          (gamma selection every 5 steps)
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
    One-step-ahead autoregressive linear regression (predicts NaCl_{t+1} from
    information available at time t — no future leakage):
      NaCl_{t+1} ~ beta_0 + beta_1*NaCl_t
                 + beta_2*HR_t + beta_3*RR_t + beta_4*O2Sat_t
                 + beta_5*Age + beta_6*gender_M
                 + beta_7*eth_BLACK + beta_8*eth_HISPANIC
                 + beta_9*eth_ASIAN + beta_10*eth_OTHER
    fitted by OLS across all (patient, lagged-step) pairs (s -> s+1) in the
    training prefix. At prediction time t -> t+1, only Y_test[:, t, :] and
    X_test[:, t, :] are used; X_test[:, t+1, :] is NOT consulted.

  Conformity score:
    |NaCl_true - NaCl_pred|   (absolute residual)

  Likelihood-ratio weighting (--with_shift):
    A logistic classifier distinguishes Train from Test featurized prefixes.
    The monkey-patched featurizer computes summary statistics over BOTH the
    NaCl 0.9% (Y) prefix AND the 3 dynamic covariate (X) prefixes, PLUS
    the static covariates (Age, gender, ethnicity dummies):
      Per dynamic variable: [mean, std, min, max, last]  -> 4 vars x 5 = 20
      Static features: Age + gender_M + 4 ethnicity dummies = 6
      Total: 20 + 6 = 26 features per patient
    The classifier's predicted probabilities give density-ratio weights
    for each calibration series.

    The covariate shift in this dataset is driven by Norepinephrine usage,
    which correlates with different fluid management patterns and
    physiologic states between TrainCal and Test populations.

FEATURE OVERRIDE
----------------
  algorithm.py's default _featurize_prefixes uses only the last value as
  the single classifier feature.  This file monkey-patches a richer
  featurizer that uses both Y and X data plus static covariates.

  Because algorithm.py's _featurize_prefixes signature only accepts Y
  prefixes (n, t+1, 1), we store auxiliary arrays on the predictor
  instance (predictor._X_ctx, predictor._S_ctx) and look them up inside
  the featurizer.  The caller sets these before each featurize call.

    features per patient = 5 stats x 4 dynamic vars + 6 static = 26 features
    stats    : [mean, std, min, max, last]
    dynamic  : NaCl 0.9% + 3 covariates
    static   : Age, gender_M, eth_BLACK, eth_HISPANIC, eth_ASIAN, eth_OTHER
=============================================================================
"""
import argparse
import json
import pickle
import types
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from core.algorithm import AdaptedCAFHT


# -- Constants ----------------------------------------------------------------

TARGET_VAR = "NaCl 0.9% (target)"

COVARIATE_VARS = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
]

STATIC_VARS = ["Age", "gender", "ethnicity"]

# Ethnicity grouping: map raw MIMIC ethnicity strings to 5 major categories
ETHNICITY_MAP = {
    "WHITE":                                        "WHITE",
    "WHITE - RUSSIAN":                              "WHITE",
    "WHITE - BRAZILIAN":                            "WHITE",
    "WHITE - EASTERN EUROPEAN":                     "WHITE",
    "WHITE - OTHER EUROPEAN":                       "WHITE",
    "BLACK/AFRICAN AMERICAN":                       "BLACK",
    "BLACK/AFRICAN":                                "BLACK",
    "BLACK/CAPE VERDEAN":                           "BLACK",
    "BLACK/HAITIAN":                                "BLACK",
    "HISPANIC OR LATINO":                           "HISPANIC",
    "HISPANIC/LATINO - PUERTO RICAN":               "HISPANIC",
    "HISPANIC/LATINO - DOMINICAN":                  "HISPANIC",
    "HISPANIC/LATINO - MEXICAN":                    "HISPANIC",
    "HISPANIC/LATINO - GUATEMALAN":                 "HISPANIC",
    "HISPANIC/LATINO - CUBAN":                      "HISPANIC",
    "HISPANIC/LATINO - SALVADORAN":                 "HISPANIC",
    "HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)":   "HISPANIC",
    "HISPANIC/LATINO - COLOMBIAN":                  "HISPANIC",
    "HISPANIC/LATINO - HONDURAN":                   "HISPANIC",
    "ASIAN":                                        "ASIAN",
    "ASIAN - CHINESE":                              "ASIAN",
    "ASIAN - ASIAN INDIAN":                         "ASIAN",
    "ASIAN - VIETNAMESE":                            "ASIAN",
    "ASIAN - FILIPINO":                             "ASIAN",
    "ASIAN - CAMBODIAN":                            "ASIAN",
    "ASIAN - KOREAN":                               "ASIAN",
    "ASIAN - JAPANESE":                             "ASIAN",
    "ASIAN - THAI":                                 "ASIAN",
    "ASIAN - OTHER":                                "ASIAN",
}
# Everything not in the map goes to OTHER
ETHNICITY_DUMMIES = ["BLACK", "HISPANIC", "ASIAN", "OTHER"]  # WHITE = reference

GAMMA_GRID = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

# Number of static features after encoding
N_STATIC = 1 + 1 + len(ETHNICITY_DUMMIES)  # Age + gender_M + 4 ethnicity dummies = 6


# =============================================================================
# Data loading: pickle -> dict
# =============================================================================

def load_data(pkl_path):
    """Load the sepsis experiment pickle and return the raw dict."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Ethnicity encoding helper
# =============================================================================

def _encode_ethnicity(raw_ethnicity):
    """
    Map a raw ethnicity string to a 4-element one-hot vector.

    The 5 groups are WHITE (reference), BLACK, HISPANIC, ASIAN, OTHER.
    Returns a length-4 array (dummies for BLACK, HISPANIC, ASIAN, OTHER).
    WHITE maps to all zeros (reference category).
    """
    group = ETHNICITY_MAP.get(raw_ethnicity, "OTHER")
    vec = np.zeros(len(ETHNICITY_DUMMIES), dtype=np.float64)
    if group != "WHITE":
        idx = ETHNICITY_DUMMIES.index(group)
        vec[idx] = 1.0
    return vec


# =============================================================================
# Data conversion: nested dict-of-DataFrames -> arrays
# =============================================================================

def _convert_to_arrays(patient_trajectory_list):
    """
    Convert the list-of-dicts representation into (Y, X, S) arrays.

    Parameters
    ----------
    patient_trajectory_list : list[dict]
        Each dict maps trajectory names to DataFrames with columns
        (hour, value), plus scalar static variables.

    Returns
    -------
    Y : ndarray of shape (n_patients, 24, 1)
        NaCl 0.9% trajectories (target).
    X : ndarray of shape (n_patients, 24, n_cov)
        Dynamic covariate trajectories (3 variables), in COVARIATE_VARS order.
    S : ndarray of shape (n_patients, N_STATIC)
        Static covariates: [Age, gender_M, eth_BLACK, eth_HISPANIC, eth_ASIAN, eth_OTHER]
    """
    n = len(patient_trajectory_list)
    L = 24
    n_cov = len(COVARIATE_VARS)

    Y = np.zeros((n, L, 1), dtype=np.float64)
    X = np.zeros((n, L, n_cov), dtype=np.float64)
    S = np.zeros((n, N_STATIC), dtype=np.float64)

    for i, patient_dict in enumerate(patient_trajectory_list):
        # NaCl 0.9% target -> Y
        target_df = patient_dict[TARGET_VAR]
        Y[i, :, 0] = target_df["value"].to_numpy(dtype=np.float64)

        # Dynamic covariates -> X
        for j, var_name in enumerate(COVARIATE_VARS):
            cov_df = patient_dict[var_name]
            X[i, :, j] = cov_df["value"].to_numpy(dtype=np.float64)

        # Static covariates -> S
        S[i, 0] = float(patient_dict["Age"])
        S[i, 1] = 1.0 if patient_dict["gender"] == "M" else 0.0
        S[i, 2:] = _encode_ethnicity(patient_dict["ethnicity"])

    return Y, X, S


# =============================================================================
# Richer featurizer --- monkey-patched onto the predictor instance.
# =============================================================================

# Compute 5 summary statistics per dynamic variable
def _summarize_series(series_2d):
    """
    Compute 5 summary statistics per variable.

    Parameters
    ----------
    series_2d : (n, T) array --- one variable's prefix across n patients.

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


# Monkey-patched featurizer combining dynamic summaries and static covariates
def _richer_featurize_prefixes(self, prefixes):
    """
    Build a rich feature vector from Y (NaCl), X (dynamic covariates), and
    S (static covariates: Age, gender, ethnicity dummies).

    prefixes : (n, t+1, 1) --- NaCl 0.9% prefix (passed by algorithm.py)

    The covariate prefix is read from self._X_ctx (n, t+1, n_cov).
    The static covariates are read from self._S_ctx (n, N_STATIC).
    Both must be set by the caller before invoking this method.

    Returns
    -------
    (n, 26) feature matrix when X and S are available:
      5 stats x 4 dynamic vars + 6 static = 26 features

    Falls back gracefully if auxiliary arrays are missing.
    """
    Y = prefixes[..., 0]          # (n, t+1)
    n, T = Y.shape

    # NaCl summary (always available)
    feat_y = _summarize_series(Y)  # (n, 5)

    # Dynamic covariate summaries (from auxiliary storage)
    X_ctx = getattr(self, '_X_ctx', None)
    if X_ctx is not None and X_ctx.shape[0] == n and X_ctx.shape[1] == T:
        n_cov = X_ctx.shape[2]
        cov_feats = [_summarize_series(X_ctx[:, :, j]) for j in range(n_cov)]
        dynamic = np.column_stack([feat_y] + cov_feats)  # (n, 5 + 5*n_cov)
    else:
        dynamic = feat_y  # (n, 5) fallback

    # Static covariates (from auxiliary storage)
    S_ctx = getattr(self, '_S_ctx', None)
    if S_ctx is not None and S_ctx.shape[0] == n:
        raw = np.column_stack([dynamic, S_ctx])  # (n, 20 + 6 = 26)
    else:
        raw = dynamic

    # Standardize using training-set statistics so that all groups (train,
    # test, cal) are on the same scale.  self._feat_mu / self._feat_std are
    # set once from the training features and reused for test and cal.
    mu  = getattr(self, '_feat_mu', None)
    std = getattr(self, '_feat_std', None)
    if mu is not None and std is not None and mu.shape[-1] == raw.shape[1]:
        return (raw - mu) / std
    else:
        return raw


# =============================================================================
# Diagnostic helpers (identical pattern to finance_conformal.py)
# =============================================================================

# Print a concise summary of LR calibration weights at time step t
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


# Build human-readable feature names for the 26-feature vector
def _build_feat_names():
    """Build human-readable names for the 26-feature vector."""
    stats = ["mean", "std", "min", "max", "last"]
    var_names = [TARGET_VAR] + COVARIATE_VARS
    names = [f"{v}|{s}" for v in var_names for s in stats]
    names += ["Age", "gender_M"] + [f"eth_{e}" for e in ETHNICITY_DUMMIES]
    return names


# Compare per-feature mean/std between train, test, and cal prefixes
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


# Report the fitted logistic classifier's training accuracy and coef norm
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
# Prediction model: cross-sectional linear regression on covariates + statics
# =============================================================================

class LinearCovariateModel:
    """
    One-step-ahead autoregressive OLS model:
      NaCl_{t+1} ~ beta_0 + beta_1*NaCl_t
                 + beta_2*HR_t + beta_3*RR_t + beta_4*O2Sat_t
                 + beta_5*Age + beta_6*gender_M
                 + beta_7*eth_BLACK + beta_8*eth_HISPANIC
                 + beta_9*eth_ASIAN + beta_10*eth_OTHER

    All predictors are evaluated at time t; the target is the next-hour NaCl.
    Dynamic covariates vary per (patient, timestep); static covariates are
    constant per patient and tiled across pair-steps for design matrix
    assembly. Training pairs are formed by teacher-forcing on the observed
    NaCl trajectory (target = NaCl_{s+1}, inputs = (NaCl_s, X_s, S)).
    """

    def __init__(self, cov_names, static_names):
        self.cov_names = cov_names
        self.static_names = static_names
        self.beta = None
        self.noise_std = 1.0

    def fit(self, Y_train, X_train, S_train, verbose=False):
        """
        Parameters
        ----------
        Y_train : (n, L_prefix, 1)   NaCl 0.9% prefix; uses indices 0..L-1.
        X_train : (n, L_prefix, n_cov)   dynamic covariate prefix; same indexing.
        S_train : (n, N_STATIC)   static covariates per patient.

        Builds (input_s, target_{s+1}) pairs for s = 0..L-2 (i.e. L-1 pairs
        per patient). The design matrix has columns
            [1, NaCl_s, HR_s, RR_s, O2Sat_s, Age, gender_M, eth_*]
        and the response is NaCl_{s+1}.
        """
        n, L, n_cov = X_train.shape
        if L < 2:
            return
        n_pairs = L - 1
        # Inputs at times 0..L-2
        y_lag = Y_train[:, :n_pairs, 0].reshape(-1, 1)        # (n*(L-1), 1)
        X_lag = X_train[:, :n_pairs, :].reshape(-1, n_cov)    # (n*(L-1), n_cov)
        # Targets at times 1..L-1
        y = Y_train[:, 1:, 0].reshape(-1)                     # (n*(L-1),)
        # Tile static covariates: each patient's statics repeated (L-1) times
        S_tiled = np.repeat(S_train, n_pairs, axis=0)         # (n*(L-1), N_STATIC)
        X_design = np.hstack([
            np.ones((len(y), 1)),
            y_lag,
            X_lag,
            S_tiled,
        ])
        self.beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ self.beta
        self.noise_std = float(np.std(resid, ddof=X_design.shape[1]))
        if verbose:
            print(f"  [Model] Fitted on {n} patients x {n_pairs} pair-steps")
            print(f"  [Model] Residual std : {self.noise_std:.4f}")
            print(f"  [Model] Coefficients :")
            print(f"            intercept = {self.beta[0]:.4f}")
            all_names = ["NaCl 0.9% (lag)"] + list(self.cov_names) + list(self.static_names)
            for name, coef in zip(all_names, self.beta[1:]):
                print(f"            {name:42s} = {coef:.4f}")

    def predict(self, y_prev, x_dyn, s_static):
        """
        Predict NaCl 0.9% at time t+1 given:
          y_prev    : float   — observed NaCl at time t
          x_dyn     : (n_cov,) — dynamic covariates at time t
          s_static  : (N_STATIC,) — static covariates
        """
        if self.beta is None:
            return 0.0
        features = np.concatenate([[1.0, float(y_prev)], x_dyn, s_static])
        return float(features @ self.beta)


# =============================================================================
# Gamma selection (mirrors _select_gamma in finance_conformal.py)
# =============================================================================

# Select gamma by running simple ACI on a 3-way split of training data
def _select_gamma(Y_train, X_train, S_train, cov_names, static_names,
                  base_alpha, t_max, gamma_grid, seed=0):
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
    Y_fit_sel = Y_train[idx1];  X_fit_sel = X_train[idx1];  S_fit_sel = S_train[idx1]
    Y_cal_sel = Y_train[idx2];  X_cal_sel = X_train[idx2];  S_cal_sel = S_train[idx2]
    Y_eval    = Y_train[idx3];  X_eval    = X_train[idx3];  S_eval    = S_train[idx3]
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
        for t in range(horizon):
            # Fit on prefix [:t+2] -> (t+1) lagged pairs s=0..t per patient
            sel_model = LinearCovariateModel(cov_names, static_names)
            sel_model.fit(Y_fit_sel[:, :t+2, :], X_fit_sel[:, :t+2, :],
                          S_fit_sel)
            predictor.noise_std = sel_model.noise_std
            # Calibration: pairs s=0..t -> predict NaCl_{s+1} from
            # (NaCl_s, X_s, S); score = |NaCl_{s+1} - pred|.
            cal_scores = []
            for i in range(len(idx2)):
                for s in range(t + 1):
                    y_prev = float(Y_cal_sel[i, s, 0])
                    x_prev = X_cal_sel[i, s, :]
                    s_i    = S_cal_sel[i, :]
                    y_true = float(Y_cal_sel[i, s + 1, 0])
                    y_pred = sel_model.predict(y_prev, x_prev, s_i)
                    cal_scores.append(abs(y_true - y_pred))
            predictor._scores  = np.array(cal_scores, dtype=float)
            predictor._weights = np.ones(len(cal_scores), dtype=float)
            predictor._q       = None
            alpha_used = alpha_series.copy()
            alpha_next = alpha_series.copy()
            step_cov = []
            # Evaluation: predict NaCl_{t+1} using inputs at time t.
            for i in range(n_eval):
                y_prev = float(Y_eval[i, t, 0])
                x_prev = X_eval[i, t, :]
                s_i    = S_eval[i, :]
                y_true = float(Y_eval[i, t + 1, 0])
                y_pred = sel_model.predict(y_prev, x_prev, s_i)
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
                           gamma_grid=None, with_shift=False,
                           n_traincal=None, n_test=None, verbose=True):
    """
    Run AdaptedCAFHT conformal prediction on sepsis ICU data.

    Parameters
    ----------
    data : dict
        Output of load_data() — pickle with patient_ids and trajectories.
    cal_frac : float
        Fraction of TrainCal patients used for calibration.
    alpha : float
        Miscoverage level.
    seed : int
        Random seed for Train/Cal split, subsampling, and gamma selection.
    gamma_grid : list[float]
        ACI step-size candidates.
    with_shift : bool
        Whether to use likelihood-ratio covariate-shift weighting.
    n_traincal : int or None
        If set, randomly subsample this many TrainCal patients.
    n_test : int or None
        If set, randomly subsample this many Test patients.

    Returns
    -------
    dict with coverage_by_time, width_by_time, overall_coverage, etc.
    """
    if gamma_grid is None:
        gamma_grid = GAMMA_GRID

    # -- Subsample patients if requested -----------------------------------
    rng_sub = np.random.default_rng(seed)

    traincal_list = data["patient_trajectory_list_traincal"]
    traincal_ids  = data["patient_ids_traincal"]
    if n_traincal is not None and n_traincal < len(traincal_list):
        sub_idx = rng_sub.choice(len(traincal_list), size=n_traincal, replace=False)
        sub_idx.sort()
        traincal_list = [traincal_list[i] for i in sub_idx]
        traincal_ids  = [traincal_ids[i] for i in sub_idx]

    test_list = data["patient_trajectory_list_test"]
    test_ids  = data["patient_ids_test"]
    if n_test is not None and n_test < len(test_list):
        sub_idx = rng_sub.choice(len(test_list), size=n_test, replace=False)
        sub_idx.sort()
        test_list = [test_list[i] for i in sub_idx]
        test_ids  = [test_ids[i] for i in sub_idx]

    # -- Convert nested dicts to arrays ------------------------------------
    Y_traincal, X_traincal, S_traincal = _convert_to_arrays(traincal_list)
    Y_test_all, X_test_all, S_test_all = _convert_to_arrays(test_list)

    n_traincal, L, _ = Y_traincal.shape
    n_test = Y_test_all.shape[0]
    T = L - 1  # 23 prediction steps (hours 0..22, predicting 1..23)

    # -- Split TrainCal -> Train + Cal -------------------------------------
    rng       = np.random.default_rng(seed)
    perm      = rng.permutation(n_traincal)
    n_cal     = int(n_traincal * cal_frac)
    n_train   = n_traincal - n_cal
    if n_train == 0:
        raise ValueError(f"cal_frac={cal_frac} leaves no training patients.")

    train_idx = perm[:n_train]
    cal_idx   = perm[n_train:]

    Y_train, X_train, S_train = Y_traincal[train_idx], X_traincal[train_idx], S_traincal[train_idx]
    Y_cal,   X_cal,   S_cal   = Y_traincal[cal_idx],   X_traincal[cal_idx],   S_traincal[cal_idx]
    Y_test,  X_test,  S_test  = Y_test_all, X_test_all, S_test_all

    cov_names = COVARIATE_VARS
    static_names = ["Age", "gender_M"] + [f"eth_{e}" for e in ETHNICITY_DUMMIES]

    if verbose:
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
        print(f"  Static (S)      : {static_names}")
        print(f"  Alpha           : {alpha}  (target = {1-alpha:.0%})")
        print(f"  Gamma grid      : {gamma_grid}")
        print(f"  With shift      : {with_shift}")
        print()

    # -- Set up predictor --------------------------------------------------
    predictor    = AdaptedCAFHT(alpha=alpha)
    linear_model = LinearCovariateModel(cov_names, static_names)

    # Monkey-patch richer featurizer (summary stats + static covariates)
    predictor._featurize_prefixes = types.MethodType(
        _richer_featurize_prefixes, predictor
    )
    n_dyn_feat = 5 * (1 + len(COVARIATE_VARS))
    n_total_feat = n_dyn_feat + N_STATIC
    if verbose:
        print(f"  Featurizer      : richer ({n_dyn_feat} dynamic + {N_STATIC} static"
              f" = {n_total_feat} features)")
        print()

    # -- ACI state ---------------------------------------------------------
    alpha_t   = np.full(n_test, alpha, dtype=float)
    gamma_opt = float(gamma_grid[0])

    coverage_by_time  = []
    width_by_time     = []
    all_covered       = []
    gamma_opt_history = []
    first_true  = []
    first_lower = []
    first_upper = []

    # -- Main loop: for each hour t, predict hour t+1 ---------------------
    for t in range(T):

        # -- Fit linear model on training prefix --------------------------
        linear_model.fit(Y_train[:, :t+2, :], X_train[:, :t+2, :], S_train,
                         verbose=(verbose and t == T - 1))
        predictor.noise_std = linear_model.noise_std

        # -- Build calibration scores -------------------------------------
        # Pairs s = 0..t per patient: predict NaCl_{s+1} from
        # (NaCl_s, X_s, S); accumulate absolute residual scores.
        cal_scores = []
        for i in range(n_cal):
            for s in range(t + 1):
                y_prev = float(Y_cal[i, s, 0])
                x_prev = X_cal[i, s, :]
                s_i    = S_cal[i, :]
                y_true = float(Y_cal[i, s + 1, 0])
                y_pred = linear_model.predict(y_prev, x_prev, s_i)
                cal_scores.append(abs(y_true - y_pred))
        cal_scores_arr = np.array(cal_scores, dtype=float)

        # Default: uniform weights
        predictor._scores  = cal_scores_arr
        predictor._weights = np.ones(len(cal_scores_arr), dtype=float)
        predictor._q       = None

        # -- Gamma selection every 5 steps --------------------------------
        if t > 0 and (t % 10 == 0):
            sel_seed = seed + 10000 + t
            gamma_opt, gamma_scores = _select_gamma(
                Y_train=Y_train, X_train=X_train, S_train=S_train,
                cov_names=cov_names, static_names=static_names,
                base_alpha=alpha, t_max=t, gamma_grid=gamma_grid,
                seed=sel_seed,
            )
            scores_str = "  ".join(
                f"gamma={g:.3f}->{v:.3f}" for g, v in gamma_scores.items()
                if np.isfinite(v)
            )
            if verbose:
                print(f"  [gamma sel t={t:3d}]  best gamma = {gamma_opt}"
                      f"   ({scores_str})")

        gamma_opt_history.append(float(gamma_opt))

        # -- Prediction loop ----------------------------------------------
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

                # Featurize train prefixes: compute raw features first,
                # then derive standardization stats from training data.
                predictor._X_ctx = X_train[:, :t+1, :]
                predictor._S_ctx = S_train
                predictor._feat_mu = None
                predictor._feat_std = None
                train_raw = predictor._featurize_prefixes(train_prefixes)
                predictor._feat_mu  = train_raw.mean(axis=0, keepdims=True)
                predictor._feat_std = train_raw.std(axis=0, keepdims=True) + 1e-8
                # Re-featurize train with standardization applied
                predictor._X_ctx = X_train[:, :t+1, :]
                predictor._S_ctx = S_train
                predictor._train_feat_t = predictor._featurize_prefixes(
                    train_prefixes)

                # Featurize test-half prefixes (uses train stats)
                predictor._X_ctx = X_test[ctx_idx, :t+1, :]
                predictor._S_ctx = S_test[ctx_idx]
                predictor._test_feat_t = predictor._featurize_prefixes(
                    Y_test[ctx_idx, :t+1, :])

                # Mark context as shifted
                predictor._is_shifted_ctx = True
                predictor._t_ctx = t
                predictor._clf = None

                # Featurize cal prefixes (uses train stats)
                predictor._X_ctx = X_cal[:, :t+1, :]
                predictor._S_ctx = S_cal
                cal_feat = predictor._featurize_prefixes(Y_cal[:, :t+1, :])

                # Compute per-series LR weights for cal data
                per_series_w = predictor._compute_density_ratio_weights(
                    trainX=predictor._train_feat_t,
                    testX=predictor._test_feat_t,
                    evalX=cal_feat,
                )

                # Diagnostics (first swap only, at selected steps)
                if verbose and pred_idx is half1 and (t == 1 or t % 10 == 0 or t == T - 1):
                    _print_feature_diagnostic(predictor, cal_feat, t)
                    _print_classifier_diagnostic(predictor, cal_feat, t)

                # Tile per-series weights to per-score weights.
                # Each cal patient contributes (t+1) score entries (one per
                # lagged pair s -> s+1, for s = 0..t), so each weight is
                # repeated (t+1) times.
                n_steps = t + 1
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
                if verbose and (t == 1 or t % 10 == 0 or t == T - 1):
                    label = "half1-ctx" if pred_idx is half1 else "half2-ctx"
                    _print_weight_diagnostic(per_series_w, t, label=label)

                # Predict NaCl_{t+1} using inputs at time t only.
                for i in pred_idx:
                    y_prev = float(Y_test[i, t, 0])
                    x_prev = X_test[i, t, :]
                    s_i    = S_test[i, :]
                    y_true = float(Y_test[i, t + 1, 0])
                    y_pred = linear_model.predict(y_prev, x_prev, s_i)
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
                y_prev = float(Y_test[i, t, 0])
                x_prev = X_test[i, t, :]
                s_i    = S_test[i, :]
                y_true = float(Y_test[i, t + 1, 0])
                y_pred = linear_model.predict(y_prev, x_prev, s_i)
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

        if verbose and ((t + 1) % 5 == 0 or t == T - 1):
            print(f"  [t={t+1:3d}/{T}]  coverage={np.mean(covered_t):.3f}  "
                  f"width={np.mean(width_t):.4f}  gamma={gamma_opt}")

    overall_coverage = float(np.mean(all_covered))
    target = 1.0 - alpha
    if verbose:
        print(f"\n  Overall coverage : {overall_coverage:.4f}  "
              f"(target = {target:.4f},  error = {overall_coverage - target:+.4f})")
        print(f"  Mean width       : {np.mean(width_by_time):.4f}")
    if verbose:
        print(f"  Final gamma_opt  : {gamma_opt}")

    # Pick patient IDs for the first test patient (for plot title)
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
            "n_train":      int(n_train),
            "n_cal":        int(n_cal),
            "n_test":       int(n_test),
            "L":            int(L),
            "alpha":        alpha,
            "gamma_grid":   [float(g) for g in gamma_grid],
            "seed":         seed,
            "with_shift":   with_shift,
            "target_var":   TARGET_VAR,
            "cov_names":    cov_names,
            "static_names": static_names,
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
    axes[0, 1].set_ylabel('Mean interval width (mL)')
    axes[0, 1].set_title('Prediction Interval Width over Time')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: First test patient --- actual NaCl vs prediction interval
    true_arr  = np.array(first["true"])
    lower_arr = np.array(first["lower"])
    upper_arr = np.array(first["upper"])
    y_min  = float(np.percentile(true_arr, 1))
    y_max  = float(np.percentile(true_arr, 99))
    margin = (y_max - y_min) * 0.3
    axes[1, 0].fill_between(x, lower_arr, upper_arr, alpha=0.25,
                             color='steelblue', label='Prediction interval')
    axes[1, 0].plot(x, true_arr, 'k-', linewidth=1.5,
                     label='Actual NaCl 0.9%')
    axes[1, 0].set_ylim(y_min - margin, y_max + margin)
    axes[1, 0].set_ylabel('NaCl 0.9% (mL)')
    axes[1, 0].set_title(f'Patient {patient} --- Actual NaCl vs. Prediction Interval')
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
                        help="Path to sepsis_experiment_data_nacl_target.pkl")
    parser.add_argument("--cal_frac",    type=float, default=0.5,
                        help="Fraction of TrainCal used for calibration "
                             "(default: 0.5)")
    parser.add_argument("--alpha",       type=float, default=0.1,
                        help="Miscoverage level (default: 0.1)")
    parser.add_argument("--seed",        type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--n_traincal",  type=int,   default=None,
                        help="Randomly subsample this many TrainCal patients "
                             "(default: use all)")
    parser.add_argument("--n_test",      type=int,   default=None,
                        help="Randomly subsample this many Test patients "
                             "(default: use all)")
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
        n_traincal=args.n_traincal,
        n_test=args.n_test,
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
