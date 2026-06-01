"""
medical/medical_runner_whole.py — single-seed runner for Algorithm 1
(whole-trajectory coverage) on the MIMIC-III sepsis cohort.

Self-contained (§ 0). Constants, ethnicity encoding, array conversion, and the
one-step-ahead `LinearCovariateModel` are COPIED (not imported) from
OLD_medical_conformal.py per § B.3. The only cross-module import for the
algorithm is core.weighted_cafht_whole.

Mapping to WEIGHTED_CAFHT_PLAN.md § 3:
  - Predictor (§ 3.1): `LinearCovariateModel` — one-step-ahead AR
    NaCl_{t+1} ~ NaCl_t + vitals_t + statics, fit on the full D_tr, applied
    one-step-ahead so the per-step prediction uses the observed NaCl_t (matches
    the algorithm box's f̂_t on the observed prefix).
  - LR featurizer (§ 3.3, Q2(a)): X_1 = the hour-0 vitals (3) + statics (6) =
    9 features, z-scored on D_tr stats. (Per Q2(a) we feed X_1 only and accept
    reduced separability — the Norepinephrine split shifts a 12-h window, not
    just hour 0, so this classifier is expected to be near-uniform.)
  - Four-way split (§ 2.0): D_ACI is peeled off the TrainCal (source) pool
    before the D_tr / D_cal split; D_test is the Test (target) pool.

Comparison (§ 5.1): full ("our version", LR + ACI) vs uniform ("no-LR", ACI).
Headline metrics: whole-trajectory JOINT coverage and mean band width.

TrainCal = sepsis patients with no Norepinephrine in the first 12 h (source);
Test = patients with early Norepinephrine (target / shifted).
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.weighted_cafht_whole import WeightedCAFHTWholeTrajectory  # noqa: E402

# =============================================================================
# Constants + data prep — copied from OLD_medical_conformal.py (§ B.3)
# =============================================================================
TARGET_VAR = "NaCl 0.9% (target)"
COVARIATE_VARS = ["Heart Rate", "Respiratory Rate", "O2 saturation pulseoxymetry"]
STATIC_VARS = ["Age", "gender", "ethnicity"]
ETHNICITY_DUMMIES = ["BLACK", "HISPANIC", "ASIAN", "OTHER"]   # WHITE = reference
N_STATIC = 1 + 1 + len(ETHNICITY_DUMMIES)                     # Age + gender_M + 4 = 6
GAMMA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2]             # medical grid (OLD)

# Ethnicity grouping: raw MIMIC strings -> 5 major categories.
ETHNICITY_MAP = {
    "WHITE": "WHITE", "WHITE - RUSSIAN": "WHITE", "WHITE - BRAZILIAN": "WHITE",
    "WHITE - EASTERN EUROPEAN": "WHITE", "WHITE - OTHER EUROPEAN": "WHITE",
    "BLACK/AFRICAN AMERICAN": "BLACK", "BLACK/AFRICAN": "BLACK",
    "BLACK/CAPE VERDEAN": "BLACK", "BLACK/HAITIAN": "BLACK",
    "HISPANIC OR LATINO": "HISPANIC", "HISPANIC/LATINO - PUERTO RICAN": "HISPANIC",
    "HISPANIC/LATINO - DOMINICAN": "HISPANIC", "HISPANIC/LATINO - MEXICAN": "HISPANIC",
    "HISPANIC/LATINO - GUATEMALAN": "HISPANIC", "HISPANIC/LATINO - CUBAN": "HISPANIC",
    "HISPANIC/LATINO - SALVADORAN": "HISPANIC",
    "HISPANIC/LATINO - CENTRAL AMERICAN (OTHER)": "HISPANIC",
    "HISPANIC/LATINO - COLOMBIAN": "HISPANIC", "HISPANIC/LATINO - HONDURAN": "HISPANIC",
    "ASIAN": "ASIAN", "ASIAN - CHINESE": "ASIAN", "ASIAN - ASIAN INDIAN": "ASIAN",
    "ASIAN - VIETNAMESE": "ASIAN", "ASIAN - FILIPINO": "ASIAN",
    "ASIAN - CAMBODIAN": "ASIAN", "ASIAN - KOREAN": "ASIAN",
    "ASIAN - JAPANESE": "ASIAN", "ASIAN - THAI": "ASIAN", "ASIAN - OTHER": "ASIAN",
}


def load_data(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _encode_ethnicity(raw):
    """4-element one-hot for (BLACK, HISPANIC, ASIAN, OTHER); WHITE = all-zero."""
    group = ETHNICITY_MAP.get(raw, "OTHER")
    vec = np.zeros(len(ETHNICITY_DUMMIES))
    if group != "WHITE":
        vec[ETHNICITY_DUMMIES.index(group)] = 1.0
    return vec


def _convert_to_arrays(patient_list):
    """list-of-dicts -> Y (n,24,1) NaCl target, X (n,24,3) vitals, S (n,6) statics."""
    n, L, n_cov = len(patient_list), 24, len(COVARIATE_VARS)
    Y = np.zeros((n, L, 1))
    X = np.zeros((n, L, n_cov))
    S = np.zeros((n, N_STATIC))
    for i, p in enumerate(patient_list):
        Y[i, :, 0] = p[TARGET_VAR]["value"].to_numpy(dtype=float)
        for j, var in enumerate(COVARIATE_VARS):
            X[i, :, j] = p[var]["value"].to_numpy(dtype=float)
        S[i, 0] = float(p["Age"])
        S[i, 1] = 1.0 if p["gender"] == "M" else 0.0
        S[i, 2:] = _encode_ethnicity(p["ethnicity"])
    return Y, X, S


# =============================================================================
# Predictor — copied from OLD_medical_conformal.py:LinearCovariateModel (§ B.3)
# =============================================================================
class LinearCovariateModel:
    """One-step-ahead autoregressive OLS:
        NaCl_{t+1} ~ b0 + b1·NaCl_t + b·vitals_t + b·statics.
    Trained by teacher-forcing on observed (NaCl_s, vitals_s, S) -> NaCl_{s+1}."""

    def __init__(self):
        self.beta = None
        self.noise_std = 1.0

    def fit(self, Y, X, S):
        n, L, n_cov = X.shape
        if L < 2:
            return
        n_pairs = L - 1
        y_lag = Y[:, :n_pairs, 0].reshape(-1, 1)
        X_lag = X[:, :n_pairs, :].reshape(-1, n_cov)
        y = Y[:, 1:, 0].reshape(-1)
        S_tiled = np.repeat(S, n_pairs, axis=0)
        design = np.hstack([np.ones((len(y), 1)), y_lag, X_lag, S_tiled])
        self.beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ self.beta
        self.noise_std = float(np.std(resid, ddof=design.shape[1]))


def _predict_trajectory(beta, Y, X, S):
    """One-step-ahead predictions Ŷ_1..Ŷ_{L-1} (n, L-1, 1) using observed
    (NaCl_t, vitals_t, S) at each t = 0..L-2 (the algorithm box's observed
    prefix f̂_t)."""
    n, L, _ = Y.shape
    preds = np.zeros((n, L - 1, 1))
    for t in range(L - 1):
        feat = np.hstack([np.ones((n, 1)), Y[:, t, :], X[:, t, :], S])
        preds[:, t, 0] = feat @ beta
    return preds


# =============================================================================
# LR featurizer (§ 3.3, Q2(a)): X_1 = hour-0 vitals + statics
# =============================================================================
def featurize_x1(X, S):
    """Whole-trajectory classifier feature = hour-0 vitals (3) + statics (6)."""
    return np.hstack([X[:, 0, :], S])                # (n, 9)


# =============================================================================
# Single-seed experiment
# =============================================================================
def run_single(pkl, seed=42, mode="full", n_traincal=1000, n_test=500,
               cal_frac=0.5, frac_aci=0.15, alpha=0.1, gamma_grid=None,
               verbose=False, data=None):
    """Run one seed; `mode` ∈ {full, uniform}. `data` may be a preloaded pickle
    dict (so the multi-seed wrapper loads it once)."""
    gamma_grid = gamma_grid if gamma_grid is not None else GAMMA_GRID
    data = data if data is not None else load_data(pkl)
    rng = np.random.default_rng(seed)

    tc_list = data["patient_trajectory_list_traincal"]
    te_list = data["patient_trajectory_list_test"]
    tc_idx = rng.choice(len(tc_list), min(n_traincal, len(tc_list)), replace=False)
    te_idx = rng.choice(len(te_list), min(n_test, len(te_list)), replace=False)
    Y_tc, X_tc, S_tc = _convert_to_arrays([tc_list[i] for i in tc_idx])
    Y_te, X_te, S_te = _convert_to_arrays([te_list[i] for i in te_idx])

    # Four-way split of TrainCal: peel D_ACI, then split rest into D_tr / D_cal.
    n_tc = len(Y_tc)
    perm = rng.permutation(n_tc)
    n_aci = int(frac_aci * n_tc)
    aci_i, rest = perm[:n_aci], perm[n_aci:]
    n_cal = int(len(rest) * cal_frac)
    cal_i, tr_i = rest[:n_cal], rest[n_cal:]

    Y_tr, X_tr, S_tr = Y_tc[tr_i], X_tc[tr_i], S_tc[tr_i]
    Y_cal, X_cal, S_cal = Y_tc[cal_i], X_tc[cal_i], S_tc[cal_i]
    Y_aci, X_aci, S_aci = Y_tc[aci_i], X_tc[aci_i], S_tc[aci_i]

    model = LinearCovariateModel()
    model.fit(Y_tr, X_tr, S_tr)
    b = model.beta
    tr_pred, tr_true = _predict_trajectory(b, Y_tr, X_tr, S_tr), Y_tr[:, 1:, :]
    cal_pred, cal_true = _predict_trajectory(b, Y_cal, X_cal, S_cal), Y_cal[:, 1:, :]
    test_pred, test_true = _predict_trajectory(b, Y_te, X_te, S_te), Y_te[:, 1:, :]
    aci_pred, aci_true = _predict_trajectory(b, Y_aci, X_aci, S_aci), Y_aci[:, 1:, :]

    # Classifier features, z-scored on D_tr stats (vitals + Age share no scale).
    F_tr_raw = featurize_x1(X_tr, S_tr)
    mu, sd = F_tr_raw.mean(0), F_tr_raw.std(0) + 1e-8
    F_tr = (F_tr_raw - mu) / sd
    F_cal = (featurize_x1(X_cal, S_cal) - mu) / sd
    F_test = (featurize_x1(X_te, S_te) - mu) / sd

    feat = ((lambda F: np.zeros((len(np.asarray(F)), 1))) if mode == "uniform"
            else (lambda F: np.asarray(F)))

    algo = WeightedCAFHTWholeTrajectory(
        alpha=alpha, gamma_grid=gamma_grid, featurize_fn=feat, verbose=verbose)
    bands = algo.predict_bands(
        (tr_pred, tr_true), (cal_pred, cal_true), (test_pred, test_true),
        (aci_pred, aci_true), X_tr=F_tr, X_cal=F_cal, X_test=F_test, seed=seed)

    metrics = _coverage_metrics(bands, test_true, alpha)
    return {
        "regime": "whole_trajectory", "domain": "medical", "mode": mode,
        "alpha": alpha, "seed": int(seed),
        "n_tr": len(tr_i), "n_aci": len(aci_i), "n_cal": len(cal_i),
        "n_test": len(te_idx), "horizon": int(test_true.shape[1]),
        "gamma_opt": algo.gamma_opt_, "n_inf": int(algo.n_inf_),
        "score_bank_shape": list(algo.score_bank_shape_), **metrics,
    }


def _coverage_metrics(bands, truth, alpha):
    """bands (n, H, 2, 1); truth (n, H, 1). Whole-trajectory JOINT coverage +
    width over finite bands; δ_∞ bands counted as covered, excluded from width."""
    low, high, y = bands[:, :, 0, 0], bands[:, :, 1, 0], truth[:, :, 0]
    covered = (y >= low) & (y <= high)
    widths = high - low
    finite = np.isfinite(widths)
    return {
        "coverage_by_time": covered.mean(axis=0).tolist(),
        "joint_coverage": float(covered.all(axis=1).mean()),
        "overall_coverage": float(covered.all(axis=1).mean()),
        "pooled_coverage": float(covered.mean()),
        "mean_width": float(widths[finite].mean()) if finite.any() else float("inf"),
        "target_coverage": 1.0 - alpha,
    }


def main():
    p = argparse.ArgumentParser(description="Medical whole-trajectory runner (Alg. 1)")
    p.add_argument("--pkl", default="medical/sepsis_experiment_data_nacl_target.pkl")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--n_traincal", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--cal_frac", type=float, default=0.5)
    p.add_argument("--frac_aci", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_json", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    res = run_single(args.pkl, seed=args.seed, mode=args.mode,
                     n_traincal=args.n_traincal, n_test=args.n_test,
                     cal_frac=args.cal_frac, frac_aci=args.frac_aci,
                     alpha=args.alpha, verbose=args.verbose)
    print(f"[medical/whole/{args.mode}] joint_cov={res['joint_coverage']:.3f} "
          f"pooled_cov={res['pooled_coverage']:.3f} "
          f"mean_width={res['mean_width']:.2f} mL/hr "
          f"gamma_opt={res['gamma_opt']} n_inf={res['n_inf']}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
