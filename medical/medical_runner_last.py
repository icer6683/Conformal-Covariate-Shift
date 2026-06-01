"""
medical/medical_runner_last.py — single-seed runner for Algorithm 2
(last-step coverage) on the MIMIC-III sepsis cohort.

Self-contained (§ 0). Constants, ethnicity encoding, and array conversion are
COPIED from OLD_medical_conformal.py per § B.3. The only cross-module import for
the algorithm is core.weighted_cafht_last.

Mapping to WEIGHTED_CAFHT_PLAN.md § 3:
  - Predictor (§ 3.2, Q3): ridge (λ by CV via RidgeCV) of the FINAL-hour NaCl on
    the flattened prefix [NaCl_{0:L-2}, vitals_{0:L-2}, statics]. We include the
    NaCl history (deviating from the plan's "vitals + statics only") because it
    is by far the most predictive feature — the algorithm is agnostic to the
    predictor's form.
  - LR featurizer (§ 3.3): the SAME flattened prefix X_{1:T} (NaCl history +
    vitals + statics), z-scored. Including the NaCl history lets the classifier
    see the dominant shift (NaCl KL ≈ 0.21 between TrainCal and Test).
  - Split: TrainCal (source) -> D_tr / D_cal; Test (target) -> D_test.
    No D_ACI / no γ / no ACI.

Comparison (§ 5.1): full ("our version", LR) vs uniform ("no-LR").
Headline metrics: final-step coverage and mean band width.
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
from core.weighted_cafht_last import WeightedCAFHTLastStep         # noqa: E402

try:
    from sklearn.linear_model import RidgeCV
    _SKLEARN = True
except Exception:                                                  # pragma: no cover
    _SKLEARN = False

# =============================================================================
# Constants + data prep — copied from OLD_medical_conformal.py (§ B.3)
# =============================================================================
TARGET_VAR = "NaCl 0.9% (target)"
COVARIATE_VARS = ["Heart Rate", "Respiratory Rate", "O2 saturation pulseoxymetry"]
ETHNICITY_DUMMIES = ["BLACK", "HISPANIC", "ASIAN", "OTHER"]
N_STATIC = 1 + 1 + len(ETHNICITY_DUMMIES)

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
    group = ETHNICITY_MAP.get(raw, "OTHER")
    vec = np.zeros(len(ETHNICITY_DUMMIES))
    if group != "WHITE":
        vec[ETHNICITY_DUMMIES.index(group)] = 1.0
    return vec


def _convert_to_arrays(patient_list):
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
# Predictor (§ 3.2): ridge of final NaCl on the flattened prefix
# =============================================================================
def featurize_xall(Y, X, S):
    """Last-step feature = flattened prefix [NaCl_{0:L-2}, vitals_{0:L-2}, S].
    Predicts the FINAL-hour NaCl; the target value itself is never included."""
    n, L, _ = Y.shape
    nacl_hist = Y[:, :L - 1, 0]                       # (n, L-1)
    vitals = X[:, :L - 1, :].reshape(n, -1)           # (n, (L-1)*n_cov)
    return np.hstack([nacl_hist, vitals, S])          # (n, (L-1) + (L-1)*3 + 6)


class LastStepRidge:
    """Ridge regression with λ chosen by CV (RidgeCV's efficient LOO; falls back
    to a fixed-λ normal-equation solve if sklearn is unavailable)."""

    def __init__(self, alphas=(0.1, 1.0, 10.0, 100.0, 1000.0)):
        self.alphas = alphas
        self._m = None
        self._beta = None

    def fit(self, F, y):
        if _SKLEARN:
            self._m = RidgeCV(alphas=self.alphas).fit(F, y)
        else:                                          # pragma: no cover
            lam = 10.0
            A = F.T @ F + lam * np.eye(F.shape[1])
            self._beta = np.linalg.solve(A, F.T @ (y - y.mean()))
            self._y0 = float(y.mean())
        return self

    def predict(self, F):
        if _SKLEARN:
            return self._m.predict(F)
        return self._y0 + F @ self._beta               # pragma: no cover


# =============================================================================
# Single-seed experiment
# =============================================================================
def run_single(pkl, seed=42, mode="full", n_traincal=1000, n_test=500,
               cal_frac=0.5, alpha=0.1, verbose=False, data=None):
    """Run one seed; `mode` ∈ {full, uniform}."""
    data = data if data is not None else load_data(pkl)
    rng = np.random.default_rng(seed)

    tc_list = data["patient_trajectory_list_traincal"]
    te_list = data["patient_trajectory_list_test"]
    tc_idx = rng.choice(len(tc_list), min(n_traincal, len(tc_list)), replace=False)
    te_idx = rng.choice(len(te_list), min(n_test, len(te_list)), replace=False)
    Y_tc, X_tc, S_tc = _convert_to_arrays([tc_list[i] for i in tc_idx])
    Y_te, X_te, S_te = _convert_to_arrays([te_list[i] for i in te_idx])

    # Split TrainCal -> D_tr / D_cal (no D_ACI in the last-step regime).
    n_tc = len(Y_tc)
    perm = rng.permutation(n_tc)
    n_cal = int(n_tc * cal_frac)
    cal_i, tr_i = perm[:n_cal], perm[n_cal:]

    # Flattened prefix features, z-scored on D_tr stats.
    F_tr_raw = featurize_xall(Y_tc[tr_i], X_tc[tr_i], S_tc[tr_i])
    mu, sd = F_tr_raw.mean(0), F_tr_raw.std(0) + 1e-8
    F_tr = (F_tr_raw - mu) / sd
    F_cal = (featurize_xall(Y_tc[cal_i], X_tc[cal_i], S_tc[cal_i]) - mu) / sd
    F_test = (featurize_xall(Y_te, X_te, S_te) - mu) / sd

    # Predictor: ridge of the final-hour NaCl on the prefix.
    y_tr = Y_tc[tr_i][:, -1, 0]
    model = LastStepRidge().fit(F_tr, y_tr)
    cal_pred, cal_true = model.predict(F_cal), Y_tc[cal_i][:, -1, 0]
    test_pred, test_true = model.predict(F_test), Y_te[:, -1, 0]

    feat = ((lambda F: np.zeros((len(np.asarray(F)), 1))) if mode == "uniform"
            else (lambda F: np.asarray(F)))

    algo = WeightedCAFHTLastStep(alpha=alpha, featurize_fn=feat, verbose=verbose)
    bands = algo.predict_bands((cal_pred, cal_true), (test_pred, test_true),
                               X_tr=F_tr, X_cal=F_cal, X_test=F_test, seed=seed)

    low, high = bands[:, 0, 0, 0], bands[:, 0, 1, 0]
    covered = (test_true >= low) & (test_true <= high)
    widths = high - low
    finite = np.isfinite(widths)
    return {
        "regime": "last_step", "domain": "medical", "mode": mode,
        "alpha": alpha, "seed": int(seed),
        "n_tr": len(tr_i), "n_cal": len(cal_i), "n_test": len(te_idx),
        "n_inf": int(algo.n_inf_),
        "coverage": float(covered.mean()),
        "overall_coverage": float(covered.mean()),
        "mean_width": float(widths[finite].mean()) if finite.any() else float("inf"),
        "target_coverage": 1.0 - alpha,
    }


def main():
    p = argparse.ArgumentParser(description="Medical last-step runner (Alg. 2)")
    p.add_argument("--pkl", default="medical/sepsis_experiment_data_nacl_target.pkl")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--n_traincal", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--cal_frac", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_json", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    res = run_single(args.pkl, seed=args.seed, mode=args.mode,
                     n_traincal=args.n_traincal, n_test=args.n_test,
                     cal_frac=args.cal_frac, alpha=args.alpha, verbose=args.verbose)
    print(f"[medical/last/{args.mode}] coverage={res['coverage']:.3f} "
          f"mean_width={res['mean_width']:.2f} mL/hr n_inf={res['n_inf']}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
