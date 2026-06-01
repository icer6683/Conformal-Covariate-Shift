"""
finance/finance_runner_last.py — per-window runner for Algorithm 2 (last-step
coverage) on the S&P 500 data.

Self-contained (§ 0). The loader and sector filters come from
finance.finance_data (STAY file). The only cross-module import for the algorithm
is core.weighted_cafht_last.

Mapping to WEIGHTED_CAFHT_PLAN.md § 3:
  - Predictor (§ 3.2): `LastStepRidge` — ridge (λ by RidgeCV's efficient LOO) of
    the FINAL-day return Y_{T+1} on the flattened covariate prefix X_{1:T}
    (4·(L-1) features); falls back to a fixed-λ normal-equation solve without
    sklearn. RidgeCV handles the high-dim prefix fine, so the predictor keeps it.
  - LR featurizer (§ 3.3): DECOUPLED from the predictor. The flattened prefix
    (4·(L-1) ≈ 156–480 features) over only ~200 source series let the classifier
    separate the classes perfectly → p̂→0/1 → ŵ=p̂/(1−p̂) blew up → ~⅓ of test
    points hit the δ_∞ atom (unbounded bands). Instead the classifier uses a
    compact, horizon-independent summary `featurize_clf(X)` = per-covariate
    {mean, std, last} (4×3 = 12 features). Covariate shift is a change in the
    covariates' DISTRIBUTION, which these moments capture, while keeping
    features ≪ samples so the weights stay bounded.
  - Split: source sectors -> D_tr / D_cal; test sector -> D_test.
    No D_ACI / no γ / no ACI.

Comparison (§ 5.1): full ("our version", LR) vs uniform ("no-LR").
Headline metrics: last-step coverage and mean band width.

Test sector = target (shifted); the remaining sectors = source. --mixed draws a
random test fraction from all sectors (null / no-shift baseline).
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.weighted_cafht_last import WeightedCAFHTLastStep            # noqa: E402
from finance.finance_data import load_stored                         # noqa: E402

try:
    from sklearn.linear_model import RidgeCV
    _SKLEARN = True
except Exception:                                                    # pragma: no cover
    _SKLEARN = False


# =============================================================================
# Predictor (§ 3.2): ridge of the final-day return on the flattened prefix
# =============================================================================
def featurize_xall(X):
    """PREDICTOR feature = the flattened covariate prefix X_{1:T} (4·(L-1)).
    Predicts the FINAL-day return; the target value itself is never included.
    RidgeCV regularizes this high-dim design, so the predictor keeps it."""
    n, L, _ = X.shape
    return X[:, :L - 1, :].reshape(n, -1)             # (n, (L-1)*n_cov)


def featurize_clf(X):
    """CLASSIFIER feature for the LR density-ratio weights — compact and
    horizon-independent: per-covariate mean, std, and last value over the
    trajectory (4 covariates × 3 = 12). Day 0 (structurally zero for 3 of 4
    covariates) is excluded from the mean/std so it doesn't bias the moments;
    `last` is the final fully-populated day. Keeping dim ≪ n_source stops the
    classifier from perfectly separating the classes, so ŵ stays bounded and
    the δ_∞ atom rarely fires."""
    Xt = X[:, 1:, :]                                  # skip structurally-zero day 0
    return np.hstack([Xt.mean(axis=1), Xt.std(axis=1), Xt[:, -1, :]])  # (n, 12)


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
# Single-window experiment
# =============================================================================
def run_single(npz, json_path=None, test_sector="Technology", mode="full",
               cal_frac=0.5, alpha=0.1, seed=42, mixed=False,
               mixed_test_frac=0.15, verbose=False, data=None):
    """Run one rolling window; `mode` ∈ {full, uniform}."""
    data = data if data is not None else load_stored(npz, json_path)
    Y, X = np.asarray(data["Y"], float), np.asarray(data["X"], float)
    meta = data["meta"]
    n_series = len(Y)
    rng = np.random.default_rng(seed)

    # ── Test / source split (target = test sector; source = the rest) ─────────
    if mixed:
        n_test = max(10, int(n_series * mixed_test_frac))
        test_mask = np.zeros(n_series, dtype=bool)
        test_mask[rng.permutation(n_series)[:n_test]] = True
        display_sector = "Mixed (all sectors)"
    else:
        test_mask = np.array(
            [m["sector"].lower() == test_sector.lower() for m in meta])
        if test_mask.sum() == 0:
            available = sorted({m["sector"] for m in meta})
            raise ValueError(
                f"No tickers for sector '{test_sector}'. Available: {available}")
        display_sector = test_sector
    if (~test_mask).sum() == 0:
        raise ValueError("All tickers belong to the test set.")

    test_idx = np.where(test_mask)[0]
    # Split SOURCE pool -> D_tr / D_cal (no D_ACI in the last-step regime).
    src_idx = rng.permutation(np.where(~test_mask)[0])
    n_cal = int(len(src_idx) * cal_frac)
    cal_idx, tr_idx = src_idx[:n_cal], src_idx[n_cal:]

    # ── Predictor features: flattened prefix, z-scored on D_tr stats ──────────
    P_tr_raw = featurize_xall(X[tr_idx])
    pmu, psd = P_tr_raw.mean(0), P_tr_raw.std(0) + 1e-8
    P_tr = (P_tr_raw - pmu) / psd
    P_cal = (featurize_xall(X[cal_idx]) - pmu) / psd
    P_test = (featurize_xall(X[test_idx]) - pmu) / psd

    # ── Predictor: ridge of the final-day return on the prefix ────────────────
    y_tr = Y[tr_idx][:, -1, 0]
    model = LastStepRidge().fit(P_tr, y_tr)
    cal_pred, cal_true = model.predict(P_cal), Y[cal_idx][:, -1, 0]
    test_pred, test_true = model.predict(P_test), Y[test_idx][:, -1, 0]

    # ── Classifier features: compact per-covariate moments, z-scored on D_tr ──
    C_tr_raw = featurize_clf(X[tr_idx])
    cmu, csd = C_tr_raw.mean(0), C_tr_raw.std(0) + 1e-8
    C_tr = (C_tr_raw - cmu) / csd
    C_cal = (featurize_clf(X[cal_idx]) - cmu) / csd
    C_test = (featurize_clf(X[test_idx]) - cmu) / csd

    feat = ((lambda F: np.zeros((len(np.asarray(F)), 1))) if mode == "uniform"
            else (lambda F: np.asarray(F)))

    algo = WeightedCAFHTLastStep(alpha=alpha, featurize_fn=feat, verbose=verbose)
    bands = algo.predict_bands((cal_pred, cal_true), (test_pred, test_true),
                               X_tr=C_tr, X_cal=C_cal, X_test=C_test, seed=seed)

    low, high = bands[:, 0, 0, 0], bands[:, 0, 1, 0]
    covered = (test_true >= low) & (test_true <= high)
    widths = high - low
    finite = np.isfinite(widths)
    return {
        "regime": "last_step", "domain": "finance", "mode": mode,
        "test_sector": display_sector, "mixed": bool(mixed),
        "npz": str(npz), "alpha": alpha, "seed": int(seed),
        "n_tr": len(tr_idx), "n_cal": len(cal_idx), "n_test": len(test_idx),
        "n_inf": int(algo.n_inf_),
        "coverage": float(covered.mean()),
        "overall_coverage": float(covered.mean()),
        "mean_width": float(widths[finite].mean()) if finite.any() else float("inf"),
        "target_coverage": 1.0 - alpha,
    }


def main():
    p = argparse.ArgumentParser(description="Finance last-step runner (Alg. 2)")
    p.add_argument("--npz", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--test_sector", default="Technology",
                   help="Sector held out as test set (ignored when --mixed is set).")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--cal_frac", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed", action="store_true", default=False,
                   help="Random test draw from all sectors (null / no-shift baseline).")
    p.add_argument("--mixed_test_frac", type=float, default=0.15)
    p.add_argument("--save_json", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    res = run_single(args.npz, json_path=args.json, test_sector=args.test_sector,
                     mode=args.mode, cal_frac=args.cal_frac, alpha=args.alpha,
                     seed=args.seed, mixed=args.mixed,
                     mixed_test_frac=args.mixed_test_frac, verbose=args.verbose)
    print(f"[finance/last/{args.mode}] sector={res['test_sector']} "
          f"coverage={res['coverage']:.3f} "
          f"mean_width={res['mean_width']:.4f} n_inf={res['n_inf']}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
