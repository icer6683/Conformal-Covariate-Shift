"""
finance/finance_runner_whole.py — per-window runner for Algorithm 1
(whole-trajectory coverage) on the S&P 500 data.

Self-contained (§ 0). The `LinearCovariateModel` (per-step contemporaneous OLS
of Y_t on the 4 covariates X_t, pooled over all series and steps) is COPIED —
not imported — from OLD_finance_conformal.py per § B.3. The loader and sector
filters come from finance.finance_data (STAY file). The only cross-module import
for the algorithm is core.weighted_cafht_whole.

Mapping to WEIGHTED_CAFHT_PLAN.md § 3:
  - Predictor (§ 3.1): `LinearCovariateModel` — Y_t ~ b0 + X_t·b, one global OLS
    pooled over (series, step) pairs in D_tr; applied per step so Ŷ_t uses the
    observed covariates X_t at that step.
  - t=0 truncation: the first observed day has 3 of 4 covariates structurally
    zero (OvernightGap + both lag-1 terms are undefined on day 1). Rather than
    band a step with no real covariates, t=0 is dropped EVERYWHERE — the OLS
    fit, the per-step predictions, and the whole-trajectory coverage all run
    over steps 1..L-1 only (horizon = L-1).
  - LR featurizer (§ 3.3): `featurize_x1(X)` = the 4 covariates at t=1. The
    classifier gets one shot (a single snapshot per series), so it must read a
    fully-populated step; t=1 is the first such step, and with t=0 truncated it
    is also the first modeled step — the genuine X_1 of the trajectory.
  - Four-way split (§ 2.0): D_ACI is peeled off the SOURCE pool (all non-test
    sectors) before the D_tr / D_cal split; D_test is the held-out sector (or a
    random draw under --mixed).

Comparison (§ 5.1): full ("our version", LR + ACI) vs uniform ("no-LR", ACI).
Headline metrics: whole-trajectory JOINT coverage and mean band width.

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
from core.weighted_cafht_whole import WeightedCAFHTWholeTrajectory  # noqa: E402
from core.weighted_cafht_whole_revised import (                     # noqa: E402
    WeightedCAFHTWholeTrajectoryRevised)                            # experimental
from finance.finance_data import load_stored                        # noqa: E402

GAMMA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]   # finance whole-traj default grid


# =============================================================================
# Predictor — copied from OLD_finance_conformal.py:LinearCovariateModel (§ B.3)
# =============================================================================
class LinearCovariateModel:
    """Contemporaneous OLS:  Y_t ~ b0 + X_t·b, fit once on all (series, step)
    pairs of D_tr. No autoregression on Y (intraday returns are ~unpredictable
    from their own past); the 4 covariates carry the signal."""

    def __init__(self, cov_names):
        self.cov_names = cov_names
        self.beta = None
        self.noise_std = 1.0

    def fit(self, Y_train, X_train):
        n, L, n_cov = X_train.shape
        if L < 2:
            return
        # t=0 is truncated from the whole-trajectory method (its covariates are
        # structurally zero), so it is also excluded from the OLS fit.
        y = Y_train[:, 1:, 0].reshape(-1)
        X = X_train[:, 1:, :].reshape(-1, n_cov)
        design = np.hstack([np.ones((len(y), 1)), X])
        self.beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ self.beta
        self.noise_std = float(np.std(resid, ddof=design.shape[1]))


def _predict_trajectory(beta, X):
    """Per-step contemporaneous predictions Ŷ_1..Ŷ_{L-1} (n, L-1, 1) using the
    observed covariates X_t at each step. t=0 is truncated (its covariates are
    structurally zero), so the whole-trajectory coverage runs over steps
    1..L-1 only."""
    n, L, _ = X.shape
    preds = np.zeros((n, L - 1, 1))
    for t in range(1, L):
        preds[:, t - 1, 0] = beta[0] + X[:, t, :] @ beta[1:]
    return preds


# =============================================================================
# LR featurizer (§ 3.3): X_1 = covariates at t=1 — first fully-populated step
# =============================================================================
def featurize_x1(X):
    """Whole-trajectory classifier feature = the 4 covariates at t=1.

    The classifier gets a single snapshot per series ("one shot"), so it must
    read a step where ALL covariates are populated. t=0 has 3 of 4 covariates
    structurally zero (OvernightGap + both lag-1 terms undefined on the first
    observed day); t=1 is the first fully-populated step. With t=0 truncated
    from the predictor/coverage, t=1 is also the first modeled step, so this is
    the genuine X_1 of the trajectory the method actually runs on."""
    return X[:, 1, :]                                  # (n, 4)


def featurize_prefix(Y, X, horizon):
    """REVISED algo (REVISED_WHOLE_TRAJECTORY.md): per-step classifier features
    (n, horizon, d). Column h summarizes the observed prefix at the point of
    predicting Y_{h+1}: the covariate prefix X_{1:h+1} (steps 1..h+1 — skips the
    structurally-zero day 0, includes the contemporaneous X_{h+1}) and the TARGET
    prefix Y_{0:h} (excludes Y_{h+1}, the value being predicted). Per-channel
    {mean, std, last}; d = n_cov·3 + 3."""
    Yc = np.asarray(Y, float)[:, :, 0]                 # (n, L)
    X = np.asarray(X, float)                           # (n, L, n_cov)
    n, L, n_cov = X.shape
    feats = np.zeros((n, horizon, n_cov * 3 + 3))
    for h in range(horizon):
        xp = X[:, 1:h + 2, :]                          # X_{1:h+1} (incl. contemporaneous)
        yp = Yc[:, :h + 1]                             # Y_{0:h} (excludes target)
        feats[:, h, :] = np.concatenate([
            xp.mean(axis=1), xp.std(axis=1), xp[:, -1, :],
            yp.mean(axis=1, keepdims=True), yp.std(axis=1, keepdims=True),
            yp[:, -1:],
        ], axis=1)
    return feats


# =============================================================================
# Single-window experiment
# =============================================================================
def run_single(npz, json_path=None, test_sector="Technology", mode="full",
               cal_frac=0.5, frac_aci=0.15, alpha=0.1, seed=42,
               gamma_grid=None, gamma_split=None, mixed=False, mixed_test_frac=0.15,
               verbose=False, data=None, revised=False):
    """Run one rolling window; `mode` ∈ {full, uniform}. `data` may be a
    preloaded result dict (so a caller can load the .npz once).
    `revised=True` uses the experimental per-step-classifier Algorithm 1."""
    gamma_grid = gamma_grid if gamma_grid is not None else GAMMA_GRID
    data = data if data is not None else load_stored(npz, json_path)
    Y, X = np.asarray(data["Y"], float), np.asarray(data["X"], float)
    meta, cov_names = data["meta"], data["cov_names"]
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
    # Four-way split of the SOURCE pool: peel D_ACI, then split rest tr / cal.
    src_idx = rng.permutation(np.where(~test_mask)[0])
    n_aci = int(frac_aci * len(src_idx))
    aci_idx, rest = src_idx[:n_aci], src_idx[n_aci:]
    n_cal = int(len(rest) * cal_frac)
    cal_idx, tr_idx = rest[:n_cal], rest[n_cal:]

    Y_tr, X_tr = Y[tr_idx], X[tr_idx]
    Y_cal, X_cal = Y[cal_idx], X[cal_idx]
    Y_aci, X_aci = Y[aci_idx], X[aci_idx]
    Y_te, X_te = Y[test_idx], X[test_idx]

    # ── Predictor: contemporaneous OLS fit on D_tr, applied to every subset ───
    model = LinearCovariateModel(cov_names)
    model.fit(Y_tr, X_tr)
    b = model.beta
    # truncate t=0 from the truth arrays to match _predict_trajectory (steps 1..L-1).
    tr_pred, tr_true = _predict_trajectory(b, X_tr), Y_tr[:, 1:, :]
    cal_pred, cal_true = _predict_trajectory(b, X_cal), Y_cal[:, 1:, :]
    test_pred, test_true = _predict_trajectory(b, X_te), Y_te[:, 1:, :]
    aci_pred, aci_true = _predict_trajectory(b, X_aci), Y_aci[:, 1:, :]

    gs = tuple(gamma_split) if gamma_split is not None else (0.33, 0.33, 0.34)
    horizon = tr_true.shape[1]
    if revised:
        # Per-step classifiers on prefix features (z-scored on D_tr stats).
        # uniform => constant (zero) features => uniform per-step weights.
        if mode == "uniform":
            z = lambda Xr: np.zeros((len(np.asarray(Xr)), horizon, 1))
            Xc_tr, Xc_cal, Xc_te = z(X_tr), z(X_cal), z(X_te)
        else:
            Pt = featurize_prefix(Y_tr, X_tr, horizon)
            mu, sd = Pt.mean(axis=0), Pt.std(axis=0) + 1e-8
            Xc_tr = (Pt - mu) / sd
            Xc_cal = (featurize_prefix(Y_cal, X_cal, horizon) - mu) / sd
            Xc_te = (featurize_prefix(Y_te, X_te, horizon) - mu) / sd
        algo = WeightedCAFHTWholeTrajectoryRevised(
            alpha=alpha, gamma_grid=gamma_grid, featurize_fn=None,
            gamma_split=gs, verbose=verbose)
        bands = algo.predict_bands(
            (tr_pred, tr_true), (cal_pred, cal_true), (test_pred, test_true),
            (aci_pred, aci_true), Xc_tr, Xc_cal, Xc_te, seed=seed)
    else:
        # Original: one classifier on X_1 (day-2 covariates), z-scored on D_tr.
        F_tr_raw = featurize_x1(X_tr)
        mu, sd = F_tr_raw.mean(0), F_tr_raw.std(0) + 1e-8
        F_tr = (F_tr_raw - mu) / sd
        F_cal = (featurize_x1(X_cal) - mu) / sd
        F_test = (featurize_x1(X_te) - mu) / sd
        feat = ((lambda F: np.zeros((len(np.asarray(F)), 1))) if mode == "uniform"
                else (lambda F: np.asarray(F)))
        algo = WeightedCAFHTWholeTrajectory(
            alpha=alpha, gamma_grid=gamma_grid, featurize_fn=feat,
            gamma_split=gs, verbose=verbose)
        bands = algo.predict_bands(
            (tr_pred, tr_true), (cal_pred, cal_true), (test_pred, test_true),
            (aci_pred, aci_true), X_tr=F_tr, X_cal=F_cal, X_test=F_test, seed=seed)

    metrics = _coverage_metrics(bands, test_true, alpha)
    return {
        "regime": "whole_trajectory", "domain": "finance", "mode": mode,
        "revised": bool(revised),
        "test_sector": display_sector, "mixed": bool(mixed),
        "npz": str(npz), "alpha": alpha, "seed": int(seed),
        "n_tr": len(tr_idx), "n_aci": len(aci_idx), "n_cal": len(cal_idx),
        "n_test": len(test_idx), "horizon": int(test_true.shape[1]),
        "gamma_opt": algo.gamma_opt_, "n_inf": int(algo.n_inf_),
        "score_bank_shape": list(algo.score_bank_shape_), **metrics,
    }


def _coverage_metrics(bands, truth, alpha):
    """bands (n, H, 2, 1); truth (n, H, 1). Whole-trajectory JOINT coverage, but
    OMITTING the first ceil(H/10) steps — a trajectory counts as covered iff
    every step from the first non-omitted one to the end is covered (skips the
    ACI cold-start warm-up). Width over finite bands; δ_∞ bands counted as
    covered, excluded from width. The un-omitted joint rate is kept for reference."""
    low, high, y = bands[:, :, 0, 0], bands[:, :, 1, 0], truth[:, :, 0]
    covered = (y >= low) & (y <= high)
    widths = high - low
    finite = np.isfinite(widths)
    n_omit = int(np.ceil(covered.shape[1] / 10))      # first 1/10 (round up)
    joint = float(covered[:, n_omit:].all(axis=1).mean())
    return {
        "coverage_by_time": covered.mean(axis=0).tolist(),
        "joint_coverage": joint,                      # omit first 1/10
        "overall_coverage": joint,
        "joint_coverage_full": float(covered.all(axis=1).mean()),  # all steps, reference
        "n_omit": n_omit,
        "pooled_coverage": float(covered.mean()),
        "mean_width": float(widths[finite].mean()) if finite.any() else float("inf"),
        "target_coverage": 1.0 - alpha,
    }


def main():
    p = argparse.ArgumentParser(description="Finance whole-trajectory runner (Alg. 1)")
    p.add_argument("--npz", required=True)
    p.add_argument("--json", default=None)
    p.add_argument("--test_sector", default="Technology",
                   help="Sector held out as test set (ignored when --mixed is set).")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--cal_frac", type=float, default=0.5)
    p.add_argument("--frac_aci", type=float, default=0.15)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gamma_grid", type=float, nargs="+", default=GAMMA_GRID)
    p.add_argument("--gamma_split", type=float, nargs=3, default=None,
                   help="Internal D_tr 3-way split for γ selection (default 0.33 0.33 0.34).")
    p.add_argument("--mixed", action="store_true", default=False,
                   help="Random test draw from all sectors (null / no-shift baseline).")
    p.add_argument("--mixed_test_frac", type=float, default=0.15)
    p.add_argument("--save_json", default=None)
    p.add_argument("--revised", action="store_true")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    res = run_single(args.npz, json_path=args.json, test_sector=args.test_sector,
                     mode=args.mode, cal_frac=args.cal_frac, frac_aci=args.frac_aci,
                     alpha=args.alpha, seed=args.seed, gamma_grid=args.gamma_grid,
                     gamma_split=args.gamma_split, mixed=args.mixed,
                     mixed_test_frac=args.mixed_test_frac, verbose=args.verbose,
                     revised=args.revised)
    print(f"[finance/whole/{args.mode}{'/REVISED' if args.revised else ''}] "
          f"sector={res['test_sector']} "
          f"joint_cov={res['joint_coverage']:.3f} "
          f"pooled_cov={res['pooled_coverage']:.3f} "
          f"mean_width={res['mean_width']:.4f} "
          f"gamma_opt={res['gamma_opt']} n_inf={res['n_inf']}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
