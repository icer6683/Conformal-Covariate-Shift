#!/usr/bin/env python3
"""
=============================================================================
FINANCE CONFORMAL PREDICTION
=============================================================================

Runs AdaptedCAFHT conformal prediction on real S&P 500 data loaded via
finance_data.py.

The only difference from the synthetic experiments is that fit_ar_model()
is replaced by fit_linear_model(), which predicts:

    Close_t ≈ β₀ + β₁·Open_t + β₂·OvernightGap_t + β₃·Volume_lag1_t + ...

Sector-based split:
  - Test tickers   : all stocks in --test_sector (e.g. "Technology")
  - Train tickers  : 1 - cal_frac of the remaining non-test stocks
  - Cal tickers    : cal_frac of the remaining non-test stocks

USAGE:
  # First pull data (once):
  python finance_data.py --pull --start 2024-01-01 --end 2024-04-01

  # Then run:
  python finance_conformal.py --npz sp500_20240102_20240328.npz --test_sector Technology
  python finance_conformal.py --npz sp500_20240102_20240328.npz --test_sector Healthcare --alpha 0.05

=============================================================================
"""
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from finance_data import load_stored
from algorithm import AdaptedCAFHT

GAMMA_GRID = [0.001, 0.005, 0.01, 0.05]

class LinearCovariateModel:
    def __init__(self, cov_names):
        self.cov_names = cov_names
        self.beta = None
        self.noise_std = 1.0

    def fit(self, Y_train, X_train):
        n, L, n_cov = X_train.shape
        y = Y_train[:, :, 0].reshape(-1)
        X = X_train.reshape(-1, n_cov)
        X_design = np.hstack([np.ones((len(y), 1)), X])
        self.beta, *_ = np.linalg.lstsq(X_design, y, rcond=None)
        resid = y - X_design @ self.beta
        self.noise_std = float(np.std(resid, ddof=X_design.shape[1]))
        print(f"  [Model] Fitted on {n} series x {L} steps")
        print(f"  [Model] Residual std : {self.noise_std:.4f}")
        print(f"  [Model] Coefficients :")
        print(f"            intercept = {self.beta[0]:.4f}")
        for name, coef in zip(self.cov_names, self.beta[1:]):
            print(f"            {name:22s} = {coef:.4f}")

    def predict(self, x_t):
        return float(self.beta[0] + x_t @ self.beta[1:])

def _select_gamma(Y_train, X_train, linear_model, base_alpha, t_max, gamma_grid, seed=0):
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
    idx2 = perm[n1 : n1 + n2]
    idx3 = perm[n1 + n2 :]
    Y_cal_sel = Y_train[idx2]
    X_cal_sel = X_train[idx2]
    Y_eval    = Y_train[idx3]
    X_eval    = X_train[idx3]
    n_eval     = Y_eval.shape[0]
    L          = Y_train.shape[1]
    horizon    = min(t_max, L - 1)
    start_eval = max(0, horizon // 2)
    target     = 1.0 - base_alpha
    scores = {}
    for gamma in gamma_grid:
        gamma = float(gamma)
        cal_scores = []
        for i in range(len(idx2)):
            for t in range(L):
                y_true = float(Y_cal_sel[i, t, 0])
                y_pred = linear_model.predict(X_cal_sel[i, t, :])
                cal_scores.append(abs(y_true - y_pred))
        predictor = AdaptedCAFHT(alpha=base_alpha)
        predictor.noise_std  = linear_model.noise_std
        predictor._scores    = np.array(cal_scores, dtype=float)
        predictor._weights   = np.ones(len(cal_scores), dtype=float)
        predictor._q         = None
        alpha_series = np.full(n_eval, base_alpha, dtype=float)
        cov_hist = []
        for t in range(horizon + 1):
            alpha_used = alpha_series.copy()
            alpha_next = alpha_series.copy()
            step_cov = []
            for i in range(n_eval):
                x_t    = X_eval[i, t, :]
                y_true = float(Y_eval[i, t, 0])
                y_pred = linear_model.predict(x_t)
                a = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
                q = predictor._weighted_quantile(predictor._scores, predictor._weights, 1.0 - a)
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

def run_finance_experiment(result, test_sector, cal_frac=0.5, alpha=0.1, seed=42, gamma_grid=None):
    if gamma_grid is None:
        gamma_grid = GAMMA_GRID
    Y         = result["Y"]
    X         = result["X"]
    dates     = result["dates"]
    cov_names = result["cov_names"]
    meta      = result["meta"]
    tickers   = result["tickers"]
    n_series, L, _ = Y.shape
    test_mask = np.array([m["sector"].lower() == test_sector.lower() for m in meta])
    n_test    = int(test_mask.sum())
    n_other   = int((~test_mask).sum())
    if n_test == 0:
        available = sorted({m["sector"] for m in meta})
        raise ValueError(f"No tickers found for sector '{test_sector}'.\nAvailable sectors: {available}")
    if n_other == 0:
        raise ValueError("All tickers belong to the test sector - nothing left to train/cal on.")
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
    print()
    linear_model = LinearCovariateModel(cov_names)
    linear_model.fit(Y_train, X_train)
    predictor = AdaptedCAFHT(alpha=alpha)
    predictor.noise_std = linear_model.noise_std
    cal_scores = []
    for i in range(n_cal):
        for t in range(L):
            y_true = float(Y_cal[i, t, 0])
            y_pred = linear_model.predict(X_cal[i, t, :])
            cal_scores.append(abs(y_true - y_pred))
    predictor._scores  = np.array(cal_scores, dtype=float)
    predictor._weights = np.ones(len(cal_scores), dtype=float)
    predictor._q       = None
    print(f"  [Cal]  {n_cal} series x {L} steps = {len(cal_scores)} scores, "
          f"median = {np.median(predictor._scores):.4f}")
    print(f"  [Test] Running on {n_test} {test_sector} tickers x {L} steps...")
    alpha_t   = np.full(n_test, alpha, dtype=float)
    gamma_opt = float(gamma_grid[0])
    coverage_by_time  = []
    width_by_time     = []
    all_covered       = []
    gamma_opt_history = []
    first_true  = []
    first_lower = []
    first_upper = []
    for t in range(L):
        if t > 0 and (t % 10 == 0):
            sel_seed = seed + 10000 + t
            gamma_opt, gamma_scores = _select_gamma(
                Y_train=Y_train, X_train=X_train, linear_model=linear_model,
                base_alpha=alpha, t_max=t, gamma_grid=gamma_grid, seed=sel_seed,
            )
            scores_str = "  ".join(
                f"gamma={g:.3f}->{v:.3f}" for g, v in gamma_scores.items() if np.isfinite(v)
            )
            print(f"  [gamma sel t={t:3d}]  best gamma = {gamma_opt}   ({scores_str})")
        gamma_opt_history.append(float(gamma_opt))
        alpha_used = alpha_t.copy()
        alpha_next = alpha_t.copy()
        covered_t = []
        width_t   = []
        for i in range(n_test):
            x_t    = X_test[i, t, :]
            y_true = float(Y_test[i, t, 0])
            y_pred = linear_model.predict(x_t)
            a = float(np.clip(alpha_used[i], 1e-6, 1 - 1e-6))
            q = predictor._weighted_quantile(predictor._scores, predictor._weights, 1.0 - a)
            lo = y_pred - q
            hi = y_pred + q
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
        "dates":             [str(d) for d in dates],
        "gamma_opt_history": gamma_opt_history,
        "first_test_ticker": test_tickers[0],
        "first_test_series": {"true": first_true, "lower": first_lower, "upper": first_upper},
        "config": {
            "test_sector": test_sector,
            "n_train":     int(n_train),
            "n_cal":       int(n_cal),
            "n_test":      int(n_test),
            "L":           int(L),
            "alpha":       alpha,
            "gamma_grid":  [float(g) for g in gamma_grid],
            "seed":        seed,
            "cov_names":   cov_names,
        },
    }

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
    axes[0,0].plot(x, cov_t, 'b-', linewidth=1.5, label='Empirical coverage')
    axes[0,0].axhline(target, color='red', linestyle='--', linewidth=2, label=f'Target ({target:.0%})')
    axes[0,0].set_ylim(0.5, 1.05)
    axes[0,0].set_ylabel('Coverage rate')
    axes[0,0].set_title(f'Coverage over Time  ({cfg["test_sector"]} sector)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].text(0.02, 0.05,
        f"Overall: {results['overall_coverage']:.3f}  (error {results['overall_coverage'] - target:+.3f})",
        transform=axes[0].transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    axes[0,1].plot(x, width_t, 'g-', linewidth=1.5)
    axes[0,1].set_ylabel('Mean interval width')
    axes[0,1].set_title('Prediction Interval Width over Time')
    axes[0,1].grid(True, alpha=0.3)
    axes[1,0].fill_between(x, first["lower"], first["upper"], alpha=0.25, color='steelblue', label='Prediction interval')
    axes[1,0].plot(x, first["true"], 'k-', linewidth=1.5, label='Actual close')
    axes[1,0].set_ylabel('Price ($)')
    axes[1,0].set_title(f'{ticker} - Actual Close vs. Prediction Interval')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,1].plot(x, gammas, drawstyle='steps-post', color='purple', linewidth=1.5)
    axes[1,1].set_ylabel('Selected gamma (log scale)')
    axes[1,1].set_title('ACI Gamma Selected over Time')
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    tick_every = max(1, len(dates) // 10)
    tick_pos   = x[::tick_every]
    tick_lbl   = [dates[i] for i in tick_pos]
    for ax in axes:
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Date')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  [Plot] Saved to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run AdaptedCAFHT conformal prediction on S&P 500 finance data.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--json", default=None)
    parser.add_argument("--test_sector", required=True, help="Sector to hold out as test set, e.g. 'Technology'")
    parser.add_argument("--cal_frac", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--gamma_grid", type=float, nargs='+', default=GAMMA_GRID)
    parser.add_argument("--save_plot", default=None)
    parser.add_argument("--save_json", default=None)
    args = parser.parse_args()
    npz_path  = Path(args.npz)
    json_path = Path(args.json) if args.json else npz_path.with_suffix(".json")
    print(f"Loading {npz_path} ...")
    result = load_stored(npz_path, json_path)
    results = run_finance_experiment(
        result=result, test_sector=args.test_sector, cal_frac=args.cal_frac,
        alpha=args.alpha, seed=args.seed, gamma_grid=args.gamma_grid,
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
