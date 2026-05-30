"""
synthetic/multi_seed_synthetic_last.py — 30-seed wrapper around
synthetic_runner_last (Algorithm 2, last-step).

Runs the single-seed runner over n_seeds independent seeds and aggregates the
last-step coverage and band width. Writes one JSON with the schema the v2 table
builder (Step 9) consumes.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthetic.synthetic_runner_last import run_single, _default_params  # noqa: E402


def _se(x):
    x = np.asarray(x, float)
    return float(np.std(x) / np.sqrt(len(x))) if len(x) else float("nan")


def run_multi(n_seeds, base_seed, covariate_mode, with_shift, mode, T,
              n_tr, n_cal, n_test, alpha, params):
    per_seed = [
        run_single(seed=base_seed + k, covariate_mode=covariate_mode,
                   with_shift=with_shift, mode=mode, T=T, n_tr=n_tr, n_cal=n_cal,
                   n_test=n_test, alpha=alpha, params=params, verbose=False)
        for k in range(n_seeds)
    ]
    cov = [r["coverage"] for r in per_seed]
    widths = [r["mean_width"] for r in per_seed]
    finite_w = [w for w in widths if np.isfinite(w)]

    return {
        "regime": "last_step", "domain": "synthetic", "mode": mode,
        "covariate_mode": covariate_mode, "with_shift": bool(with_shift),
        "alpha": alpha, "T": int(T), "n_seeds": int(n_seeds),
        "base_seed": int(base_seed), "n_tr": n_tr, "n_cal": n_cal,
        "n_test": n_test, "target_coverage": 1.0 - alpha,
        "per_seed_coverage": cov,
        "per_seed_width": widths,
        "per_seed_n_inf": [r["n_inf"] for r in per_seed],
        "coverage_mean": float(np.mean(cov)), "coverage_std": float(np.std(cov)),
        "coverage_se": _se(cov),
        "width_mean": float(np.mean(finite_w)) if finite_w else float("inf"),
        "width_std": float(np.std(finite_w)) if finite_w else float("inf"),
        "width_se": _se(finite_w) if finite_w else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description="Multi-seed synthetic last-step (Alg. 2)")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--covariate_mode", choices=["static", "dynamic"], default="static")
    p.add_argument("--with_shift", action="store_true")
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--base_seed", type=int, default=1000)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_tr", type=int, default=300)
    p.add_argument("--n_cal", type=int, default=300)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--covar_rate", type=float, default=1.0)
    p.add_argument("--covar_rate_shift", type=float, default=2.0)
    p.add_argument("--x_rate", type=float, default=0.6)
    p.add_argument("--x_rate_shift", type=float, default=0.9)
    p.add_argument("--save_json", type=str, default=None)
    args = p.parse_args()

    params = _default_params(args.covariate_mode)
    params["covar_rate"], params["covar_rate_shift"] = args.covar_rate, args.covar_rate_shift
    if args.covariate_mode == "dynamic":
        params["x_rate"], params["x_rate_shift"] = args.x_rate, args.x_rate_shift

    agg = run_multi(args.n_seeds, args.base_seed, args.covariate_mode, args.with_shift,
                    args.mode, args.T, args.n_tr, args.n_cal, args.n_test,
                    args.alpha, params)
    print(f"[{args.mode}/{args.covariate_mode}/"
          f"{'shift' if args.with_shift else 'noshift'}] {args.n_seeds} seeds: "
          f"cov={agg['coverage_mean']:.3f}±{agg['coverage_se']:.3f} "
          f"width={agg['width_mean']:.3f} target={agg['target_coverage']:.2f}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
