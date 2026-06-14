"""
synthetic/multi_seed_synthetic_whole.py — 30-seed wrapper around
synthetic_runner_whole (Algorithm 1, whole-trajectory).

Runs the single-seed runner over n_seeds independent seeds and aggregates the
whole-trajectory metrics: per-seed JOINT coverage (the regime's target) and
band width, plus the per-step coverage/width profiles averaged across seeds.
Writes one JSON with the schema the v2 table builder (Step 9) consumes.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from synthetic.synthetic_runner_whole import run_single, _default_params  # noqa: E402


def _se(x):
    x = np.asarray(x, float)
    return float(np.std(x) / np.sqrt(len(x))) if len(x) else float("nan")


def run_multi(n_seeds, base_seed, covariate_mode, with_shift, mode, T,
              n_tr, n_aci, n_cal, n_test, alpha, params, revised=False):
    per_seed = []
    for k in range(n_seeds):
        per_seed.append(run_single(
            seed=base_seed + k, covariate_mode=covariate_mode, with_shift=with_shift,
            mode=mode, T=T, n_tr=n_tr, n_aci=n_aci, n_cal=n_cal, n_test=n_test,
            alpha=alpha, params=params, verbose=False, revised=revised))

    joint = [r["joint_coverage"] for r in per_seed]
    pooled = [r["pooled_coverage"] for r in per_seed]
    widths = [r["mean_width"] for r in per_seed]
    finite_w = [w for w in widths if np.isfinite(w)]
    cov_by_time = np.array([r["coverage_by_time"] for r in per_seed])   # (S, T)
    # width_by_time may contain inf; mean over finite per column.
    wbt = np.array([r["width_by_time"] for r in per_seed], float)       # (S, T)

    return {
        "regime": "whole_trajectory", "domain": "synthetic", "mode": mode,
        "covariate_mode": covariate_mode, "with_shift": bool(with_shift),
        "revised": bool(revised),
        "alpha": alpha, "T": int(T), "n_seeds": int(n_seeds),
        "base_seed": int(base_seed), "n_tr": n_tr, "n_aci": n_aci,
        "n_cal": n_cal, "n_test": n_test, "target_coverage": 1.0 - alpha,
        # per-seed series
        "per_seed_coverage": joint,                 # JOINT coverage per seed
        "per_seed_pooled_coverage": pooled,
        "per_seed_width": widths,
        "per_seed_gamma_opt": [r["gamma_opt"] for r in per_seed],
        "per_seed_n_inf": [r["n_inf"] for r in per_seed],
        # aggregates (coverage = the whole-trajectory JOINT rate)
        "coverage_mean": float(np.mean(joint)), "coverage_std": float(np.std(joint)),
        "coverage_se": _se(joint),
        "pooled_coverage_mean": float(np.mean(pooled)),
        "width_mean": float(np.mean(finite_w)) if finite_w else float("inf"),
        "width_std": float(np.std(finite_w)) if finite_w else float("inf"),
        "width_se": _se(finite_w) if finite_w else float("nan"),
        # per-step profiles averaged across seeds
        "coverage_by_time_mean": cov_by_time.mean(axis=0).tolist(),
        "width_by_time_mean": [
            float(np.nanmean(np.where(np.isfinite(wbt[:, t]), wbt[:, t], np.nan)))
            for t in range(wbt.shape[1])
        ],
    }


def main():
    p = argparse.ArgumentParser(description="Multi-seed synthetic whole-traj (Alg. 1)")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--covariate_mode", choices=["static", "dynamic"], default="static")
    p.add_argument("--with_shift", action="store_true")
    p.add_argument("--n_seeds", type=int, default=30)
    p.add_argument("--base_seed", type=int, default=1000)
    p.add_argument("--T", type=int, default=20)
    p.add_argument("--n_tr", type=int, default=300)
    p.add_argument("--n_aci", type=int, default=150)
    p.add_argument("--n_cal", type=int, default=300)
    p.add_argument("--n_test", type=int, default=200)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--covar_rate", type=float, default=1.0)
    p.add_argument("--covar_rate_shift", type=float, default=2.0)
    p.add_argument("--x_rate", type=float, default=0.6)
    p.add_argument("--x_rate_shift", type=float, default=0.9)
    p.add_argument("--save_json", type=str, default=None)
    p.add_argument("--revised", action="store_true")
    args = p.parse_args()

    params = _default_params(args.covariate_mode)
    params["covar_rate"], params["covar_rate_shift"] = args.covar_rate, args.covar_rate_shift
    if args.covariate_mode == "dynamic":
        params["x_rate"], params["x_rate_shift"] = args.x_rate, args.x_rate_shift

    agg = run_multi(args.n_seeds, args.base_seed, args.covariate_mode, args.with_shift,
                    args.mode, args.T, args.n_tr, args.n_aci, args.n_cal, args.n_test,
                    args.alpha, params, revised=args.revised)
    print(f"[{args.mode}/{args.covariate_mode}/"
          f"{'shift' if args.with_shift else 'noshift'}"
          f"{'/REVISED' if args.revised else ''}] {args.n_seeds} seeds: "
          f"joint_cov={agg['coverage_mean']:.3f}±{agg['coverage_se']:.3f} "
          f"width={agg['width_mean']:.3f} target={agg['target_coverage']:.2f}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
