"""
medical/multi_seed_medical_last.py — 10-seed wrapper around medical_runner_last
(Algorithm 2, last-step).

Subsamples n_traincal / n_test patients per seed, aggregates the final-step
coverage and mean band width, and writes one JSON. The pickle is loaded ONCE.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from medical.medical_runner_last import run_single, load_data            # noqa: E402


def _se(x):
    x = np.asarray(x, float)
    return float(np.std(x) / np.sqrt(len(x))) if len(x) else float("nan")


def run_multi(pkl, n_seeds, base_seed, mode, n_traincal, n_test, cal_frac, alpha):
    data = load_data(pkl)
    per_seed = [
        run_single(pkl, seed=base_seed + k, mode=mode, n_traincal=n_traincal,
                   n_test=n_test, cal_frac=cal_frac, alpha=alpha, data=data)
        for k in range(n_seeds)
    ]
    cov = [r["coverage"] for r in per_seed]
    widths = [r["mean_width"] for r in per_seed]
    fw = [w for w in widths if np.isfinite(w)]

    return {
        "regime": "last_step", "domain": "medical", "mode": mode,
        "alpha": alpha, "n_seeds": int(n_seeds), "base_seed": int(base_seed),
        "n_traincal": n_traincal, "n_test": n_test, "cal_frac": cal_frac,
        "target_coverage": 1.0 - alpha,
        "per_seed_coverage": cov, "per_seed_width": widths,
        "per_seed_n_inf": [r["n_inf"] for r in per_seed],
        "coverage_mean": float(np.mean(cov)), "coverage_std": float(np.std(cov)),
        "coverage_se": _se(cov),
        "width_mean": float(np.mean(fw)) if fw else float("inf"),
        "width_std": float(np.std(fw)) if fw else float("inf"),
        "width_se": _se(fw) if fw else float("nan"),
    }


def main():
    p = argparse.ArgumentParser(description="Multi-seed medical last-step (Alg. 2)")
    p.add_argument("--pkl", default="medical/sepsis_experiment_data_nacl_target.pkl")
    p.add_argument("--mode", choices=["full", "uniform"], default="full")
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument("--base_seed", type=int, default=1000)
    p.add_argument("--n_traincal", type=int, default=1000)
    p.add_argument("--n_test", type=int, default=500)
    p.add_argument("--cal_frac", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--save_json", default=None)
    args = p.parse_args()

    agg = run_multi(args.pkl, args.n_seeds, args.base_seed, args.mode,
                    args.n_traincal, args.n_test, args.cal_frac, args.alpha)
    print(f"[medical/last/{args.mode}] {args.n_seeds} seeds: "
          f"cov={agg['coverage_mean']:.3f}±{agg['coverage_se']:.3f} "
          f"width={agg['width_mean']:.1f} mL/hr target={agg['target_coverage']:.2f}")
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"saved -> {args.save_json}")


if __name__ == "__main__":
    main()
