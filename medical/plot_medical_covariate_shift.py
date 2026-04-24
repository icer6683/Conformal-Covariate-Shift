#!/usr/bin/env python3
"""
plot_medical_covariate_shift.py
================================
Illustrates the covariate shift between ICU patients in the TrainCal set
(no early Norepinephrine) and the Test set (early Norepinephrine) using the
sepsis pickle.

Shows KDE distributions for:
  - Heart Rate, Respiratory Rate, O2 saturation (dynamic covariates)
  - NaCl 0.9% target (what we are predicting)
  - Age (static covariate)
  - Gender and ethnicity proportions (bar charts)

Prints KL divergence (histogram-based) for each continuous variable.

Usage
-----
  python medical/plot_medical_covariate_shift.py \\
      --pkl medical/sepsis_experiment_data_nacl_target.pkl \\
      --save results/medical/pdf/medical_covariate_shift.pdf
"""

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIP_PCT = 1.0  # clip outermost 1% per variable for clean KDE tails

DYNAMIC_VARS = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
    "NaCl 0.9% (target)",
]

PRETTY = {
    "Heart Rate":                    "Heart Rate (bpm)",
    "Respiratory Rate":              "Respiratory Rate (breaths/min)",
    "O2 saturation pulseoxymetry":   "O2 Saturation (%)",
    "NaCl 0.9% (target)":            "NaCl 0.9% Dose (mL/hr)",
}

TRAINCAL_COLOR = "#2166ac"   # blue  — no early vasopressor
TEST_COLOR     = "#d6604d"   # red-orange — early Norepinephrine


def clip_arr(arr, pct=CLIP_PCT):
    lo = np.percentile(arr, pct)
    hi = np.percentile(arr, 100 - pct)
    return arr[(arr >= lo) & (arr <= hi)]


def kde_plot(ax, data, color, label, alpha_fill=0.25, lw=2.0):
    from scipy.stats import gaussian_kde
    kde  = gaussian_kde(data, bw_method="scott")
    x_lo = np.percentile(data, 0.5)
    x_hi = np.percentile(data, 99.5)
    xs   = np.linspace(x_lo, x_hi, 400)
    ys   = kde(xs)
    ax.plot(xs, ys, color=color, lw=lw, label=label)
    ax.fill_between(xs, ys, alpha=alpha_fill, color=color)


def hist_kl(a, b, bins=50):
    lo = min(np.percentile(a, 0.5), np.percentile(b, 0.5))
    hi = max(np.percentile(a, 99.5), np.percentile(b, 99.5))
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(a, bins=edges, density=True)
    q, _ = np.histogram(b, bins=edges, density=True)
    eps  = 1e-10
    p = p + eps;  p /= p.sum()
    q = q + eps;  q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def extract_arrays(patient_list, var_names):
    """Return dict var -> flattened 1-D array of all time-step values."""
    out = {v: [] for v in var_names}
    ages    = []
    genders = []
    ethnics = []
    for p in patient_list:
        for v in var_names:
            vals = p[v]["value"].to_numpy(dtype=np.float64)
            out[v].append(vals)
        ages.append(float(p["Age"]))
        genders.append(p["gender"])
        ethnics.append(p.get("ethnicity", "UNKNOWN"))
    return {v: np.concatenate(out[v]) for v in var_names}, np.array(ages), genders, ethnics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl",  required=True,
                        help="Path to sepsis_experiment_data_nacl_target.pkl")
    parser.add_argument("--save", default=None,
                        help="Output path (PNG or PDF)")
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    tc_list = data["patient_trajectory_list_traincal"]
    te_list = data["patient_trajectory_list_test"]
    n_tc    = len(tc_list)
    n_te    = len(te_list)
    print(f"TrainCal: {n_tc} patients  |  Test: {n_te} patients")

    tc_dyn, tc_age, tc_gender, tc_eth = extract_arrays(tc_list, DYNAMIC_VARS)
    te_dyn, te_age, te_gender, te_eth = extract_arrays(te_list, DYNAMIC_VARS)

    # ── Layout: 4 KDE rows (dynamic vars) + 1 row (Age KDE + gender bar + ethnicity bar)
    fig = plt.figure(figsize=(14, 18))
    gs  = fig.add_gridspec(5, 3, hspace=0.55, wspace=0.35)

    # Row 0–3: one KDE per dynamic variable (span full row = 3 columns)
    for row, var in enumerate(DYNAMIC_VARS):
        ax = fig.add_subplot(gs[row, :])
        tc_vals = clip_arr(tc_dyn[var])
        te_vals = clip_arr(te_dyn[var])
        kde_plot(ax, tc_vals, TRAINCAL_COLOR,
                 f"TrainCal — no early Norepinephrine  (n={n_tc})")
        kde_plot(ax, te_vals, TEST_COLOR,
                 f"Test — early Norepinephrine  (n={n_te})")
        kl = hist_kl(te_vals, tc_vals)
        ax.set_xlabel(PRETTY.get(var, var), fontsize=11)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{PRETTY.get(var, var)}    KL(Test ‖ TrainCal) = {kl:.3f}",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Row 4, col 0: Age KDE
    ax_age = fig.add_subplot(gs[4, 0])
    tc_a = clip_arr(tc_age)
    te_a = clip_arr(te_age)
    kde_plot(ax_age, tc_a, TRAINCAL_COLOR, f"TrainCal (n={n_tc})")
    kde_plot(ax_age, te_a, TEST_COLOR, f"Test (n={n_te})")
    kl_age = hist_kl(te_a, tc_a)
    ax_age.set_xlabel("Age (years)", fontsize=11)
    ax_age.set_ylabel("Density", fontsize=10)
    ax_age.set_title(f"Age    KL = {kl_age:.3f}", fontsize=11)
    ax_age.legend(fontsize=9)
    ax_age.spines["top"].set_visible(False)
    ax_age.spines["right"].set_visible(False)

    # Row 4, col 1: Gender proportions
    ax_gen = fig.add_subplot(gs[4, 1])
    tc_m = sum(1 for g in tc_gender if g == "M") / n_tc
    te_m = sum(1 for g in te_gender if g == "M") / n_te
    x_pos = np.array([0, 1])
    ax_gen.bar(x_pos - 0.2, [tc_m, 1 - tc_m], 0.35,
               color=TRAINCAL_COLOR, alpha=0.85, label="TrainCal")
    ax_gen.bar(x_pos + 0.2, [te_m, 1 - te_m], 0.35,
               color=TEST_COLOR, alpha=0.85, label="Test")
    ax_gen.set_xticks(x_pos)
    ax_gen.set_xticklabels(["Male", "Female"], fontsize=10)
    ax_gen.set_ylabel("Proportion", fontsize=10)
    ax_gen.set_title("Gender", fontsize=11)
    ax_gen.set_ylim(0, 1)
    ax_gen.legend(fontsize=9)
    ax_gen.spines["top"].set_visible(False)
    ax_gen.spines["right"].set_visible(False)

    # Row 4, col 2: Ethnicity proportions
    ax_eth = fig.add_subplot(gs[4, 2])
    eth_groups = ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER/UNKNOWN"]

    def eth_bucket(e):
        if "WHITE" in e: return "WHITE"
        if "BLACK" in e: return "BLACK"
        if "HISPANIC" in e or "LATINO" in e: return "HISPANIC"
        if "ASIAN" in e: return "ASIAN"
        return "OTHER/UNKNOWN"

    tc_eth_b = [eth_bucket(e) for e in tc_eth]
    te_eth_b = [eth_bucket(e) for e in te_eth]
    tc_counts = np.array([tc_eth_b.count(g) / n_tc for g in eth_groups])
    te_counts = np.array([te_eth_b.count(g) / n_te for g in eth_groups])
    x_eth = np.arange(len(eth_groups))
    ax_eth.bar(x_eth - 0.2, tc_counts, 0.35, color=TRAINCAL_COLOR, alpha=0.8, label="TrainCal")
    ax_eth.bar(x_eth + 0.2, te_counts, 0.35, color=TEST_COLOR,     alpha=0.8, label="Test")
    ax_eth.set_xticks(x_eth)
    ax_eth.set_xticklabels(eth_groups, fontsize=8, rotation=20, ha="right")
    ax_eth.set_ylabel("Proportion", fontsize=10)
    ax_eth.set_title("Ethnicity", fontsize=11)
    ax_eth.legend(fontsize=9)
    ax_eth.spines["top"].set_visible(False)
    ax_eth.spines["right"].set_visible(False)

    fig.suptitle(
        "Covariate shift: TrainCal (no early vasopressor) vs Test (early Norepinephrine)\n"
        "MIMIC-III sepsis cohort — NaCl 0.9% prediction",
        fontsize=13, fontweight="bold", y=1.005,
    )

    # Print KL summary
    print("\nKL divergence summary  KL(Test ‖ TrainCal):")
    for var in DYNAMIC_VARS:
        tc_vals = clip_arr(tc_dyn[var])
        te_vals = clip_arr(te_dyn[var])
        kl = hist_kl(te_vals, tc_vals)
        print(f"  {var:40s}  {kl:.4f}")
    print(f"  {'Age':40s}  {hist_kl(clip_arr(te_age), clip_arr(tc_age)):.4f}")

    out_path = Path(args.save) if args.save else ROOT / "results" / "medical" / "pdf" / "medical_covariate_shift.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
