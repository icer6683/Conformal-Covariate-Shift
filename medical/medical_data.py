#!/usr/bin/env python3
"""
=============================================================================
MEDICAL DATA VISUALIZATION  ---  Covariate shift between TrainCal and Test
=============================================================================

QUICK START
-----------
  # Show all 7 plots (3 static + 3 dynamic + 1 target):
  python medical_data.py --pkl sepsis_experiment_data_nacl_target.pkl \
      --static --dynamic --target

  # Only static covariate comparisons:
  python medical_data.py --pkl sepsis_experiment_data_nacl_target.pkl --static

  # Only dynamic CHART trajectories:
  python medical_data.py --pkl sepsis_experiment_data_nacl_target.pkl --dynamic

  # Only the NaCl 0.9% target trajectory:
  python medical_data.py --pkl sepsis_experiment_data_nacl_target.pkl --target

  # Save the figure instead of showing it:
  python medical_data.py --pkl sepsis_experiment_data_nacl_target.pkl \
      --static --dynamic --target --save_plot covariate_shift.png

FULL OPTIONS
------------
  --pkl          Path to sepsis_experiment_data_nacl_target.pkl (required)
  --static       Include 3 static covariate plots (Age, gender, ethnicity)
  --dynamic      Include 3 dynamic CHART covariate plots
                 (Heart Rate, Respiratory Rate, O2 saturation)
  --target       Include 1 NaCl 0.9% target trajectory plot
  --save_plot    Path to save the output figure, e.g. covariate_shift.png

HOW IT WORKS
------------
  Loads the sepsis experiment pickle and compares the TrainCal set
  (no Norepinephrine) against the Test set (received Norepinephrine).

  Static variables (--static):
    Age       : side-by-side bar chart of age-bin frequencies
    gender    : side-by-side bar chart of M/F proportions
    ethnicity : side-by-side bar chart of grouped ethnicity proportions

  Dynamic CHART variables (--dynamic):
    Heart Rate, Respiratory Rate, O2 saturation pulseoxymetry:
      line plot of mean value at each of the 24 hourly time stamps,
      one line for TrainCal and one for Test.

  Target variable (--target):
    NaCl 0.9%:
      line plot of mean value at each of the 24 hourly time stamps,
      one line for TrainCal and one for Test.
=============================================================================
"""
import argparse
import pickle

import numpy as np
import matplotlib.pyplot as plt


# -- Constants (mirrored from medical_conformal.py) ---------------------------

TARGET_VAR = "NaCl 0.9% (target)"

COVARIATE_VARS = [
    "Heart Rate",
    "Respiratory Rate",
    "O2 saturation pulseoxymetry",
]

# Ethnicity grouping map (same as medical_conformal.py)
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
    "ASIAN - VIETNAMESE":                           "ASIAN",
    "ASIAN - FILIPINO":                             "ASIAN",
    "ASIAN - CAMBODIAN":                            "ASIAN",
    "ASIAN - KOREAN":                               "ASIAN",
    "ASIAN - JAPANESE":                             "ASIAN",
    "ASIAN - THAI":                                 "ASIAN",
    "ASIAN - OTHER":                                "ASIAN",
}

AGE_BINS = [0, 30, 40, 50, 60, 70, 80, 200]
AGE_LABELS = ["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]

ETHNICITY_GROUPS = ["WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER"]


# =============================================================================
# Data loading
# =============================================================================

# Load the sepsis experiment pickle and return the raw dict
def load_data(pkl_path):
    """Load the sepsis experiment pickle and return the raw dict."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


# =============================================================================
# Extraction helpers
# =============================================================================

# Extract a list of scalar static values from patient trajectory list
def _extract_static(patient_list, key):
    """Return a list of scalar values for the given static key."""
    return [p[key] for p in patient_list]


# Extract a (n_patients, 24) array of hourly values for a dynamic variable
def _extract_dynamic(patient_list, key):
    """Return an (n_patients, 24) array of hourly values."""
    n = len(patient_list)
    arr = np.zeros((n, 24), dtype=np.float64)
    for i, p in enumerate(patient_list):
        arr[i, :] = p[key]["value"].to_numpy(dtype=np.float64)
    return arr


# Map a raw ethnicity string to a grouped category
def _group_ethnicity(raw):
    """Map a raw ethnicity string to one of the 5 major groups."""
    return ETHNICITY_MAP.get(raw, "OTHER")


# Compute normalized bin frequencies for a list of values given bin edges
def _bin_frequencies(values, bins, labels):
    """Return (labels, proportions) for the given bin edges."""
    counts = np.zeros(len(labels), dtype=int)
    for v in values:
        for k in range(len(bins) - 1):
            if bins[k] <= v < bins[k + 1]:
                counts[k] += 1
                break
    props = counts / max(counts.sum(), 1)
    return labels, props


# Compute normalized category frequencies for a list of string labels
def _category_frequencies(values, categories):
    """Return (categories, proportions) for the given category list."""
    counts = {c: 0 for c in categories}
    for v in values:
        if v in counts:
            counts[v] += 1
        else:
            counts["OTHER"] += 1
    total = max(sum(counts.values()), 1)
    props = np.array([counts[c] / total for c in categories])
    return categories, props


# =============================================================================
# Plotting functions
# =============================================================================

# Plot a side-by-side bar chart comparing TrainCal vs Test category proportions
def _plot_bar_comparison(ax, labels, props_tc, props_te, title, ylabel="Proportion"):
    """Side-by-side bar chart for TrainCal vs Test."""
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, props_tc, w, label="TrainCal", color="steelblue", alpha=0.8)
    ax.bar(x + w / 2, props_te, w, label="Test", color="coral", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)


# Plot two mean-trajectory lines (TrainCal vs Test) over 24 hours
def _plot_trajectory_comparison(ax, arr_tc, arr_te, title, ylabel):
    """Line plot of mean trajectory at each hour, TrainCal vs Test."""
    hours = np.arange(24)
    mean_tc = arr_tc.mean(axis=0)
    mean_te = arr_te.mean(axis=0)
    ax.plot(hours, mean_tc, 'o-', color="steelblue", linewidth=1.5,
            markersize=4, label="TrainCal")
    ax.plot(hours, mean_te, 's-', color="coral", linewidth=1.5,
            markersize=4, label="Test")
    ax.set_xlabel("Hour")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(hours[::2])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


# =============================================================================
# Main visualization
# =============================================================================

def visualize(data, show_static=False, show_dynamic=False, show_target=False,
              save_path=None):
    """
    Create comparison plots between TrainCal and Test distributions.

    Parameters
    ----------
    data : dict
        Output of load_data().
    show_static : bool
        Include 3 static covariate plots (Age, gender, ethnicity).
    show_dynamic : bool
        Include 3 dynamic CHART covariate plots.
    show_target : bool
        Include 1 NaCl 0.9% target trajectory plot.
    save_path : str or None
        If set, save the figure to this path.
    """
    tc_list = data["patient_trajectory_list_traincal"]
    te_list = data["patient_trajectory_list_test"]
    n_tc = len(tc_list)
    n_te = len(te_list)

    n_plots = (3 if show_static else 0) + (3 if show_dynamic else 0) + (1 if show_target else 0)
    if n_plots == 0:
        print("No plots selected. Use --static, --dynamic, and/or --target.")
        return

    print(f"\n  TrainCal : {n_tc} patients")
    print(f"  Test     : {n_te} patients")
    print(f"  Plots    : {n_plots} "
          f"({'static ' if show_static else ''}"
          f"{'dynamic ' if show_dynamic else ''}"
          f"{'target' if show_target else ''})")
    print()

    # Determine layout: use a single row if <= 4 plots, else 2 rows
    if n_plots <= 4:
        n_rows, n_cols = 1, n_plots
        fig_w = 5 * n_plots
        fig_h = 4.5
    else:
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig_w = 20
        fig_h = 4.5 * n_rows

    fig, axes_flat = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                                  squeeze=False)
    axes = axes_flat.flatten()

    # Hide any extra axes
    for k in range(n_plots, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        f"Covariate Shift: TrainCal (n={n_tc}) vs Test (n={n_te})",
        fontsize=13, fontweight='bold'
    )

    idx = 0  # current subplot index

    # -- Static covariate plots -----------------------------------------------
    if show_static:
        # Age
        ages_tc = _extract_static(tc_list, "Age")
        ages_te = _extract_static(te_list, "Age")
        _, props_tc = _bin_frequencies(ages_tc, AGE_BINS, AGE_LABELS)
        _, props_te = _bin_frequencies(ages_te, AGE_BINS, AGE_LABELS)
        _plot_bar_comparison(axes[idx], AGE_LABELS, props_tc, props_te,
                             "Age Distribution")
        idx += 1

        # Gender
        genders_tc = _extract_static(tc_list, "gender")
        genders_te = _extract_static(te_list, "gender")
        gender_cats = ["M", "F"]
        _, gp_tc = _category_frequencies(genders_tc, gender_cats)
        _, gp_te = _category_frequencies(genders_te, gender_cats)
        _plot_bar_comparison(axes[idx], gender_cats, gp_tc, gp_te,
                             "Gender Distribution")
        idx += 1

        # Ethnicity (grouped)
        eth_tc = [_group_ethnicity(e) for e in _extract_static(tc_list, "ethnicity")]
        eth_te = [_group_ethnicity(e) for e in _extract_static(te_list, "ethnicity")]
        _, ep_tc = _category_frequencies(eth_tc, ETHNICITY_GROUPS)
        _, ep_te = _category_frequencies(eth_te, ETHNICITY_GROUPS)
        _plot_bar_comparison(axes[idx], ETHNICITY_GROUPS, ep_tc, ep_te,
                             "Ethnicity Distribution")
        idx += 1

    # -- Dynamic CHART covariate plots ----------------------------------------
    if show_dynamic:
        for var_name in COVARIATE_VARS:
            arr_tc = _extract_dynamic(tc_list, var_name)
            arr_te = _extract_dynamic(te_list, var_name)
            _plot_trajectory_comparison(axes[idx], arr_tc, arr_te,
                                        f"{var_name} (mean trajectory)",
                                        var_name)
            idx += 1

    # -- Target variable plot -------------------------------------------------
    if show_target:
        arr_tc = _extract_dynamic(tc_list, TARGET_VAR)
        arr_te = _extract_dynamic(te_list, TARGET_VAR)
        _plot_trajectory_comparison(axes[idx], arr_tc, arr_te,
                                    f"{TARGET_VAR} (mean trajectory)",
                                    "NaCl 0.9% (mL)")
        idx += 1

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
        description="Visualize covariate shift between TrainCal and Test sets "
                    "in the sepsis ICU dataset."
    )
    parser.add_argument("--pkl",       required=True,
                        help="Path to sepsis_experiment_data_nacl_target.pkl")
    parser.add_argument("--static",    action="store_true", default=False,
                        help="Include 3 static covariate plots "
                             "(Age, gender, ethnicity)")
    parser.add_argument("--dynamic",   action="store_true", default=False,
                        help="Include 3 dynamic CHART covariate plots "
                             "(Heart Rate, Respiratory Rate, O2 saturation)")
    parser.add_argument("--target",    action="store_true", default=False,
                        help="Include 1 NaCl 0.9%% target trajectory plot")
    parser.add_argument("--save_plot", default=None,
                        help="Path to save the output figure")
    args = parser.parse_args()

    print(f"Loading {args.pkl} ...")
    data = load_data(args.pkl)

    visualize(
        data,
        show_static=args.static,
        show_dynamic=args.dynamic,
        show_target=args.target,
        save_path=args.save_plot,
    )


if __name__ == "__main__":
    main()
