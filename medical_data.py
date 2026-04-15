"""
medical_data.py
================
Utilities for loading, inspecting, and summarizing the sepsis trajectory dataset
stored in ``sepsis_experiment_data.pkl``.

This module is designed for the experiment setting in which patients come from a
sepsis cohort and are split into TrainCal vs Test according to whether they used
Norepinephrine. Each patient record stores a small dictionary of hourly clinical
trajectories over a 24-hour ICU window. The intended use is exploratory data
analysis, sanity checking, visualization, and light preprocessing before model
training.

QUICK START
-----------
  python medical_data.py --summary
      Load ``sepsis_experiment_data.pkl`` from the current working directory and
      print dataset sizes, variable names, basic missingness / zero summaries,
      and simple split-level descriptive statistics.

  python medical_data.py --summary --path path/to/sepsis_experiment_data.pkl
      Same as above, but load the pickle file from a custom location.

  python medical_data.py --show traincal --index 0
      Display the first TrainCal patient as a table with one row per trajectory
      and columns t0, t1, ..., t23.

  python medical_data.py --show test --index 3
      Display the 4th Test patient in the same row-per-trajectory format.

  python medical_data.py --plot traincal --index 0
      Produce a line plot for all trajectories of the specified patient.

  python medical_data.py --summary --save_csv summary.csv
      Save the per-variable summary table to disk as a CSV file after printing
      the textual summary.

DATA STORED IN THE PICKLE FILE
------------------------------
The pickle file is expected to contain a dictionary with four keys:

  patient_ids_traincal
      list[str]
      Folder IDs for patients assigned to the TrainCal split.

  patient_trajectory_list_traincal
      list[dict[str, pandas.DataFrame]]
      One entry per TrainCal patient. Each entry is a dictionary whose keys are
      human-readable trajectory labels, for example
      ``"Heart Rate"``, ``"Respiratory Rate"``, ``"NaCl 0.9%"``, and
      ``"Norepinephrine"``. Each value is a DataFrame with two columns:
          - ``hour``  : integers 0, 1, ..., 23
          - ``value`` : numeric hourly trajectory values

  patient_ids_test
      list[str]
      Folder IDs for patients assigned to the Test split.

  patient_trajectory_list_test
      list[dict[str, pandas.DataFrame]]
      Same structure as the TrainCal trajectories, but for the Test split.

Conceptually, the pickle stores a nested object of the form

  {
      "patient_ids_traincal": [...],
      "patient_trajectory_list_traincal": [
          {
              "Heart Rate": DataFrame(hour, value),
              "Respiratory Rate": DataFrame(hour, value),
              ...
          },
          ...
      ],
      "patient_ids_test": [...],
      "patient_trajectory_list_test": [...]
  }

Each patient is therefore a compact multivariate time series over 24 hours.
This representation is convenient for inspection and debugging because the
trajectory names remain attached to the data and each patient can be printed as
an easy-to-read table. It is also easy to convert into array form later for
model fitting.

EXAMPLE USAGE IN PYTHON
-----------------------
The most common workflow is:

1. Load the pickle file with ``load_data``.
2. Call ``print_full_summary`` to check split sizes, variable names, and whether
   the trajectories look sensible.
3. Inspect a few individual patients with ``patient_table`` and
   ``plot_patient``.
4. Once the data quality looks correct, convert the nested structure into arrays
   suitable for your modeling code.

Typical interactive usage looks like:

    >>> from medical_data import load_data, print_full_summary, patient_table
    >>> data = load_data("sepsis_experiment_data.pkl")
    >>> print_full_summary(data)
    >>> df0 = patient_table(data, split="traincal", index=0)
    >>> print(df0)

You can also visualize one patient:

    >>> from medical_data import plot_patient
    >>> plot_patient(data, split="test", index=2)

The plot overlays all stored trajectories on the same 24-hour grid. This is
useful for quickly seeing the qualitative difference between TrainCal and Test
patients. For example, patients in the Test split should generally show nonzero
Norepinephrine, while TrainCal patients should not. The visualizations are meant
for sanity checks rather than publication figures.

SUMMARY STATISTICS PROVIDED
---------------------------
This file computes several practical summaries:

  - number of patients in each split
  - set of trajectory names stored for each split
  - per-variable counts of patients, total observed points, nonzero rate,
    zero rate, mean, standard deviation, min, max, and median
  - split-level check of how often Norepinephrine is nonzero
  - simple patient-level counts such as how many trajectories each patient has

These summaries are intended to answer questions such as:
  * Is the split consistent with the Norepinephrine rule?
  * Are all required variables present?
  * Which variables are sparse?
  * Are there obvious value-range problems?

VISUALIZATION NOTES
-------------------
The built-in plotting function draws one line per trajectory against hour.
Because the variables are on different scales, this raw overlay is mainly a
sanity-check view. If you want cleaner visual comparison, common next steps are:

  - plot each variable in its own panel
  - z-score trajectories within variable before plotting
  - compare TrainCal vs Test distributions at each hour
  - plot histograms or boxplots for selected variables

This module keeps the default visualization lightweight so it remains easy to
use inside notebooks or from the command line.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_PATH = "sepsis_experiment_data.pkl"


def load_data(path: str | Path = DEFAULT_PATH) -> dict[str, Any]:
    """Load the experiment pickle file."""
    path = Path(path)
    with path.open("rb") as f:
        data = pickle.load(f)
    return data


def _split_keys(split: str) -> tuple[str, str]:
    split = split.lower()
    if split not in {"traincal", "test"}:
        raise ValueError("split must be 'traincal' or 'test'")
    return f"patient_ids_{split}", f"patient_trajectory_list_{split}"


def patient_table(data: dict[str, Any], split: str, index: int) -> pd.DataFrame:
    """Return one patient as a row-per-trajectory table with columns t0..t23."""
    ids_key, traj_key = _split_keys(split)
    patient_ids = data[ids_key]
    patient_trajectory_list = data[traj_key]

    if index < 0 or index >= len(patient_ids):
        raise IndexError(f"index {index} out of range for split '{split}'")

    patient_dict = patient_trajectory_list[index]
    df_patient = pd.DataFrame({
        name: traj_df["value"].to_numpy() for name, traj_df in patient_dict.items()
    }).T
    df_patient.columns = [f"t{j}" for j in range(df_patient.shape[1])]
    df_patient.index.name = "trajectory"
    return df_patient


def plot_patient(data: dict[str, Any], split: str, index: int) -> None:
    """Plot all trajectories for one patient."""
    ids_key, traj_key = _split_keys(split)
    patient_ids = data[ids_key]
    patient_trajectory_list = data[traj_key]

    if index < 0 or index >= len(patient_ids):
        raise IndexError(f"index {index} out of range for split '{split}'")

    patient_id = patient_ids[index]
    patient_dict = patient_trajectory_list[index]

    plt.figure(figsize=(10, 6))
    for name, traj_df in patient_dict.items():
        plt.plot(traj_df["hour"], traj_df["value"], marker="o", label=name)

    plt.xlabel("Hour")
    plt.ylabel("Value")
    plt.title(f"{split.capitalize()} patient {index} (patient_id={patient_id})")
    plt.xticks(range(24))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.show()


def _flatten_split(patient_ids: list[str], patient_trajectory_list: list[dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Flatten one split into a long DataFrame for summary statistics."""
    rows: list[dict[str, Any]] = []
    for patient_id, patient_dict in zip(patient_ids, patient_trajectory_list):
        for variable, traj_df in patient_dict.items():
            tmp = traj_df.copy()
            tmp["patient_id"] = patient_id
            tmp["variable"] = variable
            rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["hour", "value", "patient_id", "variable"])

    flat = pd.concat(rows, ignore_index=True)
    flat["value"] = pd.to_numeric(flat["value"], errors="coerce")
    return flat


def variable_summary(data: dict[str, Any], split: str) -> pd.DataFrame:
    """Compute per-variable descriptive statistics for one split."""
    ids_key, traj_key = _split_keys(split)
    flat = _flatten_split(data[ids_key], data[traj_key])

    if flat.empty:
        return pd.DataFrame()

    summary = (
        flat.groupby("variable")
        .agg(
            n_patients=("patient_id", "nunique"),
            n_points=("value", "size"),
            n_missing=("value", lambda x: x.isna().sum()),
            n_nonzero=("value", lambda x: (x.fillna(0) != 0).sum()),
            n_zero=("value", lambda x: (x.fillna(0) == 0).sum()),
            mean=("value", "mean"),
            std=("value", "std"),
            median=("value", "median"),
            min=("value", "min"),
            max=("value", "max"),
        )
        .reset_index()
    )

    summary["nonzero_rate"] = summary["n_nonzero"] / summary["n_points"]
    summary["zero_rate"] = summary["n_zero"] / summary["n_points"]
    summary = summary.sort_values("variable").reset_index(drop=True)
    return summary


def print_full_summary(data: dict[str, Any]) -> pd.DataFrame:
    """Print dataset overview and return combined summary table."""
    n_traincal = len(data["patient_ids_traincal"])
    n_test = len(data["patient_ids_test"])

    print("=== DATASET OVERVIEW ===")
    print(f"TrainCal patients: {n_traincal}")
    print(f"Test patients    : {n_test}")
    print()

    train_vars = set()
    for patient_dict in data["patient_trajectory_list_traincal"]:
        train_vars.update(patient_dict.keys())

    test_vars = set()
    for patient_dict in data["patient_trajectory_list_test"]:
        test_vars.update(patient_dict.keys())

    print("TrainCal variables:")
    print(sorted(train_vars))
    print()
    print("Test variables:")
    print(sorted(test_vars))
    print()

    summary_train = variable_summary(data, split="traincal")
    if not summary_train.empty:
        summary_train.insert(0, "split", "traincal")

    summary_test = variable_summary(data, split="test")
    if not summary_test.empty:
        summary_test.insert(0, "split", "test")

    summary = pd.concat([summary_train, summary_test], ignore_index=True)

    # Quick split sanity check for Norepinephrine if present.
    norepi_name = "Norepinephrine"
    if not summary.empty and norepi_name in set(summary["variable"]):
        print("=== SPLIT SANITY CHECK (Norepinephrine) ===")
        tmp = summary.loc[summary["variable"] == norepi_name, ["split", "n_nonzero", "n_points", "nonzero_rate"]]
        print(tmp.to_string(index=False))
        print()

    print("=== VARIABLE SUMMARY ===")
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(summary)
    print()

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and summarize sepsis experiment data.")
    parser.add_argument("--path", type=str, default=DEFAULT_PATH, help="Path to sepsis_experiment_data.pkl")
    parser.add_argument("--summary", action="store_true", help="Print full dataset summary")
    parser.add_argument("--show", choices=["traincal", "test"], help="Display one patient table from the specified split")
    parser.add_argument("--plot", choices=["traincal", "test"], help="Plot one patient from the specified split")
    parser.add_argument("--index", type=int, default=0, help="Patient index for --show or --plot")
    parser.add_argument("--save_csv", type=str, default=None, help="Optional path to save summary CSV")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    data = load_data(args.path)

    summary_df: pd.DataFrame | None = None
    if args.summary:
        summary_df = print_full_summary(data)

    if args.save_csv is not None:
        if summary_df is None:
            summary_df = pd.concat(
                [
                    variable_summary(data, split="traincal").assign(split="traincal"),
                    variable_summary(data, split="test").assign(split="test"),
                ],
                ignore_index=True,
            )
            if not summary_df.empty:
                cols = ["split"] + [c for c in summary_df.columns if c != "split"]
                summary_df = summary_df[cols]
        Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(args.save_csv, index=False)
        print(f"Saved summary CSV to {args.save_csv}")

    if args.show:
        table = patient_table(data, split=args.show, index=args.index)
        print(table)

    if args.plot:
        plot_patient(data, split=args.plot, index=args.index)

    if not args.summary and not args.show and not args.plot:
        parser.print_help()


if __name__ == "__main__":
    main()
