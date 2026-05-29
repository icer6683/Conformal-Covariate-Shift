#!/usr/bin/env bash
# =============================================================================
# run_all_v2.sh — regime-aware experiment dispatcher (v2 restructure).
# =============================================================================
# Starting point: OLD_run_all_experiments.sh — its function structure, NPZ list,
# sector grids, and mixed-window setup are copied (NOT sourced), per § B.3.
#
# Adds a --regime flag {last_step, whole_trajectory, both} on top of the
# existing --synthetic / --finance / --medical / --all / --build-tables flags.
# For each (domain, regime) it dispatches to the matching runner per condition
# (full / uniform / zerog where applicable; § 5.1) and writes outputs under
# results/{domain}/{regime}/{json,pdf,tables}/ (§ 5.2).
#
# Default PY="python" (avoids the conda-env issue from v1).
#
# Usage (see § 5.3):
#   ./run_all_v2.sh --medical   --regime last_step --build-tables
#   ./run_all_v2.sh --finance   --regime whole_trajectory
#   ./run_all_v2.sh --all       # all domains × all regimes
#
# TODO: implement per § 4 step 9.
# =============================================================================
