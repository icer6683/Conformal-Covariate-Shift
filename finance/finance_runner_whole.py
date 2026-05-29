"""
finance/finance_runner_whole.py — per-window runner for Algorithm 1
(whole-trajectory) on the S&P 500 data.

Self-contained (§ 0). Contents (see WEIGHTED_CAFHT_PLAN.md § 3):
  - LinearCovariateModel — per-step OLS on X_t, fit once globally (copied, NOT
    imported, from OLD_finance_conformal.py per § B.3; § 3.1).
  - featurize_x1(X) — the 4 covariates at day 1 (§ 3.3).
  - four-way data partition (D_ACI + D_tr + D_cal + D_test) per § 2.0.
  - main loop calling core.weighted_cafht_whole.WeightedCAFHTWholeTrajectory.
  - CLI matches OLD_finance_conformal.py (--npz --test_sector --mode
    --gamma_grid --seed --save_json --save_plot) plus --mixed and
    --frac_aci / --gamma_split for whole-traj.

Imports the loader + sector filters from finance.finance_data (STAY file).

No multi-seed wrapper for finance — the shell script iterates over the 13
rolling windows × 2 sectors + mixed null.

TODO: implement per § 4 step 7.
"""
