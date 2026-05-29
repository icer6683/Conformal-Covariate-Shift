"""
synthetic/synthetic_runner_whole.py — single-seed runner for Algorithm 1
(whole-trajectory) on the synthetic DGPs.

Self-contained (§ 0): defines its own predictor, featurizer, main loop, and CLI;
the only cross-module import for the algorithm is core.weighted_cafht_whole.

Contents (see WEIGHTED_CAFHT_PLAN.md § 3):
  - build_whole_trajectory_predictor(Y_train) — global AR(1) by OLS, iterated
    forward to produce an (n, T+1, 1) predicted trajectory (§ 3.1).
  - featurize_x1(X) — X_1 == Y_0 under the X = Y_lag reinterpretation (§ 3.3).
  - four-way data partition (D_ACI + D_tr + D_cal + D_test) per § 2.0: peel
    D_ACI off the raw pool first; fit predictor on the full D_tr; precompute the
    frozen D_ACI residual score bank.
  - main loop calling core.weighted_cafht_whole.WeightedCAFHTWholeTrajectory.
  - CLI: --n_train --n_cal --n_test --T --alpha --mode {full,uniform,zerog}
    --covariate_mode {static,dynamic} --with_shift --frac_aci --gamma_split
    --seed --save_json --save_plot.

Imports synthetic DGPs from core.ts_generator (STAY file).

TODO: implement per § 4 step 8.
"""
