"""
synthetic/synthetic_runner_last.py — single-seed runner for Algorithm 2
(last-step) on the synthetic DGPs.

Self-contained (§ 0). Contents (see WEIGHTED_CAFHT_PLAN.md § 3):
  - build_last_step_predictor(Y_train, lags=T) — OLS of Y_{T+1} on Y_{1:T},
    producing a scalar (n,) prediction (§ 3.2).
  - featurize_xall(X) — full Y history (§ 3.3).
  - main loop calling core.weighted_cafht_last.WeightedCAFHTLastStep.
  - CLI mirrors synthetic_runner_whole minus the ACI/γ flags (no --frac_aci,
    no --gamma_split; modes {full, uniform} only — no zerog for last-step).

Imports synthetic DGPs from core.ts_generator (STAY file).

TODO: implement per § 4 step 8.
"""
