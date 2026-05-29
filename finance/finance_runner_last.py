"""
finance/finance_runner_last.py — per-window runner for Algorithm 2 (last-step)
on the S&P 500 data.

Self-contained (§ 0). Contents (see WEIGHTED_CAFHT_PLAN.md § 3):
  - LastStepRidge — ridge of Y_{T+1} on flattened X_{1:T} (4·T features),
    λ by 5-fold CV (§ 3.2).
  - featurize_xall(X) — flattened X_{1:T} (4·T) (§ 3.3).
  - main loop calling core.weighted_cafht_last.WeightedCAFHTLastStep.
  - CLI mirrors finance_runner_whole minus the γ/ACI flags (modes {full, uniform}).

Imports the loader + sector filters from finance.finance_data (STAY file).

TODO: implement per § 4 step 7.
"""
