"""
medical/medical_runner_last.py — single-seed runner for Algorithm 2 (last-step)
on the MIMIC-III sepsis cohort.

Self-contained (§ 0). Contents (see WEIGHTED_CAFHT_PLAN.md § 3):
  - LastStepRidge — ridge of Y_{T+1} on flattened X_{1:T} + S (3·T + 6
    features), λ by 5-fold CV (§ 3.2).
  - featurize_xall(X, S) — flattened X_{1:T} + S (§ 3.3; but see Q2).
  - main loop calling core.weighted_cafht_last.WeightedCAFHTLastStep.
  - CLI mirrors medical_runner_whole minus the γ/ACI flags (modes {full, uniform}).

Imports the loader from medical.medical_data (STAY file).

TODO: implement per § 4 step 6.
"""
