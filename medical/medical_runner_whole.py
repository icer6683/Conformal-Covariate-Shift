"""
medical/medical_runner_whole.py — single-seed runner for Algorithm 1
(whole-trajectory) on the MIMIC-III sepsis cohort.

Self-contained (§ 0). Contents (see WEIGHTED_CAFHT_PLAN.md § 3):
  - LinearCovariateModel — one-step-ahead AR Y_{t+1} ~ Y_t + X_t + S (copied,
    NOT imported, from OLD_medical_conformal.py per § B.3; § 3.1), plus the
    ETHNICITY_MAP / TARGET_VAR / COVARIATE_VARS / STATIC_VARS constants.
  - featurize_x1(X, S) — 3 vitals at hour 0 + 6 statics = 9 features
    (§ 3.3; per Q2 answer = (a), feed X_1 only, literal spec).
  - four-way data partition (D_ACI + D_tr + D_cal + D_test) per § 2.0.
  - main loop calling core.weighted_cafht_whole.WeightedCAFHTWholeTrajectory.
  - CLI matches OLD_medical_conformal.py (--pkl --n_traincal --n_test --alpha
    --mode {full,uniform,zerog} --seed --save_json --save_plot) plus
    --frac_aci 0.15 --gamma_split 0.33 0.33 0.34.

Imports the loader from medical.medical_data (STAY file).

TODO: implement per § 4 step 6 (medical first — smallest sample, fastest).
"""
