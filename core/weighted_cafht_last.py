"""
core/weighted_cafht_last.py — Algorithm 2: Weighted CAFHT (last-step coverage).

Self-contained module implementing last-step coverage (see WEIGHTED_CAFHT_PLAN.md
§ 2.4). No γ, no ACI, no D_ACI, no per-step bands. Per the "one algorithm per
file" ground rule (§ 0), the two shared primitives are duplicated here (copied
verbatim from core/weighted_cafht_whole.py, NOT imported):

  - weighted_quantile_with_inf(scores, w_cal, w_test, level)   (§ 2.1)
  - density_ratio_weights(X_pos, X_neg, X_eval, ...)           (§ 2.1)
  - class WeightedCAFHTLastStep                                (§ 2.4)
        calibration_scores / predict_bands; cross-half + δ_∞ quantile only.

Coverage target: P(Y_{T+1} ∈ Ĉ_{T+1}) ≥ 1 − α.

TODO: implement per work-plan step 5. Includes an `if __name__ == "__main__"`
block with the inline sanity tests listed in § 4 step 5 (helpers bit-identical
to the whole-trajectory versions; uniform weights reduce to standard split CP).
"""
