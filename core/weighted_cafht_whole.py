"""
core/weighted_cafht_whole.py — Algorithm 1: Weighted CAFHT (whole-trajectory coverage).

Self-contained module implementing the whole-trajectory coverage algorithm
(see WEIGHTED_CAFHT_PLAN.md § 2.3). Per the "one algorithm per file" ground
rule (§ 0), this module owns ALL of its primitives — they are intentionally
duplicated here rather than factored into shared aci.py / density_ratio.py /
weighted_quantile.py modules:

  - weighted_quantile_with_inf(scores, w_cal, w_test, level)   (§ 2.1)
        Tibshirani et al. (2019) weighted-exchangeability quantile with a
        δ_∞ atom on the test point. Returns np.inf when the atom must be reached.
  - density_ratio_weights(X_pos, X_neg, X_eval, ...)           (§ 2.1)
        Logistic-regression likelihood-ratio weights, 5x-mean clipped,
        UNNORMALIZED (the δ_∞ quantile needs raw masses).
  - class ACI                                                  (§ 2.2)
        Single-trajectory adaptive conformal inference with an externally
        supplied, frozen score_bank.
  - class WeightedCAFHTWholeTrajectory                         (§ 2.3)
        select_gamma / calibration_scores / predict_bands.

Coverage target: P(∀t: Y_t ∈ Ĉ_t) ≥ 1 − α.

Data partition (§ 2.0): the raw trajectory pool is split FOUR ways at the
runner level — D_tr (predictor fit + internal γ-selection split), D_ACI
(separate held-out residual score bank for the main-algorithm ACI runs),
D_cal (calibration ε_i), D_test (cross-half deployment). D_ACI is disjoint
from D_tr / D_cal / D_test and must be peeled off BEFORE the conventional split.

Two ACI invocations with DIFFERENT score banks (§ 2.0):
  - γ-selection sandbox ACI: score bank from D_tr^(2) residuals.
  - main-algorithm ACI (cal + test): score bank from frozen D_ACI residuals.

Helper logic is adapted (copied, NOT imported) from
OLD_algorithm.py:AdaptedCAFHT._compute_density_ratio_weights and
._weighted_quantile (§ B.3).

TODO: implement per work-plan step 4. Includes an `if __name__ == "__main__"`
block with the four inline sanity tests listed in § 4 step 4.
"""
