#!/usr/bin/env python3
"""
Sanity tests for the oracle Poisson likelihood-ratio weighting added to
AdaptedCAFHT. Run with `python tests/test_oracle_weights.py` (no pytest dep).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from core.algorithm import AdaptedCAFHT


def _approx_equal(a, b, tol=1e-10):
    return np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=0)


def test_oracle_poisson_matches_closed_form():
    """For lam_src=1, lam_tgt=2: w(x) ∝ exp(-1) * 2**x; after normalization the
    ratios w(x)/w(0) must equal 2**x exactly (the exp(-1) prefactor cancels)."""
    xs = np.array([0, 1, 2, 3, 4], dtype=float)
    w = AdaptedCAFHT._oracle_poisson_weights(xs, lam_src=1.0, lam_tgt=2.0)
    # Sums to 1
    assert _approx_equal(w.sum(), 1.0), f"weights not normalized: sum={w.sum()}"
    # Pairwise ratios match the analytic LR
    expected_ratios = 2.0 ** xs            # w(x) / w(0)
    actual_ratios = w / w[0]
    assert _approx_equal(actual_ratios, expected_ratios), (
        f"ratio mismatch:\n  expected={expected_ratios}\n  actual  ={actual_ratios}"
    )
    print(f"[ok] oracle Poisson weights for λ=1→2, x={xs.astype(int).tolist()}: "
          f"{np.round(w, 6).tolist()}  (ratios vs x=0: {actual_ratios.tolist()})")


def test_oracle_no_shift_is_uniform():
    """When lam_src == lam_tgt the LR is identically 1, so weights are uniform."""
    xs = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    w = AdaptedCAFHT._oracle_poisson_weights(xs, lam_src=1.0, lam_tgt=1.0)
    assert _approx_equal(w, np.full_like(xs, 1.0 / xs.size)), \
        f"expected uniform, got {w}"
    print(f"[ok] no-shift case (λ_src=λ_tgt=1) → uniform weights {w.tolist()}")


def test_oracle_degenerate_params_falls_back_to_uniform():
    xs = np.array([0, 1, 2], dtype=float)
    for lam_s, lam_t in [(None, 2.0), (1.0, None), (0.0, 2.0), (1.0, -0.5)]:
        w = AdaptedCAFHT._oracle_poisson_weights(xs, lam_src=lam_s, lam_tgt=lam_t)
        assert _approx_equal(w, np.ones_like(xs)), \
            f"degenerate params (src={lam_s}, tgt={lam_t}) did not return ones: {w}"
    print("[ok] degenerate Poisson params correctly fall back to ones")


def test_ess_known_vector():
    """ESS = (Σw)² / Σw².
    For w = [1,1,1,1] → ESS = 16/4 = 4 (uniform → full sample size).
    For w = [3,1,0,0] → ESS = 16 / 10 = 1.6 (concentrated)."""
    e1 = AdaptedCAFHT._effective_sample_size([1.0, 1.0, 1.0, 1.0])
    e2 = AdaptedCAFHT._effective_sample_size([3.0, 1.0, 0.0, 0.0])
    assert _approx_equal(e1, 4.0), f"expected ESS=4.0, got {e1}"
    assert _approx_equal(e2, 1.6), f"expected ESS=1.6, got {e2}"
    print(f"[ok] ESS([1,1,1,1])={e1}, ESS([3,1,0,0])={e2}")


def test_calibrate_routes_to_oracle_when_cal_X_provided():
    """Smoke test: calibrate with cal_X under weight_mode='oracle_poisson'
    must populate _weights from oracle (length matches cal series count) and
    bypass the classifier."""
    rng = np.random.default_rng(0)
    n_cal, T = 50, 5
    cal_Y = rng.normal(size=(n_cal, T, 1))
    cal_X = rng.poisson(lam=1.0, size=n_cal).astype(float)

    p = AdaptedCAFHT(alpha=0.1, weight_mode="oracle_poisson",
                     lambda_source=1.0, lambda_target=2.0)
    p.fit_ar_model(cal_Y)
    p.calibrate(cal_Y, cal_X=cal_X)

    assert p._weights is not None and len(p._weights) == n_cal, \
        f"weights wrong shape: {None if p._weights is None else p._weights.shape}"
    assert _approx_equal(p._weights.sum(), 1.0), \
        f"oracle weights not normalized to 1: sum={p._weights.sum()}"
    assert p._clf is None, "classifier was fitted; oracle path should bypass it"
    assert p._last_ess is not None and 0 < p._last_ess <= n_cal, \
        f"ESS not stored or out of range: {p._last_ess}"
    print(f"[ok] calibrate(weight_mode='oracle_poisson') → "
          f"weights normalized, no classifier fit, ESS={p._last_ess:.2f}/{n_cal}")


def test_predict_with_interval_oracle_includes_test_point():
    """The test-inclusive normalization shrinks cal mass below 1, so the
    weighted (1-α) cal-only quantile is ≤ the test-inclusive quantile.
    Equivalently, the half-width returned by predict_with_interval_oracle is
    ≥ the half-width from predict_with_interval (cal-only normalization), with
    equality only when w_test is negligible.
    Also: when the test point's weight is tiny relative to the cal mass, the
    two methods must agree to within rounding."""
    rng = np.random.default_rng(0)
    n_cal, T = 200, 5
    cal_Y = rng.normal(size=(n_cal, T, 1))
    cal_X = rng.poisson(lam=1.0, size=n_cal).astype(float)

    p = AdaptedCAFHT(alpha=0.1, weight_mode="oracle_poisson",
                     lambda_source=1.0, lambda_target=2.0)
    p.fit_ar_model(cal_Y)
    p.calibrate(cal_Y, cal_X=cal_X)

    input_series = rng.normal(size=(T, 1))

    # Tiny test weight (x_test = 0 → log_lr most negative): test mass tiny,
    # quantiles should match closely.
    pred_a, lo_a, hi_a = p.predict_with_interval(input_series, alpha_level=0.1)
    pred_b, lo_b, hi_b = p.predict_with_interval_oracle(
        input_series, test_x=0.0, alpha_level=0.1)
    half_a = (hi_a - lo_a) / 2
    half_b = (hi_b - lo_b) / 2
    assert abs(half_b - half_a) / max(half_a, 1e-12) < 0.05, (
        f"tiny-test-x: cal-only half={half_a}, test-inclusive half={half_b} "
        "differ by more than 5%; expected near-equality")
    # Test-inclusive quantile is >= cal-only quantile (more conservative)
    assert half_b >= half_a - 1e-9, \
        f"test-inclusive half ({half_b}) should be ≥ cal-only ({half_a})"

    # Big test weight (x_test = 6 → log_lr highly positive): test mass eats a
    # bigger fraction of the denominator, half-width should grow.
    pred_c, lo_c, hi_c = p.predict_with_interval_oracle(
        input_series, test_x=6.0, alpha_level=0.1)
    half_c = (hi_c - lo_c) / 2
    assert half_c >= half_b - 1e-9, \
        f"larger test_x should yield wider interval, got {half_c} vs {half_b}"
    print(f"[ok] predict_with_interval_oracle: half-width grows monotonically "
          f"with test_x — half(x=0)≈{half_b:.3f} ≤ half(x=6)={half_c:.3f}; "
          f"cal-only half={half_a:.3f}")


def test_oracle_dynamic_matches_scipy_norm():
    """Hand-construct a tiny path of length 4 and check that
    AdaptedCAFHT._oracle_dynamic_log_weights matches independent computation
    via scipy.stats.norm.logpdf for the Gaussian transitions plus the analytic
    Poisson PMF ratio for the initial X_0."""
    from scipy.stats import norm

    src = dict(x_rate=0.6, x_trend=0.0, x_noise_std=0.2, x0_lambda=1.0)
    tgt = dict(x_rate_shift=0.9, x_trend_shift=0.05,
               x_noise_std_shift=0.4, x0_lambda_shift=2.0)

    # One path of length 4 (T+1 = 4 → t_use can be up to 3)
    x = np.array([[1.0, 0.7, 0.65, 0.42]])  # x_0=1 (Poisson-feasible), then floats

    for t_use in [0, 1, 2, 3]:
        got = float(AdaptedCAFHT._oracle_dynamic_log_weights(
            x, t_use=t_use, src=src, tgt=tgt)[0])

        # Independent reference
        # Initial Poisson term (k=1):  log Pois(1; λ_t) - log Pois(1; λ_s)
        # = (λ_s - λ_t) + 1*log(λ_t/λ_s)
        lam_s, lam_t = src["x0_lambda"], tgt["x0_lambda_shift"]
        ref = (lam_s - lam_t) + x[0, 0] * np.log(lam_t / lam_s)
        # Transition terms via scipy
        for s in range(t_use):
            mu_s = src["x_rate"] * x[0, s] + src["x_trend"] * s
            mu_t = tgt["x_rate_shift"] * x[0, s] + tgt["x_trend_shift"] * s
            ref += (norm.logpdf(x[0, s + 1], loc=mu_t, scale=tgt["x_noise_std_shift"])
                    - norm.logpdf(x[0, s + 1], loc=mu_s, scale=src["x_noise_std"]))

        assert _approx_equal(got, ref, tol=1e-9), (
            f"t_use={t_use}: implementation={got}, scipy reference={ref}, "
            f"diff={got - ref}")
        print(f"[ok] oracle_dynamic at t_use={t_use}: log_w={got:+.6f} "
              f"(matches scipy reference)")


def test_oracle_dynamic_no_shift_is_zero():
    """If src == tgt (no shift), every per-row log weight must be exactly 0."""
    src = dict(x_rate=0.7, x_trend=0.0, x_noise_std=0.2, x0_lambda=1.0)
    tgt = dict(x_rate_shift=0.7, x_trend_shift=0.0,
               x_noise_std_shift=0.2, x0_lambda_shift=1.0)
    rng = np.random.default_rng(7)
    X = rng.normal(loc=0.5, scale=0.3, size=(50, 6))
    X[:, 0] = rng.poisson(lam=1.0, size=50)
    log_w = AdaptedCAFHT._oracle_dynamic_log_weights(X, t_use=5, src=src, tgt=tgt)
    assert _approx_equal(log_w, np.zeros(50), tol=1e-12), \
        f"no-shift dynamic LR not zero: {log_w[:5]}"
    print(f"[ok] oracle_dynamic with no shift returns log_w=0 for all rows")


def test_predict_with_interval_oracle_dynamic():
    """End-to-end: calibrate with oracle_dynamic mode, then predict; check that
    weights normalize cleanly and ESS is finite."""
    rng = np.random.default_rng(42)
    n_cal, T = 100, 6
    cal_X = rng.normal(loc=0.5, scale=0.3, size=(n_cal, T))
    cal_X[:, 0] = rng.poisson(lam=1.0, size=n_cal)
    cal_Y = rng.normal(size=(n_cal, T, 1))

    src = dict(x_rate=0.6, x_trend=0.0, x_noise_std=0.2, x0_lambda=1.0)
    tgt = dict(x_rate_shift=0.9, x_trend_shift=0.0,
               x_noise_std_shift=0.4, x0_lambda_shift=2.0)

    p = AdaptedCAFHT(alpha=0.1, weight_mode="oracle_dynamic",
                     dynamic_source_params=src, dynamic_target_params=tgt)
    p.fit_ar_model(cal_Y)
    p.calibrate(cal_Y, cal_X=cal_X)

    assert p._cal_log_lr_unnorm is not None and len(p._cal_log_lr_unnorm) == n_cal
    assert p._t_pred == T - 2  # cal_Y has T cols → outer t = T-2
    assert p._last_ess is not None and 0 < p._last_ess <= n_cal

    # Predict for one test series
    test_x_path = rng.normal(loc=0.5, scale=0.3, size=T)
    test_x_path[0] = 1.0
    input_series = rng.normal(size=(T - 1, 1))
    pred, lo, hi = p.predict_with_interval_oracle_dynamic(
        input_series, test_x_path=test_x_path, alpha_level=0.1)
    assert hi > lo and np.isfinite(pred) and np.isfinite(lo) and np.isfinite(hi), \
        f"non-finite/inverted interval: pred={pred}, lo={lo}, hi={hi}"
    print(f"[ok] predict_with_interval_oracle_dynamic: ESS={p._last_ess:.1f}/{n_cal}, "
          f"interval half-width={(hi-lo)/2:.3f}")


if __name__ == "__main__":
    test_oracle_poisson_matches_closed_form()
    test_oracle_no_shift_is_uniform()
    test_oracle_degenerate_params_falls_back_to_uniform()
    test_ess_known_vector()
    test_calibrate_routes_to_oracle_when_cal_X_provided()
    test_predict_with_interval_oracle_includes_test_point()
    test_oracle_dynamic_matches_scipy_norm()
    test_oracle_dynamic_no_shift_is_zero()
    test_predict_with_interval_oracle_dynamic()
    print("\nAll oracle-weight sanity checks passed.")
