"""
core/weighted_cafht_last.py — Algorithm 2: Weighted CAFHT (last-step coverage).

Self-contained module implementing last-step coverage (see WEIGHTED_CAFHT_PLAN.md
§ 2.4). No γ, no ACI, no D_ACI, no per-step bands — just a single interval at the
final step T+1. Per the "one algorithm per file" ground rule (§ 0), the two
shared primitives are duplicated here VERBATIM from core/weighted_cafht_whole.py
(copied, NOT imported):

  - weighted_quantile_with_inf(scores, w_cal, w_test, level)   (§ 2.1)
  - density_ratio_weights(X_pos, X_neg, X_eval, ...)           (§ 2.1)
  - class WeightedCAFHTLastStep                                (§ 2.4)
        calibration_scores / predict_bands; cross-half + δ_∞ quantile only.

Coverage target: P(Y_{T+1} ∈ Ĉ_{T+1}) ≥ 1 − α.

Algorithm 2 (from the algo box):
  1. Fit one predictor f_{T+1} (regress Y_{T+1} on X_{1:T}); the runner supplies
     the precomputed scalar predictions Ŷ_{T+1} for D_cal and D_test.
  2. ε_i = |Y_{T+1}^(i) − Ŷ_{T+1}^(i)| for i ∈ D_cal (one scalar per series).
  3. Cross-half split of D_test. Per half: train the LR classifier on
     {(X_{1:T}^i, 0)}_{i∈D_tr} ∪ {(X_{1:T}^i, 1)}_{i∈D_test^(opp half)}; weight
     D_cal (clipped) and the deploy half (raw).
  4. For each deploy point j:
        η_j = weighted_quantile_with_inf(ε, Ŵ_cal, Ŵ_j, 1−α)
        Ĉ_{T+1}^j = [Ŷ_{T+1}^j − η_j, Ŷ_{T+1}^j + η_j].

NB on parameter order: this file uses the (pred, true) tuple convention for
cal_data / test_data, matching weighted_cafht_whole.py, so a runner author sees
one consistent ordering across both algorithms.

Run `python -m core.weighted_cafht_last` to execute the inline sanity tests.
"""

import numpy as np

# --- optional sklearn: LR classifier for the density-ratio weights -----------
try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN = True
except Exception:                                       # pragma: no cover
    _SKLEARN = False


# =============================================================================
# § 2.1  Shared primitive 1 — weighted quantile with a δ_∞ atom
#         (copied VERBATIM from core/weighted_cafht_whole.py — do not edit one
#          without the other; § 0 self-contained-file rule.)
# =============================================================================
def weighted_quantile_with_inf(scores, w_cal, w_test, level):
    """Tibshirani-Foygel-Barber-Candès-Ramdas (2019) weighted-exchangeability
    quantile with a point mass δ_∞ carrying the test point's own weight.

    Implements the correction term of Algorithm 2's deployment step:

        η_j = Quantile( Σ_{i∈D_cal} (Ŵ_i / S) δ_{ε_i}  +  (Ŵ_j / S) δ_∞ , 1-α )

    with S = Σ_{i∈D_cal} Ŵ_i + Ŵ_j (the denominator INCLUDES the test point's
    weight). Concretely we return the smallest cal score `s` such that

        Σ_{i : ε_i ≤ s} w_cal_i   ≥   level · ( Σ_k w_cal_k + w_test ).

    If even the full cal mass Σ w_cal is below that threshold, the remaining
    probability lives on the δ_∞ atom and the quantile is +∞ — an unbounded
    correction (the honest answer when the test point is far out of the
    calibration distribution). This is the generalized, δ_∞-augmented sibling of
    OLD_algorithm.py:_weighted_quantile / .predict_with_interval_oracle.

    Tie-breaking is the 'higher' convention (we take the first score whose
    cumulative weight reaches the threshold).

    Parameters
    ----------
    scores  : (n_cal,)  the calibration conformity scores ε_i (≥ 0).
    w_cal   : (n_cal,)  UNNORMALIZED cal weights Ŵ_i (raw masses).
    w_test  : float     UNNORMALIZED weight Ŵ_j of the single test point.
    level   : float     target coverage 1 - α.

    Returns
    -------
    float : the correction η_j, or np.inf if the δ_∞ atom must be reached.
    """
    scores = np.asarray(scores, float)
    w_cal = np.asarray(w_cal, float)

    # Drop non-finite cal entries (keep them paired).
    mask = np.isfinite(scores) & np.isfinite(w_cal) & (w_cal >= 0.0)
    scores, w_cal = scores[mask], w_cal[mask]

    w_test = float(w_test) if np.isfinite(w_test) and w_test > 0 else 0.0

    # No cal scores at all => only the δ_∞ atom remains => +∞.
    if scores.size == 0:
        return np.inf

    total = float(np.sum(w_cal)) + w_test          # S = Σ w_cal + w_test
    if not np.isfinite(total) or total <= 0.0:
        return np.inf

    threshold = level * total                      # mass we must accumulate

    # Sort scores ascending and walk the cumulative cal weight.
    order = np.argsort(scores, kind="mergesort")   # stable sort for tie-breaking
    s_sorted = scores[order]
    cumw = np.cumsum(w_cal[order])

    # Smallest index whose cumulative weight reaches `threshold` ('higher').
    idx = int(np.searchsorted(cumw, threshold, side="left"))
    if idx >= s_sorted.size:
        # Even all cal mass is short of the threshold => the δ_∞ atom is needed.
        return np.inf
    return float(s_sorted[idx])


# =============================================================================
# § 2.1  Shared primitive 2 — density-ratio (likelihood-ratio) weights
#         (copied VERBATIM from core/weighted_cafht_whole.py.)
# =============================================================================
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logreg_fit_gd(X, y, lr=0.1, iters=500, l2=1e-4):
    """Plain L2-regularized logistic regression by gradient descent.
    Fallback for when sklearn is unavailable. Copied from
    OLD_algorithm.py:_logreg_fit_gd. Returns weights [b, w_1..w_d]."""
    n, d = X.shape
    w = np.zeros(d + 1)
    Xb = np.column_stack([np.ones(n), X])
    for _ in range(iters):
        p = _sigmoid(Xb @ w)
        grad = Xb.T @ (p - y) / n + l2 * np.r_[0.0, w[1:]]
        w -= lr * grad
    return w


def density_ratio_weights(X_pos, X_neg, X_eval, *,
                          classifier=None,
                          clip_factor=5.0,
                          prob_clip=1e-6):
    """Estimate the likelihood ratio dP̃ / dP via a probabilistic classifier.

    Mirrors Algorithm 2's "learning the weights" block: a logistic classifier
    is trained on
        X_pos (label 0)  ==  source / reference  ==  X_{1:T} of D_tr
        X_neg (label 1)  ==  target              ==  X_{1:T} of D_test^(half)
    and the per-point weight is the odds
        ŵ(x) = p̂(label=1 | x) / p̂(label=0 | x),
    which, with `class_weight='balanced'` (removing the class-prior ratio),
    estimates p_target(x) / p_source(x) — the importance weight that reshapes
    the calibration distribution toward the test distribution. Logic adapted
    from OLD_algorithm.py:_compute_density_ratio_weights.

    IMPORTANT — UNNORMALIZED output. Unlike the old code we do NOT divide the
    weights by their sum: the δ_∞ weighted quantile needs raw masses so that
    the cal weights and the test-point weight live on a common scale and the
    `Σ w_cal + w_test` denominator is meaningful (§ 2.1).

    The returned `w_eval` is clipped at `clip_factor × mean` (the old 5×-mean
    guard against a single point dominating). The companion `weight_fn` returns
    RAW (only prob-clipped, NOT 5×-mean-clipped) weights — the caller uses it
    for the test point so a genuinely out-of-distribution test weight can stay
    large and trip the δ_∞ atom (exactly what OLD_algorithm's oracle path does
    by clipping cal weights only).

    Parameters
    ----------
    X_pos   : (n0, d)   features labelled 0 (source / D_tr).
    X_neg   : (n1, d)   features labelled 1 (target / D_test half).
    X_eval  : (m,  d)   features to score (e.g. D_cal).
    classifier : an unfitted sklearn-style estimator, or None for the default
                 balanced LogisticRegression.
    clip_factor : cap on `w_eval` at this multiple of its mean (None/≤0 = off).
    prob_clip   : clip predicted probabilities to [prob_clip, 1-prob_clip].

    Returns
    -------
    dict with:
        "w_eval"     : (m,)  clipped, unnormalized weights for X_eval.
        "w_eval_raw" : (m,)  raw (only prob-clipped) weights for X_eval.
        "prob1_eval" : (m,)  p̂(label=1 | X_eval).
        "weight_fn"  : callable(Xq) -> raw weights, reusing the fitted model.
        "train_acc"  : float training accuracy (separability diagnostic).
        "coef_norm"  : float ‖coef‖₂ (0.0 for the GD fallback's raw coef).
    """
    X_pos = np.asarray(X_pos, float)
    X_neg = np.asarray(X_neg, float)
    X_eval = np.asarray(X_eval, float)

    # Degenerate inputs => no information => uniform weights.
    if X_pos.size == 0 or X_neg.size == 0:
        ones = np.ones(X_eval.shape[0], float)
        return {"w_eval": ones, "w_eval_raw": ones.copy(),
                "prob1_eval": np.full(X_eval.shape[0], 0.5),
                "weight_fn": (lambda Xq: np.ones(np.asarray(Xq).shape[0], float)),
                "train_acc": 0.5, "coef_norm": 0.0}

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.zeros(len(X_pos), int), np.ones(len(X_neg), int)])

    if _SKLEARN:
        clf = classifier if classifier is not None else LogisticRegression(
            max_iter=1000, solver="lbfgs", class_weight="balanced")
        clf.fit(X, y)

        def _prob1(Xq):
            return clf.predict_proba(np.asarray(Xq, float))[:, 1]

        train_acc = float(clf.score(X, y))
        coef_norm = float(np.linalg.norm(np.ravel(clf.coef_)))
    else:                                               # pragma: no cover
        w = _logreg_fit_gd(X, y)

        def _prob1(Xq):
            Xq = np.asarray(Xq, float)
            return _sigmoid(Xq @ w[1:] + w[0])

        pred = (_prob1(X) >= 0.5).astype(int)
        train_acc = float(np.mean(pred == y))
        coef_norm = float(np.linalg.norm(w[1:]))

    def weight_fn(Xq):
        """Raw (only prob-clipped) likelihood-ratio weights for new points."""
        p1 = np.clip(_prob1(Xq), prob_clip, 1.0 - prob_clip)
        # class_weight='balanced' => p1/(1-p1) is already a pure LR (no prior).
        return p1 / (1.0 - p1)

    raw_eval = weight_fn(X_eval)
    prob1_eval = np.clip(_prob1(X_eval), prob_clip, 1.0 - prob_clip)

    # Clip cal weights at clip_factor × mean (regularization; old 5×-mean guard).
    w_eval = raw_eval.copy()
    if clip_factor is not None and clip_factor > 0:
        m = float(np.mean(raw_eval))
        if np.isfinite(m) and m > 0:
            w_eval = np.minimum(w_eval, clip_factor * m)

    return {"w_eval": w_eval, "w_eval_raw": raw_eval, "prob1_eval": prob1_eval,
            "weight_fn": weight_fn, "train_acc": train_acc, "coef_norm": coef_norm}


# =============================================================================
# Last-step conformity score  (ε_i)
# =============================================================================
def _as_2d(y):
    """Coerce per-series last-step values to (n, ndim)."""
    y = np.asarray(y, float)
    return y.reshape(-1, 1) if y.ndim == 1 else y


def _default_score(y_pred, y_true):
    """Default last-step nonconformity score: the ∞-norm absolute residual at
    T+1, one scalar per series (= |Y_{T+1} − Ŷ_{T+1}| when ndim == 1).

    y_pred, y_true : (n, ndim)   ->   (n,)
    """
    return np.max(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)),
                  axis=-1)


# =============================================================================
# § 2.4  Algorithm 2 — Weighted CAFHT, last-step coverage
# =============================================================================
class WeightedCAFHTLastStep:
    """Last-step coverage:  P(Y_{T+1} ∈ Ĉ_{T+1}) ≥ 1 − α.

    No γ, no ACI, no D_ACI, no per-step bands — a single split-conformal
    interval at the final step, reweighted by the estimated likelihood ratio
    with the Tibshirani δ_∞ correction. Predictor-agnostic: the runner fits one
    predictor f_{T+1} and passes the precomputed scalar predictions Ŷ_{T+1} for
    D_cal and D_test. `featurize_fn` turns a raw covariate array into the
    X_{1:T} feature matrix the classifier consumes (per § 3.3, the whole
    covariate path is used for the last-step LR).
    """

    def __init__(self, alpha, featurize_fn, weight_clip=5.0,
                 score_fn=None, verbose=True):
        self.alpha = float(alpha)
        self.featurize_fn = featurize_fn
        self.weight_clip = weight_clip
        # score_fn(y_pred, y_true) -> (n,) scalar scores; default abs residual.
        # NB: predict_bands builds the symmetric box [Ŷ ± η]; that is the exact
        # level set {y : score ≤ η} only for an abs-/∞-norm-type score (default).
        self.score_fn = score_fn if score_fn is not None else _default_score
        self.verbose = verbose

        # diagnostics populated by predict_bands (for the runner to log)
        self.eps_ = None
        self.n_inf_ = 0

    def calibration_scores(self, y_pred, y_true):
        """ε_i = score_fn(Ŷ_{T+1}^(i), Y_{T+1}^(i)) — one scalar per cal series."""
        return self.score_fn(_as_2d(y_pred), _as_2d(y_true))

    def predict_bands(self, cal_data, test_data, X_tr, X_cal, X_test,
                      y_trim=None, seed=123):
        """Produce the last-step prediction interval for every test series.

        Parameters
        ----------
        cal_data  : (cal_pred, cal_true)   D_cal last-step preds/truths.
        test_data : (test_pred, test_true) D_test; only test_pred centers the
                    band (last-step has no online adaptation), test_true is
                    accepted for API symmetry and ignored here.
        X_tr, X_cal, X_test : raw covariate arrays for the LR classifier. X_tr
            is the classifier's NEGATIVE class (label 0) per the algorithm box's
            "{(X_{1:T}^i, 0)}_{i∈D_tr}"; like § 2.3 it is passed explicitly
            (the plan's § 2.4 signature omitted it).
        y_trim : optional [lo, hi] clip applied to the final band.

        Returns
        -------
        (n_test, 1, 2, ndim) bands — horizon axis of length 1 (the T+1 step),
        matching weighted_cafht_whole's (n, horizon, 2, ndim) layout so the
        runners share one coverage/width routine.
        """
        cal_pred, cal_true = cal_data
        test_pred = _as_2d(test_data[0])               # (n_test, ndim)
        n_test, ndim = test_pred.shape

        # (1) calibration scores ε_i (one scalar per cal series).
        eps = self.calibration_scores(cal_pred, cal_true)
        self.eps_ = eps

        # X_{1:T} features for the LR classifier.
        F_tr = np.asarray(self.featurize_fn(X_tr), float)
        F_cal = np.asarray(self.featurize_fn(X_cal), float)
        F_test = np.asarray(self.featurize_fn(X_test), float)

        # (2) cross-half deployment. Split D_test into two halves; for each
        #     deploy half the classifier's positives are the OPPOSITE half so a
        #     test point never informs its own weight (no leakage).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_test)
        cut = n_test // 2
        half_a, half_b = perm[:cut], perm[cut:]

        out = np.empty((n_test, 1, 2, ndim))
        level = 1.0 - self.alpha
        self.n_inf_ = 0

        for pos_idx, deploy_idx in ((half_a, half_b), (half_b, half_a)):
            if deploy_idx.size == 0:
                continue

            # Classifier: negatives = D_tr (label 0), positives = the opposite
            # test half (label 1). Score D_cal (clipped) for Ŵ_i and the deploy
            # half (RAW) for Ŵ_j — raw test weights keep the δ_∞ atom alive.
            dr = density_ratio_weights(F_tr, F_test[pos_idx], F_cal,
                                       clip_factor=self.weight_clip)
            W_cal = dr["w_eval"]
            W_dep = dr["weight_fn"](F_test[deploy_idx])

            for local, j in enumerate(deploy_idx):
                eta = weighted_quantile_with_inf(eps, W_cal,
                                                 float(W_dep[local]), level)
                if not np.isfinite(eta):
                    # δ_∞ atom reached => unbounded interval (honest answer).
                    low = np.full(ndim, -np.inf)
                    high = np.full(ndim, np.inf)
                    self.n_inf_ += 1
                else:
                    # symmetric box: Ĉ_{T+1}^j = [Ŷ_j − η_j, Ŷ_j + η_j].
                    low = test_pred[j] - eta
                    high = test_pred[j] + eta
                if y_trim is not None:
                    low = np.maximum(low, y_trim[0])
                    high = np.minimum(high, y_trim[1])
                out[j, 0, 0, :] = low
                out[j, 0, 1, :] = high

        return out


# =============================================================================
# Inline sanity tests  —  `python -m core.weighted_cafht_last`
# =============================================================================
def _test_helpers_identical_to_whole():
    """The two duplicated primitives must behave identically to the step-4
    versions in weighted_cafht_whole.py (they are verbatim copies)."""
    from core.weighted_cafht_whole import (
        weighted_quantile_with_inf as wq_whole,
        density_ratio_weights as dr_whole)

    rng = np.random.default_rng(123)
    scores = rng.random(50)
    w_cal = rng.random(50) + 0.1
    for wt in (0.0, 0.5, 5.0, 1000.0):
        a = weighted_quantile_with_inf(scores, w_cal, wt, 0.9)
        b = wq_whole(scores, w_cal, wt, 0.9)
        assert (a == b) or (np.isinf(a) and np.isinf(b)), (wt, a, b)

    X_pos = rng.normal(size=(60, 4))
    X_neg = rng.normal(loc=0.6, size=(60, 4))
    X_eval = rng.normal(size=(30, 4))
    d_here = density_ratio_weights(X_pos, X_neg, X_eval)
    d_whole = dr_whole(X_pos, X_neg, X_eval)
    assert np.allclose(d_here["w_eval"], d_whole["w_eval"])
    assert np.allclose(d_here["w_eval_raw"], d_whole["w_eval_raw"])
    print("  [ok] helpers behave identically to weighted_cafht_whole versions")


def _test_uniform_reduces_to_split_conformal():
    """With uniform weights (a classifier that cannot separate the classes), the
    last-step band reduces to the standard split-conformal interval
    [Ŷ ± q_(1-α)] where q is the δ_∞-augmented (n+1)-corrected quantile of the
    calibration residuals — identical half-width for every test point."""
    rng = np.random.default_rng(0)
    n_cal, n_test, n_tr = 200, 100, 150
    cal_true = rng.normal(size=n_cal)
    cal_pred = cal_true + rng.normal(scale=0.5, size=n_cal)
    test_pred = rng.normal(size=n_test)
    test_true = test_pred + rng.normal(scale=0.5, size=n_test)

    alpha = 0.1
    # constant features => classifier at chance => uniform weights (ratio ≈ 1).
    algo = WeightedCAFHTLastStep(
        alpha=alpha, featurize_fn=lambda X: np.zeros((len(X), 1)), verbose=False)
    bands = algo.predict_bands((cal_pred, cal_true), (test_pred, test_true),
                               X_tr=np.zeros((n_tr, 1)),
                               X_cal=np.zeros((n_cal, 1)),
                               X_test=np.zeros((n_test, 1)), seed=0)

    assert bands.shape == (n_test, 1, 2, 1), bands.shape
    assert np.all(bands[:, 0, 1, 0] >= bands[:, 0, 0, 0]), "upper < lower"

    # expected: weighted_quantile_with_inf with uniform cal weights and a unit
    # test atom (Ŵ_j ≈ 1) — the textbook (n+1)-corrected conformal quantile.
    eps = np.abs(cal_true - cal_pred)
    expected_q = weighted_quantile_with_inf(eps, np.ones(n_cal), 1.0, 1 - alpha)
    half = (bands[:, 0, 1, 0] - bands[:, 0, 0, 0]) / 2.0
    centers = (bands[:, 0, 1, 0] + bands[:, 0, 0, 0]) / 2.0
    assert np.allclose(half, expected_q, atol=1e-9), \
        f"half-width {half[:3]} != split-conformal q {expected_q}"
    assert np.allclose(centers, test_pred), "band not centered on Ŷ"
    print(f"  [ok] uniform weights -> split-conformal band [Ŷ ± {expected_q:.3f}]")


def _test_delta_inf_band():
    """A test point whose raw weight dwarfs the cal mass trips the δ_∞ atom and
    yields an unbounded interval (or the y_trim box when trimming is on)."""
    rng = np.random.default_rng(2)
    n_cal = 100
    eps = np.abs(rng.normal(size=n_cal))
    # Directly exercise the deployment quantile (the mechanism predict_bands uses).
    huge = 100.0 * float(np.sum(np.ones(n_cal)))
    assert weighted_quantile_with_inf(eps, np.ones(n_cal), huge, 0.9) == np.inf
    print("  [ok] dominant test weight -> δ_∞ -> unbounded last-step interval")


if __name__ == "__main__":
    print("Running inline sanity tests for weighted_cafht_last ...")
    _test_helpers_identical_to_whole()
    _test_uniform_reduces_to_split_conformal()
    _test_delta_inf_band()
    print("OK")
