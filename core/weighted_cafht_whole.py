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
.predict_with_interval_oracle / ._weighted_quantile (§ B.3). The ACI loop and
the additive nonconformity score mirror
CAFHT/ConformalizedTS/methods.py:Adaptive_Conformal_Inference and
CAFHT.nonconf_scores (additive branch, line 313).

Run `python -m core.weighted_cafht_whole` to execute the inline sanity tests.
"""

import numpy as np
import numpy.linalg as la

# --- optional scipy: ACI band quantile uses CAFHT's mquantiles convention ----
try:
    from scipy.stats.mstats import mquantiles
    _SCIPY = True
except Exception:                                       # pragma: no cover
    _SCIPY = False

# --- optional sklearn: LR classifier for the density-ratio weights -----------
try:
    from sklearn.linear_model import LogisticRegression
    _SKLEARN = True
except Exception:                                       # pragma: no cover
    _SKLEARN = False


# =============================================================================
# Small numeric helpers
# =============================================================================
def _resid_inf_norm(y_pred, y_true):
    """Per-time-step nonconformity residual, collapsing the ndim axis with the
    ∞-norm — exactly CAFHT's `la.norm(pred - true, np.inf, axis=1)` convention.

    y_pred, y_true : (..., horizon, ndim)
    returns        : (..., horizon)   — |y_pred - y_true| max-ed over ndim.
    """
    return la.norm(np.asarray(y_pred, float) - np.asarray(y_true, float),
                   np.inf, axis=-1)


def _quantile_band(bank_column, level):
    """Half-width of an ACI band at a single step: the `level`-quantile of that
    step's column of the score bank. Uses scipy's `mquantiles` to match CAFHT
    exactly; falls back to `np.quantile` (linear) when scipy is unavailable.

    `bank_column` is the held-out residuals AT a single time step. `level` is
    1 - α_t (the per-step adaptive coverage level). A larger level => a wider
    band. Returns 0.0 for an empty column (degenerate guard).
    """
    bank = np.asarray(bank_column, float)
    bank = bank[np.isfinite(bank)]
    if bank.size == 0:
        return 0.0
    level = float(np.clip(level, 0.0, 1.0))
    if _SCIPY:
        return float(mquantiles(bank, prob=level)[0])
    return float(np.quantile(bank, level))


# =============================================================================
# § 2.1  Shared primitive 1 — weighted quantile with a δ_∞ atom
# =============================================================================
def weighted_quantile_with_inf(scores, w_cal, w_test, level):
    """Tibshirani-Foygel-Barber-Candès-Ramdas (2019) weighted-exchangeability
    quantile with a point mass δ_∞ carrying the test point's own weight.

    Implements the correction term of Algorithm 1's deployment step:

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

    Mirrors Algorithm 1's "learning the weights" block: a logistic classifier
    is trained on
        X_pos (label 0)  ==  source / reference  ==  X_1 of D_tr
        X_neg (label 1)  ==  target              ==  X_1 of D_test^(half)
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
# § 2.2  ACI updater — drives the per-step adaptive coverage level α_t
# =============================================================================
class ACI:
    """Single-pass Adaptive Conformal Inference over a batch of trajectories.

    Mirrors CAFHT/ConformalizedTS/methods.py:Adaptive_Conformal_Inference, with
    one deliberate change that resolves the spec gap flagged in § 2.0 / Q8:

        the per-step calibration quantile is taken over an EXTERNAL, FROZEN
        `score_bank` supplied by the caller — NOT over the trajectory's own past
        residuals plus random warm-start noise (CAFHT's approach).

    Because the bank is always fully populated, there is no cold-start problem
    and no warm-start is needed. The update rule is the standard ACI recursion

        band_t      = [ŷ_t − q_t , ŷ_t + q_t],   q_t = Quantile(bank[:, t], 1 − α_t)
        err_t       = 1[ |y_t − ŷ_t|_∞  >  q_t ]                (miscoverage)
        α_{t+1}     = clip( α_t + γ · (α − err_t) ,  1e-6 , 1 − 1e-6 ).

    `score_bank` is a 2-D (n_bank, horizon) array: column t holds the held-out
    residuals AT TIME t, and the band at step t is built from THAT COLUMN only
    (NOT a pool over all time steps). So the base half-width is time-varying —
    it tracks how the residual magnitude changes along the horizon — and the
    adaptive level α_t modulates it on top.
    """

    def __init__(self, alpha=0.1, verbose=False):
        self.alpha = float(alpha)
        self.verbose = verbose

    def predict_intervals(self, score_bank, y_pred, y_true,
                          gamma=0.1, seed=123):
        """Run ACI for every trajectory in (y_pred, y_true).

        Parameters
        ----------
        score_bank : (n_bank, horizon) array of absolute residuals — column t
                     is the score pool for step t. The frozen D_ACI bank for the
                     main algorithm, or the D_tr^(2) sandbox bank during γ
                     selection. Its horizon must match y_pred's.
        y_pred, y_true : (n, horizon, ndim) predictions and truths.
        gamma : ACI learning rate.
        seed  : accepted for API symmetry with CAFHT; unused (no warm-start,
                so the output is deterministic given the inputs).

        Returns
        -------
        (n, horizon, 2, ndim) array; [..., 0, :] = lower, [..., 1, :] = upper.
        """
        y_pred = np.asarray(y_pred, float)
        y_true = np.asarray(y_true, float)
        n, horizon, ndim = y_pred.shape

        # 2-D bank: column t is the held-out residual pool for step t (§ 2.0).
        bank = np.asarray(score_bank, float)
        assert bank.ndim == 2 and bank.shape[1] == horizon, (
            f"score_bank must be (n_bank, horizon={horizon}); got {bank.shape}")

        resid = _resid_inf_norm(y_pred, y_true)        # (n, horizon)
        out = np.empty((n, horizon, 2, ndim))

        for k in range(n):
            alpha_t = self.alpha                       # α_1 = α  (per algo box)
            for t in range(horizon):
                a = float(np.clip(alpha_t, 1e-6, 1.0 - 1e-6))
                # half-width from THIS step's score column, at level 1-α_t.
                q = _quantile_band(bank[:, t], 1.0 - a)

                out[k, t, 0, :] = y_pred[k, t, :] - q
                out[k, t, 1, :] = y_pred[k, t, :] + q

                # miscoverage indicator, then the ACI level update (clipped).
                err = 1.0 if resid[k, t] > q else 0.0
                alpha_t = float(np.clip(alpha_t + gamma * (self.alpha - err),
                                        1e-6, 1.0 - 1e-6))
        return out


# =============================================================================
# Additive nonconformity score  (ε_i)  — CAFHT nonconf_scores, line 313
# =============================================================================
def _additive_nonconf_scores(bands, y_true):
    """Whole-trajectory conformity score for each trajectory (additive branch):

        ε_i = max_{t} max{ Y_t − U_t , L_t − Y_t , 0 }

    i.e. the largest signed amount by which the truth falls OUTSIDE the ACI
    band anywhere along the trajectory (0 if always covered). This is the
    additive branch of CAFHT.nonconf_scores (methods.py:313); the +0 clip keeps
    the correction non-negative, matching CAFHT (the algorithm box writes the
    unclipped max{Y−U, L−Y}, identical whenever any step is uncovered).

    bands  : (n, horizon, 2, ndim)
    y_true : (n, horizon, ndim)
    returns: (n,) conformity scores ε_i ≥ 0.
    """
    low = bands[:, :, 0, :]
    high = bands[:, :, 1, :]
    over = np.maximum(0.0, y_true - high)              # truth above the band
    under = np.maximum(0.0, low - y_true)              # truth below the band
    # max over both (horizon, ndim) axes per trajectory.
    return np.maximum(over.max(axis=(1, 2)), under.max(axis=(1, 2)))


# =============================================================================
# § 2.3  Algorithm 1 — Weighted CAFHT, whole-trajectory coverage
# =============================================================================
class WeightedCAFHTWholeTrajectory:
    """Whole-trajectory coverage:  P(∀t: Y_t ∈ Ĉ_t) ≥ 1 − α.

    Predictor-agnostic. The runner fits the per-step predictors {f_t} on the
    FULL D_tr, peels a disjoint D_ACI off the raw pool (§ 2.0), and hands this
    class the precomputed (pred, true) arrays for D_tr, D_cal, D_test, D_ACI,
    plus the raw covariate arrays for the LR classifier. `featurize_fn` turns a
    raw covariate array into the X_1 feature matrix the classifier consumes.
    """

    def __init__(self, alpha, gamma_grid, featurize_fn,
                 gamma_split=(0.33, 0.33, 0.34),
                 weight_clip=5.0, randomize=False, verbose=True):
        self.alpha = float(alpha)
        self.gamma_grid = list(gamma_grid)
        self.featurize_fn = featurize_fn
        self.gamma_split = tuple(gamma_split)          # internal D_tr split
        self.weight_clip = weight_clip
        self.randomize = randomize
        self.verbose = verbose

        # diagnostics populated by predict_bands (for the runner to log)
        self.gamma_opt_ = None
        self.eps_ = None
        self.score_bank_shape_ = None
        self.gamma_widths_ = None
        self.n_inf_ = 0

    # ----- "fitting prediction models and choosing γ" block ------------------
    def select_gamma(self, tr_pred, tr_true, seed=123):
        """Choose γ by minimum mean band width on an internal 3-way split of
        D_tr (Q4). The algorithm box's "simple ACI with train=D_tr^(1),
        cal=D_tr^(2), test=D_tr^(3)" maps to:

            * predictors are ALREADY fit globally on the full D_tr, so the
              D_tr^(1) "training" portion is conceptually already consumed —
              we carve it out for faithfulness but do not re-use it here;
            * D_tr^(2) supplies the SANDBOX score bank (its abs residuals);
            * D_tr^(3) is where we run ACI and measure mean band width.

        Note the sandbox bank comes from D_tr^(2), NOT from the frozen D_ACI
        bank — these are two distinct ACI invocations (§ 2.0).
        """
        tr_pred = np.asarray(tr_pred, float)
        tr_true = np.asarray(tr_true, float)
        n = tr_pred.shape[0]

        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        f1, f2, _f3 = self.gamma_split
        n1 = int(round(f1 * n))
        n2 = int(round(f2 * n))
        idx2 = perm[n1:n1 + n2]                         # D_tr^(2): sandbox bank
        idx3 = perm[n1 + n2:]                           # D_tr^(3): eval
        # Guard tiny n: ensure both subsets are non-empty.
        if idx2.size == 0 or idx3.size == 0:
            half = max(1, n // 2)
            idx2, idx3 = perm[:half], perm[half:]

        # sandbox bank: (n_tr2, horizon) — column t used at step t (not pooled).
        sandbox_bank = _resid_inf_norm(tr_pred[idx2], tr_true[idx2])

        aci = ACI(self.alpha, verbose=False)
        widths = []
        for g in self.gamma_grid:
            bands = aci.predict_intervals(sandbox_bank,
                                          tr_pred[idx3], tr_true[idx3],
                                          gamma=g, seed=seed)
            widths.append(float(np.mean(bands[:, :, 1, :] - bands[:, :, 0, :])))

        self.gamma_widths_ = list(zip(self.gamma_grid, widths))
        g_opt = float(self.gamma_grid[int(np.argmin(widths))])
        if self.verbose:
            print(f"[select_gamma] widths={dict(self.gamma_widths_)} "
                  f"-> gamma_opt={g_opt}")
        return g_opt

    # ----- "calibration" block -----------------------------------------------
    def calibration_scores(self, cal_pred, cal_true, score_bank,
                           gamma_opt, seed=123):
        """ε_i for every calibration trajectory: run the main-algorithm ACI
        (frozen D_ACI bank, γ_opt) to get bands, then take the additive
        whole-trajectory nonconformity score. One scalar per cal series
        (preserves the v2 single-score-per-series calibration; § C check 5)."""
        aci = ACI(self.alpha, verbose=False)
        bands = aci.predict_intervals(score_bank,
                                      np.asarray(cal_pred, float),
                                      np.asarray(cal_true, float),
                                      gamma=gamma_opt, seed=seed)
        eps = _additive_nonconf_scores(bands, np.asarray(cal_true, float))
        if self.randomize:                             # CAFHT tie-break jitter
            eps = eps + eps * np.random.uniform(0.0, 1e-3, size=eps.shape)
        return eps

    # ----- "learning the weights" + "deployment" blocks ----------------------
    def predict_bands(self, tr_data, cal_data, test_data, aci_data,
                      X_tr, X_cal, X_test, y_trim=None, seed=123):
        """Produce the online prediction band for every test trajectory.

        Parameters
        ----------
        tr_data   : (tr_pred,  tr_true)   full D_tr  — γ selection splits it.
        cal_data  : (cal_pred, cal_true)  D_cal      — calibration ε_i.
        test_data : (test_pred,test_true) D_test     — cross-half deployment.
        aci_data  : (aci_pred, aci_true)  D_ACI      — builds the frozen bank.
        X_tr, X_cal, X_test : raw covariate arrays for the LR classifier. X_tr
            is the classifier's NEGATIVE class (label 0) per the algorithm box;
            it is not in the plan's original § 2.3 signature but the box's
            "feed {(X_1^i,0)}_{i∈D_tr}" requires it, so it is passed explicitly.
        y_trim : optional [lo, hi] clip applied to the final band.

        Returns
        -------
        (n_test, horizon, 2, ndim) prediction bands.
        """
        tr_pred, tr_true = tr_data
        cal_pred, cal_true = cal_data
        test_pred = np.asarray(test_data[0], float)
        test_true = np.asarray(test_data[1], float)
        aci_pred, aci_true = aci_data

        n_test, horizon, ndim = test_pred.shape

        # (1) frozen D_ACI score bank: abs residuals on the held-out D_ACI
        #     trajectories, kept as a 2-D (n_ACI, horizon) array so ACI draws a
        #     SEPARATE score pool per time step (column t builds the band at
        #     step t — NOT a pool over the whole horizon; § 2.0).
        score_bank = _resid_inf_norm(np.asarray(aci_pred, float),
                                     np.asarray(aci_true, float))
        self.score_bank_shape_ = tuple(score_bank.shape)

        # (2) choose γ on the internal 3-way split of D_tr.
        gamma_opt = self.select_gamma(tr_pred, tr_true, seed=seed)
        self.gamma_opt_ = gamma_opt

        # (3) calibration scores ε_i. They do not depend on the cross-half (the
        #     ACI bands are weight-free), so we compute them ONCE here even
        #     though the algo box nests calibration inside the half loop.
        eps = self.calibration_scores(cal_pred, cal_true, score_bank,
                                      gamma_opt, seed=seed)
        self.eps_ = eps

        # X_1 features for the LR classifier.
        F_tr = np.asarray(self.featurize_fn(X_tr), float)
        F_cal = np.asarray(self.featurize_fn(X_cal), float)
        F_test = np.asarray(self.featurize_fn(X_test), float)

        # (4) cross-half deployment. Split D_test into two halves; for each
        #     deploy half, the classifier's positives are the OPPOSITE half so
        #     a test point never informs its own weight (no leakage).
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_test)
        cut = n_test // 2
        half_a, half_b = perm[:cut], perm[cut:]

        out = np.empty((n_test, horizon, 2, ndim))
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

            # ACI bands for the deploy half (frozen D_ACI bank, γ_opt).
            aci_bands = ACI(self.alpha, verbose=False).predict_intervals(
                score_bank, test_pred[deploy_idx], test_true[deploy_idx],
                gamma=gamma_opt, seed=seed)

            for local, j in enumerate(deploy_idx):
                # correction term η_j from the δ_∞ weighted quantile.
                eta = weighted_quantile_with_inf(eps, W_cal,
                                                 float(W_dep[local]), level)
                band = aci_bands[local]                # (horizon, 2, ndim)
                if not np.isfinite(eta):
                    # δ_∞ atom reached => unbounded band (honest answer).
                    low = np.full((horizon, ndim), -np.inf)
                    high = np.full((horizon, ndim), np.inf)
                    self.n_inf_ += 1
                else:
                    # inflate the ACI band: [min C^aci - η, max C^aci + η].
                    low = band[:, 0, :] - eta
                    high = band[:, 1, :] + eta
                if y_trim is not None:
                    low = np.maximum(low, y_trim[0])
                    high = np.minimum(high, y_trim[1])
                out[j, :, 0, :] = low
                out[j, :, 1, :] = high

        return out


# =============================================================================
# Inline sanity tests  —  `python -m core.weighted_cafht_whole`
# =============================================================================
def _test_weighted_quantile_uniform():
    """Uniform cal weights with no test atom (w_test=0) reduce to the standard
    empirical 'inverted_cdf' quantile (smallest score whose cumulative count
    fraction reaches `level`)."""
    scores = np.arange(1.0, 11.0)                      # [1..10]
    w_cal = np.ones(10)
    for level in (0.3, 0.5, 0.7, 0.9):
        got = weighted_quantile_with_inf(scores, w_cal, 0.0, level)
        # reference: first score whose cumulative weight reaches level*Σw.
        cum = np.cumsum(w_cal)
        idx = int(np.searchsorted(cum, level * cum[-1], side="left"))
        exp = scores[idx]
        assert got == exp, f"level={level}: got {got}, exp {exp}"
    print("  [ok] weighted_quantile_with_inf reduces to empirical quantile")


def _test_weighted_quantile_inf_atom():
    """A test weight 100× the total cal mass forces the δ_∞ atom => +∞."""
    scores = np.arange(1.0, 11.0)
    w_cal = np.ones(10)
    w_test = 100.0 * float(np.sum(w_cal))
    got = weighted_quantile_with_inf(scores, w_cal, w_test, 0.9)
    assert got == np.inf, f"expected inf, got {got}"
    # all-zero cal weights also => inf.
    assert weighted_quantile_with_inf(scores, np.zeros(10), 1.0, 0.9) == np.inf
    print("  [ok] δ_∞ atom triggers +∞ for a dominant test weight")


def _test_density_ratio_identical():
    """Identical source/target samples => classifier at chance => weights ≈ 1."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    out = density_ratio_weights(X, X.copy(), X.copy())
    w = out["w_eval"]
    assert np.all(np.abs(w - 1.0) < 0.3), f"weights stray from 1: {w[:5]}"
    print(f"  [ok] density_ratio_weights ≈ 1 on identical dists "
          f"(train_acc={out['train_acc']:.2f})")


def _test_aci_gamma_zero_per_step_width():
    """With γ=0 the level α_t never moves, so at each step t the band half-width
    is the quantile of THAT STEP'S column => width_t = 2·Quantile(bank[:,t], 1−α),
    constant across trajectories but VARYING across t (per-step bank, not
    pooled)."""
    rng = np.random.default_rng(0)
    n_bank, horizon = 300, 8
    # column t has scale (t+1): the residual magnitude grows along the horizon.
    bank = np.abs(rng.normal(size=(n_bank, horizon)) * (np.arange(horizon) + 1.0))
    alpha = 0.1
    y_pred = rng.normal(size=(4, horizon, 1))
    y_true = y_pred + rng.normal(scale=0.5, size=(4, horizon, 1))
    bands = ACI(alpha).predict_intervals(bank, y_pred, y_true, gamma=0.0)
    widths = bands[:, :, 1, 0] - bands[:, :, 0, 0]     # (4, horizon)
    expected = np.array([2.0 * _quantile_band(bank[:, t], 1.0 - alpha)
                         for t in range(horizon)])
    # every trajectory shares the per-step width profile (γ=0 => α_t fixed).
    assert np.allclose(widths, expected[None, :]), "per-step width mismatch"
    # and the width genuinely changes across t (confirms per-step, not pooled).
    assert expected[-1] > 2.0 * expected[0], "width should grow with t"
    print(f"  [ok] ACI γ=0 per-step width tracks Q(bank[:,t],1−α): "
          f"{expected[0]:.2f}..{expected[-1]:.2f}")


def _test_predict_bands_smoke():
    """End-to-end wiring smoke test: shapes line up and (most) bands finite."""
    rng = np.random.default_rng(7)
    T, ndim = 6, 1

    def make(n):
        true = np.cumsum(rng.normal(scale=0.3, size=(n, T, ndim)), axis=1)
        pred = true + rng.normal(scale=0.2, size=(n, T, ndim))
        X = true[:, 0, :].copy()                       # X_1 proxy
        return pred, true, X

    tr_p, tr_t, X_tr = make(120)
    cal_p, cal_t, X_cal = make(80)
    te_p, te_t, X_te = make(60)
    aci_p, aci_t, _ = make(40)

    algo = WeightedCAFHTWholeTrajectory(
        alpha=0.1, gamma_grid=[0.005, 0.01, 0.05],
        featurize_fn=lambda X: np.asarray(X, float).reshape(len(X), -1),
        verbose=False)
    bands = algo.predict_bands((tr_p, tr_t), (cal_p, cal_t), (te_p, te_t),
                               (aci_p, aci_t), X_tr, X_cal, X_te, seed=0)
    assert bands.shape == (60, T, 2, ndim), bands.shape
    assert np.all(bands[..., 1, :] >= bands[..., 0, :]), "upper < lower"
    assert np.mean(np.isfinite(bands)) > 0.8, "too many infinite bands"
    print(f"  [ok] predict_bands smoke: shape={bands.shape}, "
          f"gamma_opt={algo.gamma_opt_}, n_inf={algo.n_inf_}, "
          f"D_ACI bank shape={algo.score_bank_shape_}")


if __name__ == "__main__":
    print("Running inline sanity tests for weighted_cafht_whole ...")
    _test_weighted_quantile_uniform()
    _test_weighted_quantile_inf_atom()
    _test_density_ratio_identical()
    _test_aci_gamma_zero_per_step_width()
    _test_predict_bands_smoke()
    print("OK")
