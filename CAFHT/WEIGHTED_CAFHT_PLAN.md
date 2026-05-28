# Implementation plan — Weighted CAFHT (CAFHT + likelihood-ratio reweighting)

This plan describes how to add the algorithm in the user's pseudocode to the cloned `CAFHT/` codebase. The new method is structurally CAFHT with two changes:

1. The trajectory-level quantile over calibration scores becomes a **likelihood-ratio-weighted** quantile.
2. The quantile uses the **Tibshirani et al. (2019)** weighted-exchangeability normalization (test-point atom at ∞).

Everything else — the per-time-step predictor, the inner ACI loop, γ selection by minimum band width on a held-out training fold — is reused from existing code.

---

## Open questions (resolve before coding)

These were intentionally left open. Each has a concrete impact on the implementation and should be answered before writing code; see the "Decision-dependent code paths" section below for what changes once each is decided.

### Q1. Predictor model — LSTM or linear / AR?
- **CAFHT default**: a 4-layer LSTM (`ConformalizedTS/networks.py:MyLSTM`) trained via `Blackbox.full_train` and iterated through `predict_iterate`.
- **Parent-repo style**: a per-time-step OLS regressor (AR(1) for synthetic; linear-covariate for finance/medical), refit at every prediction step.
- The pseudocode's `f_t` ("regress Y_t on (Z_{1:(t-1)}, X_t) for all i ∈ D_tr") is more naturally read as the parent-repo style, **but** the CAFHT LSTM produces a *trajectory* of forecasts in a single pass and works fine if we interpret `f_t` as "the t-th component of the LSTM forecast". Both are valid; we have not committed.
- **Impact**: changes only what we pass into `WeightedCAFHT.predict_bands` (the precomputed `(test_pred, test_true)` / `(calib_pred, calib_true)` tuples). The conformal layer itself is predictor-agnostic.

### Q2. LR-classifier feature representation — dataset-dependent
- The pseudocode says "feeding `{(X_1^{(i)}, 0)}` and `{(X_1^{(i)}, 1)}`". Literally: use only the first covariate vector.
- The parent repo (`core/algorithm.py:_compute_density_ratio_weights`, `medical/medical_conformal.py:_richer_featurize_prefixes`, `finance/finance_conformal.py`) found that richer featurizers — prefix means/stds/last-Y/AR(1) coeffs — materially improve LR separability.
- **Decision rule**: expose `featurize_fn` as a constructor argument with three built-ins:
  - `"x1"` (literal): first time step only.
  - `"x_summary"`: mean / std / min / max / last of X over the available prefix.
  - `"yx_summary"`: parent-repo-style; Y-summary + X-summary (used in finance).
- The pedestrian dataset has only positional X with no auxiliary covariates → use `"x1"`. Synthetic AR has scalar X (the AR series itself) → either. The user should pick per dataset.

### Q3. Per-time-step model fitting cost
- The pseudocode trains a *separate* `f_t` for every t ∈ [T+1]. With T=100 and an LSTM that is prohibitive.
- If we keep the LSTM path (Q1, option A), `f_t` should be **read as a single LSTM applied iteratively** — we already get the full T-step forecast from `Blackbox.predict_iterate`. We will document this interpretation in the docstring rather than literally fitting T+1 separate networks.
- If we take the linear-AR path (Q1, option B), per-step refit is cheap and we can follow the pseudocode literally.

### Q4. Three-way training split sizing
- The pseudocode says "randomly split D_tr into D_tr^(1), D_tr^(2), D_tr^(3)" without specifying proportions.
- CAFHT's `select_gamma` doesn't use a three-way split — it calibrates on D_tr^(1) and minimizes width on that same data (data-splitting mode further splits a separate D_cal into halves).
- Proposal: 50% / 25% / 25% (fit / cal / eval). Tunable via constructor.

### Q5. Test-half handling — exactly one band per test point
- Decided: each test point gets one band (option A from the second clarification round). The "reverse and repeat" produces bands for the other half; we concatenate, preserving the original test-point ordering.

---

## Algorithm → code mapping

Numbering follows the user's pseudocode block-by-block.

### Block 1 — fit predictors and choose γ_opt

| Pseudocode | Code |
|---|---|
| Split D_tr into D_tr^(1), D_tr^(2), D_tr^(3) | `_three_way_split(d_train, seed)` — new helper in `weighted_methods.py`. |
| For each t, fit `f_t` | Reuses existing path (Q1, Q3). Produces `(pred_train, true_train)` arrays of shape `(n, T, ndim)` already produced by `Blackbox.predict_iterate`. We just **slice** the arrays into D_tr^(1)/(2)/(3) along the n-axis. No retraining per split. |
| For each γ ∈ Γ run simple ACI and record performance | This is exactly what `Adaptive_Conformal_Inference.predict_intervals` does. We loop over `gamma_grid`, build bands on D_tr^(3) using α_t initialized from the simple ACI run on D_tr^(2), and record **mean band width**. |
| Pick γ_opt | `argmin` over the width vector. |

**Reuse decision**: do NOT reuse `CAFHT.select_gamma` directly because its calibration uses *trajectory-level* scores and a separate width loop, not the literal "run ACI on D_tr^(2)/D_tr^(3)" structure the pseudocode wants. We will write `_select_gamma_aci` as a thin new method that wraps `Adaptive_Conformal_Inference` only. Confirmed γ-rule: **minimum mean band width** (CAFHT default — `methods.py:393`).

### Block 2 — learn LR weights

| Pseudocode | Code |
|---|---|
| Split D_test into D_test^(1), D_test^(2) | `train_test_split(test_pred, test_true, test_size=0.5, random_state=seed)` — same idiom CAFHT already uses at `methods.py:422`. |
| Train LR classifier on `{(X_1, 0): i ∈ D_tr}` ∪ `{(X_1, 1): i ∈ D_test^(1)}` | Use the same sklearn `LogisticRegression(class_weight="balanced")` machinery that `core/algorithm.py:_compute_density_ratio_weights` uses in the parent repo. We will **copy that function into the new file** rather than import across the CAFHT/parent boundary, so `CAFHT/` stays self-contained. Featurizer is the `featurize_fn` from Q2. |
| Apply classifier on `i ∈ D_cal` to get `W_i` | `clf.predict_proba(featurize(X_cal))[:, 1] / clf.predict_proba(featurize(X_cal))[:, 0]`. Clip at 5× mean weight (parent-repo convention). Normalize to sum to 1 inside the weighted-quantile call. |

**Cross-half (Q5 decision)**: we run Block 2 twice — once with positives = D_test^(1), once with positives = D_test^(2). The first fit produces weights used to band D_test^(2); the second produces weights used to band D_test^(1).

Also: weights are produced for **each cal point** *and* **each test point** (the test weight `W_j` is needed for the δ_∞ atom in Block 4). The test weight uses the same classifier — apply `clf.predict_proba` to the *opposite* test half.

### Block 3 — calibration scores

| Pseudocode | Code |
|---|---|
| For each i ∈ D_cal, run ACI with γ_opt to get bands C_t^(i) | Single call to `Adaptive_Conformal_Inference.predict_intervals(calib_pred, calib_true, gamma=gamma_opt, q0=q0, y_trim=y_trim)`. Output shape `(n_cal, T, 2, ndim)`. |
| ε_i = max_t max{Y_t − U_t, L_t − Y_t} | New helper `_trajectory_score_additive(bands, y_true)`. This is exactly CAFHT's `nonconf_scores` with `adaptive=False` (see `methods.py:298`): `score_ = max(max(0, y − pred_high), max(0, pred_low − y))`. We will call CAFHT's existing function with `self.adaptive = False`. |

Confirmed: **additive only**. We will not implement the multiplicative branch in the first pass.

### Block 4 — deployment / weighted quantile

For each j in the test half being deployed:

| Pseudocode | Code |
|---|---|
| η_j = Quantile(Σ_i (W_i/Σ W) δ_{ε_i} + (W_j/Σ W) δ_∞, 1−α) | New helper `_weighted_quantile_with_inf(scores, w_cal, w_test_point, level)`. Builds an empirical distribution over scores + an atom at ∞ with mass W_j/Σ_k W_k, then returns the smallest value v such that cumulative mass ≥ 1 − α. Critically: includes the δ_∞ atom in the denominator (Σ_k spans cal **and** the test point). |
| α_1^(i) = α; for t ∈ [T+1]: run ACI at level α_t; band = [min C_t^aci − η_j, max C_t^aci + η_j] | Single call to `Adaptive_Conformal_Inference.predict_intervals(test_pred[j:j+1], test_true[j:j+1], gamma=gamma_opt, q0=q0)`, then **inflate** the resulting band by ±η_j. This is the `adaptive=False` branch of CAFHT's `predict_bands_subroutine` (`methods.py:343`): `PI_base[:, :, 1, :] += scores_calibrated; PI_base[:, :, 0, :] -= scores_calibrated`. We will reuse that branch as-is. |

The trick is that **η_j depends on j** (through W_j) — unlike vanilla CAFHT where the calibrated quantile is shared across all test points. So we need a per-test-point inflation, not a single broadcast. Implementation: loop over j in the deployed half; compute η_j; inflate `PI_base[j]` only.

### Block 5 — reverse and repeat

Run the entire Block 2 / 3 / 4 sequence twice with the two test halves swapped. Concatenate the resulting per-trajectory bands, restoring the original test-point order via the indices returned by `train_test_split` (we'll capture them with `np.arange` shuffled and tracked manually rather than relying on `train_test_split`'s `random_state`).

Cal scores in Block 3 can be **reused** across the two halves — they depend only on D_cal and γ_opt, not on which half is which. Only the LR fit and the per-test-point η_j change. We will compute ε_i once and reuse it.

---

## Proposed new file: `CAFHT/ConformalizedTS/weighted_methods.py`

Skeleton:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy.linalg as la
from tqdm import tqdm

from ConformalizedTS.methods import Adaptive_Conformal_Inference, CAFHT
from ConformalizedTS.utils import trimming


class WeightedCAFHT:
    """
    CAFHT with likelihood-ratio reweighting of trajectory-level calibration scores.
    Predictor-agnostic: takes precomputed (pred, true) arrays for train, cal, test.

    Parameters
    ----------
    alpha : float
    gamma_grid : 1D array-like
    featurize_fn : Callable[[np.ndarray], np.ndarray]   # X-prefix -> feature matrix
    weight_clip : float, default 5.0                    # clip at weight_clip * mean
    train_split : tuple[float, float, float], default (0.5, 0.25, 0.25)
    verbose : bool

    Methods
    -------
    select_gamma(train_pred, train_true, q0, seed)
        Returns gamma_opt by minimum mean band width on D_tr^(3).

    fit_lr(train_X, test_X_half_pos, cal_X, deploy_X)
        Returns (W_cal, W_deploy) — both clipped, unnormalized.

    calibration_scores(calib_pred, calib_true, gamma_opt, q0, seed)
        Returns eps : shape (n_cal,) — additive trajectory scores.

    predict_bands(train_data, calib_data, test_data, X_train, X_test, q0,
                  y_trim=None, seed=123)
        Full pipeline:
          1. select_gamma on train_data three-way split.
          2. for half in [first, second]:
             a. fit LR on D_tr ∪ opposite half as positives.
             b. compute eps on D_cal (once, shared across halves).
             c. for each j in this half:
                  W_j = lr.predict_proba(featurize(X_test[j])) ratio
                  eta_j = weighted_quantile_with_inf(eps, W_cal, W_j, 1 - alpha)
                  band[j] = ACI(test_pred[j], test_true[j], gamma_opt) ± eta_j
        Returns pred_intervals shape (n_test, T, 2, ndim) in original order.
    """
```

Internal helpers (all module-private):

- `_three_way_split(n, fractions, seed) -> (idx1, idx2, idx3)` — index arrays.
- `_compute_density_ratio_weights(X_pos, X_neg, X_eval, clip)` — direct port of `core/algorithm.py:_compute_density_ratio_weights` from the parent repo, with the sklearn `LogisticRegression(class_weight="balanced")` fit and 5×-mean clipping.
- `_trajectory_score_additive(bands, y_true)` — port the `adaptive=False` branch of `CAFHT.nonconf_scores`.
- `_weighted_quantile_with_inf(scores, w_cal, w_test, level)` — see "Weighted quantile" section below.
- `_inflate_band(bands, eta)` — additive widen by η.

---

## Featurizer interface

A `featurize_fn` takes the full X-prefix array of shape `(n, T, dx)` (or whatever the dataset stores) and returns a 2D feature matrix `(n, d_feat)`.

Built-in implementations:

| Name | Output features |
|---|---|
| `"x1"` | `X[:, 0, :]` flattened → `(n, dx)` |
| `"x_summary"` | per-dim mean, std, min, max, last over the prefix → `(n, 5*dx)` |
| `"yx_summary"` | Y-prefix mean / std / AR(1) coefficient over last 30 steps + X-summary → matches parent-repo finance featurizer |

Default = `"x1"` (matches the literal pseudocode). User overrides per dataset.

---

## Weighted quantile with δ_∞ atom

This is the Tibshirani (2019) construction. Algorithm:

```
def _weighted_quantile_with_inf(scores, w_cal, w_test, level):
    # scores: shape (n_cal,)
    # w_cal:  shape (n_cal,)  -- unnormalized
    # w_test: scalar           -- unnormalized
    # level:  float in (0,1)   -- e.g. 1 - alpha

    total = w_cal.sum() + w_test
    p_cal = w_cal / total
    p_inf = w_test / total

    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_p = p_cal[order]
    cum = np.cumsum(sorted_p)

    # the inf atom sits at the top of the sorted list.
    # smallest score whose cumulative mass >= level
    idx = np.searchsorted(cum, level, side='left')
    if idx >= len(sorted_scores) or cum[-1] + p_inf < level:
        return np.inf
    return sorted_scores[idx]
```

Edge cases to handle:
- All cal weights zero → return ∞.
- `level` exceeds total cal mass (i.e. would only be reached by including the δ_∞ atom) → return ∞ (band becomes (−∞, ∞)).
- `n_cal == 0` → return ∞.

This is the same construction `AdaptedCAFHT` uses in the parent repo (`core/algorithm.py`); we duplicate it inside `weighted_methods.py` to avoid cross-repo imports.

---

## What we keep from CAFHT vs. write fresh

| Component | Reused from `ConformalizedTS/methods.py` | New |
|---|---|---|
| Inner ACI loop | `Adaptive_Conformal_Inference.predict_intervals` (full reuse) | — |
| Additive band inflation | `predict_bands_subroutine` (adaptive=False branch) | — |
| Trajectory score (additive) | `CAFHT.nonconf_scores` with `adaptive=False` | — |
| γ-selection by min width | logic mirrors `CAFHT.select_gamma` but on a **single** D_tr^(3) eval (no double-cal) | thin wrapper `_select_gamma_aci` |
| LR weight estimation | not present in `CAFHT/` | new (ported from parent repo) |
| Weighted quantile w/ δ_∞ | not present in `CAFHT/` | new |
| Cross-half test split | partially present (`predict_bands` does a single split) | extend to symmetric two-way scheme |
| `q0` (initial quantile) | computed identically to `ts_realdata.py:222` from training residuals | reused |

---

## File touchpoints

| File | Action |
|---|---|
| `CAFHT/ConformalizedTS/weighted_methods.py` | **new** — contains `WeightedCAFHT` plus all helpers |
| `CAFHT/ConformalizedTS/methods.py` | **no change** — strictly read-only reuse |
| `CAFHT/ConformalizedTS/__init__.py` | n/a — the package isn't `__init__`'d; imports are by file path |
| `CAFHT/experiments/ts_sim.py`, `ts_realdata.py` | **no change in this milestone** (Q-Scope = "implement only"). If we run experiments later, we add the new method via the same `evaluation_multivariate(...)` pattern already used. |
| `CAFHT/CAFHT_OVERVIEW.md` | append a "Weighted CAFHT" subsection in the methods table once the class lands. |

---

## Decision-dependent code paths

Quick summary of what each open question gates:

- **Q1 (predictor)**: affects only the *inputs* to `WeightedCAFHT.predict_bands` (we always take precomputed `(pred, true)` tuples). The class is predictor-agnostic.
- **Q2 (featurizer)**: affects the `featurize_fn` argument. Built-ins covered above. Per-dataset.
- **Q3 (per-step refit)**: affects only the script that *calls* the class. Class side: no change.
- **Q4 (split fractions)**: parameter `train_split=(0.5, 0.25, 0.25)`. Class side: trivial.

None of the open questions blocks the class implementation itself.

---

## Sanity checks (to verify correctness before any experiment run)

1. **Reduce-to-CAFHT check**: with uniform LR weights (force `W_i = 1` for all i, `W_test = 1`), the weighted quantile should equal CAFHT's `np.quantile(scores, (1−α)(1+1/n), interpolation='higher')` up to interpolation differences. Build a unit-style assert.
2. **δ_∞ atom check**: when `w_test` dominates the total mass, η_j → ∞ (band becomes (−∞, ∞)); when `w_test` is negligible the result matches the unweighted weighted quantile of cal scores.
3. **Two-half consistency**: every test trajectory appears in exactly one of the two halves; concatenated output has shape `(n_test, T, 2, ndim)` with the original ordering.
4. **Additive-only invariant**: assert `self.adaptive is False` on every internal call to CAFHT helpers; raise `NotImplementedError` if a user passes `adaptive=True` in the first version.

---

## Sequence of work

1. Lock in answers to Q1–Q4 (esp. Q1, which decides whether we use the LSTM or a linear predictor in the eventual run script — but does not block the class).
2. Implement `weighted_methods.py` with the four helpers and the `WeightedCAFHT` class.
3. Run the four sanity checks above in a tiny script (synthetic data, T=5, n_train=100, n_cal=50, n_test=10) — no commits to result CSVs.
4. Update `CAFHT_OVERVIEW.md` to list the new method.
5. (Out of scope for this milestone) experiment integration.

---

## Notes / things to flag

- The pseudocode does not specify how to break ties in the weighted quantile when multiple cal scores equal the threshold. We will use `interpolation='higher'` equivalent (smallest score with cumulative mass ≥ level), matching CAFHT's existing convention at `methods.py:373`.
- The clip-at-5×-mean rule for LR weights is a parent-repo convention. We will expose it as `weight_clip` (default 5.0) with `weight_clip=None` disabling.
- y_trim (band clipping to a known value range) is preserved from CAFHT and applied **after** η-inflation, identical to `predict_bands_subroutine`.
- If `clf.predict_proba` returns extreme probabilities (>0.999 or <0.001), the ratio explodes before clipping. We will clip probabilities to `[1e-6, 1−1e-6]` before computing the ratio, matching the parent-repo pattern.
