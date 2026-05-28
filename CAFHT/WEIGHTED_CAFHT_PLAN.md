# Weighted CAFHT — Repository Restructuring Plan (v2)

This document is the implementation plan for the **two-regime / two-algorithm** revision of the project. The previous version of this file (v1, kept in git history) was scoped to writing code inside `CAFHT/` and assumed a single trajectory-level method; both assumptions are obsolete and have been discarded.

---

## 0. Scope and ground rules

- **The `CAFHT/` folder is read-only reference material.** No code, scripts, or notebooks under `CAFHT/` will be created or modified. Reading `CAFHT/ConformalizedTS/methods.py` to mirror specific routines is fine; importing from it at runtime is not.
- All code changes happen in our repository (`core/`, `synthetic/`, `finance/`, `medical/`, `results/`, top-level scripts).
- The problem statement now specifies **two regimes** (last-step coverage; whole-trajectory coverage) and **two distinct algorithms** (one per regime). They share two primitives (LR weights, weighted quantile with a δ_∞ atom) but differ in everything else: predictor, conformity score, deployment loop, and γ usage. They will be implemented as **separate top-level entry points**, not as flags on a single class.
- All existing experiment outputs (`results/{medical,finance,synthetic}/json/*.json`) become stale under the new algorithm definitions. The plan covers what to archive vs. regenerate.

---

## 1. What changed

### 1.1 Notation

| Symbol | Old meaning (parent repo, pre-v2) | New meaning (algo box) |
|---|---|---|
| `Z_t^(i)` | not used as a top-level object | `(X_t^(i), Y_t^(i))`, one per (series, time) |
| `X_t^(i)` | dynamic + static covariates (medical), 4 daily features (finance), Poisson (synthetic) | the covariate vector at time t — same domain interpretation but explicitly indexed |
| `Y_t^(i)` | scalar response | same |
| **Shift assumption** | unspecified | shift is on `X_1` (whole-trajectory); shift is on `X_{1:T}` jointly (last-step) — conditional kernels for the remainder are identical across reference and test |

The user's note "this coincides with the previous notation by letting `X_t = Y_{t-1}`" means the last-step regime reduces to AR-style prediction if X is taken as past Y. Our current synthetic/medical pipelines already do this, so the data loaders need no schema change — only the runners do.

### 1.2 Old algorithm vs. new algorithms — side-by-side

| Aspect | Old (`core/algorithm.py:AdaptedCAFHT`) | New: last-step coverage (Alg. 2) | New: whole-trajectory coverage (Alg. 1) |
|---|---|---|---|
| Coverage target | per-step marginal | `P(Y_{T+1} ∈ Ĉ_{T+1}) ≥ 1−α` | `P(∀t: Y_t ∈ Ĉ_t) ≥ 1−α` |
| Predictor | per-step refit (AR(1), or `LinearCovariateModel`) | single model `f_{T+1}` trained once: regress `Y_{T+1}` on `X_{1:T}` | per-step models `{f_t}_{t∈[T+1]}`, each regressing `Y_t` on `(Z_{1:t-1}, X_t)` |
| γ selection | per-series ACI step inside main loop, every 5–10 steps; criterion: average coverage closest to 1−α on a 3-way training split | **not used** (no ACI in last-step) | 3-way train split: D_tr^(1) → fit, D_tr^(2) → cal, D_tr^(3) → eval. Run ACI for each γ; pick by **minimum mean band width** (CAFHT convention, `CAFHT/ConformalizedTS/methods.py:393`). Open Q4 below. |
| Cal score | `\|Y_{t+1} - Ŷ_{t+1}\|`, single-time-step (post-v2 calibration fix) | `s(Y_{T+1}^(i), f_{T+1}(X^(i)))`, e.g. abs residual; one scalar per cal series | `ε_i = max_t max{Y_t − U_t, L_t − Y_t}` over the ACI bands run on cal series i with γ_opt (additive branch of CAFHT's `nonconf_scores`) |
| Weighting | `p̂/(1−p̂)` from cross-half LR on prefix-summary features; normalized to sum to 1 | LR on **entire X vector** `X_{1:T}^(i)`; weights kept unnormalized for δ_∞ quantile | LR on **X_1** only (shift assumption); weights kept unnormalized for δ_∞ quantile |
| Deployment quantile | weighted empirical CDF (no test atom) | `η_j = Quantile(Σ Ŵ_i/Σ Ŵ_k · δ_{ε_i} + Ŵ_j/Σ Ŵ_k · δ_∞, 1−α)` per test point | same formula |
| Deployment band | `[Ŷ ± q]` from weighted quantile of cal scores | `Ĉ_{T+1}^j = {y : s(y, f_{T+1}(X^j)) ≤ η_j}` | run ACI on test point j to get Ĉ_t^aci, then **inflate**: `Ĉ_t^j = [min Ĉ_t^aci − η_j, max Ĉ_t^aci + η_j]` for every t |
| Test-side α update | per-series ACI, online | none | none at deployment (η_j is a fixed scalar per test point) |
| Cross-half test split | yes | yes (positives = `D_test^(1)`, deploy on `D_test^(2)`; reverse and repeat) | same |

### 1.3 What survives from the current code

- `core/ts_generator.py` — synthetic DGPs are unchanged.
- `finance/finance_data.py`, `medical/medical_data.md`, `medical/sepsis_experiment_data_nacl_target.pkl` — data loaders/pickle unchanged.
- Two algorithmic primitives can be salvaged from `core/algorithm.py`:
  - `AdaptedCAFHT._compute_density_ratio_weights` — keep the logistic-regression + 5×-mean clip pattern but expose two things that the old impl didn't: the **raw `(W_cal, W_test)` pair** (we no longer normalize before quantile-time, because the δ_∞ atom needs all weights on the same scale) and a **caller-supplied featurizer** so X_1 (Alg. 1) and X_{1:T} (Alg. 2) can plug in.
  - `AdaptedCAFHT._weighted_quantile` — needs a new sibling that takes the test-point weight as a separate argument and inserts a δ_∞ atom; the existing function stays for legacy.
- Per-domain *data wrangling* (medical's `_convert_to_arrays`, `_encode_ethnicity`, `ETHNICITY_MAP`; finance's sector split logic) — keep, factor out into thin loader modules.
- `LinearCovariateModel` (both `medical_conformal.py:451` and `finance_conformal.py:228`) — keep as a per-step `f_t` for the whole-trajectory regime in those domains; for the last-step regime we will need a new predictor (see § 4.4).

What does **not** survive:

- The control flow of `AdaptedCAFHT.calibrate` / `predict_with_interval` / `update_weighting_context` and the per-series α_t update loop. The new algorithms have a different shape: cal-side ACI runs for scoring; deployment-side ACI runs for the band; LR weights enter only at the η_j quantile step.
- `_select_gamma` in `medical_conformal.py` and `finance_conformal.py` — both use the "coverage closest to target" criterion. The new whole-trajectory algorithm uses "minimum mean band width" (CAFHT convention).
- Per-domain monkey-patched featurizers (`_richer_featurize_prefixes`, `_make_featurizer`). Featurizers move into the LR-classifier helper as explicit `featurize_fn` arguments.

---

## 2. Clarification questions (block coding)

These need answers before implementation starts. Each has a concrete impact noted.

**Q1. Which (domain × regime) cells do we evaluate?**
Combinations are `{synthetic, finance, medical} × {last-step, whole-trajectory}` = 6. We have committed effort to whole-trajectory for all three domains historically. Last-step is new. Do we want all 6 cells, or only some?
*Impact*: number of runners to write; number of result subfolders; size of LaTeX tables; CPU budget for the next sweep.

**Q2. LR featurizer when the shift assumption is violated.**
The whole-trajectory algorithm assumes shift is on `X_1` only and prescribes feeding `X_1^(i)` to the classifier. For medical, the Norepinephrine split induces shift on a 12-hour window (so `X_{1:12}`) not on `X_1`. Two options:
  (a) follow the spec literally and accept some loss of separability;
  (b) use the entire `X_{1:T}` for the LR classifier and document the deviation.
*Impact*: medical Alg. 1's empirical coverage. Finance has a similar question (Tech/Util shift is on the joint covariate distribution, not on X_1).

**Q3. Predictor for last-step (Alg. 2) on each domain.**
The algorithm says "regress `Y_{T+1}` on `X_{1:T}`". For our settings:
  - **Synthetic**: with `X_t = Y_{t-1}` (the AR(1) reinterpretation), the natural predictor is just an OLS regression of `Y_{T+1}` on the full lagged history.
  - **Finance**: `X` is 4-dimensional × 40 days = 160 features per ticker if flattened. Options: flatten + L2-regularized linear; use only the most recent few X_t; use an RNN.
  - **Medical**: `X` is 3-dim × 24 hours + 6 static = 78 features per patient (or 84 with NaCl history). Options: flatten + ridge; use last 12 hours only.
*Impact*: predictor accuracy, which dominates residual magnitude and therefore interval width. **Strong recommendation: use ridge regression on flattened `X_{1:T}` with a small held-out CV for λ; defer RNNs to a later iteration.**

**Q4. γ-selection criterion in the whole-trajectory algorithm.**
CAFHT picks γ by minimum mean band width (`CAFHT/ConformalizedTS/methods.py:393`); the user's previous `_select_gamma` in our medical/finance code picked γ by "tail-half coverage closest to 1−α". The new algorithm box says "best performance" without specifying. **Recommendation: match CAFHT exactly (minimum mean width on D_tr^(3) after running ACI calibrated on D_tr^(2))**, since the new algorithm is otherwise derived from CAFHT.

**Q5. Treatment of legacy `core/algorithm.py:AdaptedCAFHT` and per-domain runners.**
Three options:
  (a) Delete `AdaptedCAFHT` and the legacy runners; preserve only via git history.
  (b) Move them under `legacy/` and stop touching them.
  (c) Keep them in place as a deprecated reference but mark them clearly.
*Impact*: code clarity vs. ability to reproduce prior numbers without git checkout. **Recommendation: option (b)** — move to `legacy/` so the active tree only contains the new algorithms and the structural change is visible in the directory layout.

**Q6. Backward compatibility of saved JSONs.**
Current `results/{medical,finance,synthetic}/json/` files were produced under the old AdaptedCAFHT semantics, then under the partially-fixed semantics. None of them are comparable to the new algorithms.
**Recommendation: archive existing `results/` to `results_legacy/` and start fresh under `results/{domain}/{regime}/{json,pdf,tables}/`.**

**Q7. Naming.**
Two options for the user-facing names:
  (a) "Weighted CAFHT (whole-trajectory)" and "Weighted CAFHT (last-step)" — emphasizes shared lineage; matches the algo-box titles.
  (b) "Weighted CAFHT" (whole-trajectory) and "Weighted Split CP" (last-step) — semantically more accurate since last-step has no per-step inner conformal loop.
*Impact*: paper section names, table captions, file names.
**Recommendation: (a)** — match the user's algo boxes so the paper and code share terminology.

---

## 3. Proposed file layout

```
core/
  ts_generator.py                 [unchanged]
  weighted_cafht_whole.py         NEW   Algorithm 1 (whole-trajectory)
  weighted_cafht_last.py          NEW   Algorithm 2 (last-step)
  aci.py                          NEW   stand-alone ACI updater (mirrors CAFHT/.../methods.py:Adaptive_Conformal_Inference)
  density_ratio.py                NEW   logistic-regression weight estimator (lifted from old AdaptedCAFHT)
  weighted_quantile.py            NEW   Tibshirani-2019 quantile with δ_∞ atom
  adaptive_conformal.py           [unchanged — legacy synthetic baseline]

legacy/                           NEW dir — see Q5
  algorithm.py                    moved from core/algorithm.py (old AdaptedCAFHT)
  medical_conformal.py            moved from medical/
  finance_conformal.py            moved from finance/
  finance_adaptive.py             moved from finance/   (already legacy)

synthetic/
  multi_seed_experiments.py       [refactor: add --regime {last_step, whole_trajectory}]
  test_conformal.py               [refactor: single-seed wrapper]
  predictors.py                   NEW   per-regime synthetic predictors (AR(1) → trajectory; ridge → last-step)
  featurizers.py                  NEW   X_1 vs X_{1:T} featurizers for the LR step

finance/
  finance_data.py                 [unchanged]
  predictors.py                   NEW   LinearCovariateModel reused for whole-traj; ridge on flattened X for last-step
  featurizers.py                  NEW
  finance_runner.py               NEW   single entry point with --regime flag (replaces finance_conformal.py)

medical/
  medical_data.md                 [unchanged]
  sepsis_experiment_data_nacl_target.pkl   [unchanged]
  plot_medical_covariate_shift.py [unchanged]
  data_loader.py                  NEW   exposes load_data + _convert_to_arrays + ETHNICITY_MAP from old medical_conformal.py
  predictors.py                   NEW   one-step-ahead AR + statics (whole-traj); ridge on flattened X+statics (last-step)
  featurizers.py                  NEW
  medical_runner.py               NEW   single entry point with --regime flag (replaces medical_conformal.py)
  multi_seed_medical.py           [refactor: thread --regime through to medical_runner]

results_legacy/                   NEW dir — archive of pre-v2 JSONs/PDFs/tables (see Q6)

results/
  synthetic/last_step/{json,pdf,tables}/
  synthetic/whole_trajectory/{json,pdf,tables}/
  finance/last_step/{json,pdf,tables}/
  finance/whole_trajectory/{json,pdf,tables}/
  medical/last_step/{json,pdf,tables}/
  medical/whole_trajectory/{json,pdf,tables}/

build_tex_tables.py               [extend: produce one table per (domain × regime); 6 tables total]
run_all_experiments.sh            [restructure: dispatch by regime; see § 7]
```

### Why two `core/` algorithm files instead of one

The two algorithms differ in `(predictor type, cal score, deployment loop, γ usage)` — i.e. essentially everything except the LR weight estimation and the δ_∞ weighted quantile. Putting both into one class would require pervasive `if regime == ...` branching that obscures the actual control flow. Two focused modules read better and let each have a docstring tied to its algo box.

### Why per-domain `predictors.py` and `featurizers.py`

The predictor is domain-specific (Poisson AR vs. SP500 covariates vs. NaCl + CHART) and regime-specific (per-step `f_t` vs. one-shot `f_{T+1}`). Putting both in a small module per domain keeps the runner thin and the cross-domain comparisons code-aligned (every domain exposes the same two functions: `build_whole_trajectory_predictor(...)` and `build_last_step_predictor(...)`).

---

## 4. Core components in detail

### 4.1 `core/weighted_quantile.py`

```
def weighted_quantile_with_inf(scores, w_cal, w_test, level) -> float:
    """
    Tibshirani-Foygel-Barber-Candès-Ramdas (2019) weighted-exchangeability quantile.
    Returns the smallest score s such that
        Σ_{i: ε_i ≤ s} W_i  ≥  level · (Σ_k W_k + W_test).
    Returns np.inf if the δ_∞ atom must be reached to attain `level`.
    """
```

Edge cases: all w_cal == 0 → ∞; w_test dominates total mass → ∞; n_cal == 0 → ∞. Tie-breaking matches CAFHT's `interpolation='higher'` convention.

### 4.2 `core/density_ratio.py`

```
def density_ratio_weights(
    X_pos, X_neg, X_eval,
    *,
    classifier=None,       # default: sklearn LogisticRegression(class_weight='balanced')
    clip_factor=5.0,       # 5x mean clip on the raw ratio
    prob_clip=1e-6,        # clip probabilities to [prob_clip, 1-prob_clip] pre-ratio
) -> dict:
    """
    Fit a logistic classifier with `X_pos` (label 0) vs. `X_neg` (label 1), score `X_eval`,
    return raw clipped weights (NOT normalized) plus diagnostics:
        {
          "w_eval": (n_eval,) clipped weights,
          "prob1_eval": (n_eval,) classifier prob1 on eval,
          "train_acc": float,
          "coef_norm": float,
        }
    Critically: returns UNNORMALIZED weights because the downstream weighted_quantile_with_inf
    needs raw masses to compute the Σ_k W_k + W_test denominator with the test-point atom.
    """
```

Lifted from `AdaptedCAFHT._compute_density_ratio_weights` (`core/algorithm.py:378`) with two changes: no internal normalization, and a `classifier` slot for future swap-in.

### 4.3 `core/aci.py`

```
class ACI:
    """
    Adaptive conformal inference for a single trajectory:
        α_{t+1} = α_t + γ · (α − err_t),   clipped to (1e-6, 1 - 1e-6).
    Mirrors CAFHT/ConformalizedTS/methods.py:Adaptive_Conformal_Inference.
    Takes precomputed (y_pred, y_true) trajectories; predictor agnostic.
    """
    def predict_intervals(self, y_pred, y_true, gamma, q0, seed=...): ...
```

Stand-alone so both `weighted_cafht_whole.py` (uses it for cal-side band construction + deployment-side per-test-point bands) and the synthetic baseline can call it without circular imports.

### 4.4 `core/weighted_cafht_whole.py`  (Algorithm 1)

```
class WeightedCAFHTWholeTrajectory:
    """
    Implements the whole-trajectory coverage algorithm.
    Predictor-agnostic: caller supplies precomputed (pred, true) arrays of shape
    (n, T+1, ndim) for train, cal, test, plus the X arrays used by the LR step.
    """

    def __init__(self, alpha, gamma_grid, featurize_fn, train_split=(0.5, 0.25, 0.25),
                 weight_clip=5.0, randomize=False, verbose=True): ...

    def select_gamma(self, train_pred, train_true, q0, seed) -> float:
        # 3-way split of D_tr; for each γ, calibrate ACI on D_tr^(2), apply on D_tr^(3),
        # return γ with minimum mean band width (CAFHT convention; see Q4).

    def calibration_scores(self, cal_pred, cal_true, gamma_opt, q0, seed) -> np.ndarray:
        # for each cal series: run ACI to get bands; ε_i = max_t max{y - U_t, L_t - y}
        # (additive branch; CAFHT/.../methods.py:313).

    def predict_bands(self, train_data, cal_data, test_data,
                      X_train, X_cal, X_test, q0, y_trim=None, seed=...) -> np.ndarray:
        # 1. γ_opt = self.select_gamma(...).
        # 2. ε = self.calibration_scores(D_cal, γ_opt).
        # 3. For each test half (cross-half scheme):
        #     a. Fit LR classifier (positives = opposite half, features from featurize_fn).
        #     b. W_cal = clf weights on X_cal (raw, clipped).
        #     c. For each j in this half:
        #          W_j = clf weight on X_test[j].
        #          η_j = weighted_quantile_with_inf(ε, W_cal, W_j, 1-α).
        #          band[j] = ACI(test_pred[j], test_true[j], γ_opt) inflated ± η_j.
        # 4. Concatenate; return shape (n_test, T+1, 2, ndim).
```

Block-by-block correspondence to Algorithm 1 is identical to the structure above; this is just the entry point.

### 4.5 `core/weighted_cafht_last.py`  (Algorithm 2)

```
class WeightedCAFHTLastStep:
    """
    Implements last-step coverage. No γ, no ACI, no per-step bands.
    Predictor-agnostic: caller supplies precomputed scalar predictions for cal and test.
    """

    def __init__(self, alpha, featurize_fn, weight_clip=5.0, score_fn=None, verbose=True):
        # score_fn defaults to lambda y, yhat: abs(y - yhat)

    def calibration_scores(self, y_cal_true, y_cal_pred) -> np.ndarray: ...
        # ε_i = score_fn(Y_{T+1}^(i), Ŷ_{T+1}^(i))

    def predict_bands(self, cal_data, test_data, X_train, X_cal, X_test) -> np.ndarray:
        # 1. ε = self.calibration_scores(...).
        # 2. For each test half:
        #     a. Fit LR classifier (positives = opposite half).
        #     b. W_cal = clf weights on X_cal (raw, clipped).
        #     c. For each j in this half:
        #          W_j = clf weight on X_test[j].
        #          η_j = weighted_quantile_with_inf(ε, W_cal, W_j, 1-α).
        #          band[j] = [Ŷ_j − η_j, Ŷ_j + η_j]   (score_fn=|·|)
        # 3. Concatenate; return shape (n_test, 1, 2, ndim).  Or (n_test, 2) if ndim=1.
```

Trivial compared to Alg. 1 — just the cross-half + δ_∞ quantile step.

---

## 5. Per-domain predictors and featurizers

For each cell, name the function (in `{synthetic,finance,medical}/predictors.py`) and what it produces.

### 5.1 Whole-trajectory predictors (Algorithm 1)

| Domain | Function | Implementation | Output |
|---|---|---|---|
| Synthetic | `build_whole_trajectory_predictor(Y_train)` | global AR(1) by OLS on training pairs; iterated forward | `(n, T+1, 1)` predicted trajectory |
| Finance | `build_whole_trajectory_predictor(Y_train, X_train)` | per-step OLS on `X_t` (reuses `legacy/finance_conformal.py:LinearCovariateModel`); fit once globally, applied at each t | `(n, T+1, 1)` |
| Medical | `build_whole_trajectory_predictor(Y_train, X_train, S_train)` | one-step-ahead AR `Y_{t+1} ~ Y_t + X_t + S` (reuses `legacy/medical_conformal.py:LinearCovariateModel`) | `(n, T+1, 1)` |

### 5.2 Last-step predictors (Algorithm 2)

| Domain | Function | Implementation | Output |
|---|---|---|---|
| Synthetic | `build_last_step_predictor(Y_train, lags=T)` | OLS of `Y_{T+1}` on `Y_{1:T}` | `(n,)` scalar prediction |
| Finance | `build_last_step_predictor(Y_train, X_train)` | ridge of `Y_{T+1}` on flattened `X_{1:T}` (4·T features); λ chosen by 5-fold CV | `(n,)` |
| Medical | `build_last_step_predictor(Y_train, X_train, S_train)` | ridge of `Y_{T+1}` on flattened `X_{1:T}` + S (3·T + 6 features); λ chosen by 5-fold CV | `(n,)` |

### 5.3 LR featurizers (used in both regimes, with different time-extents)

| Domain | Whole-trajectory featurizer (X_1) | Last-step featurizer (X_{1:T}) |
|---|---|---|
| Synthetic | `Y_0` only (== X_1 under the X=Y_lag reinterpretation) | full Y history |
| Finance | `X_1` = 4 covariates at day 1 | flattened `X_{1:T}` (4·T) |
| Medical | `X_1` = 3 vitals at hour 0 + 6 statics = 9 features | flattened `X_{1:T}` + S (3·T + 6) — but see Q2 |

All featurizers will live in `{domain}/featurizers.py` and expose `featurize_x1(X)` / `featurize_xall(X)` so the runners can pass them as `featurize_fn` to the algorithm classes.

---

## 6. Migration: what moves where

| Current path | New path | Reason |
|---|---|---|
| `core/algorithm.py` | `legacy/algorithm.py` | superseded by the two new algorithm files; keep for reference |
| `medical/medical_conformal.py` | `legacy/medical_conformal.py` | superseded by `medical/medical_runner.py` |
| `finance/finance_conformal.py` | `legacy/finance_conformal.py` | superseded by `finance/finance_runner.py` |
| `finance/finance_adaptive.py` | `legacy/finance_adaptive.py` | already legacy per `CLAUDE.md` |
| `finance/tune_featurizer.py` | `legacy/tune_featurizer.py` | tied to old featurizer design |
| `finance/plot_covariate_shift.py` | `finance/plot_covariate_shift.py` (in place) | data-only utility, no algorithm dependency |
| `medical/plot_medical_covariate_shift.py` | (in place) | same |
| `results/{domain}/` | `results_legacy/{domain}/` | numbers no longer match either algorithm's semantics |
| `run_all_experiments.sh` | (rewritten in place) | see § 7 |
| `build_tex_tables.py` | (rewritten in place) | see § 7 |

Splitting data wrangling out:

- `medical/data_loader.py` exposes `load_data`, `_convert_to_arrays`, `_encode_ethnicity`, `ETHNICITY_MAP`, the constants (`TARGET_VAR`, `COVARIATE_VARS`, `STATIC_VARS`). Both `medical_runner.py` and `multi_seed_medical.py` import from here.
- `finance/finance_data.py` already plays this role for finance; no change.

---

## 7. Experiment plan and table layout

### 7.1 Conditions

Each domain × regime is run with three ablation conditions, matching v1 of the project:

| Tag | What it isolates | Whole-trajectory (Alg. 1) | Last-step (Alg. 2) |
|---|---|---|---|
| `full` | Both LR weighting and γ-selected ACI | LR + ACI per algo | LR only (no ACI exists) |
| `uniform` | LR off (uniform weights), ACI on | uniform calibration weights | uniform calibration weights |
| `zerog` | LR on, ACI off (γ=0) | LR weights, γ=0 | **not applicable** (no γ) |

So **3 conditions × whole-trajectory + 2 conditions × last-step = 5 conditions per domain**, times 3 domains = 15 saved JSON families. Synthetic is replicated over 30 seeds per condition; finance over 13 rolling windows × 2 sectors (78 finance whole-traj runs) + mixed null; medical over 10 seeds. Exact size table TBD with Q1.

### 7.2 Result folder schema

```
results/{domain}/{regime}/json/{condition}_{detail}.json
results/{domain}/{regime}/pdf/{condition}_{detail}.pdf
results/{domain}/{regime}/tables/{domain}_{regime}.tex
```

`{detail}` is dates for finance windows; nothing for synthetic / medical multi-seed.

### 7.3 `run_all_experiments.sh` restructure

```
./run_all_experiments.sh --synthetic --regime last_step
./run_all_experiments.sh --synthetic --regime whole_trajectory
./run_all_experiments.sh --finance   --regime whole_trajectory
./run_all_experiments.sh --medical   --regime last_step --build-tables
./run_all_experiments.sh --all       # all domains × all regimes
```

Per-condition loops inside each function get a `regime=$1` argument and dispatch to the right runner. Filename stems include the regime tag for unambiguous file paths.

### 7.4 `build_tex_tables.py` restructure

- One `.tex` per (domain × regime) = 6 tables total.
- Each table shares the existing 4-column layout `(Algorithm × Coverage × |Δ̄| × Width)` plus a leading column when the domain has a "data condition" axis (synthetic: noshift/static/dynamic; finance: tech/util/mixed).
- The medical table now has only 2 rows in the last-step regime (no `zerog`).
- Cross-table comparisons (whole-trajectory full vs last-step full) are not auto-generated in this milestone; can be added later via a separate `_emit_comparison_table` helper.

---

## 8. Sanity checks before running experiments

These run on tiny tinker datasets (T=5, n_train=100, n_cal=50, n_test=10) and gate the experiment sweep.

1. **Unweighted reduction**: with `featurize_fn` returning constants (so LR predicts 50/50 → all weights equal), `weighted_cafht_whole.predict_bands(...)` must match an unweighted CAFHT-style band (identical to `CAFHT/.../methods.py:CAFHT.predict_bands` with `adaptive=False`, single-cal mode) up to numerical noise.
2. **Last-step reduction**: with uniform weights and `score_fn=|·|`, `weighted_cafht_last.predict_bands(...)` must match `[Ŷ ± q_(1-α)(scores)]` from the standard split-conformal procedure.
3. **δ_∞ atom**: when `W_test = 100 × Σ W_cal`, `η_j` must equal `∞` and the band must be `(−∞, ∞)` after `y_trim` is applied.
4. **Cross-half disjointness**: every test index appears in exactly one half across the two iterations of the cross-half loop; concatenated output has shape `(n_test, T+1, 2, ndim)` with the original ordering.
5. **Calibration size invariance**: the v2 calibration fix (one cal score per cal series, not pooled across time) is preserved. The number of `ε_i` values must equal `n_cal`, not `n_cal × T`.

---

## 9. Sequence of work

1. **Decisions**: resolve Q1–Q7 in one pass with the user.
2. **Archive**: create `legacy/` and `results_legacy/`, move files per § 6. Single commit, no behavior change.
3. **Primitives**: implement `core/weighted_quantile.py`, `core/density_ratio.py`, `core/aci.py`. Unit tests for each (especially the δ_∞ quantile and `q0` derivation).
4. **Algorithms**: implement `core/weighted_cafht_whole.py` and `core/weighted_cafht_last.py`. Run sanity checks § 8.
5. **Per-domain glue**: implement `{synthetic,finance,medical}/predictors.py` and `featurizers.py`; implement `{finance,medical}/{finance,medical}_runner.py`; refactor `synthetic/multi_seed_experiments.py` and `medical/multi_seed_medical.py`.
6. **Scripts**: rewrite `run_all_experiments.sh` and `build_tex_tables.py` per § 7.
7. **Pilot run**: medical only, both regimes, 10 seeds. Verify tables build. Compare order-of-magnitude numbers against the legacy results to catch silent regressions.
8. **Full sweep**: all 15 condition families. Update `CLAUDE.md` and `README.md` to describe the new structure.

Step 1 blocks step 2–8. Steps 2–6 are roughly two days of focused work; step 7 is the diagnostic gate before launching the full sweep.

---

## 10. Things explicitly out of scope for this milestone

- LSTM / RNN predictors. Ridge on flattened X is the v2 baseline for the last-step regime; sequence models are a v3 question.
- The CAFHT "multiplicative" branch of `nonconf_scores` (`adaptive=True`). v2 implements only the additive branch.
- Importing anything from `CAFHT/ConformalizedTS/` at runtime. We mirror the routines we need; we do not import.
- Re-running the synthetic experiment grid that v1 already produced (Group C). The synthetic sweep stays at 30 seeds but now spans two regimes; old C1–C4 results become reference numbers in `results_legacy/`.
