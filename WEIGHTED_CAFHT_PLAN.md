# Weighted CAFHT — Repository Restructuring Plan (v2)

This document is the implementation plan for the **two-regime / two-algorithm** revision of the project. The previous version of this file (v1, kept in git history) was scoped to writing code inside `CAFHT/` and assumed a single trajectory-level method; both assumptions are obsolete and have been discarded.

**Section order**: §0 (ground rules) → §1 (file layout) → §2 (core components) → §3 (per-domain wiring) → §4 (work plan) → §5 (experiment + tables) → §6 (clarification questions, answered) → **Additional references**: §A (what changed), §B (migration), §C (sanity checks), §D (out of scope).

---

# Progress & Milestones

A running log of what is done and what is still pending. Update it whenever a milestone (or sub-step) is completed, per the rule in § 0.

### Current progress

- Step 1: Q1–Q8 resolved (2026-05-29; see § 6 for the chosen answers).
- Step 2: Renamed 11 legacy files with `OLD_` prefix (2026-05-29; commit `22358a6`). Pure `git mv`, 100% similarity, no content edits. List in § B.1.
- Step 3: Created the 14 new file stubs (2026-05-29; commit `642c4d8`), docstring-only, all byte-compile. 2 algo files + 6 runners + 4 multi-seed wrappers + `run_all_v2.sh` + `build_tex_tables_v2.py`.
- Follow-on: marked 8 legacy per-domain run scripts `OLD_` (2026-05-29; commit `642c4d8`) — superseded by `run_all_v2.sh`. See § B.1 follow-on table.
- Step 4: Implemented `core/weighted_cafht_whole.py` (2026-05-30; commit `b1d2605`). All 4 inline tests + 1 end-to-end smoke pass in ~1.6 s. Reuses `OLD_algorithm.py` LR-weight + δ_∞-quantile logic and CAFHT's ACI/`nonconf_scores`; ACI draws from the frozen D_ACI bank (no warm-start). **Bank is per-time-step**: a 2-D `(n_ACI, T+1)` array, column t builds the band at step t (corrected from an earlier pooled-over-time draft, which gave a wrong time-flat width).
- Step 5: Implemented `core/weighted_cafht_last.py` (2026-05-30; commit `4d7bd3f`). 3 inline tests pass in ~1.3 s. The two shared helpers are byte-identical (docstrings aside) and behaviorally identical to the step-4 versions; no ACI/γ/D_ACI — a single δ_∞-corrected split-conformal interval at T+1, output shape `(n_test, 1, 2, ndim)`.
- Step 8 (done out of order, before 6/7, per user): Implemented the synthetic runner pair + 30-seed wrappers (2026-05-30; committed). Conditions simplified to `full` (our version) vs `uniform` (no-LR) per user — `zerog` dropped. Headline metrics: whole-traj joint coverage + mean width; last-step final coverage + mean width. **30-seed shift results** (saved under `results/synthetic/{whole_trajectory,last_step}/json/`, target 0.90): whole-traj static full **0.886** vs uniform 0.700 (LR clearly helps); whole-traj dynamic full 0.711 ≈ uniform 0.701 (Q2(a) X_1-only featurizer + unshifted X_0 → no signal); last-step static full 0.900 ≈ uniform 0.894; last-step dynamic full 0.868 vs uniform 0.814 (full path carries the shift; 17.5% δ_∞ bands). **Modeling note**: LR featurizer uses the actual Poisson/path covariate (what shifts), not Y_0; predictor is one-step-ahead pure AR(1) (whole) / OLS on Y-history (last).

### Remaining milestones

- **Step 6** — Medical runner pair + 10-seed wrappers (§ 4 step 6).
- **Step 7** — Finance runner pair, per-window (§ 4 step 7).
- **Step 9** — `run_all_v2.sh` + `build_tex_tables_v2.py` (§ 4 step 9).
- **Step 10** — Pilot run: medical, both regimes, 10 seeds (§ 4 step 10).
- **Step 11** — Full sweep `bash run_all_v2.sh --all` (§ 4 step 11).

---

## 0. Scope and ground rules

- **The `CAFHT/` folder is read-only reference material.** No code, scripts, or notebooks under `CAFHT/` will be created or modified. Reading `CAFHT/ConformalizedTS/methods.py` to mirror specific routines is fine; importing from it at runtime is not.
- All code changes happen in our repository (`core/`, `synthetic/`, `finance/`, `medical/`, `results/`, top-level scripts).
- The problem statement now specifies **two regimes** (last-step coverage; whole-trajectory coverage) and **two distinct algorithms** (one per regime). They share two primitives (LR weights, weighted quantile with a δ_∞ atom) but differ in everything else: predictor, conformity score, deployment loop, and γ usage. They will be implemented as **separate top-level entry points**, not as flags on a single class.
- **One algorithm per file.** Each new algorithm `.py` file is **self-contained**: weighted-quantile helper, density-ratio weight estimator, ACI updater (where applicable), and the main class all live in the same module. The shared primitives are deliberately duplicated across the two algorithm files (≈100 lines per file) rather than factored into separate `aci.py` / `density_ratio.py` / `weighted_quantile.py` modules. Same self-contained discipline applies to per-domain runners: each runner file includes its predictor, featurizer, main loop, and CLI.
- **Additive-only repository changes.** No old files move directories or get deleted. Existing files either:
  (a) **stay in place with their current name** if they are fully reused (data loaders, data files, plotting utilities, documentation), OR
  (b) **get an `OLD_` prefix** to mark them as legacy reference — superseded by a new file with a distinct name. We do not import from `OLD_*` files at runtime; if a function from an `OLD_*` file is needed, its logic is copied into the new module that needs it.
- **Existing `results/` outputs are preserved in place.** New results land in new subfolders (`results/{domain}/{regime}/...`) alongside the legacy `results/{domain}/{json,pdf,tables}/` folders. No mass renames or moves under `results/`.
- **Always update Progress & Milestones after a completed milestone.** Whenever any implementation step (or sub-step) finishes, edit the **Progress & Milestones** section at the top of this file: move the completed item from "Remaining milestones" to "Current progress" with a one-liner summary (date + what was done). Treat this as part of the milestone's checkpoint — the milestone is not "done" until the log is updated.

---

## 1. Proposed file layout

Legend: **STAY** = file unchanged, original name kept; **OLD_** = renamed with `OLD_` prefix to mark as legacy reference (no content changes); **NEW** = new file created in this restructuring.

```
core/
  ts_generator.py                       STAY    synthetic DGPs
  OLD_algorithm.py                      OLD_    legacy AdaptedCAFHT (reference)
  OLD_adaptive_conformal.py             OLD_    legacy sliding-window baseline (reference)
  weighted_cafht_whole.py               NEW     Algorithm 1 (self-contained)
  weighted_cafht_last.py                NEW     Algorithm 2 (self-contained)

synthetic/
  OLD_test_conformal.py                 OLD_
  OLD_multi_seed_experiments.py         OLD_
  synthetic_runner_whole.py             NEW     single-seed Alg. 1 runner
  synthetic_runner_last.py              NEW     single-seed Alg. 2 runner
  multi_seed_synthetic_whole.py         NEW     30-seed wrapper, Alg. 1
  multi_seed_synthetic_last.py          NEW     30-seed wrapper, Alg. 2

finance/
  finance_data.py                       STAY    yfinance loader
  plot_covariate_shift.py               STAY    plotting utility
  data/                                 STAY    sp500_*.npz + .json
  OLD_finance_conformal.py              OLD_
  OLD_finance_adaptive.py               OLD_
  OLD_tune_featurizer.py                OLD_
  finance_runner_whole.py               NEW     per-window Alg. 1 runner
  finance_runner_last.py                NEW     per-window Alg. 2 runner

medical/
  medical_data.py                       STAY    data loader
  medical_data.md                       STAY    data dictionary
  sepsis_experiment_data_nacl_target.pkl STAY   raw pickle
  plot_medical_covariate_shift.py       STAY    plotting utility
  OLD_medical_conformal.py              OLD_
  OLD_multi_seed_medical.py             OLD_
  medical_runner_whole.py               NEW     single-seed Alg. 1 runner
  medical_runner_last.py                NEW     single-seed Alg. 2 runner
  multi_seed_medical_whole.py           NEW     10-seed wrapper, Alg. 1
  multi_seed_medical_last.py            NEW     10-seed wrapper, Alg. 2

results/                                STAY    pre-v2 outputs untouched
results/{domain}/{regime}/{json,pdf,tables}/   NEW   6 new subfolders total

OLD_run_all_experiments.sh              OLD_
OLD_build_tex_tables.py                 OLD_
run_all_v2.sh                           NEW     regime-aware dispatcher (§ 5)
build_tex_tables_v2.py                  NEW     6 tables: domain × regime (§ 5)

CAFHT/                                  unchanged (read-only)
```

### Why self-contained algorithm files

The two new algorithms differ in `(predictor type, cal score, deployment loop, γ usage)` — essentially everything except LR weight estimation and the δ_∞ weighted quantile. Splitting those two shared primitives into separate modules would add three import hops for ≈100 lines of code, with no realistic prospect of either primitive being reused elsewhere. Per the user directive, each algorithm file owns its primitives outright. The duplication is paid for by readability (each file maps 1:1 to its algorithm box) and isolation (a change to Algorithm 1's quantile cannot accidentally affect Algorithm 2).

### Why no per-domain `predictors.py` / `featurizers.py`

Same logic applied at the domain level: each runner file (`{domain}_runner_{regime}.py`) is self-contained — predictor class/function, featurizer, main loop, CLI all in one module — so a reader navigating from the shell script to the actual algorithm crosses exactly two file boundaries (`{domain}_runner_{regime}.py` → `core/weighted_cafht_{regime}.py`).

### What "fully reused" means and what gets copied

- **Fully reused (`STAY`)** — `medical/medical_data.py`, `medical/medical_data.md`, the sepsis pickle, `finance/finance_data.py`, `finance/plot_covariate_shift.py`, `medical/plot_medical_covariate_shift.py`, `core/ts_generator.py`. Imported as-is by the new runners.
- **Logic copied, not imported (`OLD_*`)** — when a new runner needs something from `OLD_medical_conformal.py` (e.g., `LinearCovariateModel`, `_convert_to_arrays`, `ETHNICITY_MAP`), the relevant block is **copied** into the new runner file, with attribution comment `# from OLD_medical_conformal.py`. We do not `from OLD_medical_conformal import ...`. Same convention for the other `OLD_*` files.

This means the new tree is fully separable from the old tree: deleting all `OLD_*` files would not break any runtime import path.

---

## 2. Core components in detail

Both algorithm files are self-contained. The three primitives (weighted quantile with δ_∞ atom, density-ratio weights, ACI updater for the whole-trajectory file only) appear as module-level helpers at the top of each file. Below: a clarification of the hidden held-out set required by ACI (§ 2.0), the contract for each helper, then the class layout for each algorithm. The two files duplicate the helper code intentionally; do not refactor into shared modules.

### 2.0 The held-out set D_ACI (whole-trajectory only)

**The algorithm box is silent on where ACI's conformity scores come from.** It runs ACI in two places — once per calibration trajectory to produce `ε_i`, and once per deployed test trajectory to produce `Ĉ^aci_t` — but does not say what conformity-score history seeds those ACI runs. ACI's band at step t is built from a quantile of past scores; with no banked history, ACI has nothing to quantile over at step 1. This plan resolves the silence by introducing an explicit held-out set **D_ACI** that is **disjoint from D_tr, D_cal, and D_test**: its absolute-residual array is frozen as the score bank for every main-algorithm ACI invocation downstream of predictor fitting.

**Data partition at the runner level (whole-trajectory).** The raw trajectory pool is split into **four** disjoint subsets:
- **D_tr** — predictor fit AND γ selection. Per the algorithm box, predictors `{f_t}_{t∈[T+1]}` are fit on the **entire** D_tr by regressing `Y_t^(i)` on `(Z_{1:(t-1)}^(i), X_t^(i))` for all `i ∈ D_tr`. For γ selection, D_tr is then randomly partitioned **internally** into three parts D_tr^(1) / D_tr^(2) / D_tr^(3); for each γ ∈ Γ, run the simple ACI procedure with train = D_tr^(1), cal = D_tr^(2), test = D_tr^(3), and record the performance; γ_opt picks the best performance (min mean band width on D_tr^(3); see Q4). This 3-way split lives entirely inside D_tr and exists only for γ selection — it does **not** involve D_ACI.
- **D_ACI** — score bank for the main-algorithm ACI runs. **Separate held-out set, disjoint from D_tr, D_cal, D_test.** Apply the fitted predictors `{f_t}` to each `i ∈ D_ACI` to get `Ŷ_t^(i)`; compute the residual array `{|Y_t^(i) − Ŷ_t^(i)| : i ∈ D_ACI, t ∈ [T+1]}` once; this `(|D_ACI|, T+1)` array is then immutable and passed into every main-algorithm ACI call (cal-side `ε_i` on D_cal; test-side `Ĉ^aci_t` on D_test). **The bank is used per time step**: at ACI step t the band half-width is the `(1−α_t)`-quantile of **column t** of this array (the residuals at time t) — NOT a pool over all time steps, since residual magnitudes differ across t.
- **D_cal** — calibration. Per the algorithm box: for each `i ∈ D_cal`, run ACI with γ_opt and the D_ACI score bank to produce bands `(L_t^(i), U_t^(i))`, then take `ε_i = max_t max{Y_t − U_t, L_t − Y_t}`.
- **D_test** — deployment. Per the algorithm box: cross-half split, fit LR (positives = opposite half), compute η_j via weighted quantile, run ACI on each test trajectory (with D_ACI bank and γ_opt) and inflate by ±η_j.

**Disjointness**: `D_ACI ∩ D_tr = D_ACI ∩ D_cal = D_ACI ∩ D_test = ∅`. Runners peel D_ACI off the raw trajectory pool **before** the conventional D_tr / D_cal / D_test split.

**Two ACI invocations with different score banks.** This is the source of confusion the algorithm box leaves implicit:
- **γ-selection ACI** (inside `select_gamma`) runs the standard simple-ACI procedure on D_tr^(1)/D_tr^(2)/D_tr^(3); its score bank is built from D_tr^(2) residuals (cal role within simple ACI). This is a sandbox simulation used only to choose γ_opt.
- **Main-algorithm ACI** (inside `calibration_scores` and `predict_bands`) runs on D_cal trajectories (to produce ε_i) and on D_test trajectories (to produce Ĉ^aci_t); its score bank is the frozen D_ACI residuals.

**Why frozen for the main algorithm.** Letting ACI re-derive its score bank from D_cal at cal time or from past test-trajectory observations at deploy time would create circular dependencies between the LR-weighted-quantile step and the ACI step. Freezing the bank from a disjoint D_ACI keeps the main algorithm modular.

**Default size budget.** Peel `--frac_aci` of trajectories off the raw pool as D_ACI (default 0.15), then split the remainder into D_tr / D_cal / D_test by their existing fractions. The internal γ-selection split of D_tr defaults to `--gamma_split 0.33 0.33 0.34`.

**Algorithm 2 (last-step) does not need D_ACI.** No ACI is invoked at any point.

### 2.1 Shared helpers (duplicated in both algo files)

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

```
def density_ratio_weights(
    X_pos, X_neg, X_eval,
    *,
    classifier=None,        # default: sklearn LogisticRegression(class_weight='balanced')
    clip_factor=5.0,        # 5x mean clip on the raw ratio
    prob_clip=1e-6,         # clip probabilities to [prob_clip, 1-prob_clip] pre-ratio
) -> dict:
    """
    Fit a logistic classifier on X_pos (label 0) vs X_neg (label 1), score X_eval,
    return raw clipped weights (NOT normalized) plus diagnostics:
        {"w_eval": (n_eval,), "prob1_eval": (n_eval,), "train_acc": float, "coef_norm": float}
    Returns UNNORMALIZED weights because weighted_quantile_with_inf needs raw masses
    to compute the Σ_k W_k + W_test denominator with the test-point atom.
    """
```

Both helpers' logic is adapted from `OLD_algorithm.py:AdaptedCAFHT._compute_density_ratio_weights` and `_weighted_quantile` — the original `_weighted_quantile` is generalized to insert the δ_∞ atom and to take an unnormalized weight vector + a separate test-point mass.

### 2.2 ACI updater (only in `core/weighted_cafht_whole.py`)

```
class ACI:
    """
    Single-trajectory adaptive conformal inference:
        α_{t+1} = α_t + γ · (α − err_t),  clipped to (1e-6, 1 - 1e-6).
    The band at step t is built from a frozen `score_bank` (absolute residuals)
    supplied by the caller — specifically from COLUMN t of that bank (the
    residuals at time t), NOT a pool over all steps. Predictor-agnostic: takes
    precomputed (y_pred, y_true) trajectories for online error computation.
    Mirrors CAFHT/ConformalizedTS/methods.py:Adaptive_Conformal_Inference but
    with an externally-supplied, per-time-step score bank instead of an
    internally-grown buffer (this removes CAFHT's warm-start entirely).
    """
    def predict_intervals(self, score_bank, y_pred, y_true, gamma, seed=...): ...
```

`score_bank` is a **2-D `(n_bank × (T+1))` array** of absolute residuals: **column t is the score pool used to build the ACI band at step t**, so the base half-width is time-varying (it tracks how residual magnitude changes along the horizon). It is NOT flattened/pooled across time — pooling would give a wrong time-flat width. The caller decides what trajectories it represents: during γ selection it is built from D_tr^(2) residuals (the simple-ACI sandbox); during the main-algorithm calibration and deployment phases it is built from D_ACI residuals (see § 2.0). Its horizon must equal the deployed trajectories' horizon. Algorithm 2 does not need ACI, so `weighted_cafht_last.py` does not include this class.

### 2.3 `core/weighted_cafht_whole.py`  (Algorithm 1)

```
class WeightedCAFHTWholeTrajectory:
    """
    Implements the whole-trajectory coverage algorithm.
    Predictor-agnostic: caller supplies precomputed (pred, true) arrays of shape
    (n, T+1, ndim) for D_tr (full), D_ACI, D_cal, D_test, plus the X arrays used
    by the LR step. Predictor fitting itself happens in the runner on the FULL
    D_tr; D_ACI must be peeled off the raw pool BEFORE the D_tr/D_cal/D_test
    split per § 2.0.
    """

    def __init__(self, alpha, gamma_grid, featurize_fn,
                 gamma_split=(0.33, 0.33, 0.34),  # internal D_tr^(1)/D_tr^(2)/D_tr^(3) of D_tr, for γ selection only
                 weight_clip=5.0, randomize=False, verbose=True): ...

    def select_gamma(self, tr_pred, tr_true, seed) -> float:
        # 1. Internally split D_tr indices into D_tr^(1)/D_tr^(2)/D_tr^(3) per gamma_split.
        # 2. Build the sandbox score bank from D_tr^(2) residuals only:
        #        sandbox_bank = |tr_true[D_tr^(2)] - tr_pred[D_tr^(2)]|
        #    (NOT D_ACI; this is the simple-ACI cal set per the algorithm box.)
        # 3. For each γ in gamma_grid: run ACI(sandbox_bank, tr_pred[D_tr^(3)][i],
        #    tr_true[D_tr^(3)][i], γ) for each i in D_tr^(3); compute mean band width.
        # 4. Return γ with minimum mean width on D_tr^(3) (CAFHT convention; see Q4).
        # Note: D_tr^(1)'s role in the simple-ACI sandbox follows CAFHT's convention.

    def calibration_scores(self, cal_pred, cal_true, score_bank, gamma_opt, seed) -> np.ndarray:
        # `score_bank` is the frozen D_ACI residual array (built by the caller).
        # For each i in D_cal: bands = ACI(score_bank, cal_pred[i], cal_true[i], γ_opt).
        # ε_i = max_t max{y_t - U_t, L_t - y_t}  (additive branch; CAFHT/.../methods.py:313).

    def predict_bands(self, tr_data, cal_data, test_data, aci_data,
                      X_cal, X_test, y_trim=None, seed=...) -> np.ndarray:
        # tr_data   = (tr_pred, tr_true)     full D_tr; γ selection splits internally
        # cal_data  = (cal_pred, cal_true)   D_cal
        # test_data = (test_pred, test_true) D_test
        # aci_data  = (aci_pred, aci_true)   D_ACI; used to build the frozen score bank
        #
        # 1. score_bank = |aci_true - aci_pred|   (frozen D_ACI residuals; computed once).
        # 2. γ_opt = self.select_gamma(tr_pred, tr_true).
        # 3. ε = self.calibration_scores(cal_pred, cal_true, score_bank, γ_opt).
        # 4. For each test half (cross-half scheme):
        #     a. Fit LR (positives = opposite half; features via featurize_fn).
        #     b. W_cal = clf weights on X_cal (raw, clipped).
        #     c. For each j in this half:
        #          W_j = clf weight on X_test[j].
        #          η_j = weighted_quantile_with_inf(ε, W_cal, W_j, 1-α).
        #          aci_j = ACI(score_bank, test_pred[j], test_true[j], γ_opt)
        #          band[j, t] = [min(aci_j[t]) - η_j, max(aci_j[t]) + η_j]  ∀ t
        # 5. Concatenate; return shape (n_test, T+1, 2, ndim).
```

The runner is responsible for the four-way data partition `(D_ACI, D_tr, D_cal, D_test)` per § 2.0 — D_ACI must be peeled off the raw pool **before** the conventional D_tr / D_cal / D_test split, and predictors must be fit on the **full** D_tr before passing the (pred, true) pairs into `predict_bands`. The class itself never sees D_ACI as a set — only as a precomputed residual array (`aci_data`).

### 2.4 `core/weighted_cafht_last.py`  (Algorithm 2)

```
class WeightedCAFHTLastStep:
    """
    Implements last-step coverage. No γ, no ACI, no D_ACI, no per-step bands.
    Predictor-agnostic: caller supplies precomputed scalar predictions for cal and test.
    """

    def __init__(self, alpha, featurize_fn, weight_clip=5.0, score_fn=None, verbose=True):
        # score_fn defaults to lambda y, yhat: abs(y - yhat)

    def calibration_scores(self, y_cal_true, y_cal_pred) -> np.ndarray: ...
        # ε_i = score_fn(Y_{T+1}^(i), Ŷ_{T+1}^(i))

    def predict_bands(self, cal_data, test_data, X_cal, X_test) -> np.ndarray:
        # 1. ε = self.calibration_scores(...).
        # 2. For each test half:
        #     a. Fit LR (positives = opposite half).
        #     b. W_cal = clf weights on X_cal (raw, clipped).
        #     c. For each j in this half:
        #          W_j = clf weight on X_test[j].
        #          η_j = weighted_quantile_with_inf(ε, W_cal, W_j, 1-α).
        #          band[j] = [Ŷ_j − η_j, Ŷ_j + η_j]   (score_fn=|·|)
        # 3. Concatenate; return shape (n_test, 1, 2, ndim).  Or (n_test, 2) if ndim=1.
```

Trivial compared to Alg. 1 — just the cross-half + δ_∞ quantile step.

---

## 3. Per-domain predictors and featurizers

Each predictor and featurizer is implemented **inside its runner file** — `{domain}_runner_{regime}.py` — as a module-level function or small class. The runners do not import from a shared `predictors.py` / `featurizers.py` module (no such modules exist in this plan; per § 0, runners are self-contained).

### 3.1 Whole-trajectory predictors (Algorithm 1) — defined in `{domain}_runner_whole.py`

| Domain | Predictor symbol | Implementation | Output |
|---|---|---|---|
| Synthetic | `build_whole_trajectory_predictor(Y_train)` in `synthetic_runner_whole.py` | global AR(1) by OLS on training pairs; iterated forward | `(n, T+1, 1)` predicted trajectory |
| Finance | `LinearCovariateModel` (copied from `OLD_finance_conformal.py`) in `finance_runner_whole.py` | per-step OLS on `X_t`; fit once globally, applied at each t | `(n, T+1, 1)` |
| Medical | `LinearCovariateModel` (copied from `OLD_medical_conformal.py`) in `medical_runner_whole.py` | one-step-ahead AR `Y_{t+1} ~ Y_t + X_t + S` | `(n, T+1, 1)` |

### 3.2 Last-step predictors (Algorithm 2) — defined in `{domain}_runner_last.py`

| Domain | Predictor symbol | Implementation | Output |
|---|---|---|---|
| Synthetic | `build_last_step_predictor(Y_train, lags=T)` in `synthetic_runner_last.py` | OLS of `Y_{T+1}` on `Y_{1:T}` | `(n,)` scalar prediction |
| Finance | `LastStepRidge(...)` class in `finance_runner_last.py` | ridge of `Y_{T+1}` on flattened `X_{1:T}` (4·T features); λ by 5-fold CV | `(n,)` |
| Medical | `LastStepRidge(...)` class in `medical_runner_last.py` | ridge of `Y_{T+1}` on flattened `X_{1:T}` + S (3·T + 6 features); λ by 5-fold CV | `(n,)` |

### 3.3 LR featurizers — defined inside each runner

Each runner defines its featurizer as a small function taking the X array and returning a 2D feature matrix, then passes it as `featurize_fn` to the algorithm class.

| Domain | Whole-trajectory featurizer (Alg. 1, X_1) | Last-step featurizer (Alg. 2, X_{1:T}) |
|---|---|---|
| Synthetic | `featurize_x1(X)` → `Y_0` only (== X_1 under the X=Y_lag reinterpretation) | `featurize_xall(X)` → full Y history |
| Finance | `featurize_x1(X)` → `X_1` = 4 covariates at day 1 | `featurize_xall(X)` → flattened `X_{1:T}` (4·T) |
| Medical | `featurize_x1(X, S)` → `X_1` = 3 vitals at hour 0 + 6 statics = 9 features | `featurize_xall(X, S)` → flattened `X_{1:T}` + S (3·T + 6 features) — but see Q2 |

The function names (`featurize_x1`, `featurize_xall`) are by convention; the algorithm class only sees an opaque callable. Each runner instantiates a single featurizer matching its regime; cross-regime sharing is not needed.

---

## 4. Sequence of work

Each numbered step ends in a checkpoint that should be obviously verifiable (file present, test passing, JSON committed, table rendering). Steps 1–3 are housekeeping and produce no behavior change; steps 4–10 build the new pipeline incrementally; step 11 is the full experiment sweep.

### 1. Resolve Q1–Q8  *(DONE — 2026-05-29; answers in § 6)*

Single sit-down with the user. Each answer feeds into the file layout (Q1, Q5, Q6, Q7), the LR featurizer (Q2), the last-step predictor (Q3), the γ rule (Q4), or the D_ACI partition (Q8). **Checkpoint**: Q1–Q8 marked resolved at the top of this file with the chosen answer in one line each. ✓

### 2. `OLD_` renames (one commit, no content edits)  *(DONE — 2026-05-29; commit `22358a6`)*

`git mv` the 11 files listed in § B.1. Update no other files. **Checkpoint**: tree shows the new names; `git status` is clean after the commit; running any of the existing `OLD_*` scripts still works (just under the new name). ✓ All 11 renamed at 100% similarity; tree clean. (Note: cross-imports between `OLD_*` files are intentionally left unfixed — these are frozen reference files, not in any active import path, per § B.1.)

### 3. Skeleton new files (one commit, all empty)  *(DONE — 2026-05-29; commit `642c4d8`)*

Touch the 14 new files (`core/weighted_cafht_{whole,last}.py`; 6 per-domain runners; 4 multi-seed wrappers; `run_all_v2.sh`; `build_tex_tables_v2.py`) with only a module docstring describing what each file will contain. **Checkpoint**: `find . -name "OLD_*"` lists 11; `find . -newer ...` for the empty stubs lists 14; nothing else changed. ✓ Verified: 11 `OLD_` files, 14 new untracked stubs, all 13 Python stubs byte-compile and `run_all_v2.sh` parses.

### 4. `core/weighted_cafht_whole.py` — Algorithm 1, end-to-end  *(DONE — 2026-05-30; commit `b1d2605`)*

Inline order: `weighted_quantile_with_inf` → `density_ratio_weights` → `ACI` class (with `score_bank` argument per § 2.2) → `WeightedCAFHTWholeTrajectory` class (with D_ACI partition logic per § 2.0). The class exposes `select_gamma`, `calibration_scores`, `predict_bands` per § 2.3.

**Implementation notes** (deviations / decisions): (a) `predict_bands` takes an extra `X_tr` arg — the algo box's classifier negatives are `{(X_1^i,0)}_{i∈D_tr}`, which the plan's original § 2.3 signature omitted. (b) Cal weights are 5×-mean clipped but the test-point weight `W_j` is kept RAW, mirroring `OLD_algorithm.predict_with_interval_oracle`, so the δ_∞ atom can fire. (c) ε_i is computed once (not per cross-half) since the ACI bands are weight-free. (d) ACI uses a per-time-step 2-D bank (column t builds step t's band) and no warm-start (the frozen D_ACI bank removes the cold-start that warm-start patched in CAFHT).

**Inline tests** (at the bottom of the file under `if __name__ == "__main__"`):
- `weighted_quantile_with_inf` with uniform weights matches `np.quantile(scores, level, interpolation='higher')`.
- `weighted_quantile_with_inf` with `w_test = 100·Σ w_cal` returns `np.inf`.
- `density_ratio_weights` with `X_pos == X_neg` returns weights near 1 for any `X_eval`.
- `ACI.predict_intervals` with γ=0 and a constant `score_bank` returns bands whose width equals the quantile of `score_bank` at level 1−α₀ for every t.

**Checkpoint**: `python -m core.weighted_cafht_whole` runs all four tests in <5 s and reports `OK`.

### 5. `core/weighted_cafht_last.py` — Algorithm 2, end-to-end  *(DONE — 2026-05-30; commit `4d7bd3f`)*

Inline order: `weighted_quantile_with_inf` (copied verbatim from step 4) → `density_ratio_weights` (copied verbatim) → `WeightedCAFHTLastStep` class per § 2.4. No ACI class, no D_ACI.

**Implementation notes**: (a) `predict_bands` takes `X_tr` (classifier negatives), same justified deviation as step 4. (b) Same clipped-cal / raw-test weighting so the δ_∞ atom fires. (c) `cal_data`/`test_data` use the `(pred, true)` order, matching step 4, so runner authors see one convention; `test_true` is accepted but unused (last-step has no online adaptation). (d) Output is `(n_test, 1, 2, ndim)` — horizon axis length 1 — so the runners share one coverage/width routine with the whole-trajectory regime. (e) Default `score_fn` is the ∞-norm abs residual; the symmetric `[Ŷ ± η]` box is the exact level set only for such a score.

**Inline tests**:
- The two duplicated helpers produce bit-identical output to the step-4 versions on a fixed-seed input. ✓ (also verified byte-identical with docstrings stripped via `ast`).
- With uniform weights, `predict_bands` returns the standard split-conformal band `[Ŷ ± q_(1-α)]`. ✓
- (extra) A dominant test weight trips the δ_∞ atom → unbounded interval. ✓

**Checkpoint**: `python -m core.weighted_cafht_last` runs all tests and reports `OK`. ✓ (~1.3 s)

### 6. Medical runner pair — `medical_runner_whole.py` and `medical_runner_last.py`

Why medical first: smallest sample size, fastest per-seed wall time, and we have the most experience tuning the LR step here. For each runner:

- Import `medical_data.py` for the loader and constants.
- Copy `LinearCovariateModel` (whole-traj) or implement `LastStepRidge` (last-step) inline per § 3.
- Define `featurize_x1` (whole-traj) or `featurize_xall` (last-step) inline per § 3.3.
- For whole-traj only: implement the **four-way data partition** (D_ACI + D_tr + D_cal + D_test) per § 2.0 — peel D_ACI off the raw patient pool **before** the conventional D_tr / D_cal / D_test split; fit predictors on the **full** D_tr; precompute the D_ACI score bank by applying the fitted predictors to D_ACI and storing absolute residuals; the internal D_tr^(1) / D_tr^(2) / D_tr^(3) split for γ selection lives inside the algorithm class, not the runner.
- Build the main loop calling the appropriate `core/weighted_cafht_*` class.
- CLI matches the existing `medical/OLD_medical_conformal.py` flags (`--pkl`, `--n_traincal`, `--n_test`, `--alpha`, `--mode {full,uniform,zerog}`, `--seed`, `--save_json`, `--save_plot`), plus `--frac_aci 0.15 --gamma_split 0.33 0.33 0.34` for whole-traj.
- Then write `multi_seed_medical_{whole,last}.py` as thin 10-seed wrappers around the runner.

**Checkpoint**: a single-seed `medical_runner_whole.py --mode full --n_traincal 200 --n_test 100` finishes in <60 s and writes a JSON; same for last-step.

### 7. Finance runner pair — `finance_runner_whole.py` and `finance_runner_last.py`

Same recipe as step 6, scaled to the per-window finance loop. No multi-seed wrapper for finance (the shell script iterates over windows). CLI matches `OLD_finance_conformal.py` (`--npz`, `--test_sector`, `--mode`, `--gamma_grid`, `--seed`, `--save_json`, `--save_plot`), plus `--mixed` for the null baseline and `--frac_aci` + `--gamma_split` for whole-traj.

**Checkpoint**: `finance_runner_whole.py --npz finance/data/sp500_20240201_20240328.npz --test_sector Technology --mode full` finishes in <90 s.

### 8. Synthetic runner pair — `synthetic_runner_whole.py` and `synthetic_runner_last.py`  *(DONE — 2026-05-30; committed; done before 6/7 per user)*

Smallest runtime per cell, so we leave it for last to avoid blocking step-7 debugging. The 30-seed wrappers go on top of each runner.

**Implementation notes**: (a) conditions are `full` vs `uniform` only (`zerog` dropped per § 5.1). (b) LR featurizer uses the actual covariate that shifts (static: Poisson scalar; dynamic: X_0 for whole / full X-path for last), NOT Y_0 — the plan's "Y_0 (X_1 under X=Y_lag)" would be unshifted under this DGP and make the LR a no-op. (c) Predictor follows the algorithm box's observed-prefix `f̂_t`: one-step-ahead pure AR(1) (whole) and OLS of Y_T on the Y-history (last). (d) Source pool (P) → D_tr/D_ACI/D_cal; independent target pool (P̃) → D_test. (e) Headline metrics: whole-traj joint coverage + mean width; last-step final coverage + mean width (per-step profiles kept for plots). Dynamic last-step shows high δ_∞-band counts (sharp path separation) — flagged for tuning.

**Checkpoint**: a 3-seed smoke run of each multi-seed wrapper completes in <60 s and produces a JSON with the new aggregated schema (per-seed coverage list, overall coverage_mean / coverage_se, etc.).

### 9. `run_all_v2.sh` and `build_tex_tables_v2.py`

`run_all_v2.sh`: take `OLD_run_all_experiments.sh` as a starting point, replicate its dispatch structure, and add a `--regime` flag (`last_step`, `whole_trajectory`, both). For each (domain, regime) combination, call the appropriate runner per condition (`full` / `uniform` / `zerog` where applicable). Default `PY="python"` (the conda-env issue from v1).

`build_tex_tables_v2.py`: extend `OLD_build_tex_tables.py` to emit 6 tables instead of 4 — one per (domain × regime). Every table now has 2 rows per data-condition group (`full`, `uniform`); the `zerog` row is dropped everywhere (§ 5.1, simplified).

**Checkpoint**: `bash run_all_v2.sh --medical --regime last_step --build-tables` produces 2 JSONs (for `full` and `uniform`) and `results/medical/last_step/tables/medical_last_step.tex`.

### 10. Pilot run — medical, both regimes, 10 seeds

Run `bash run_all_v2.sh --medical --regime whole_trajectory` and `bash run_all_v2.sh --medical --regime last_step` end-to-end. Verify:

- All expected JSONs land in `results/medical/{whole_trajectory,last_step}/json/`.
- Both `.tex` tables render to LaTeX without errors.
- Coverage on the `uniform` condition is in the same order of magnitude as the v1 `uniform` runs (sanity check against silent regressions; not a strict equality).
- Per § C, the four sanity checks (unweighted reduction, δ_∞ atom behavior, cross-half disjointness, calibration-set size = `n_cal`) all pass with assertions in the runner.

**Checkpoint**: 5 JSONs (3 for whole-trajectory, 2 for last-step) and 2 `.tex` tables committed.

### 11. Full sweep

`bash run_all_v2.sh --all`. Estimated runtime: finance dominates (~ same as the v1 sweep, ~2–3 h, now doubled to ~4–6 h with both regimes); medical 30–60 min per regime; synthetic 1–2 h per regime depending on Q1 scope. Plan for a single 8–12 h overnight run.

**Checkpoint**: 6 `.tex` tables under `results/{domain}/{regime}/tables/` with realistic numbers; `CLAUDE.md` and `README.md` updated to point at the new layout; the `OLD_` files remain on disk for cross-reference but are not in any active import path.

### Dependency graph

```
1 (decisions)  →  2 (renames)  →  3 (skeletons)  →  4 (whole algo)
                                                 ↘
                                                  5 (last algo)
                                                 ↓
                                                 6 (medical)  →  10 (pilot)  →  11 (full sweep)
                                                 ↓
                                                 7 (finance)  ↗
                                                 ↓
                                                 8 (synthetic) ↗
                                                 ↓
                                                 9 (scripts) ↗
```

Steps 6, 7, 8, 9 are independent after step 5; they can be parallelized across reviewers but each must reach its checkpoint before step 10.

---

## 5. Experiment plan and table layout

### 5.1 Conditions  *(simplified 2026-05-30 per user: LR vs no-LR only)*

Each (domain × regime) is run with **two** conditions — "our version" vs the
"no-LR version" — to isolate the likelihood-ratio reweighting. The earlier
3-condition ablation (the `zerog` / LR-only-γ=0 cell) is dropped.

| Tag | Role | Whole-trajectory (Alg. 1) | Last-step (Alg. 2) |
|---|---|---|---|
| `full` | our version | LR weights + γ-selected ACI | LR weights (no ACI exists) |
| `uniform` | no-LR version | uniform weights, ACI on | uniform weights |

So **2 conditions per (domain × regime)**, times 6 cells = 12 saved JSON
families. Synthetic is replicated over 30 seeds per condition; finance over 13
rolling windows × 2 sectors + mixed null; medical over 10 seeds. Per Q1
(answered), all 6 (domain × regime) cells are in scope.

**Performance metrics** (the only two reported per cell):
- Whole-trajectory: **joint coverage** — the fraction of test trajectories
  covered at *every* step simultaneously (P(∀t: Y_t ∈ Ĉ_t)) — and **mean band
  width**.
- Last-step: **coverage at the final step** (fraction of Y_{T+1} in the
  interval) and **mean band width**.

(Per-step coverage/width profiles are still saved for diagnostic plots, but they
are secondary to the two headline metrics above.)

### 5.2 Result folder schema

```
results/{domain}/{regime}/json/{condition}_{detail}.json
results/{domain}/{regime}/pdf/{condition}_{detail}.pdf
results/{domain}/{regime}/tables/{domain}_{regime}.tex
```

`{detail}` is dates for finance windows; nothing for synthetic / medical multi-seed.

### 5.3 `run_all_experiments.sh` restructure

```
./run_all_experiments.sh --synthetic --regime last_step
./run_all_experiments.sh --synthetic --regime whole_trajectory
./run_all_experiments.sh --finance   --regime whole_trajectory
./run_all_experiments.sh --medical   --regime last_step --build-tables
./run_all_experiments.sh --all       # all domains × all regimes
```

Per-condition loops inside each function get a `regime=$1` argument and dispatch to the right runner. Filename stems include the regime tag for unambiguous file paths.

### 5.4 `build_tex_tables.py` restructure

- One `.tex` per (domain × regime) = 6 tables total.
- Each table reports the two headline metrics **Coverage × Width** for the two conditions (`full` = our version, `uniform` = no-LR), plus a leading column when the domain has a "data condition" axis (synthetic: noshift/static/dynamic; finance: tech/util/mixed). Whole-trajectory tables report joint coverage; last-step tables report final-step coverage.
- Every table now has exactly **2 rows per data-condition group** (full, uniform) — the `zerog` row is gone in all regimes.
- Cross-table comparisons (whole-trajectory full vs last-step full) are not auto-generated in this milestone; can be added later via a separate `_emit_comparison_table` helper.

---

## 6. Clarification questions (answered)

All eight questions were resolved on 2026-05-29. The original prompts and recommendations are preserved below for traceability; each ends with the user's chosen answer.

**Q1. Which (domain × regime) cells do we evaluate?**
Combinations are `{synthetic, finance, medical} × {last-step, whole-trajectory}` = 6. We have committed effort to whole-trajectory for all three domains historically. Last-step is new. Do we want all 6 cells, or only some?
*Impact*: number of runners to write; number of result subfolders; size of LaTeX tables; CPU budget for the next sweep.

**Answer:** We want all 6 cells.

**Q2. LR featurizer when the shift assumption is violated.**
The whole-trajectory algorithm assumes shift is on `X_1` only and prescribes feeding `X_1^(i)` to the classifier. For medical, the Norepinephrine split induces shift on a 12-hour window (so `X_{1:12}`) not on `X_1`. Two options:
  (a) follow the spec literally and accept some loss of separability;
  (b) use the entire `X_{1:T}` for the LR classifier and document the deviation.
*Impact*: medical Alg. 1's empirical coverage. Finance has a similar question (Tech/Util shift is on the joint covariate distribution, not on X_1).

**Answer:** (a).

**Q3. Predictor for last-step (Alg. 2) on each domain.**
The algorithm says "regress `Y_{T+1}` on `X_{1:T}`". For our settings:
  - **Synthetic**: with `X_t = Y_{t-1}` (the AR(1) reinterpretation), the natural predictor is just an OLS regression of `Y_{T+1}` on the full lagged history.
  - **Finance**: `X` is 4-dimensional × 40 days = 160 features per ticker if flattened. Options: flatten + L2-regularized linear; use only the most recent few X_t; use an RNN.
  - **Medical**: `X` is 3-dim × 24 hours + 6 static = 78 features per patient (or 84 with NaCl history). Options: flatten + ridge; use last 12 hours only.
*Impact*: predictor accuracy, which dominates residual magnitude and therefore interval width. **Strong recommendation: use ridge regression on flattened `X_{1:T}` with a small held-out CV for λ; defer RNNs to a later iteration.**

**Answer:** Use linear predictor for all domains. Synthetic: regress `Y_{T+1}` on the full lagged history. Finance: flatten + L2-regularized linear. Medical: flatten + ridge.

**Q4. γ-selection criterion in the whole-trajectory algorithm.**
The algorithm box says "Run the simple ACI method with training set D_tr^(1), calibration set D_tr^(2), and test set D_tr^(3). Record the performance" and pick γ with "best performance"; CAFHT's own implementation picks γ by minimum mean band width (`CAFHT/ConformalizedTS/methods.py:393`); our pre-v2 `_select_gamma` in medical/finance picked γ by "tail-half coverage closest to 1−α". **Recommendation: match CAFHT — run the simple-ACI procedure exactly as the algorithm box specifies (train = D_tr^(1), cal = D_tr^(2), test = D_tr^(3); the score bank for this sandbox ACI is built from D_tr^(2) residuals — *not* D_ACI), measure mean band width on D_tr^(3), and pick the γ with minimum mean width.**

**Answer:** Yes. Let's do minimum mean width.

**Q5. Treatment of legacy `core/algorithm.py:AdaptedCAFHT` and per-domain runners.**
The user has chosen the rename-with-`OLD_`-prefix approach (see § 0 ground rules and § B.1). Open sub-question: do we keep `OLD_` files committed in the repository long-term, or move them out after the new pipeline is validated and the legacy numbers are no longer needed?
**Recommendation: keep them in tree indefinitely.** They are reference-only, no runtime cost, and useful for git-blame continuity and for paper-revision cross-checks. We can revisit deletion after the paper is submitted.

**Answer:** Keep them indefinitely.

**Q6. Existing `results/` JSON outputs.**
Existing `results/{medical,finance,synthetic}/{json,pdf,tables}/` files were produced under the old AdaptedCAFHT semantics, then under the partially-fixed semantics. None of them are comparable to the new algorithms.
Per § 0, no mass move under `results/`. New outputs land in `results/{domain}/{regime}/{json,pdf,tables}/` alongside the existing subdirs.
Open sub-question: should the legacy `.tex` tables (`results/{domain}/tables/{domain}.tex`) be deleted, or kept and shadowed by the new ones at `results/{domain}/{regime}/tables/{domain}_{regime}.tex`?
**Recommendation: keep the legacy `.tex` files in place; paper drafts will pull from the new paths only.**

**Answer:** Keep the legacy .tex files in place; paper drafts will pull from the new paths only.

**Q7. Naming.**
Two options for the user-facing names:
  (a) "Weighted CAFHT (whole-trajectory)" and "Weighted CAFHT (last-step)" — emphasizes shared lineage; matches the algo-box titles.
  (b) "Weighted CAFHT" (whole-trajectory) and "Weighted Split CP" (last-step) — semantically more accurate since last-step has no per-step inner conformal loop.
*Impact*: paper section names, table captions, file names.
**Recommendation: (a)** — match the user's algo boxes so the paper and code share terminology.

**Answer:** (a). It'll be convenient to change the algorithm name later when needed.

**Q8. ACI conformity-score bank (the hidden D_ACI).**
The whole-trajectory algorithm box runs ACI in two places in the main algorithm (per cal series to produce `ε_i`; per test series to produce `Ĉ^aci_t`) but **does not specify where the conformity scores used inside those ACI runs come from**. ACI computes its band width as a quantile of past conformity scores; without a banked score history, ACI has no quantile to draw on at step 1. This is a genuine spec gap, not an implementation detail.
We propose adding an explicit held-out set **D_ACI** — **disjoint from D_tr, D_cal, and D_test** — whose precomputed absolute residuals `{|Y_t^(i) − Ŷ_t^(i)| : i ∈ D_ACI, t ∈ [T+1]}` are frozen as the score bank for the main-algorithm ACI invocations on D_cal and D_test. The γ-selection ACI (the simple-ACI sandbox inside `select_gamma`) is a **separate** invocation; per the algorithm box, it uses the 3-way split D_tr^(1)/D_tr^(2)/D_tr^(3) of D_tr and draws its own (internal) score bank from D_tr^(2), not from D_ACI. See § 2.0 for the resulting four-way data partition (D_ACI + D_tr + D_cal + D_test) and § 2.2/§ 2.3 for the API changes.
*Impact*: data partition at the runner level (now four-way, with D_ACI peeled off first); class signatures (`ACI.predict_intervals`, `WeightedCAFHTWholeTrajectory.{calibration_scores, predict_bands}`) take an explicit `score_bank` argument; runner CLI gains `--frac_aci` and `--gamma_split`.
**Recommendation: adopt the proposal as described in § 2.0.** Default: peel 15% of trajectories off the raw pool as D_ACI before splitting the remainder into D_tr / D_cal / D_test by their existing fractions; inner γ-selection split of D_tr defaults to 0.33 / 0.33 / 0.34. Algorithm 2 (last-step) does not invoke ACI, so D_ACI is not needed there.

**Answer:** Adopt the proposal as described in § 2.0.

---

# Additional references

The sections below are background material — they were primary in v1 of this plan, but in v2 they are mostly reference. Readers can skip to § D (out of scope) if they only need the implementation plan.

---

## A. What changed

### A.1 Notation

| Symbol | Old meaning (parent repo, pre-v2) | New meaning (algo box) |
|---|---|---|
| `Z_t^(i)` | not used as a top-level object | `(X_t^(i), Y_t^(i))`, one per (series, time) |
| `X_t^(i)` | dynamic + static covariates (medical), 4 daily features (finance), Poisson (synthetic) | the covariate vector at time t — same domain interpretation but explicitly indexed |
| `Y_t^(i)` | scalar response | same |
| **Shift assumption** | unspecified | shift is on `X_1` (whole-trajectory); shift is on `X_{1:T}` jointly (last-step) — conditional kernels for the remainder are identical across reference and test |

The user's note "this coincides with the previous notation by letting `X_t = Y_{t-1}`" means the last-step regime reduces to AR-style prediction if X is taken as past Y. Our current synthetic/medical pipelines already do this, so the data loaders need no schema change — only the runners do.

### A.2 Old algorithm vs. new algorithms — side-by-side

| Aspect | Old (`core/algorithm.py:AdaptedCAFHT`) | New: last-step coverage (Alg. 2) | New: whole-trajectory coverage (Alg. 1) |
|---|---|---|---|
| Coverage target | per-step marginal | `P(Y_{T+1} ∈ Ĉ_{T+1}) ≥ 1−α` | `P(∀t: Y_t ∈ Ĉ_t) ≥ 1−α` |
| Predictor | per-step refit (AR(1), or `LinearCovariateModel`) | single model `f_{T+1}` trained once: regress `Y_{T+1}` on `X_{1:T}` | per-step models `{f_t}_{t∈[T+1]}`, each regressing `Y_t` on `(Z_{1:t-1}, X_t)` |
| γ selection | per-series ACI step inside main loop, every 5–10 steps; criterion: average coverage closest to 1−α on a 3-way training split | **not used** (no ACI in last-step) | Predictors fit on full D_tr; D_tr is then split internally into D_tr^(1)/D_tr^(2)/D_tr^(3); for each γ, run simple ACI with train=D_tr^(1), cal=D_tr^(2) (sandbox score bank), test=D_tr^(3); pick γ by **minimum mean band width** on D_tr^(3) (CAFHT convention, `CAFHT/ConformalizedTS/methods.py:393`). D_ACI is a *separate* held-out set used only by the main-algorithm ACI (cal-side + test-side), not by γ selection. See Q4 + Q8 in § 6. |
| Cal score | `\|Y_{t+1} - Ŷ_{t+1}\|`, single-time-step (post-v2 calibration fix) | `s(Y_{T+1}^(i), f_{T+1}(X^(i)))`, e.g. abs residual; one scalar per cal series | `ε_i = max_t max{Y_t − U_t, L_t − Y_t}` over the ACI bands run on cal series i with γ_opt (additive branch of CAFHT's `nonconf_scores`) |
| Weighting | `p̂/(1−p̂)` from cross-half LR on prefix-summary features; normalized to sum to 1 | LR on **entire X vector** `X_{1:T}^(i)`; weights kept unnormalized for δ_∞ quantile | LR on **X_1** only (shift assumption); weights kept unnormalized for δ_∞ quantile |
| Deployment quantile | weighted empirical CDF (no test atom) | `η_j = Quantile(Σ Ŵ_i/Σ Ŵ_k · δ_{ε_i} + Ŵ_j/Σ Ŵ_k · δ_∞, 1−α)` per test point | same formula |
| Deployment band | `[Ŷ ± q]` from weighted quantile of cal scores | `Ĉ_{T+1}^j = {y : s(y, f_{T+1}(X^j)) ≤ η_j}` | run ACI on test point j to get Ĉ_t^aci, then **inflate**: `Ĉ_t^j = [min Ĉ_t^aci − η_j, max Ĉ_t^aci + η_j]` for every t |
| Test-side α update | per-series ACI, online | none | none at deployment (η_j is a fixed scalar per test point) |
| Cross-half test split | yes | yes (positives = `D_test^(1)`, deploy on `D_test^(2)`; reverse and repeat) | same |

### A.3 What survives from the current code

- `core/ts_generator.py` — synthetic DGPs are unchanged.
- `finance/finance_data.py`, `medical/medical_data.md`, `medical/sepsis_experiment_data_nacl_target.pkl` — data loaders/pickle unchanged.
- Two algorithmic primitives can be salvaged from `core/algorithm.py`:
  - `AdaptedCAFHT._compute_density_ratio_weights` — keep the logistic-regression + 5×-mean clip pattern but expose two things that the old impl didn't: the **raw `(W_cal, W_test)` pair** (we no longer normalize before quantile-time, because the δ_∞ atom needs all weights on the same scale) and a **caller-supplied featurizer** so X_1 (Alg. 1) and X_{1:T} (Alg. 2) can plug in.
  - `AdaptedCAFHT._weighted_quantile` — needs a new sibling that takes the test-point weight as a separate argument and inserts a δ_∞ atom; the existing function stays for legacy.
- Per-domain *data wrangling* (medical's `_convert_to_arrays`, `_encode_ethnicity`, `ETHNICITY_MAP`; finance's sector split logic) — keep, factor out into thin loader modules.
- `LinearCovariateModel` (both `medical_conformal.py:451` and `finance_conformal.py:228`) — keep as a per-step `f_t` for the whole-trajectory regime in those domains; for the last-step regime we will need a new predictor (see § 2.4).

What does **not** survive:

- The control flow of `AdaptedCAFHT.calibrate` / `predict_with_interval` / `update_weighting_context` and the per-series α_t update loop. The new algorithms have a different shape: cal-side ACI runs for scoring; deployment-side ACI runs for the band; LR weights enter only at the η_j quantile step.
- `_select_gamma` in `medical_conformal.py` and `finance_conformal.py` — both use the "coverage closest to target" criterion. The new whole-trajectory algorithm uses "minimum mean band width" (CAFHT convention).
- Per-domain monkey-patched featurizers (`_richer_featurize_prefixes`, `_make_featurizer`). Featurizers move into the LR-classifier helper as explicit `featurize_fn` arguments.

---

## B. Migration: renames and copy-overs

This restructuring is purely additive at the directory level: no files are moved, no folders are created beyond the new `results/{domain}/{regime}/` subfolders. Old behavior is preserved by renaming legacy code files to an `OLD_` prefix; new behavior lives in new files.

### B.1 `OLD_` renames (one `git mv` each — no content edits)

| Old path | New (renamed) path |
|---|---|
| `core/algorithm.py` | `core/OLD_algorithm.py` |
| `core/adaptive_conformal.py` | `core/OLD_adaptive_conformal.py` |
| `synthetic/test_conformal.py` | `synthetic/OLD_test_conformal.py` |
| `synthetic/multi_seed_experiments.py` | `synthetic/OLD_multi_seed_experiments.py` |
| `finance/finance_conformal.py` | `finance/OLD_finance_conformal.py` |
| `finance/finance_adaptive.py` | `finance/OLD_finance_adaptive.py` |
| `finance/tune_featurizer.py` | `finance/OLD_tune_featurizer.py` |
| `medical/medical_conformal.py` | `medical/OLD_medical_conformal.py` |
| `medical/multi_seed_medical.py` | `medical/OLD_multi_seed_medical.py` |
| `run_all_experiments.sh` | `OLD_run_all_experiments.sh` |
| `build_tex_tables.py` | `OLD_build_tex_tables.py` |

After this step, none of these files are referenced by any new code path. They are kept in tree only for human reference and for git-blame continuity.

**Follow-on (2026-05-29): 8 legacy per-domain run scripts** also marked `OLD_` for the same reason — they are superseded by `run_all_v2.sh` (§ 5.3), exactly as `run_all_experiments.sh` was. Pure `git mv`, no content edits.

| Old path | New (renamed) path |
|---|---|
| `run_finance_experiments.sh` | `OLD_run_finance_experiments.sh` |
| `run_new_finance_experiments.sh` | `OLD_run_new_finance_experiments.sh` |
| `run_static_shift_diagnostics.sh` | `OLD_run_static_shift_diagnostics.sh` |
| `run_synthetic_9cell.sh` | `OLD_run_synthetic_9cell.sh` |
| `run_synthetic_experiments.sh` | `OLD_run_synthetic_experiments.sh` |
| `run_utilities_experiments.sh` | `OLD_run_utilities_experiments.sh` |
| `run_utilities_g03_experiments.sh` | `OLD_run_utilities_g03_experiments.sh` |
| `run_utilities_g10_experiments.sh` | `OLD_run_utilities_g10_experiments.sh` |

### B.2 Files that stay in place (unchanged name and content)

- `core/ts_generator.py`
- `medical/medical_data.py`
- `medical/medical_data.md`
- `medical/sepsis_experiment_data_nacl_target.pkl`
- `medical/plot_medical_covariate_shift.py`
- `finance/finance_data.py`
- `finance/plot_covariate_shift.py`
- `finance/data/` (all `sp500_*.npz` + `.json`)
- All existing `results/{domain}/{json,pdf,tables}/` and their contents

The new runners `import` directly from these files. No copy-over for any "STAY" file.

### B.3 Logic copied (not imported) from `OLD_*` files

For each item below, the indicated block is **copy-pasted** into the new file with a one-line attribution comment (e.g., `# from OLD_medical_conformal.py:451 (LinearCovariateModel)`). After this milestone, no runtime import path touches an `OLD_*` file.

| Source | Block | Destination |
|---|---|---|
| `OLD_medical_conformal.py` | `LinearCovariateModel` class; `_convert_to_arrays`, `_encode_ethnicity` helpers; `ETHNICITY_MAP`, `TARGET_VAR`, `COVARIATE_VARS`, `STATIC_VARS` constants | `medical_runner_whole.py` (only `LinearCovariateModel` + constants); `medical/medical_data.py` may already provide the data-loader helpers — verify before copying |
| `OLD_finance_conformal.py` | `LinearCovariateModel`; `_make_featurizer` (used as reference for the new featurizer); `_select_gamma` (reference for our new "min mean width" rule) | `finance_runner_whole.py` |
| `OLD_algorithm.py` | `AdaptedCAFHT._compute_density_ratio_weights` (logic only — generalized into `density_ratio_weights`); `AdaptedCAFHT._weighted_quantile` (logic only — generalized into `weighted_quantile_with_inf`) | both `core/weighted_cafht_whole.py` and `core/weighted_cafht_last.py` |
| `OLD_run_all_experiments.sh` | function structure, NPZ list, sector grids, mixed-window setup | `run_all_v2.sh` (extended with `--regime` dispatch) |
| `OLD_build_tex_tables.py` | `_emit_table`, `_abs_dev`, the per-domain builders | `build_tex_tables_v2.py` (one table per (domain × regime); 6 tables total) |

---

## C. Sanity checks before running experiments

These run on tiny tinker datasets (T=5, n_train=100, n_cal=50, n_test=10) and gate the experiment sweep.

1. **Unweighted reduction**: with `featurize_fn` returning constants (so LR predicts 50/50 → all weights equal), `weighted_cafht_whole.predict_bands(...)` must match an unweighted CAFHT-style band (identical to `CAFHT/.../methods.py:CAFHT.predict_bands` with `adaptive=False`, single-cal mode) up to numerical noise.
2. **Last-step reduction**: with uniform weights and `score_fn=|·|`, `weighted_cafht_last.predict_bands(...)` must match `[Ŷ ± q_(1-α)(scores)]` from the standard split-conformal procedure.
3. **δ_∞ atom**: when `W_test = 100 × Σ W_cal`, `η_j` must equal `∞` and the band must be `(−∞, ∞)` after `y_trim` is applied.
4. **Cross-half disjointness**: every test index appears in exactly one half across the two iterations of the cross-half loop; concatenated output has shape `(n_test, T+1, 2, ndim)` with the original ordering.
5. **Calibration size invariance**: the v2 calibration fix (one cal score per cal series, not pooled across time) is preserved. The number of `ε_i` values must equal `n_cal`, not `n_cal × T`.
6. **Data partition disjointness**: `D_ACI ∩ D_tr = D_ACI ∩ D_cal = D_ACI ∩ D_test = ∅`, and `D_ACI ∪ D_tr ∪ D_cal ∪ D_test` covers the raw trajectory pool; the internal γ-selection split `D_tr^(1)/D_tr^(2)/D_tr^(3)` is pairwise disjoint within D_tr and unions to D_tr. Score bank shape after `np.abs(aci_true - aci_pred)` is `(|D_ACI|, T+1)`.

---

## D. Things explicitly out of scope for this milestone

- LSTM / RNN predictors. Ridge on flattened X is the v2 baseline for the last-step regime; sequence models are a v3 question.
- The CAFHT "multiplicative" branch of `nonconf_scores` (`adaptive=True`). v2 implements only the additive branch.
- Importing anything from `CAFHT/ConformalizedTS/` at runtime. We mirror the routines we need; we do not import.
- Re-running the synthetic experiment grid that v1 already produced (Group C). The synthetic sweep stays at 30 seeds but now spans two regimes; old C1–C4 results stay in their existing `results/synthetic/` subdirs per § 0 and become reference numbers.
