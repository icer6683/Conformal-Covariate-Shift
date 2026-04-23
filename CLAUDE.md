# Project: Weighted CAFHT — Conformal Prediction under Covariate Shift

## What this project is

A research codebase for a new conformal prediction method called **Weighted CAFHT** for time series under covariate shift. The paper's central claim:

- Likelihood-ratio (LR) reweighting reshapes the calibration score distribution toward the deployment distribution.
- ACI-style adaptive updating corrects residual miscoverage online.
- The combined method maintains long-run empirical coverage while improving efficiency (narrower bands) when covariate shift is well-estimated.

Target coverage guarantee: P(Y^{n+1}_t ∈ Ĉ_t, ∀t ∈ [T]) ≥ 1 − α.

---

## Repository map

**Run scripts are invoked from the project root with the `boa` conda env (Python 3.11).**
**All paths below are relative to the project root.**

### Core library (`core/`)
| File | Role |
|---|---|
| `core/algorithm.py` | `AdaptedCAFHT` class — the full proposed method |
| `core/adaptive_conformal.py` | `OnlineConformalPredictor` and `BasicConformalPredictor` — baselines |
| `core/ts_generator.py` | `TimeSeriesGenerator` — synthetic DGPs + shift |

### Synthetic experiments (`synthetic/`)
| File | Role |
|---|---|
| `synthetic/test_conformal.py` | Single-seed synthetic experiment runner; exports `run_time_based_coverage_experiment` |
| `synthetic/multi_seed_experiments.py` | Multi-seed wrapper around `test_conformal.py`; aggregates coverage + width |

### Finance experiments (`finance/`)
| File | Role |
|---|---|
| `finance/finance_data.py` | yfinance S&P 500 loader; save/load `.npz`+`.json`; sector/industry filters |
| `finance/finance_conformal.py` | S&P 500 experiment using `AdaptedCAFHT` with linear covariate model |
| `finance/finance_adaptive.py` | S&P 500 experiment using `OnlineConformalPredictor` (AR(1) only, no covariates) |
| `finance/tune_featurizer.py` | Grid-search over featurizer variants for the S&P 500 LR classifier |
| `finance/plot_covariate_shift.py` | Standalone covariate distribution visualization |

### Medical experiments (`medical/`)
| File | Role |
|---|---|
| `medical/medical_conformal.py` | Sepsis ICU experiment using `AdaptedCAFHT` with linear covariate model + static covariates |
| `medical/medical_data.md` | Data dictionary for the sepsis pickle (cohort, imputation, split) |
| `medical/sepsis_experiment_data_nacl_target.pkl` | Pre-extracted MIMIC-III sepsis pickle (9264 TrainCal + 5827 Test patients; split by Norepinephrine exposure in first 12 hours) |

### Data and results
| Path | Role |
|---|---|
| `finance/data/` | All S&P 500 data files (`sp500_*.npz` + `sp500_*.json`). Pass as `finance/data/sp500_DATES.npz` |
| `results/finance/json/` | Finance experiment JSON outputs |
| `results/finance/pdf/` | Finance experiment PDF/PNG figures |
| `results/synthetic/json/` | Synthetic experiment JSON outputs |
| `results/synthetic/pdf/` | Synthetic experiment PNG figures |
| `results/medical/json/` | Medical experiment JSON outputs (none yet) |
| `results/medical/pdf/` | Medical experiment figures (none yet) |

---

## Data-generating processes

### Static-X (implemented in `ts_generator.py:generate_with_poisson_covariate`)
- X ~ Pois(λ), fixed per series
- Y_0 ~ N(μ_0, σ_0²)
- Y_t = α·Y_{t-1} + β·X + δ·t + ε_t, ε_t ~ N(0, σ²)
- Shift: redraw X ~ Pois(λ̃) for test, regenerate Y from same Y_0
- Default: λ=1.0, λ̃=2.0, α=0.7, β=1.0, σ=0.2

### Dynamic-X (implemented in `ts_generator.py:generate_with_dynamic_covariate`)
- X_0 ~ Pois(λ_0)
- X_{t+1} = ρ_X·X_t + κ_X·t + η_t, η_t ~ N(0, σ_X²)
- Y_{t+1} = α·Y_t + β·X_t + δ·t + ε_t
- Shift: regenerate entire X_t path under (ρ̃_X, κ̃_X, σ̃_X, λ̃_0); regenerate Y from same Y_0

### S&P 500 finance data (`finance_data.py`)
- Y = intraday return = Close_t / Open_t − 1
- X (4 covariates): OvernightGapReturn (not lagged), Above52wLowReturn (not lagged; 52w-low from pre-sample data to avoid lookahead), TurnoverRatio_lag1, DailyRangeReturn_lag1
- Period: 2024-02-01 to 2024-03-28
- Collected via yfinance (`yf.download`, `auto_adjust=True`)
- 503 S&P 500 tickers hardcoded in `finance_data.py:_get_sp500_constituents`
- Sector split: Technology = test, remaining = train+cal (50/50 split)
- Mixed mode (`--mixed`): random 15% draw of all tickers as test (null no-shift baseline)

---

## Methods implemented

### AdaptedCAFHT (`algorithm.py`)
Single class implementing the full proposed method. Three stages at each time step t:

1. **Model fit**: AR(1) by OLS on all training series up to time t+1.
2. **Score construction**: nonconformity score = |Y_s − Ŷ_s| accumulated for all s ≤ t+1.
3. **Weighted quantile**: prediction interval = [Ŷ_{t+1} − q, Ŷ_{t+1} + q] where q is the weighted (1−α_t)-quantile.

In finance experiments, Step 1 is replaced by a linear regression on the 4 S&P 500 covariates (OLS, pooled across all training tickers and all time steps).

### Likelihood-ratio estimation (`algorithm.py:_compute_density_ratio_weights`)
- Label train series as class 0, (half of) test series as class 1
- Fit LogisticRegression with `class_weight="balanced"` (removes class-prior ratio)
- ŵ(z) ∝ p̂(z) / (1 − p̂(z))
- Clip at 5× mean weight to prevent degeneracy; normalize to sum to 1
- Fallback numpy gradient-descent logistic regression if sklearn unavailable

### Cross-half split for LR (no data leakage)
Test set split into halves H1, H2 at each time step:
- Predict H1 using LR trained on D_train ∪ H2 as positives
- Predict H2 using LR trained on D_train ∪ H1 as positives
- Implemented in both `test_conformal.py` and `finance_conformal.py`

### Finance featurizer (monkey-patched at runtime)
When `--with_shift` is set, `_featurize_YX_summaries` replaces the default last-Y featurizer:
- Y features: mean, std, AR(1) coefficient over last 30 steps
- X features: temporal mean of each covariate over full prefix
- Injected via `types.MethodType` without subclassing

### ACI adaptive update
α^{(i)}_{t+1} = α^{(i)}_t + γ·(α − 1[Y^{(i)}_{t+1} ∉ Ĉ^{(i)}_{t+1}])
clipped to (1e-6, 1−1e-6). Applied per-series.

### Gamma selection
Every 10 time steps, 3-way split of training data:
- D_tr^(1): model fitting; D_tr^(2): calibration; D_tr^(3): evaluation
- Γ = {0.001, 0.005, 0.01, 0.05, 0.1} in synthetic; {0.001, 0.005, 0.01, 0.05} in finance
- Simulate ACI on D_tr^(3) for each γ; pick γ with average coverage (second half of horizon) closest to 1−α

---

## Baselines

| Name | Class | Key properties |
|---|---|---|
| Split conformal (no ACI) | `BasicConformalPredictor` | AR(1), fixed empirical quantile, no adaptation |
| Adaptive split conformal | `OnlineConformalPredictor` | AR(1), ACI α_t update, optional online score update (append test residuals) |
| Weighted CAFHT without ACI | Not separately implemented | Requires γ=0 with LR weights; no saved results |

`finance_adaptive.py` runs `OnlineConformalPredictor` on S&P 500 using AR(1) only (no covariate model), as a weaker finance-specific baseline.

---

## Metrics

- **Empirical coverage at time t**: fraction of test series covered at step t+1
- **Overall empirical coverage**: pooled over all series and all time steps
- **Mean prediction interval width at time t**: mean of 2q^{(i)}_{t+1} over test series
- **Coverage degradation**: mean coverage (first 1/3 of horizon) − mean coverage (last 1/3)
- Multi-seed: mean ± std across seeds; inter-seed IQR for coverage-over-time profiles

---

## Experiment defaults

### Synthetic (algorithm predictor)
- n_train=200, n_cal=200, n_test=500, T=20, α=0.1
- Multi-seed: n_train=1000, n_cal=1000, n_test=500, T=40, N=100 seeds from seed 1000

### Finance
- α=0.1, cal_frac=0.5, seed=42, Y_window=30
- gamma_grid=[0.001, 0.005, 0.01, 0.05]

---

## Results currently available (updated 2026-04-23)

All S&P 500 data files are in `finance/data/`. Pass as `finance/data/sp500_DATES.npz` to all scripts.

### Synthetic multi-seed (static-X, n_seeds=30, T=20) — Group C1–C4 complete
| File | Content |
|---|---|
| `results/synthetic/json/results_algorithm_shift_20260414_103750.json` + PNG | C1: AdaptedCAFHT, with shift |
| `results/synthetic/json/results_algorithm_noshift_20260414_103748.json` + PNG | C2: AdaptedCAFHT, no shift |
| `results/synthetic/json/results_adaptive_shift_20260414_103752.json` + PNG | C3: OnlineConformalPredictor, with shift |
| `results/synthetic/json/results_adaptive_noshift_20260414_103751.json` + PNG | C4: OnlineConformalPredictor, no shift |

Additional older single-seed runs exist in `results/synthetic/json/` (dated 20260206, 20260210); superseded by the C1–C4 multi-seed files above. C5/C6 (dynamic-X) have not been run yet.

### Finance (all 13 windows complete, seed=42)
| Pattern | Content |
|---|---|
| `results/finance/json/finance_tech_shift_DATES.json` + PDF | AdaptedCAFHT with LR weighting; Technology test sector; 13 windows |
| `results/finance/json/finance_tech_noshift_DATES.json` + PDF | AdaptedCAFHT, no shift correction; Technology test sector; 13 windows |
| `results/finance/json/finance_mixed_withweighting.json` + PDF | Mixed-sector test (null/no-shift baseline), with LR weighting; one window |
| `results/finance/json/finance_mixed_noweighting.json` + PDF | Mixed-sector test (null/no-shift baseline), no weighting; one window |
| `results/finance/json/finance_healthcare_shift.json` | Healthcare test, with LR weighting (pending E2 KL verification) |
| `results/finance/pdf/covariate_shift.png` | Tech vs. non-Tech covariate KDE (motivating figure; PDF version pending E1) |

Note: several files with a ` 2` suffix exist (e.g., `finance_tech_shift_20240201_20240328 2.json`) — artifact reruns that can be deleted.

---

## Experiment plan (agreed 2026-04-22)

### Finance experiments — run_finance_experiments.sh
Run Tech shift + no-shift on 13 .npz windows (all except sp500_20231004_20240328.npz).
Output naming: results/finance/{json,pdf}/finance_tech_{shift,noshift}_DATES.{json,pdf}

NPZ files to use (all in finance/data/):
  finance/data/sp500_20240102_20240229.npz
  finance/data/sp500_20240201_20240328.npz
  finance/data/sp500_20240301_20240430.npz
  finance/data/sp500_20240401_20240531.npz
  finance/data/sp500_20240501_20240628.npz
  finance/data/sp500_20240603_20240731.npz
  finance/data/sp500_20240701_20240830.npz
  finance/data/sp500_20240801_20240930.npz
  finance/data/sp500_20240903_20241031.npz
  finance/data/sp500_20241001_20241129.npz
  finance/data/sp500_20241101_20241231.npz
  finance/data/sp500_20241202_20250131.npz
  finance/data/sp500_20250102_20250228.npz
Excluded: finance/data/sp500_20231004_20240328.npz (long overlapping window).

Mixed-sector null baseline: finance/data/sp500_20240201_20240328.npz only (illustrative).
  Outputs: results/finance/{json,pdf}/finance_mixed_{withweighting,noweighting}.{json,pdf}

### Synthetic experiments — run_synthetic_experiments.sh (Group C)
  C1: algorithm, static, with_shift     -> results/synthetic/{json,pdf}/
  C2: algorithm, static, no shift       -> results/synthetic/{json,pdf}/
  C3: adaptive, static, with_shift      -> results/synthetic/{json,pdf}/
  C4: adaptive, static, no shift        -> results/synthetic/{json,pdf}/
  C5: algorithm, dynamic, with_shift    (x_rate=0.6, x_rate_shift=0.9)
  C6: algorithm, dynamic, no shift      (x_rate=0.6)
All with --n_seeds 100, save JSON aggregation files.

---

## TO-DO: deferred experiments (do not run yet)

### Group D — Ablation
D1: LR weighting without ACI (gamma=0).
    Command: python synthetic/multi_seed_experiments.py --predictor algorithm
             --covariate_mode static --with_shift --aci_stepsize 0.0
             --n_seeds 100 --save_dir results/synthetic
    Prerequisite: verify that aci_stepsize=0.0 is handled cleanly (alpha stays
    fixed = alpha throughout; no division by zero).
    Purpose: isolates contribution of LR weighting from ACI update.

### Group E — Covariate shift verification figures
E1: Technology vs. rest KDE + KL divergence
    Command: python finance/plot_covariate_shift.py --npz finance/data/sp500_20240201_20240328.npz
             --test_sector Technology --save results/finance/pdf/covariate_shift_tech.pdf
E2: Healthcare vs. rest KDE + KL divergence
    Command: python finance/plot_covariate_shift.py --npz finance/data/sp500_20240201_20240328.npz
             --test_sector Healthcare --save results/finance/pdf/covariate_shift_healthcare.pdf
    Note: run E2 and inspect KL values before including healthcare as a primary
    shift experiment. If all four KL values are near zero, demote to robustness
    condition only.

---

## Gaps before experimental section is paper-ready

### Must-have
1. **Dynamic-X simulation results** — run C5/C6 above (not yet run).
2. ~~**Finance results across all 13 windows**~~ — **DONE** (all 13 windows complete for tech shift/noshift; see `results/finance/`).
3. ~~**Mixed-sector null baseline**~~ — **DONE** (`finance_mixed_withweighting.json` and `finance_mixed_noweighting.json` complete).
4. ~~**JSON aggregation files for multi-seed synthetics**~~ — **DONE** (C1–C4 multi-seed JSON files present in `results/synthetic/json/`).

### Deferred (Group D/E — see TO-DO above)
5. **Ablation: LR weighting without ACI** — Group D1.
6. **Healthcare covariate shift verification** — Group E2.
7. **Vector figure output** — all figures should be saved as PDF not PNG.

### Structural
8. **Medical experiment (Section 5.3)** — data AND experiment script both exist and are fully implemented (see Medical Experiment Audit section below); no results have been run yet and no baseline (OnlineConformalPredictor) is wired up for this domain.
9. **Theoretical proofs** — guarantee "has yet to be finalized" per technical note.

---

## Paper-level framing (statistical language)

The calibration sample follows P = P_Z × P_{Y|Z}. Test units follow P̃ = P̃_Z × P_{Y|Z}. The method estimates dP̃_Z/dP_Z via logistic regression, reweights calibration scores, and applies per-series ACI to correct residual miscoverage. The cross-half test-split prevents data leakage into the classifier. Gamma is selected on a held-out training fold every 10 steps.

---

## Medical Experiment Audit (Section 5.3)

Audit performed 2026-04-23. All claims below are source-verified.

---

### 1. Repository map — medical components

| File | Role | Status |
|---|---|---|
| `medical/medical_conformal.py` | Full experiment script: data loading, model, gamma selection, LR weighting, ACI loop, plotting, CLI | Implemented |
| `medical/medical_data.md` | Data dictionary: cohort definition, imputation rules, patient dict structure, example usage | Implemented |
| `medical/sepsis_experiment_data_nacl_target.pkl` | Pre-extracted MIMIC-III sepsis data (9264 TrainCal + 5827 Test patients, 24 hourly steps each; split by Norepinephrine use in first 12 hours) | Present |
| `results/medical/` | No medical output files exist yet | Not run |

No notebooks, no multi-seed wrapper, no baseline script, no covariate-shift diagnostic plot for the medical domain.

---

### 2. Implemented methods

#### Proposed method: AdaptedCAFHT with linear covariate model
- **Code name**: `AdaptedCAFHT` (imported from `core/algorithm.py`); prediction model `LinearCovariateModel` (defined in `medical/medical_conformal.py:443`)
- **Paper name**: Weighted CAFHT
- **File/function**: `medical/medical_conformal.py:run_medical_experiment`
- **Category**: Proposed combined method (LR weighting + ACI). Replaces AR(1) with cross-sectional OLS on dynamic + static covariates.
- **Hyperparameters**:
  - α = 0.1 (default), cal_frac = 0.5, seed = 42
  - gamma_grid = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] (finer grid than finance/synthetic)
  - Gamma reselected every **5** steps (vs every 10 in finance/synthetic)
  - T = 23 prediction steps (hours 1–23, predicting from prefix of length t+2)

#### Unweighted baseline (no separate script)
- **Code name**: same `run_medical_experiment` with `--with_shift` omitted (`with_shift=False`)
- **Paper name**: Split Conformal (no LR weighting, with ACI)
- **File/function**: `medical/medical_conformal.py:run_medical_experiment` (else-branch at line 851)
- **Category**: ACI conformal without likelihood-ratio reweighting. Uniform weights used.
- **Note**: No separate `OnlineConformalPredictor` (AR(1)) baseline is implemented for the medical domain.

#### Prediction model: LinearCovariateModel
- **Code name**: `LinearCovariateModel` (`medical/medical_conformal.py:443`)
- **Paper name**: Cross-sectional OLS predictor
- **Equation**: `NaCl_t ~ β₀ + β₁·HR_t + β₂·RR_t + β₃·O2Sat_t + β₄·Age + β₅·gender_M + β₆·eth_BLACK + β₇·eth_HISPANIC + β₈·eth_ASIAN + β₉·eth_OTHER`
- **Fitted**: pooled across all (patient, timestep) pairs in the training prefix at each hour t via `np.linalg.lstsq`.

#### Featurizer for LR classifier: `_richer_featurize_prefixes`
- **Code name**: `_richer_featurize_prefixes` (`medical/medical_conformal.py:303`); monkey-patched onto predictor via `types.MethodType`
- **Features (26 total)**:
  - 5 stats × 4 dynamic variables (NaCl + HR + RR + O2Sat): mean, std, min, max, last = 20 features
  - 6 static: Age, gender_M, eth_BLACK, eth_HISPANIC, eth_ASIAN, eth_OTHER
- **Standardization**: computed from training-set features at each time step; stored as `predictor._feat_mu`, `predictor._feat_std`
- **Auxiliary storage**: `predictor._X_ctx` (dynamic covariate prefix), `predictor._S_ctx` (static covariates) — set by caller before each featurize call

#### Gamma selection: `_select_gamma`
- **Code name**: `_select_gamma` (`medical/medical_conformal.py:505`)
- **Mechanism**: 3-way split of training data (fit / cal / eval, each ≈n/3); simulates ACI without LR weighting; picks γ with second-half coverage closest to 1−α
- **Called**: every 5 steps in the main loop

---

### 3. Data-generating process (real data)

**Source**: MIMIC-III sepsis cohort, pre-extracted and stored in `sepsis_experiment_data_nacl_target.pkl` (n_traincal=9264, n_test=5827, verified by loading the pickle on 2026-04-23 after the split-rule update).

**Cohort criteria** (from `medical_data.md`):
- Sepsis diagnosis (ICD codes, upstream pipeline)
- All three CHART covariates non-zero across 24 hours
- NaCl 0.9% target non-zero across 24 hours

**Variables**:
| Role | Variable | Code name | Notes |
|---|---|---|---|
| Target Y | NaCl 0.9% dosage (mL/hr) | `TARGET_VAR = "NaCl 0.9% (target)"` | 24-step trajectory; zeros left as-is |
| Dynamic covariate X₁ | Heart Rate | `"Heart Rate"` | Zeros imputed with patient-level median of nonzero values |
| Dynamic covariate X₂ | Respiratory Rate | `"Respiratory Rate"` | Same imputation rule |
| Dynamic covariate X₃ | O2 saturation pulseoxymetry | `"O2 saturation pulseoxymetry"` | Same imputation rule |
| Static covariate S | Age, gender, ethnicity | `STATIC_VARS` | Age numeric; gender binary; ethnicity 5-group one-hot (WHITE reference) |
| Shift variable (split only) | Norepinephrine | `"Norepinephrine"` | Not used as covariate; defines TrainCal vs. Test by exposure during the first 12 hours (t0..t11) |

**TrainCal / Test split** (`medical_data.md`):
- TrainCal: patients with **no Norepinephrine in the first 12 hours** (t0..t11) → n=9264. May still have late Norepinephrine (t12..t23).
- Test: patients with **any nonzero Norepinephrine in the first 12 hours** (t0..t11) → n=5827.
- Covariate shift: early-ICU vasopressor initiation correlates with different fluid management and physiologic states.
- The rule was tightened from any-time exposure to first-12-hour exposure, which moved 664 patients with late-only Norepinephrine from Test to TrainCal.

**Time horizon**: T = 23 (predict hours 1–23 from prefix; each patient has exactly 24 hourly observations).

**Imputation** (`medical_data.md`):
- CHART covariates: all-zero → patient excluded upstream; remaining zeros → patient's median of nonzero values
- NaCl 0.9% target: all-zero → patient excluded upstream; remaining zeros → left unchanged

---

### 4. Implemented experiments

#### `medical/medical_conformal.py` — single experiment runner
- **Script**: `medical/medical_conformal.py`; entry point `main()` / `run_medical_experiment()`
- **Methods compared**: AdaptedCAFHT with shift (`--with_shift`) vs. without shift (uniform weights)
- **No baseline**: `OnlineConformalPredictor` is not wired up for this domain
- **Sample sizes**: n_traincal=9264, n_test=5827 (full); `--n_traincal` / `--n_test` flags allow subsampling
- **Cal/train split**: cal_frac=0.5 → ~4300 train / ~4300 cal
- **Time horizon**: T=23 steps
- **Alpha**: 0.1 (default)
- **Seeds**: single seed (default 42); no multi-seed wrapper exists
- **Replications**: 1 (no multi-seed)
- **Metrics computed**:
  - `coverage_by_time`: empirical coverage fraction at each of 23 steps
  - `width_by_time`: mean prediction interval width at each step
  - `overall_coverage`: pooled fraction across all patients and all steps
  - `gamma_opt_history`: selected gamma at each step
  - `first_test_series`: true Y, lower, upper for the first test patient
- **Output files**: `--save_plot` (PNG), `--save_json` (JSON); **no results files have been run yet**

---

### 5. Implemented figures and tables

#### `plot_results` in `medical/medical_conformal.py:930`
- **What it shows**: 4-panel figure — (1) coverage over time + target line, (2) mean interval width over time, (3) first test patient's actual NaCl vs. prediction interval, (4) selected ACI gamma over time (log scale)
- **Created by**: `medical/medical_conformal.py:plot_results`; called from `main()` after `run_medical_experiment`
- **Status**: **Not yet run** — no output files exist in `results/`
- **Paper-readiness**: Preliminary. Output is PNG only (no PDF path). No comparison between shift/no-shift conditions on the same axes.

---

### 6. Missing or partial components

| Component | Status | Notes |
|---|---|---|
| Data pickle (`medical/sepsis_experiment_data_nacl_target.pkl`) | **Completed** | Present; 9264 TrainCal + 5827 Test patients (split by Norep in first 12 h); verified loadable |
| Data dictionary (`medical/medical_data.md`) | **Completed** | Cohort, imputation, split, dict structure, usage examples |
| Experiment script (`medical/medical_conformal.py`) | **Completed** | Full CLI, LR weighting, ACI, model, featurizer, gamma selection, plot |
| Run results (any output JSON/PDF) | **Not implemented** | No files in `results/`; experiment has not been executed |
| Baseline comparison (OnlineConformalPredictor on medical data) | **Not implemented** | finance_adaptive.py exists for finance; no equivalent for medical |
| Multi-seed wrapper for medical | **Not implemented** | `multi_seed_experiments.py` wraps synthetic only |
| Covariate shift diagnostic plot (KDE / KL) for Norepinephrine split | **Not implemented** | `plot_covariate_shift.py` is finance-only |
| Ablation: LR weighting without ACI (γ=0) | **Not implemented** | Deferred (Group D); same gap as finance |
| PDF figure output | **Not implemented** | `save_path` in `plot_results` accepts any extension but no PDF run exists |
| MIMIC-III extraction pipeline (upstream of pkl) | **Not in repo** | Mentioned in `medical_conformal.py` docstring; code not present |
