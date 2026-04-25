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
| `core/adaptive_conformal.py` | `OnlineConformalPredictor` — sliding-window split-conformal baseline (no ACI, no LR weighting). Legacy; ablations of Weighted CAFHT are now done via `AdaptedCAFHT` with weights=1 (LR-only off) or γ=0 (ACI off) rather than through this class. |
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
| `finance/finance_adaptive.py` | **Legacy / unused.** S&P 500 experiment using `OnlineConformalPredictor` (AR(1) only, no covariates). Never run and saved. Retained for reference only; the paper baseline is now `AdaptedCAFHT` with γ=0 or uniform weights on the same covariate model. |
| `finance/tune_featurizer.py` | Grid-search over featurizer variants for the S&P 500 LR classifier |
| `finance/plot_covariate_shift.py` | Standalone covariate distribution visualization |

### Medical experiments (`medical/`)
| File | Role |
|---|---|
| `medical/medical_conformal.py` | Sepsis ICU experiment using `AdaptedCAFHT` with linear covariate model + static covariates; `verbose` parameter suppresses per-step diagnostics for multi-seed use |
| `medical/multi_seed_medical.py` | Multi-seed wrapper: subsamples n_traincal/n_test patients per seed from full pool; aggregates coverage + width; output schema matches synthetic multi-seed JSONs |
| `medical/plot_medical_covariate_shift.py` | KDE + KL-divergence plot comparing TrainCal vs Test on HR, RR, O2Sat, NaCl, Age, gender, ethnicity |
| `medical/medical_data.md` | Data dictionary for the sepsis pickle (cohort, imputation, split) |
| `medical/sepsis_experiment_data_nacl_target.pkl` | Pre-extracted MIMIC-III sepsis pickle (9264 TrainCal + 5827 Test patients; split by Norepinephrine exposure in first 12 hours) |

### Sector diagnostics (`finance/`)
| File | Role |
|---|---|
| `finance/plot_sector_separability.py` | KS heatmap + violin plots across 7 sectors × 8 features; caches data to `_sector_diag_cache_*.npz`; produces JSON of KS statistics |

### Run scripts (project root)
| File | Role |
|---|---|
| `run_finance_experiments.sh` | Tech shift + noshift, 13 windows |
| `run_new_finance_experiments.sh` | Tech LR-only (13 windows) + Healthcare shift/noshift (2 windows) |
| `run_utilities_experiments.sh` | Utilities shift + noshift + LR-only, 13 windows, default γ-grid |
| `run_utilities_g10_experiments.sh` | Utilities shift + noshift, 13 windows, expanded γ-grid {0.001,0.005,0.01,0.05,0.1} |

### Audit outputs (`results/audit/`)
| File | Role |
|---|---|
| `results/audit/build_master_table.py` | Regenerates all audit tables from saved JSONs |
| `results/audit/master_results_table.{csv,md}` | All saved experiment rows (154 rows) |
| `results/audit/finance_tech_aggregate.{csv,md}` | Per-window and aggregate tables for Tech + Utilities; g10 grid is primary for Utilities |
| `results/audit/schema_inconsistencies.md` | Schema differences across JSON families |

### Data and results
| Path | Role |
|---|---|
| `finance/data/` | All S&P 500 data files (`sp500_*.npz` + `sp500_*.json`). Pass as `finance/data/sp500_DATES.npz` |
| `results/finance/json/` | Finance experiment JSON outputs |
| `results/finance/pdf/` | Finance experiment PDF/PNG figures |
| `results/synthetic/json/` | Synthetic experiment JSON outputs |
| `results/synthetic/pdf/` | Synthetic experiment PNG figures |
| `results/medical/json/` | Medical experiment JSON outputs |
| `results/medical/pdf/` | Medical experiment figures |

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
Every 10 time steps (every 5 for medical), 3-way split of training data:
- D_tr^(1): model fitting; D_tr^(2): calibration; D_tr^(3): evaluation
- Γ = {0.001, 0.005, 0.01, 0.05, 0.1} in synthetic; {0.001, 0.005, 0.01, 0.05} in finance/tech (default); {0.001, 0.005, 0.01, 0.05, 0.1} in utilities (g10 grid); [1e-6 … 0.01] (9 values) in medical
- Simulate ACI on D_tr^(3) for each γ; pick γ with average coverage (second half of horizon) closest to 1−α

---

## Baselines

| Name | Class | Key properties | Run status |
|---|---|---|---|
| Sliding-window split conformal | `OnlineConformalPredictor` | AR(1), fixed α, empirical quantile recomputed each step over sliding window. **No ACI α_t update.** | Run (synthetic C3/C4 multi-seed) |
| Uniform + ACI (no shift) | `AdaptedCAFHT`, `with_shift=False` | Uniform calibration weights + per-series ACI. Standard conformal baseline on same model. | Run (all domains) |
| LR only, γ=0 | `AdaptedCAFHT`, `with_shift=True`, `gamma_grid=[0.0]` | LR-weighted quantile, α held fixed (no ACI). Isolates LR contribution. | Run (finance tech/util 13 windows; medical 10 seeds) |

Paper ablations are performed through `AdaptedCAFHT` parameter tweaks, not a separate class. `OnlineConformalPredictor` is retained as a structurally different synthetic comparator.

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
- Multi-seed (C1–C4 saved): n_train=600, n_cal=600, n_series=300, T=20, n_seeds=30, base_seed=1000

### Finance
- α=0.1, cal_frac=0.5, seed=42, Y_window=30
- gamma_grid=[0.001, 0.005, 0.01, 0.05] for Technology; [0.001, 0.005, 0.01, 0.05, 0.1] for Utilities (g10)

### Medical
- α=0.1, cal_frac=0.5, default seed=42; multi-seed uses base_seed=1000
- n_traincal=1000, n_test=500 (subsampled from full pool each seed)
- gamma_grid=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] (finer grid; gamma reselected every 5 steps)

---

## Results currently available (updated 2026-04-24)

All S&P 500 data files are in `finance/data/`. Pass as `finance/data/sp500_DATES.npz` to all scripts.

### Synthetic multi-seed (static-X, n_seeds=30, T=20) — Group C1–C4 complete
| File | Condition | Coverage | Width |
|---|---|---|---|
| `results/synthetic/json/results_algorithm_shift_20260414_103750.json` | C1: AdaptedCAFHT (LR+ACI), shift | 0.8870 | 1.278 |
| `results/synthetic/json/results_algorithm_noshift_20260414_103748.json` | C2: AdaptedCAFHT (uniform+ACI), no shift | 0.8974 | 1.287 |
| `results/synthetic/json/results_adaptive_shift_20260414_103752.json` | C3: Sliding-window, shift | 0.8969 | 1.290 |
| `results/synthetic/json/results_adaptive_noshift_20260414_103751.json` | C4: Sliding-window, no shift | 0.8999 | 1.290 |

Config: n_train=600, n_cal=600, n_series=300, T=20, n_seeds=30. C5/C6 (dynamic-X) not yet run.

### Finance (all 13 windows, seed=42)
| Pattern | Condition | Coverage (mean±std/13w) | Width (mean) |
|---|---|---|---|
| `finance_tech_shift_DATES.json` ×13 | Tech: LR+ACI | 0.8876 ± 0.013 | 0.0505 |
| `finance_tech_noshift_DATES.json` ×13 | Tech: uniform+ACI | 0.8803 ± 0.017 | 0.0493 |
| `finance_tech_LRonly_DATES.json` ×13 | Tech: LR only, γ=0 | 0.8721 ± 0.018 | 0.0465 |
| `finance_util_shift_g10_DATES.json` ×13 | Util: LR+ACI, g10 grid | 0.9233 ± 0.021 | 0.0451 |
| `finance_util_noshift_g10_DATES.json` ×13 | Util: uniform+ACI, g10 grid | 0.9243 ± 0.020 | 0.0454 |
| `finance_util_LRonly_DATES.json` ×13 | Util: LR only, γ=0 | 0.9316 ± 0.024 | 0.0447 |
| `finance_mixed_withweighting.json` | Mixed null: LR+ACI, 1 window | 0.9121 | 0.0412 |
| `finance_mixed_noweighting.json` | Mixed null: uniform+ACI, 1 window | 0.9106 | 0.0413 |
| `finance_healthcare_shift_DATES.json` ×2 | Healthcare: LR+ACI, 2 windows (preliminary) | ~0.907 | ~0.063 |
| `finance_healthcare_noshift_DATES.json` ×2 | Healthcare: uniform+ACI, 2 windows (preliminary) | ~0.907 | ~0.063 |

Sector separability (KS vs rest): Utilities max KS=0.75 (strong, driven by beta_spy); Technology max KS≈0.49; Healthcare max KS=0.42 (weak — LR classifier near-uniform weights for healthcare).

### Medical (MIMIC-III sepsis, n_traincal=1000, n_test=500, 10 seeds)
| File | Condition | Coverage (mean±std/seeds) | Width mean (mL/hr) |
|---|---|---|---|
| `results/medical/json/medical_ms_noshift.json` | uniform+ACI | 0.9766 ± 0.006 | 684 ± 57 |
| `results/medical/json/medical_ms_shift.json` | LR+ACI | 0.9413 ± 0.007 | 420 ± 48 |
| `results/medical/json/medical_ms_LRonly.json` | LR only, γ=0 | 0.9383 ± 0.006 | 420 ± 48 |
| `results/medical/pdf/medical_covariate_shift.pdf` | KDE shift plot | NaCl KL=0.206, RR KL=0.022, O2Sat KL=0.050 | — |

Note: LR weighting shifts coverage from +7.7pp (noshift) to +4.1pp above target; ACI contributes negligibly on top of LR (Δ=0.003).

---

## Experiment plan — status (updated 2026-04-24)

### Finance experiments — COMPLETE
- ✅ Tech shift + noshift, 13 windows (`run_finance_experiments.sh`)
- ✅ Tech LR-only (γ=0), 13 windows (`run_new_finance_experiments.sh`)
- ✅ Utilities shift + noshift + LR-only, 13 windows, default grid (`run_utilities_experiments.sh`)
- ✅ Utilities shift + noshift, 13 windows, g10 grid (`run_utilities_g10_experiments.sh`) — **primary**
- ✅ Healthcare shift + noshift, 2 windows (`run_new_finance_experiments.sh`) — preliminary only
- ✅ Mixed-sector null baseline, 1 window

### Synthetic experiments — Group C1–C4 COMPLETE; C5/C6 pending
- ✅ C1: AdaptedCAFHT, static-X, with shift (30 seeds)
- ✅ C2: AdaptedCAFHT, static-X, no shift (30 seeds)
- ✅ C3: Sliding-window, static-X, with shift (30 seeds)
- ✅ C4: Sliding-window, static-X, no shift (30 seeds)
- ☐ C5: AdaptedCAFHT, dynamic-X, with shift (x_rate=0.6, x_rate_shift=0.9)
- ☐ C6: AdaptedCAFHT, dynamic-X, no shift (x_rate=0.6)

### Medical experiments — 10-seed runs COMPLETE
- ✅ noshift (uniform+ACI), 10 seeds
- ✅ shift (LR+ACI), 10 seeds
- ✅ LR-only (γ=0), 10 seeds
- ✅ Covariate shift KDE plot (TrainCal vs Test on HR/RR/O2Sat/NaCl/Age)

---

## TO-DO: remaining gaps before paper is ready

### Must-have
1. **Dynamic-X simulation** — run C5/C6 (not yet run).
2. **Synthetic LR-only ablation (D1)** — run `multi_seed_experiments.py --aci_stepsize 0.0 --with_shift` to isolate LR contribution in synthetic setting.
   Command: `python synthetic/multi_seed_experiments.py --predictor algorithm --covariate_mode static --with_shift --aci_stepsize 0.0 --n_seeds 30 --save_dir results/synthetic`
3. **Covariate shift PDF figures for paper** — run `plot_covariate_shift.py` with `--save` for Technology and Utilities sectors.

### Deferred / lower priority
4. Healthcare sector — KL too low for it to be a primary finance result; include only as robustness appendix.
5. Vector figure output — save all figures as PDF not PNG for camera-ready.
6. **Theoretical proofs** — guarantee "has yet to be finalized" per technical note.

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
| `results/medical/` | JSON + PDF results for 3 conditions × multi-seed (10 seeds) | Run |
| `medical/multi_seed_medical.py` | Multi-seed wrapper: subsamples patients per seed, aggregates coverage/width, plots IQR bands | Implemented |
| `medical/plot_medical_covariate_shift.py` | KDE + bar covariate shift diagnostic (Heart Rate, RR, O2Sat, NaCl, Age, gender, ethnicity); prints KL divergences | Implemented |

No notebooks, no OnlineConformalPredictor baseline, no MIMIC-III extraction pipeline in repo.

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
- **Seeds**: single seed for exploratory runs (default 42); multi-seed wrapper `multi_seed_medical.py` runs 10 seeds (base_seed=1000) with independent patient subsampling
- **Replications**: 10 seeds × 3 conditions (noshift, shift/LR+ACI, LRonly/γ=0) — all complete
- **Metrics computed**:
  - `coverage_by_time`: empirical coverage fraction at each of 23 steps
  - `width_by_time`: mean prediction interval width at each step
  - `overall_coverage`: pooled fraction across all patients and all steps
  - `gamma_opt_history`: selected gamma at each step
  - `first_test_series`: true Y, lower, upper for the first test patient
- **Output files**: `--save_plot` (PDF), `--save_json` (JSON); results in `results/medical/`

---

### 5. Implemented figures and tables

#### `plot_results` in `medical/medical_conformal.py:930`
- **What it shows**: 4-panel figure — (1) coverage over time + target line, (2) mean interval width over time, (3) first test patient's actual NaCl vs. prediction interval, (4) selected ACI gamma over time (log scale)
- **Created by**: `medical/medical_conformal.py:plot_results`; called from `main()` after `run_medical_experiment`
- **Status**: Run and saved as PDF for all 3 conditions in `results/medical/pdf/`
- **Paper-readiness**: Preliminary. Per-condition only — no single comparison figure with shift/no-shift on the same axes.

#### `plot_aggregated` in `medical/multi_seed_medical.py`
- **What it shows**: 6-panel figure — coverage+IQR, width±std, early/mid/late bar+errorbar, per-seed histogram, variability over time, summary text
- **Status**: Run for all 3 multi-seed conditions
- **Paper-readiness**: Preliminary; designed for diagnostics, not direct paper inclusion.

#### `plot_medical_covariate_shift.py`
- **What it shows**: 5-row KDE+bar layout: Heart Rate, RR, O2Sat, NaCl (full KDE each), Age/gender/ethnicity (row 5 split)
- **KL divergences (Test ‖ TrainCal)**: Heart Rate=0.0009, RR=0.022, O2Sat=0.050, NaCl=0.206 (dominant), Age=0.014
- **Status**: Run; output at `results/medical/pdf/medical_covariate_shift.pdf`

---

### 6. Missing or partial components

| Component | Status | Notes |
|---|---|---|
| Data pickle (`medical/sepsis_experiment_data_nacl_target.pkl`) | **Completed** | Present; 9264 TrainCal + 5827 Test patients (split by Norep in first 12 h); verified loadable |
| Data dictionary (`medical/medical_data.md`) | **Completed** | Cohort, imputation, split, dict structure, usage examples |
| Experiment script (`medical/medical_conformal.py`) | **Completed** | Full CLI, LR weighting, ACI, model, featurizer, gamma selection, plot |
| Run results (JSON/PDF) for 3 conditions | **Completed** | `results/medical/json/medical_ms_{noshift,shift,LRonly}.json`; PDFs in `results/medical/pdf/` |
| Multi-seed wrapper for medical | **Completed** | `medical/multi_seed_medical.py`; 10 seeds, base_seed=1000, n_traincal=1000, n_test=500 |
| Covariate shift diagnostic plot (KDE / KL) | **Completed** | `medical/plot_medical_covariate_shift.py`; output `results/medical/pdf/medical_covariate_shift.pdf` |
| Baseline comparison (OnlineConformalPredictor on medical data) | **Not implemented** | No equivalent of finance_adaptive.py for the medical domain |
| Ablation: LR weighting without ACI (γ=0) as separate paper figure | **Partial** | LRonly multi-seed exists (`medical_ms_LRonly.json`); no side-by-side comparison figure |
| MIMIC-III extraction pipeline (upstream of pkl) | **Not in repo** | Mentioned in `medical_conformal.py` docstring; code not present |
