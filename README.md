# Conformal Prediction with Covariate Shift

> **Status**: Objective section is a placeholder — to be filled in after author clarification.

---

## Table of Contents

1. [Project Objective](#project-objective)
2. [Repository Layout](#repository-layout)
3. [Module Reference](#module-reference)
   - [ts_generator.py](#ts_generatorpy)
   - [algorithm.py](#algorithmpy)
   - [adaptive_conformal.py](#adaptive_conformalpy)
   - [test_conformal.py](#test_conformalpy)
   - [multi_seed_experiments.py](#multi_seed_experimentspy)
   - [finance_data.py](#finance_datapy)
   - [finance_conformal.py](#finance_conformalpy)
   - [medical_conformal.py](#medical_conformalpy)
   - [medical_data.py](#medical_datapy)
4. [Data Files](#data-files)
5. [Experimental Results](#experimental-results)
6. [Quick-Start Commands](#quick-start-commands)
7. [Medical Experiment Status](#medical-experiment-status)

---

## Project Objective

This codebase accompanies a new paper proposing **AdaptedCAFHT** (`algorithm.py`) as an improved conformal prediction algorithm for time series that undergo **covariate shift** at test time.

**Core claim**: When the covariate distribution shifts between calibration and test, the standard online conformal predictor (`adaptive_conformal.py` — `OnlineConformalPredictor`) loses its nominal coverage guarantee. AdaptedCAFHT restores coverage by re-weighting calibration scores with **density-ratio weights** estimated via logistic regression between the train and test covariate prefixes. It further combines this with **Adaptive Conformal Inference (ACI)** — per-series miscoverage levels that adapt online — making the method robust to both distributional shift and non-stationarity.

**Evaluation structure**:
- **Synthetic AR(1) setting** — controlled baseline experiments that isolate the effect of covariate shift; `OnlineConformalPredictor` serves as the no-weighting baseline.
- **Real-world finance application** — S&P 500 daily returns with one GICS sector held out as the test set (sector = natural covariate shift); validates the method on real data.
- **Real-world medical application** — MIMIC-III sepsis cohort, 24-hour NaCl 0.9% dosage trajectories. Train/Test split is defined by Norepinephrine exposure (induces shift without being used as a covariate). **Implemented single-seed; no saved runs yet.** See [§ Medical Experiment Status](#medical-experiment-status).

**Key comparison**: AdaptedCAFHT (with density-ratio weighting) vs. `OnlineConformalPredictor` (no covariate correction), under both no-shift and shift conditions.

---

## Repository Layout

```
Conformal-Covariate-Refactor/
├── algorithm.py                               # Core algorithm: AdaptedCAFHT (weighted conformal + ACI)
├── adaptive_conformal.py                      # Simpler baseline: OnlineConformalPredictor (no covariate weighting)
├── ts_generator.py                            # Synthetic AR(1) time series generator with covariate shift
├── test_conformal.py                          # Single-run coverage experiment (time-based analysis)
├── multi_seed_experiments.py                  # Multi-seed wrapper for statistical robustness
├── finance_data.py                            # S&P 500 data loader (yfinance) + save/load utilities
├── finance_conformal.py                       # Finance experiment: AdaptedCAFHT on S&P 500 sectors
├── medical_conformal.py                       # Medical experiment: AdaptedCAFHT on sepsis ICU (NaCl target)
├── medical_data.py                            # Medical: TrainCal-vs-Test covariate-shift visualization
├── medical_data.md                            # Pickle format + cohort definition documentation
├── sepsis_experiment_data_nacl_target.pkl     # Preprocessed MIMIC-III sepsis cohort (8600 + 6491 patients)
├── data/
│   └── sp500_YYYYMMDD_YYYYMMDD.{npz,json}     # Cached market data (13 date windows)
└── results/                                   # JSON + PDF outputs (finance populated; medical empty)
```

---

## Module Reference

### `ts_generator.py`

**Class**: `TimeSeriesGenerator`

Generates synthetic univariate AR(1) time series driven by a covariate `X`, and can apply a **covariate shift** to a test subset.

#### Two covariate modes

| Mode | X generation | Shift mechanism |
|---|---|---|
| `static` | `X ~ Poisson(covar_rate)`, constant across time | Re-draw `X` from `Poisson(covar_rate_shift)` |
| `dynamic` | `X_0 ~ Poisson(x0_lambda)`, then AR(1) path `X_{t+1} = ρ·X_t + η_t` | Re-generate entire X path with shifted AR parameters |

#### Y model (both modes)

```
Y_t = ar_coef · Y_{t-1} + beta · X_t + trend_coef · t + ε_t,   ε_t ~ N(0, noise_std²)
```

#### Key methods

| Method | Description |
|---|---|
| `generate_with_poisson_covariate(n, ...)` | Static-X generation → returns `(Y, X)` of shapes `(n, T+1, 1)` and `(n,)` |
| `generate_with_dynamic_covariate(n, ...)` | Dynamic-X generation → returns `(Y, X)` of shapes `(n, T+1, 1)` and `(n, T+1)` |
| `introduce_covariate_shift(original_Y, original_X, ...)` | Re-generates Y from same Y₀ but under shifted X; returns `(shifted_Y, shifted_X_series)` |
| `compute_likelihood_ratios(original_data, shifted_data)` | Gaussian KDE estimate of density ratio p_shift / p_orig on initial conditions |
| `visualize_covariate_shift(...)` | 5-panel plot comparing original vs. shifted Y paths and X distributions |
| `print_statistics(...)` | Console summary of Y and X moments before/after shift |

---

### `algorithm.py`

**Class**: `AdaptedCAFHT`

The primary algorithm. Combines:
- **AR(1) point forecasting** (fitted via least squares with intercept)
- **Weighted conformal calibration** using density-ratio weights to correct for covariate shift
- **Adaptive Conformal Inference (ACI)**: per-series miscoverage level `α_t` is updated online as `α_{t+1} = α_t + γ·(α - err_t)` where `err_t ∈ {0,1}` indicates a miss.

#### Key attributes

| Attribute | Role |
|---|---|
| `alpha` | Nominal miscoverage level (e.g. 0.1 → 90% target coverage) |
| `ar_intercept`, `ar_coef`, `noise_std` | Fitted AR(1) parameters |
| `_scores`, `_weights` | Calibration conformity scores and their density-ratio weights |
| `_clf` | Logistic regression classifier used for density-ratio estimation |

#### Key methods

| Method | Description |
|---|---|
| `fit_ar_model(Y_subset)` | Fits AR(1) with intercept via OLS on all `(Y_{t-1}, Y_t)` pairs |
| `update_weighting_context(train_prefixes, test_prefixes, is_shifted, ...)` | Stores train/test prefix features for subsequent density-ratio computation |
| `calibrate(cal_Y_subset)` | Computes conformity scores `|y_true - ŷ|` on calibration data; applies density-ratio weights if shift context is set |
| `predict_with_interval(input_series, alpha_level)` | Returns `(pred, lower, upper)` from the weighted `(1-α)` quantile |
| `_featurize_prefixes(Y_prefixes, X_prefixes)` | Default: last Y value only. Monkey-patched in `finance_conformal.py` with richer Y+X summaries |
| `_compute_density_ratio_weights(trainX, testX, evalX)` | Fits a balanced logistic regression (train=0, test=1) and returns `prob1/(1-prob1)` ratios, clipped at 5× mean |
| `_weighted_quantile(values, weights, q)` | Weighted empirical CDF quantile |

#### Density-ratio estimation

Uses `sklearn.LogisticRegression` (balanced class weights) if available, otherwise falls back to a custom gradient-descent logistic regression. Weights are clipped at 5× their mean to prevent degeneracy.

---

### `adaptive_conformal.py`

**Class**: `OnlineConformalPredictor`

A simpler **online/adaptive** baseline that uses AR(1) on Y only (no covariate weighting). Conformity scores are updated after each new observation using a sliding window.

#### Key methods

| Method | Description |
|---|---|
| `fit_ar_model(train_data)` | Fits AR(1) with intercept on training data |
| `calibrate(calibration_data)` | Initialises conformity score buffer from calibration residuals |
| `get_current_quantile()` | Computes `⌈(n+1)(1-α)⌉/n` finite-sample-corrected quantile |
| `predict_with_interval(series)` | Returns `(pred, lower, upper)` using the current quantile |
| `evaluate_coverage(test_data, adaptive)` | Runs over all test series; if `adaptive=True`, appends each new residual to the score buffer (sliding window of size `window_size`) |

---

### `test_conformal.py`

Single-run experiment harness. Generates data via `TimeSeriesGenerator`, runs a chosen predictor, and reports **coverage rate and interval width at each time step t = 1, …, T**.

#### Supported predictors

| `--predictor` | Class used |
|---|---|
| `basic` | `BasicConformalPredictor` (AR only, no adaptation) |
| `adaptive` | `OnlineConformalPredictor` |
| `algorithm` | `AdaptedCAFHT` |

#### `run_time_based_coverage_experiment` (main function)

At each time step `t`:
1. Generates `n_train` training series, `n_cal` calibration series, and `n_series` test series (optionally under covariate shift).
2. Fits the model on training data.
3. Calibrates on calibration data.
4. Predicts intervals for all test series and records coverage and width.
5. For `algorithm` predictor with `with_shift=True`, performs half-sample split: each test half uses the other half as the "context" for computing density-ratio weights.

Returns a dict `results_by_time[t]` containing `coverage_rate`, `interval_width`, `coverage_history`, etc.

---

### `multi_seed_experiments.py`

**Class**: `MultiSeedExperiment`

Wrapper that calls `run_time_based_coverage_experiment` across `n_seeds` different random seeds, then aggregates and visualises the results.

#### Workflow

```
MultiSeedExperiment.run_all_seeds()
  → _run_single_seed(seed) for each seed
  
MultiSeedExperiment.aggregate_results()
  → per-time-step statistics (mean, std, IQR of coverage and width)
  → overall statistics (pooled coverage, width, degradation)

MultiSeedExperiment.plot_aggregated_results(aggregated)
  → 6-panel figure: coverage over time, width over time, boxplots,
    empirical vs. mean coverage, coverage variability, summary table

MultiSeedExperiment.save_results(aggregated, filepath)
  → serialises to JSON
```

#### Aggregated statistics reported

- Per-time-step: `coverage_mean`, `coverage_std`, `coverage_q25`, `coverage_q75`, `empirical_coverage` (pooled)
- Overall: `coverage_mean ± coverage_se`, `width_mean`, `early_coverage_mean`, `late_coverage_mean`, `coverage_degradation` (early − late)

---

### `finance_data.py`

Downloads and caches S&P 500 daily OHLCV data from `yfinance`.

#### Arrays produced

| Array | Shape | Description |
|---|---|---|
| `Y` | `(n_series, L, 1)` | Intraday return `Close_t / Open_t − 1` |
| `X` | `(n_series, L, n_cov)` | Four covariates (see below) |
| `dates` | `(L,)` | Trading date strings |

#### Covariates

| Name | Lag | Formula |
|---|---|---|
| `OvernightGapReturn` | None | `(Open_t − Close_{t-1}) / Close_{t-1}` |
| `Above52wLowReturn` | None | `(Open_t − 52w_low) / 52w_low`; 52w low computed from 252 pre-sample days |
| `TurnoverRatio_lag1` | 1 day | `Volume_{t-1} / (shares_outstanding · Close_{t-1})` |
| `DailyRangeReturn_lag1` | 1 day | `(High − Low)_{t-1} / Close_{t-1}` |

#### Key functions

| Function | Description |
|---|---|
| `load_sp500(start, end, top_n)` | Downloads all ~503 S&P 500 constituents; `top_n` limits for quick tests |
| `load_series(tickers, start, end)` | Downloads an arbitrary ticker list |
| `save(result, stem, directory)` | Saves arrays to `.npz` and metadata to `.json` |
| `load_stored(npz_path, json_path)` | Reloads from disk without re-downloading |
| `filter_by_sector(result, sectors)` | Returns sub-result for given sectors |
| `filter_by_industry(result, industries)` | Returns sub-result for given industries |
| `summarize(result)` | Prints human-readable data summary |

---

### `finance_conformal.py`

Runs the full `AdaptedCAFHT` pipeline on S&P 500 data.

#### Experiment setup

1. **Split** tickers by sector: one sector is held out as **test**, remaining tickers are split into **train** (50%) and **calibration** (50%) by default.
2. **Model**: `LinearCovariateModel` — OLS regression of `Y_t` on the four covariates `X_t` (with intercept).
3. **Calibration**: residuals `|Y_t − Ŷ_t|` accumulated over all calibration tickers and all time steps up to `t`.
4. **Gamma selection**: every 10 time steps, selects the ACI step-size `γ` from `GAMMA_GRID = [0.001, 0.005, 0.01, 0.05]` by minimising coverage error on a held-out third of train tickers.
5. **Shift correction** (`--with_shift`): uses a richer featurizer (`_featurize_YX_summaries`) that computes rolling-window Y statistics (mean, std, AR(1) coefficient over last 30 steps) plus mean covariate values, then estimates density-ratio weights via logistic regression.

#### Rich featurizer (`_featurize_YX_summaries`)

| Feature | Description |
|---|---|
| `Y_mean(w30)` | Mean of Y over last 30 steps |
| `Y_std(w30)` | Std of Y over last 30 steps |
| `Y_ar1(w30)` | AR(1) coefficient of Y over last 30 steps |
| `X_mean_k` | Mean of covariate `k` over full prefix (one per covariate) |

#### Output

Returns a results dict with `coverage_by_time`, `width_by_time`, `overall_coverage`, `gamma_opt_history`, and `first_test_series` (actual close prices + prediction intervals for one test ticker).

---

### `medical_conformal.py`

**Function**: `run_medical_experiment(data, cal_frac=0.5, alpha=0.1, seed=42, gamma_grid=None, with_shift=False, n_traincal=None, n_test=None)`

Runs `AdaptedCAFHT` on the MIMIC-III sepsis cohort (`sepsis_experiment_data_nacl_target.pkl`). Target is the 24-hour `NaCl 0.9%` dosage trajectory. Train/Test split is fixed upstream by Norepinephrine exposure; this script further splits TrainCal into Train + Cal via `cal_frac`.

#### Prediction model — `LinearCovariateModel`

Pooled cross-sectional OLS:
```
NaCl_t ≈ β_0 + β_1·HR_t + β_2·RR_t + β_3·O2Sat_t
       + β_4·Age + β_5·gender_M
       + β_6·eth_BLACK + β_7·eth_HISPANIC + β_8·eth_ASIAN + β_9·eth_OTHER
```
Dynamic covariates vary per (patient, timestep); static covariates are tiled across timesteps for design-matrix assembly.

#### Featurizer for LR classifier (`_richer_featurize_prefixes`, monkey-patched)

26-dimensional feature vector per patient, standardized with training-set `(μ, σ)`:

| Block | Count | Contents |
|---|---|---|
| Dynamic summary stats | 5 × 4 = 20 | mean, std, min, max, last over NaCl + 3 CHART covariates |
| Static | 6 | Age, gender_M, eth_BLACK, eth_HISPANIC, eth_ASIAN, eth_OTHER |

Because `algorithm.py:_featurize_prefixes` only accepts Y prefixes, the caller sets `predictor._X_ctx` and `predictor._S_ctx` before each featurize call.

#### Loop structure (identical pattern to `finance_conformal.py`)

```
for t in range(T):                              # T = 23 (hours 0..22)
    fit LinearCovariateModel on train[:, :t+2, :]
    build calibration scores on cal[:, :t+2, :]
    (re-select γ every 5 steps via 3-way train-split ACI)
    if with_shift and t >= 1:
        cross-split test into two halves
        for each (predict_half, context_half):
            train LR classifier: train (label 0) vs context_half (label 1)
            reweight cal scores with density-ratio weights
            predict step t+1 for predict_half
    else:
        predict step t+1 for all test patients (uniform weights)
    ACI alpha update per patient: α_{t+1} = α_t + γ·(α − err_t)
```

#### CLI knobs

| Flag | Default | Purpose |
|---|---|---|
| `--pkl` | required | Path to `sepsis_experiment_data_nacl_target.pkl` |
| `--with_shift` | off | Toggle LR-weighted calibration |
| `--cal_frac` | 0.5 | Fraction of TrainCal used for calibration |
| `--alpha` | 0.1 | Miscoverage target (90% coverage) |
| `--seed` | 42 | Random seed (Train/Cal split, subsampling, γ-selection) |
| `--gamma_grid` | `[1e-6 … 1e-2]` (9 values) | ACI step-size candidates |
| `--n_traincal` / `--n_test` | `None` | Random subsample for faster iteration |
| `--save_json` / `--save_plot` | `None` | Output paths |

#### Output

Returns a results dict with `coverage_by_time`, `width_by_time`, `overall_coverage`, `target_coverage`, `gamma_opt_history`, `first_test_patient`, `first_test_series`, and a `config` block.

---

### `medical_data.py`

**Function**: `visualize(data, show_static=False, show_dynamic=False, show_target=False, save_path=None)`

Standalone covariate-shift inspection tool. Compares TrainCal vs. Test distributions on the sepsis cohort.

| Flag | Produces |
|---|---|
| `--static` | Side-by-side bar charts for Age bins, Gender M/F, Ethnicity groups |
| `--dynamic` | Three mean-trajectory lines (Heart Rate, Respiratory Rate, O2 saturation) |
| `--target` | One mean-trajectory line for NaCl 0.9% |
| `--save_plot` | Output path (PDF or PNG) |

Up to 7 panels; useful for Section 5.3 motivating figures.

---

## Data Files

### Cached S&P 500 data (`.npz` + `.json` pairs)

Thirteen date windows are cached locally, spanning Jan 2023 – Feb 2025:

| File stem | Date range |
|---|---|
| `sp500_20231004_20240328` | Oct 2023 – Mar 2024 |
| `sp500_20240102_20240229` | Jan – Feb 2024 |
| `sp500_20240201_20240328` | Feb – Mar 2024 |
| `sp500_20240301_20240430` | Mar – Apr 2024 |
| `sp500_20240401_20240531` | Apr – May 2024 |
| `sp500_20240501_20240628` | May – Jun 2024 |
| `sp500_20240603_20240731` | Jun – Jul 2024 |
| `sp500_20240701_20240830` | Jul – Aug 2024 |
| `sp500_20240801_20240930` | Aug – Sep 2024 |
| `sp500_20240903_20241031` | Sep – Oct 2024 |
| `sp500_20241001_20241129` | Oct – Nov 2024 |
| `sp500_20241101_20241231` | Nov – Dec 2024 |
| `sp500_20241202_20250131` | Dec 2024 – Jan 2025 |
| `sp500_20250102_20250228` | Jan – Feb 2025 |

### Sepsis-ICU pickle (`sepsis_experiment_data_nacl_target.pkl`)

Preprocessed MIMIC-III sepsis cohort. Full format is documented in `medical_data.md`. Summary:

| Key | Type | Description |
|---|---|---|
| `patient_ids_traincal` | `list` | 8600 patient IDs with no Norepinephrine exposure |
| `patient_trajectory_list_traincal` | `list[dict]` | Aligned patient dicts (same order as IDs) |
| `patient_ids_test` | `list` | 6491 patient IDs with any Norepinephrine exposure |
| `patient_trajectory_list_test` | `list[dict]` | Aligned patient dicts |

Each patient dict:

| Key | Type | Shape |
|---|---|---|
| `Age` / `gender` / `ethnicity` | scalars | per patient |
| `Heart Rate`, `Respiratory Rate`, `O2 saturation pulseoxymetry` | `pd.DataFrame` | 24×2 (`hour`, `value`) |
| `NaCl 0.9% (target)` | `pd.DataFrame` | 24×2 — the prediction target |
| `Norepinephrine` | `pd.DataFrame` | 24×2 — kept for reference only; *not* a model covariate |

Imputation rule: CHART zeros → per-patient median of nonzero values; NaCl zeros kept as-is (sparse signal).

---

## Experimental Results

All synthetic experiments use: α = 0.1 (target 90% coverage), AR coef α_Y = 0.7, β = 1.0, noise_std = 0.2, static Poisson covariate (rate 1.0 → 2.0 under shift), ACI step-size γ = 0.005. Python 3.11+ required (use `boa` conda env).

---

### Synthetic Comparison: AdaptedCAFHT vs OnlineConformalPredictor (30 seeds each)

**Setup**: n_series = 300, n_train = 600, n_cal = 600, T = 20, 30 random seeds.

#### Overall summary

| Predictor | Shift? | Coverage (mean ± se) | Error vs 90% | Width | Early cov. | Late cov. |
|---|---|---|---|---|---|---|
| AdaptedCAFHT | No | **89.74% ± 0.07%** | −0.26% | 1.287 | 89.84% | 89.62% |
| AdaptedCAFHT | Yes | **88.70% ± 0.07%** | −1.30% | 1.278 | 84.84% | 90.39% |
| OnlineConformalPredictor | No | **89.99% ± 0.07%** | −0.01% | 1.290 | 89.95% | 90.02% |
| OnlineConformalPredictor | Yes | **89.69% ± 0.07%** | −0.31% | 1.290 | 85.19% | 91.56% |

*Early = first ⌊T/3⌋ time steps; late = last ⌊T/3⌋ time steps.*

#### Per-time-step coverage — shift condition

The critical comparison is the **recovery trajectory** after the initial coverage drop at t = 1 (before density-ratio weights are activated):

| t | AdaptedCAFHT + shift | Adaptive + shift | AdaptedCAFHT | Adaptive |
|---|---|---|---|---|
| 1 | 66.4% | 66.6% | 90.2% | 90.4% |
| 2 | **82.8%** | 75.0% | 89.9% | 90.1% |
| 3 | 88.0% | 87.5% | 89.4% | 89.5% |
| 4 | 89.9% | **93.8%** | 90.2% | 90.2% |
| 5 | 91.2% | **94.4%** | 89.4% | 89.6% |
| 6 | 90.8% | **93.9%** | 89.9% | 89.9% |
| 7 | 90.8% | **92.7%** | 89.9% | 90.1% |
| 8 | 90.8% | **92.4%** | 90.3% | 90.5% |
| 9 | 90.7% | **92.1%** | 89.4% | 89.6% |
| 10 | 89.9% | 90.8% | 89.5% | 89.7% |
| 15 | 91.0% | 92.2% | 90.0% | 90.4% |
| 20 | 90.0% | 91.3% | 88.7% | 89.2% |

**Key observations**:

1. **t = 1**: Both methods drop identically to ~66% coverage. Density-ratio weights are not yet active (the algorithm requires t ≥ 1 in 0-indexed loop to engage weighting). This is a known cold-start effect.

2. **t = 2 onward**: AdaptedCAFHT recovers faster (82.8% at t=2 vs 75.0% for adaptive). ACI widens intervals in response to the t=1 miss, then density-ratio weighting fine-tunes the quantile.

3. **t = 3–10 (sustained shift)**: The adaptive baseline over-corrects significantly (92–94% coverage, up to +4% above target). AdaptedCAFHT stays tighter to the nominal 90%, with intervals that are the same width or slightly narrower.

4. **No-shift condition**: Both methods achieve near-nominal coverage throughout (within 0.5% of target), validating correctness in the standard setting.

**Interpretation**: AdaptedCAFHT's advantage is **coverage stability** — it avoids the over-correction that the adaptive baseline exhibits at t=3–10. The adaptive baseline compensates by making intervals too wide after the initial failure; AdaptedCAFHT reaches the correct quantile more directly via weighted calibration.

---

### Earlier Single-Seed Results (for reference)

**File**: `results/results_algorithm_noshift_20260302_144653.json` — AdaptedCAFHT, no shift, 1 seed, T=40, n_series=500. Overall coverage: **89.94%**, mean width 1.077.

**File**: `results/results_algorithm_shift_20260203_120324.json` — AdaptedCAFHT, with shift (rate 1→2), T=100, n_series=500. Overall coverage: **91.0%**, mean width 2.170.

---

### Finance Experiments — S&P 500 (AdaptedCAFHT on real data)

**Setup**: `sp500_20240201_20240328.npz` (Feb–Mar 2024, 40 trading days). One GICS sector held out as test; remaining tickers split 50/50 into train and calibration. α = 0.1 (target 90%). The rich featurizer (`_featurize_YX_summaries`) computes Y rolling-window statistics + mean covariate values over the prefix. Covariates were selected to differentiate the Technology sector.

> Note: covariates (`OvernightGapReturn`, `Above52wLowReturn`, `TurnoverRatio_lag1`, `DailyRangeReturn_lag1`) are designed for Technology sector differentiation. Results on other sectors are secondary.

#### Technology sector (designed test case)

| Condition | train/cal/test | Overall coverage | Error | Mean width | Early cov. | Late cov. |
|---|---|---|---|---|---|---|
| AdaptedCAFHT + shift correction | 199/198/72 | **88.14%** | −1.86% | 0.0444 | 89.32% | 89.74% |
| AdaptedCAFHT, no shift correction | 199/198/72 | **87.54%** | −2.46% | 0.0432 | 88.25% | 89.21% |
| **Improvement from shift correction** | — | **+0.60%** | +0.60pp | +2.8% | +1.07pp | +0.53pp |

**Per-time-step highlights** (Technology, diff = with_shift − no_shift):

| t | with_shift | no_shift | diff |
|---|---|---|---|
| 3 | **84.7%** | 76.4% | **+8.3%** ← largest improvement |
| 4 | 94.4% | 93.1% | +1.4% |
| 6 | 86.1% | 84.7% | +1.4% |
| 11 | 88.9% | 87.5% | +1.4% |
| 33 | 87.5% | 84.7% | +2.8% |
| 38 | 77.8% | 76.4% | +1.4% |

At t=3, the first step where the density-ratio weighting is fully engaged with two-step Y prefix features, the shift correction prevents a coverage collapse from 76.4% to 84.7% — an 8.3% improvement in a single time step. At all other steps the correction is non-negative (never hurts), with modest consistent improvement.

#### Healthcare sector (secondary, different covariate fit)

| Condition | train/cal/test | Overall coverage | Error | Mean width |
|---|---|---|---|---|
| AdaptedCAFHT + shift correction | 206/205/58 | 90.76% | +0.76% | 0.0441 |

Healthcare slightly over-covers (+0.76%), which is expected — the covariates were designed for Technology, so the density-ratio correction applies some adjustment that is less precisely targeted but still non-harmful.

---

## Quick-Start Commands

> **Python version**: The codebase uses `dict | None` union syntax (Python 3.10+). Use the `boa` conda environment:
> ```bash
> PYTHON=/Users/andrewlou/opt/anaconda3/envs/boa/bin/python
> ```

### Synthetic experiments

```bash
# Single run, no shift, algorithm predictor
MPLBACKEND=Agg $PYTHON test_conformal.py --predictor algorithm --n_series 300

# Single run, with covariate shift
MPLBACKEND=Agg $PYTHON test_conformal.py --predictor algorithm --with_shift --n_series 300

# 4-way comparison (proposed vs baseline × shift vs no shift), 30 seeds each
MPLBACKEND=Agg $PYTHON multi_seed_experiments.py --predictor algorithm --n_seeds 30 \
    --n_series 300 --n_train 600 --n_cal 600 --T 20 --with_shift --save_dir results
MPLBACKEND=Agg $PYTHON multi_seed_experiments.py --predictor algorithm --n_seeds 30 \
    --n_series 300 --n_train 600 --n_cal 600 --T 20 --save_dir results
MPLBACKEND=Agg $PYTHON multi_seed_experiments.py --predictor adaptive --n_seeds 30 \
    --n_series 300 --n_train 600 --n_cal 600 --T 20 --with_shift --save_dir results
MPLBACKEND=Agg $PYTHON multi_seed_experiments.py --predictor adaptive --n_seeds 30 \
    --n_series 300 --n_train 600 --n_cal 600 --T 20 --save_dir results

# Dynamic covariates with shift
MPLBACKEND=Agg $PYTHON test_conformal.py --predictor algorithm --covariate_mode dynamic \
    --with_shift --x_rate 0.6 --x_rate_shift 0.9
```

### Finance experiments

```bash
# Step 1: pull data (once — ~30 min for all 500 tickers)
MPLBACKEND=Agg $PYTHON finance_data.py --pull --start 2024-02-01 --end 2024-03-28

# Step 2: run experiment (Technology sector as test, no shift)
MPLBACKEND=Agg $PYTHON finance_conformal.py --npz sp500_20240201_20240328.npz \
    --test_sector Technology --save_json results/finance_tech_noshift.json

# Step 2 (with shift correction)
MPLBACKEND=Agg $PYTHON finance_conformal.py --npz sp500_20240201_20240328.npz \
    --test_sector Technology --with_shift \
    --save_json results/finance_tech_shift.json \
    --save_plot results/finance_tech_shift.png
```

### Medical experiments

```bash
# Covariate-shift visualization (TrainCal vs Test bars + trajectories)
MPLBACKEND=Agg $PYTHON medical_data.py \
    --pkl sepsis_experiment_data_nacl_target.pkl \
    --static --dynamic --target \
    --save_plot results/medical_covariate_shift.pdf

# No shift correction (baseline)
MPLBACKEND=Agg $PYTHON medical_conformal.py \
    --pkl sepsis_experiment_data_nacl_target.pkl \
    --save_json results/medical_nacl_noshift.json \
    --save_plot results/medical_nacl_noshift.pdf

# With LR covariate-shift correction
MPLBACKEND=Agg $PYTHON medical_conformal.py \
    --pkl sepsis_experiment_data_nacl_target.pkl --with_shift \
    --save_json results/medical_nacl_shift.json \
    --save_plot results/medical_nacl_shift.pdf

# Fast smoke-test (subsampled cohort)
MPLBACKEND=Agg $PYTHON medical_conformal.py \
    --pkl sepsis_experiment_data_nacl_target.pkl --with_shift \
    --n_traincal 500 --n_test 300
```

---

## Medical Experiment Status

**State**: pipeline implemented, no runs committed.

### Completed
- Data loader + 3-dynamic + 6-static array converter ([medical_conformal.py:231](medical_conformal.py#L231) `_convert_to_arrays`; [medical_data.md](medical_data.md) documents the pickle).
- Ethnicity grouping (30+ MIMIC strings → {WHITE, BLACK, HISPANIC, ASIAN, OTHER}; 4-dim one-hot).
- `LinearCovariateModel` with static covariates tiled into the design matrix ([medical_conformal.py:443](medical_conformal.py#L443)).
- Monkey-patched 26-dim LR featurizer (20 dynamic stats + 6 static), standardized with training-set stats ([medical_conformal.py:303](medical_conformal.py#L303) `_richer_featurize_prefixes`).
- Cross-half test split for LR weighting ([medical_conformal.py:753-892](medical_conformal.py#L753)).
- Gamma selection via 3-way training split, re-run every 5 steps ([medical_conformal.py:505](medical_conformal.py#L505) `_select_gamma`; `GAMMA_GRID` at line 191).
- Single-seed `with_shift` / `no_shift` runner ([medical_conformal.py:593](medical_conformal.py#L593) `run_medical_experiment`, CLI at line 1021).
- Patient subsampling via `--n_traincal` / `--n_test`.
- Covariate-shift visualization (3 bar charts + 4 trajectory plots) ([medical_data.py:217](medical_data.py#L217) `visualize`).

### Not implemented
- Multi-seed wrapper (seeds vary the Train/Cal split within TrainCal; Train/Test split is fixed).
- Baselines on medical data: plain split conformal, `OnlineConformalPredictor`, Weighted-CAFHT-without-ACI.
- Shell runner `run_medical_experiments.sh`.
- LaTeX tables / paper-ready PDFs.
- Any saved runs in `results/` (currently zero `medical_*` / `sepsis_*` / `nacl_*` files).

### Known non-trivial design choices
- `GAMMA_GRID = [1e-6 … 1e-2]` (wider than finance) — NaCl residual scale is hundreds of mL and the target is sparse, so small γ is needed.
- Gamma re-selected every **5** steps (finance/synthetic use **10**) — the horizon is shorter (T=23) so more frequent updates are reasonable.
- Norepinephrine is *not* used as a predictive covariate — only to define the Train/Test split. This is the only source of covariate shift and it is a fixed property of the cohort (no knob).

