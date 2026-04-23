# Project: Weighted CAFHT — Conformal Prediction under Covariate Shift

## What this project is

A research codebase for a new conformal prediction method called **Weighted CAFHT** for time series under covariate shift. The paper's central claim:

- Likelihood-ratio (LR) reweighting reshapes the calibration score distribution toward the deployment distribution.
- ACI-style adaptive updating corrects residual miscoverage online.
- The combined method maintains long-run empirical coverage while improving efficiency (narrower bands) when covariate shift is well-estimated.

Target coverage guarantee: P(Y^{n+1}_t ∈ Ĉ_t, ∀t ∈ [T]) ≥ 1 − α.

---

## Repository map

| File | Role |
|---|---|
| `algorithm.py` | `AdaptedCAFHT` class — the full proposed method |
| `adaptive_conformal.py` | `OnlineConformalPredictor` and `BasicConformalPredictor` — baselines |
| `ts_generator.py` | `TimeSeriesGenerator` — synthetic DGPs + shift |
| `test_conformal.py` | Single-seed synthetic experiment runner; exports `run_time_based_coverage_experiment` |
| `multi_seed_experiments.py` | Multi-seed wrapper around `test_conformal.py`; aggregates coverage + width |
| `finance_data.py` | yfinance S&P 500 loader; save/load `.npz`+`.json`; sector/industry filters |
| `data/` | All S&P 500 data files (`sp500_*.npz` + `sp500_*.json`). Pass paths as `data/sp500_DATES.npz` |
| `finance_conformal.py` | S&P 500 experiment using `AdaptedCAFHT` with linear covariate model |
| `finance_adaptive.py` | S&P 500 experiment using `OnlineConformalPredictor` (AR(1) only, no covariates) |
| `plot_covariate_shift.py` | Standalone covariate distribution visualization |
| `medical_conformal.py` | Sepsis-ICU experiment using `AdaptedCAFHT` with linear model on 3 dynamic CHART covariates + 3 static covariates (Age, gender, ethnicity); target = NaCl 0.9% trajectory |
| `medical_data.py` | Sepsis-ICU covariate-shift visualization (bar charts for static + mean trajectories for dynamic/target) |
| `medical_data.md` | Documentation of `sepsis_experiment_data_nacl_target.pkl` format and cohort definition |
| `sepsis_experiment_data_nacl_target.pkl` | Preprocessed MIMIC-III sepsis cohort (8600 TrainCal, 6491 Test patients) |
| `results/` | Output PNGs and JSON files |

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

### Sepsis ICU medical data (`sepsis_experiment_data_nacl_target.pkl`)
Source: MIMIC-III sepsis cohort, 24 hourly timestamps per patient, 8600 TrainCal + 6491 Test. Full format documented in `medical_data.md`.
- Y (target): `NaCl 0.9% (target)` — hourly mL dosage trajectory. Zeros preserved (sparse medication signal).
- X (3 dynamic CHART covariates): `Heart Rate`, `Respiratory Rate`, `O2 saturation pulseoxymetry`. Zero entries imputed with per-patient median of nonzero values; patients with all-zero trajectory excluded upstream.
- S (3 static covariates, encoded to 6 dims): `Age` (numeric), `gender` (M→1, F→0), `ethnicity` (30+ raw categories → 5 groups {WHITE, BLACK, HISPANIC, ASIAN, OTHER}; one-hot, WHITE reference).
- Train/Test split: defined by upstream Norepinephrine exposure — TrainCal = no Norepinephrine, Test = any Norepinephrine. Norepinephrine is *not* used as a model covariate; it only induces the shift.
- Covariate shift is a fixed property of the cohort (no knob).

---

## Methods implemented

### AdaptedCAFHT (`algorithm.py`)
Single class implementing the full proposed method. Three stages at each time step t:

1. **Model fit**: AR(1) by OLS on all training series up to time t+1.
2. **Score construction**: nonconformity score = |Y_s − Ŷ_s| accumulated for all s ≤ t+1.
3. **Weighted quantile**: prediction interval = [Ŷ_{t+1} − q, Ŷ_{t+1} + q] where q is the weighted (1−α_t)-quantile.

In finance experiments, Step 1 is replaced by a linear regression on the 4 S&P 500 covariates (OLS, pooled across all training tickers and all time steps).

In medical experiments (`medical_conformal.py:LinearCovariateModel`), Step 1 is a pooled cross-sectional OLS regressing NaCl_t on the 3 dynamic covariates plus the 6 encoded static features (intercept + 3 + 6 = 10 coefficients). Static covariates are tiled across timesteps for design-matrix assembly.

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

### Medical featurizer (monkey-patched at runtime)
`medical_conformal.py:_richer_featurize_prefixes` replaces the default featurizer with a 26-dim feature vector:
- 5 summary stats (mean, std, min, max, last) over the NaCl prefix and each of the 3 dynamic covariates → 20 features
- 6 static features (Age, gender_M, 4 ethnicity dummies) appended verbatim
- Standardized using training-set mean/std; the same `(mu, std)` is reused for calibration and test so distributional differences are preserved.
- Auxiliary arrays `predictor._X_ctx` and `predictor._S_ctx` are set by the caller before each featurize call, since `algorithm.py:_featurize_prefixes` only accepts Y prefixes.

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

### Medical (sepsis ICU)
- α=0.1, cal_frac=0.5, seed=42
- T=23 prediction steps (hours 0..22 predicting 1..23); 24 hourly timestamps
- gamma_grid=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2] (wider than finance because NaCl residual scale is much larger — hundreds of mL — and sparsity makes small γ more appropriate)
- Gamma re-selected every 5 steps (not 10 as in synthetic/finance)
- Single-seed only — no multi-seed wrapper currently exists
- `--n_traincal` / `--n_test` subsample full cohort for faster iteration

---

## Results currently available (post-cleanup, 2026-04-22)

All S&P 500 data files are in `data/`. Pass as `data/sp500_DATES.npz` to all scripts.

### Synthetic multi-seed (static-X, n_seeds=30, T=20) — interim, will be superseded by C1–C4
| File | Content |
|---|---|
| `results/results_algorithm_shift_20260414_103750.json` + PNG | AdaptedCAFHT, with shift |
| `results/results_algorithm_noshift_20260414_103748.json` + PNG | AdaptedCAFHT, no shift |
| `results/results_adaptive_shift_20260414_103752.json` + PNG | OnlineConformalPredictor, with shift |
| `results/results_adaptive_noshift_20260414_103751.json` + PNG | OnlineConformalPredictor, no shift |

### Finance (single-seed, seed=42, pending multi-window runs)
| File | Content |
|---|---|
| `results/finance_healthcare_shift.json` | Healthcare test, with LR weighting (pending E2 verification) |
| `results/covariate_shift.png` | Tech vs. non-Tech covariate KDE (motivating figure, needs PDF) |

All other results files were deleted (stale single-seed runs and duplicate configs).

### Medical (sepsis ICU)
No saved runs. `results/` currently contains zero files matching `medical_*`, `sepsis_*`, or `nacl_*`. All medical runs to date have been interactive / not committed.

---

## Experiment plan (agreed 2026-04-22)

### Finance experiments — run_finance_experiments.sh
Run Tech shift + no-shift on 13 .npz windows (all except sp500_20231004_20240328.npz).
Output naming: finance_tech_shift_DATES.json/.pdf and finance_tech_noshift_DATES.json/.pdf

NPZ files to use (all in data/):
  data/sp500_20240102_20240229.npz
  data/sp500_20240201_20240328.npz
  data/sp500_20240301_20240430.npz
  data/sp500_20240401_20240531.npz
  data/sp500_20240501_20240628.npz
  data/sp500_20240603_20240731.npz
  data/sp500_20240701_20240830.npz
  data/sp500_20240801_20240930.npz
  data/sp500_20240903_20241031.npz
  data/sp500_20241001_20241129.npz
  data/sp500_20241101_20241231.npz
  data/sp500_20241202_20250131.npz
  data/sp500_20250102_20250228.npz
Excluded: data/sp500_20231004_20240328.npz (long overlapping window).

Mixed-sector null baseline: data/sp500_20240201_20240328.npz only (illustrative).
  Outputs: finance_mixed_withweighting.json/.pdf, finance_mixed_noweighting.json/.pdf

### Synthetic experiments — run_synthetic_experiments.sh (Group C)
  C1: algorithm, static, with_shift     -> results/
  C2: algorithm, static, no shift       -> results/
  C3: adaptive, static, with_shift      -> results/
  C4: adaptive, static, no shift        -> results/
  C5: algorithm, dynamic, with_shift    (x_rate=0.6, x_rate_shift=0.9)
  C6: algorithm, dynamic, no shift      (x_rate=0.6)
All with --n_seeds 100, save JSON aggregation files.

### Medical experiments — NO shell runner yet
Minimum viable runs (single-seed, seed=42):
  M1: medical, with_shift
      python medical_conformal.py --pkl sepsis_experiment_data_nacl_target.pkl --with_shift \
          --save_json results/medical_nacl_shift.json \
          --save_plot results/medical_nacl_shift.pdf
  M2: medical, no shift
      python medical_conformal.py --pkl sepsis_experiment_data_nacl_target.pkl \
          --save_json results/medical_nacl_noshift.json \
          --save_plot results/medical_nacl_noshift.pdf
  M3: covariate-shift visualization
      python medical_data.py --pkl sepsis_experiment_data_nacl_target.pkl \
          --static --dynamic --target --save_plot results/medical_covariate_shift.pdf

---

## TO-DO: deferred experiments (do not run yet)

### Group D — Ablation
D1: LR weighting without ACI (gamma=0).
    Command: python multi_seed_experiments.py --predictor algorithm
             --covariate_mode static --with_shift --aci_stepsize 0.0
             --n_seeds 100 --save_dir results/
    Prerequisite: verify that aci_stepsize=0.0 is handled cleanly (alpha stays
    fixed = alpha throughout; no division by zero).
    Purpose: isolates contribution of LR weighting from ACI update.

### Group E — Covariate shift verification figures
E1: Technology vs. rest KDE + KL divergence
    Command: python plot_covariate_shift.py --npz data/sp500_20240201_20240328.npz
             --test_sector Technology --save results/covariate_shift_tech.pdf
E2: Healthcare vs. rest KDE + KL divergence
    Command: python plot_covariate_shift.py --npz data/sp500_20240201_20240328.npz
             --test_sector Healthcare --save results/covariate_shift_healthcare.pdf
    Note: run E2 and inspect KL values before including healthcare as a primary
    shift experiment. If all four KL values are near zero, demote to robustness
    condition only.

---

## Gaps before experimental section is paper-ready

### Must-have
1. **Dynamic-X simulation results** — run C5/C6 above.
2. **Finance results across all 13 windows** — run finance script above.
3. **Mixed-sector null baseline** — run B1/B2 above (illustrative, one window).
4. **JSON aggregation files for multi-seed synthetics** — needed for LaTeX tables.

### Deferred (Group D/E — see TO-DO above)
5. **Ablation: LR weighting without ACI** — Group D1.
6. **Healthcare covariate shift verification** — Group E2.
7. **Vector figure output** — all figures should be saved as PDF not PNG.

### Structural
8. **Medical data (Section 5.3)** — pipeline implemented (`medical_conformal.py`, `medical_data.py`, `sepsis_experiment_data_nacl_target.pkl`) but **no runs are saved** in `results/` and **no multi-seed wrapper / baselines / shell runner** exist yet. Technical-note section 5.3 still reads "To be added".
   Remaining work:
   - (a) Run M1/M2/M3 above and commit outputs.
   - (b) Build multi-seed wrapper for medical (seeds vary the Train/Cal split since the Train/Test split is fixed).
   - (c) Add baselines (plain split CP, `OnlineConformalPredictor` without LR weighting) — currently only Weighted CAFHT is run on this cohort.
   - (d) Consider LR-only (γ=0) ablation once (b) exists.
9. **Theoretical proofs** — guarantee "has yet to be finalized" per technical note.

---

## Paper-level framing (statistical language)

The calibration sample follows P = P_Z × P_{Y|Z}. Test units follow P̃ = P̃_Z × P_{Y|Z}. The method estimates dP̃_Z/dP_Z via logistic regression, reweights calibration scores, and applies per-series ACI to correct residual miscoverage. The cross-half test-split prevents data leakage into the classifier. Gamma is selected on a held-out training fold every 10 steps.
