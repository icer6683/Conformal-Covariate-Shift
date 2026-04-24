# Schema inconsistencies across saved JSONs

## Three distinct schemas

1. **Synthetic multi-seed** (top-level keys: `time_steps`, `n_seeds`, `config`, `by_time`, `overall`). 7 files in `results/synthetic/json/`.
2. **Finance experiment** (top-level keys: `coverage_by_time`, `width_by_time`, `overall_coverage`, `target_coverage`, `dates`, `gamma_opt_history`, `first_test_ticker`, `first_test_series`, `config`; optional `clf_prob1_mean_by_time`, `clf_prob1_std_by_time`). 37 files in `results/finance/json/`.
3. **Featurizer tuning** (top-level keys: `npz`, `test_sector`, `alpha`, `seed`, `no_shift`, `variants`). 5 files in `results/finance/json/`.

## Inconsistencies

### Finance experiment files missing `clf_prob1_mean_by_time` / `clf_prob1_std_by_time`

- `finance_healthcare_shift.json`
- `sector_separability_ks.json`

### Duplicate artifacts (` 2.json` suffix)

- None.

### Synthetic config field naming

- All 7 synthetic files use `config.n_series` (not `n_test`). Interpreted as test-set size.
- `n_train` and `n_cal` are stored as top-level config fields.

### Synthetic `overall` structure

- `results_adaptive_noshift_20260414_103751.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_adaptive_shift_20260414_103752.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamic_LRonly_20260423_220742.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamic_noshift_20260423_220707.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamic_shift_20260423_220748.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_LRandACI_20260424_110424.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_LRandACI_20260424_123626.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_LRandACI_20260424_133328.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_LRonly_20260424_110424.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_LRonly_20260424_133238.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_LRonly_20260424_133955.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_weightone_20260424_110416.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_weightone_20260424_132913.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_dynamicshift_weightone_20260424_133636.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_20260206_155850.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_20260414_103748.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_LRandACI_20260424_110409.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_LRandACI_20260424_115734.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_LRandACI_20260424_120440.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_LRonly_20260424_110413.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_LRonly_20260424_120333.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_LRonly_20260424_121137.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_weightone_20260424_110358.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_weightone_20260424_120026.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_noshift_weightone_20260424_120743.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_shift_20260206_160044.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_shift_20260210_121930.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_shift_20260414_103750.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_LRandACI_20260424_110415.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_LRandACI_20260424_120643.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_LRandACI_20260424_121546.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_LRonly_20260424_110422.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_LRonly_20260424_121425.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_LRonly_20260424_133005.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_weightone_20260424_110410.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_weightone_20260424_121025.json`: has `overall` + overall.coverage_se present + has `by_time`
- `results_algorithm_staticshift_weightone_20260424_132653.json`: has `overall` + overall.coverage_se present + has `by_time`

### Missing fields across all experiment JSONs (never populated)

- Joint coverage (per-series `prod_t 1[covered]`) — never computed.
- Cross-seed std of per-seed means (synthetic `overall.coverage_std` is pooled per-(series,t), not across seeds).
- Raw per-series coverage arrays are stored inside `by_time[t].coverage_history` (synthetic) and `coverage_by_time` aggregates only the fraction (finance), so reconstructing joint coverage is possible on synthetic but not directly on finance without reloading.

### Horizon field name

- Synthetic: `config.T`.
- Finance: `config.L` (number of time steps = 40; `coverage_by_time`/`dates`/`width_by_time` have length 39 = L−1).

### Predictor identifier

- Synthetic: `config.predictor` ∈ {`algorithm`, `adaptive`}. Files with `predictor=adaptive` are sliding-window split conformal (no ACI); CLAUDE.md is now updated to reflect this.
- Finance: no `config.predictor` field — method is inferred from filename (`shift`/`noshift`/`mixed`) and `config.with_shift` / `config.mixed` flags.

### Seeds

- Synthetic: `n_seeds` top-level (all 4 of the `20260414_*` files have `n_seeds=30`). Older `20260206`/`20260210` files also multi-seed.
- Finance: single `config.seed=42` (no multi-seed wrapper for finance).

### Width and coverage time series representation

- Synthetic: `by_time[t]` is a dict per time step with `coverage_rate`, `coverage_std`, `interval_width`, `width_std`, `alpha_mean`, `alpha_std`, `gamma_opt`, etc. Keyed by stringified t.
- Finance: flat lists `coverage_by_time[i]` / `width_by_time[i]` / `gamma_opt_history[i]` / `dates[i]`, length L−1.

### Sample-size drift across synthetic runs

The 7 synthetic files do NOT share common sample sizes:

- `results_adaptive_noshift_20260414_103751.json`: n_seeds=30, n_train=600, n_cal=600, n_series=300, T=20
- `results_adaptive_shift_20260414_103752.json`: n_seeds=30, n_train=600, n_cal=600, n_series=300, T=20
- `results_algorithm_dynamic_LRonly_20260423_220742.json`: n_seeds=100, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamic_noshift_20260423_220707.json`: n_seeds=100, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamic_shift_20260423_220748.json`: n_seeds=100, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_LRandACI_20260424_110424.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_LRandACI_20260424_123626.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_LRandACI_20260424_133328.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_LRonly_20260424_110424.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_LRonly_20260424_133238.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_LRonly_20260424_133955.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_weightone_20260424_110416.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_weightone_20260424_132913.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_dynamicshift_weightone_20260424_133636.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_20260206_155850.json`: n_seeds=10, n_train=1000, n_cal=1000, n_series=500, T=20
- `results_algorithm_noshift_20260414_103748.json`: n_seeds=30, n_train=600, n_cal=600, n_series=300, T=20
- `results_algorithm_noshift_LRandACI_20260424_110409.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_LRandACI_20260424_115734.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_LRandACI_20260424_120440.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_LRonly_20260424_110413.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_LRonly_20260424_120333.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_LRonly_20260424_121137.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_weightone_20260424_110358.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_weightone_20260424_120026.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_noshift_weightone_20260424_120743.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_shift_20260206_160044.json`: n_seeds=10, n_train=1000, n_cal=1000, n_series=500, T=20
- `results_algorithm_shift_20260210_121930.json`: n_seeds=1, n_train=1000, n_cal=1000, n_series=500, T=20
- `results_algorithm_shift_20260414_103750.json`: n_seeds=30, n_train=600, n_cal=600, n_series=300, T=20
- `results_algorithm_staticshift_LRandACI_20260424_110415.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_LRandACI_20260424_120643.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_LRandACI_20260424_121546.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_LRonly_20260424_110422.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_LRonly_20260424_121425.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_LRonly_20260424_133005.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_weightone_20260424_110410.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_weightone_20260424_121025.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40
- `results_algorithm_staticshift_weightone_20260424_132653.json`: n_seeds=30, n_train=1000, n_cal=1000, n_series=500, T=40

- Older `20260206` files: n_train=1000, n_cal=1000, n_series=500, n_seeds=10.
- Single-seed outlier `20260210_121930`: n_seeds=1 (not a multi-seed run).
- Newer `20260414` files (the current reference runs): n_train=600, n_cal=600, n_series=300, n_seeds=30.
- CLAUDE.md states the `20260414` runs supersede earlier ones, but all 7 files are still present in `results/synthetic/json/` without being moved or deleted.

### Finance horizon (`config.L`) varies across windows

`L` ranges from 39 to 44 across the 13 tech windows because trading-day counts differ per 2-month window (holidays, short months). `coverage_by_time`/`width_by_time`/`dates`/`gamma_opt_history` have length `L − 1`. This means 'mean width across windows' averages over different horizon lengths; per-time-step panels across windows do not line up in index.

### Units / semantics

- Synthetic target Y is on the raw AR(1) scale; widths (~1.28) are comparable to AR(1) residual std × q_{0.9} ≈ 2·0.65 = 1.30 when noise_std≈0.2, ar_coef=0.7.
- Finance target Y is intraday return (Close/Open − 1) in decimal units; widths (~0.04–0.06) are comparable across tickers but are absolute returns, not bps.

