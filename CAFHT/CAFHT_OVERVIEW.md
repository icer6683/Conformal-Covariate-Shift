# Project: CAFHT — Conformalized Adaptive Forecasting of Heterogeneous Trajectories

## What this project is

A research codebase for **CAFHT**, a conformal prediction method that produces *simultaneous* forecasting bands guaranteed to cover the *entire path* of a new random trajectory with high probability. The motivation is motion-planning applications in which different objects have different levels of intrinsic unpredictability (heteroscedasticity across trajectories) and forecasts must remain valid jointly across the prediction horizon.

Central design idea:

- A black-box sequence model (LSTM) produces point trajectory predictions.
- An online single-series conformal updater (ACI or PID) maintains per-step prediction sets.
- A second, outer conformal layer calibrates a *trajectory-level* scaling factor over a calibration set to obtain a joint coverage guarantee over the whole horizon.
- The online step-size γ of the inner updater is selected from a grid to minimize calibrated band width.

Target guarantee: P(Y^{n+1}_t ∈ Ĉ_t, ∀t ∈ [T]) ≥ 1 − α, with finite-sample correctness via either data splitting or a theoretical (DKW/Markov hybrid) correction.

Accompanying paper: *Conformalized Adaptive Forecasting of Heterogeneous Trajectories*.

---

## Repository map

All paths below are relative to the `CAFHT/` root.

### Core library (`ConformalizedTS/`)
| File | Role |
|---|---|
| `ConformalizedTS/methods.py` | All conformal methods: `Adaptive_Conformal_Inference`, `Conformal_PID_control`, `Split_Conformal` (CFRNN-style baseline), `Max_calibrate` (NCTP-style normalized baseline), `CAFHT` (the proposed method) |
| `ConformalizedTS/black_box.py` | `Blackbox` class — wraps a PyTorch network with a training loop (`full_train`, `train_single_epoch`) and two inference modes (`predict_single`, `predict_iterate`) |
| `ConformalizedTS/networks.py` | Two LSTM architectures: `SimpleLSTM` (univariate; unsqueezes input) and `MyLSTM` (multivariate; preserves last-dim) |
| `ConformalizedTS/evals.py` | `evaluation` (1D) and `evaluation_multivariate` (multi-D) — compute simultaneous coverage, K-times-most coverage, local sliding-window coverage, average coverage, and mean band size; also `sliding_window_average` helper |
| `ConformalizedTS/utils.py` | `TSDataset` (PyTorch dataset), `split_train_sequence` (x/y autoregressive split), `trimming`, `plot_loss`, `plot_PI_for_single_seq`, `saturation_fn_log` + `mytan` (used by PID integrator) |

### Third-party theoretical bounds (`third_party/`)
| File | Role |
|---|---|
| `third_party/theory.py` | DKW, Markov, and hybrid concentration bounds and their inverses. `inv_hybrid(T, n, α)` returns the corrected miscoverage level used by CAFHT's `theoretical_correction` mode |

### Experiments (`experiments/`)
| File | Role |
|---|---|
| `experiments/data_gen.py` | `data_gen` class — synthetic AR(p) data with heterogeneous noise (`generate_AR`), plus AR + seasonality, random peaks, and volatility clustering variants (extra DGPs are coded but commented out in the runners) |
| `experiments/ts_sim.py` | Single-run synthetic experiment: parses 11 CLI args, generates AR data with heteroscedastic noise (φ=[0.9, 0.1, −0.2]), trains a 4-layer LSTM (hidden=128), then evaluates CFRNN + NCTP + 12 CAFHT variants (3 calibration modes × 2 base updaters × {fixed, adaptive}) |
| `experiments/ts_realdata.py` | Same pipeline applied to the pedestrian-trajectory dataset under `experiments/realdata/pedestrian_data/`; takes 6 CLI args (no horizon / data_model / ndim / delta-test — these are derived from the data) |
| `experiments/ts_sim.sh` | One-shot SLURM-worker shell wrapper that loads CUDA/CuDNN modules, activates the `expt` conda env, and calls `python3 ts_sim.py "$@"` |
| `experiments/ts_realdata.sh` | Same wrapper for `ts_realdata.py` |
| `experiments/submit_ts_sim.sh` | SLURM job-array submitter for `ts_sim.sh`. Encodes 7 numbered configurations (CONF=0..6): vary n_data, horizon, ndim, delta, delta_test, noise_level. Skips jobs whose output file already exists |
| `experiments/submit_ts_realdata.sh` | SLURM job-array submitter for `ts_realdata.sh`. Two configurations (CONF=0..1): vary n_data or noise_level |
| `experiments/realdata/pedestrian_data/x_value.pkl`, `y_value.pkl` | Pre-extracted pedestrian-trajectory pickle (nested dicts → (x, y) per pedestrian); loaded by `ts_realdata.py` |
| `experiments/models/`, `experiments/results/`, `experiments/logs/` | Output directories for trained checkpoints, CSV results, and SLURM stdout/stderr respectively (currently contain only `.DS_Store`) |

### Demo / animation assets (`media/`)
| File | Role |
|---|---|
| `media/example_usage.py` | Standalone end-to-end demo: same pipeline as `ts_realdata.py` but accepts an optional custom dataset path as a 7th arg; written so a user can drop in a new pickle and rerun |
| `media/animations.py` | 2D matplotlib animation (`FuncAnimation`) showing actual vs predicted positions of several objects with circular uncertainty regions; an additional "robot" object plans paths that avoid all the circles |
| `media/animation_3d.py` | 3D variant of the above: real X/Y data plus simulated sinusoidal Z, with disk uncertainty regions and a robot navigating bottom-left to top-right via A*-style heap search |
| `media/animation_rectangle.py` | Same as `animations.py` but with rectangular (axis-aligned) uncertainty regions instead of circles |
| `media/*.mp4` | Pre-rendered output animations (`animation_v1.mp4`, `animation_v2.mp4`, `animation_3d.mp4`) |
| `media/{adult,kid,robot,robot_smiling}*.png` | Optional image markers used when `--use_image` is passed to the animation scripts |
| `media/example_results/` | One sample text result (`n1000_modped_..._media_demo.txt`) used as input to the animation demos |
| `media/models/real_data/...` | Cached trained model directory referenced by the demo |

---

## Methods implemented (`ConformalizedTS/methods.py`)

### `Adaptive_Conformal_Inference` (ACI)
- Per-series online updater. At each step t:
  - score s_t = ‖ŷ_t − y_t‖_∞
  - if α_t ∈ (0,1): pred = ŷ_t ± Quantile_{1−α_t}(recent scores)
  - update α_{t+1} = α_t + γ·(α − err_t), where err_t = 1[s_t > q_{1−α_t}]
- Supports a "warm start" phase that synthesizes a small uniform-noise burn-in trajectory per test series so the score quantile is non-empty at t=0.

### `Conformal_PID_control` (PID)
- Same online setting, but the threshold q_t is updated via a PID-style rule (P-term: gradient on coverage; I-term: integrated under-coverage transformed through a log-tangent saturation `saturation_fn_log` from `utils.py`).
- `proportional_lr=True` rescales γ by the running score range; `Csat`, `KI`, `integrate` control the integrator.

### `Split_Conformal` (CFRNN baseline)
- Static split-conformal across the horizon. With `bonf_correction=True`, uses level `(1 − α/T)(1 + 1/n)` per step (Bonferroni-corrected for the T steps).

### `Max_calibrate` (NCTP baseline)
- Normalizes per-step scores by a fixed per-step scaling vector (`normalize`), takes the max across the horizon, then quantiles the resulting per-trajectory scores. Produces bands that are *normalized-scaled* but constant in the calibrated quantile across steps.

### `CAFHT` (proposed method)
- Wraps either ACI or PID as `base_method`.
- `adaptive=True`: scales the *width* of the base band by a multiplicative scalar before/after calibration. `adaptive=False`: additive shift.
- `nonconf_scores`: trajectory-level score. Adaptive: max over the horizon of the violation-to-width ratio (ℓ∞ over dim). Non-adaptive: max additive overshoot.
- `calibrate`: runs the base updater on a calibration split, computes per-trajectory scores, takes the (1−α)(1+1/n)-quantile.
- `select_gamma`: for each γ in `gamma_grid`, runs `calibrate` then constructs bands; picks the γ minimizing mean band width.
- `predict_bands(method, ...)` supports three calibration modes:
  - `'data_splitting'` — split calib into two halves: half-1 for γ selection, half-2 for the trajectory-level quantile. This is the recommended mode in the paper.
  - `'theoretical_correction'` — uses `inv_hybrid(n_gamma, n_cal, α)` from `third_party/theory.py` to inflate α so a *single* calibration set can be reused for both γ selection and quantile estimation, with a closed-form coverage bound.
  - `'naive'` — reuses the same calibration set for both steps without correction (for comparison only; not valid).

---

## Data-generating processes (`experiments/data_gen.py`)

### Heterogeneous AR(p)
- Order p=3 with φ=[0.9, 0.1, −0.2] (set in `ts_sim.py`).
- Each sequence i draws u_i ~ U(0,1); if u_i ≤ δ the noise variance is multiplied by `noise_level` (default 10), otherwise variance 1 — this defines "hard" vs "easy" trajectories.
- `noise_profile='static'`: constant variance over time.
- `noise_profile='dynamic'`: variance grows linearly in t.
- `delta` controls the fraction of "hard" sequences in train+calib; `delta_test` controls the fraction in test (varies independently to study mismatch).

### Extra DGPs (defined but not run)
- `generate_AR_seasonality`: superimposes a sine wave; hetero version uses 10× amplitude.
- `generate_AR_random_peaks`: random ± spikes at random positions; hetero version uses 10× amplitude.
- `generate_AR_volClus`: localized volatility clusters of random width.

(All three are present in `data_gen.py` and referenced in commented-out blocks of `ts_sim.py`.)

### Pedestrian real data
- Stored as nested dicts in `experiments/realdata/pedestrian_data/{x,y}_value.pkl`.
- Loader picks the inner-most leaf per pedestrian, takes window `[p_len : 2*p_len]` (p_len=20 → horizon=19), shuffles, and splits.
- Optional heterogeneous noise injection at scale `noise_level/200`, applied to the fraction δ=0.1 of trajectories.
- Last 291 trajectories form the test set.

---

## Black-box predictor (`ConformalizedTS/black_box.py`)

`Blackbox(net, device, train_loader, batch_size, max_epoch, learning_rate, criterion, optimizer, val_loader=None, verbose=True)`:
- `train_single_epoch()`: standard zero-grad / forward / loss / backward / step.
- `full_train(save_dir, model_name)`: runs `max_epoch` epochs; tracks train (and val) loss; saves `{'stats', 'model_state'}` via `torch.save`.
- `predict_single(test_loader, horizon)`: returns the last `horizon` predictions, flattened (used for single-step setups).
- `predict_iterate(test_loader, horizon, y_trim, ndim)`: per-trajectory iterative forecasting; supports optional `[lo, hi]` clipping. This is the routine used by both experiment scripts.

Networks (`networks.py`):
- `SimpleLSTM` adds a ReLU and unsqueezes the last dim — used when input is univariate.
- `MyLSTM` preserves the input shape and is used in the multivariate experiments.

---

## Experiment runners

### `experiments/ts_sim.py` — synthetic
CLI args (11, in order): `n_train_calib lr n_epoch seed horizon data_model noise_profile ndim noise_level delta delta_test`.

Pipeline:
1. Generate train/calib (75/25 split) and test (n=500) trajectories with `data_gen.generate_AR`.
2. Scale data to [−1, 1] (`y_trim=[-1,1]`).
3. Train `MyLSTM` (hidden=128, num_layers=4) with AdamW (lr from arg, weight_decay=1e-5) and MSE loss for `n_epoch` epochs.
4. Iteratively forecast horizon-long trajectories for calib and test.
5. Estimate initial quantile `q0` from training residuals.
6. Run **CFRNN**, **NCTP**, and **12 CAFHT variants** = {data_splitting, theoretical_correction, naive} × {ACI, PID} × {adaptive=True, adaptive=False}.
7. Evaluate every variant with `evaluation_multivariate`, save a CSV to `results/simulated_data/n{N}_h{H}_..._.txt`, delete the cached model.

`gamma_grid` is fixed: `np.concatenate([np.arange(0.001, 0.1, 0.01), np.arange(0.2, 1.1, 0.1)])` (≈ 19 values).

### `experiments/ts_realdata.py` — pedestrian
CLI args (6): `n_train_calib lr n_epoch seed noise_profile noise_level`. Horizon is fixed at 19, ndim=2 (x and y position).

Pipeline identical to synthetic, except step 1 loads the pedestrian pickles and optionally injects heteroscedastic noise. Output CSV under `results/real_data/`.

### `media/example_usage.py`
A 7-arg variant of `ts_realdata.py` that accepts an optional custom dataset path. Otherwise identical pipeline. Saves its result CSV under `media/example_results/`.

---

## Metrics (`ConformalizedTS/evals.py`)

`evaluation_multivariate(test_pred, test_true, PI, method, data_gen, hard_idx, easy_idx)` returns a single-row DataFrame with:
- `Simutaneous coverage`: fraction of trajectories covered at *every* step in *every* dimension (the paper's primary metric).
- `Conditional coverage-hard` / `Conditional coverage-easy`: same, restricted to the high-noise / low-noise sub-population.
- `Average coverage`: mean per-step coverage.
- `Size`: mean band width across steps and dims.

`evaluation` is the 1-D analogue and adds `K-times-most coverage` (fraction of trajectories covered at all but ≤K steps) and `local (sliding window) coverage`.

---

## Theoretical correction (`third_party/theory.py`)

For mode `'theoretical_correction'` in `CAFHT.predict_bands`, the corrected miscoverage level is computed from:
- `DKW_bound(T, n, α)` / `inv_DKW`: DKW-based concentration bound on the empirical quantile after model selection across T candidates.
- `Markov_bound(T, n, α, b=100)` / `inv_Markov`: Markov-style bound via the Beta CDF.
- `hybrid_bound` / `inv_hybrid`: takes the max of the two — this is what `methods.py` actually calls (line 437).

The corrected α is then passed back into `CAFHT.predict_bands` via `self.alpha = alpha2` before `select_gamma` and `predict_bands_subroutine`.

---

## Defaults at a glance

| Setting | Value |
|---|---|
| α (target miscoverage) | 0.1 |
| LSTM hidden / layers | 128 / 4 |
| Batch size | 20 |
| Train/calib split | 75 / 25 of `n_train_calib` |
| n_test (synthetic) | 500 |
| n_test (pedestrian) | 291 |
| Horizon (synthetic default) | 100 |
| Horizon (pedestrian) | 19 (p_len=20) |
| φ (AR coefficients) | [0.9, 0.1, −0.2] |
| Noise level (default) | 10× for the "hard" sub-population |
| `gamma_grid` | concatenation of `np.arange(0.001, 0.1, 0.01)` and `np.arange(0.2, 1.1, 0.1)` |
| Optimizer | AdamW, weight_decay=1e-5 |

---

## How experiments are launched

The shell scripts assume a SLURM cluster with a `gcc/8.3.0 + cuda/11.2.0 + cudnn` module stack and a conda environment named `expt`.

```bash
# Synthetic — pick one of CONF=0..6 inside submit_ts_sim.sh, then:
cd experiments
bash submit_ts_sim.sh

# Pedestrian real data — pick CONF=0 or 1 inside submit_ts_realdata.sh, then:
bash submit_ts_realdata.sh
```

The submitter scripts skip jobs whose expected output CSV is already present in `results/`, so re-runs are idempotent.

For a one-off interactive run, the underlying `ts_sim.sh` / `ts_realdata.sh` workers can be invoked directly with their positional args.

---

## Prerequisites (from upstream README)

Library: `numpy`, `scipy`, `sklearn`, `torch`, `random`, `pathlib`, `tqdm`, `math`, `pandas`, `matplotlib`, `statsmodels`.
Experiments additionally: `shutil`, `tempfile`, `pickle`, `sys`, `os`.

---

## Relationship to the surrounding repo

This `CAFHT/` directory is the original public release of the CAFHT method by FionaZ3696. It is the *base* method that the parent project (`Conformal-Covariate-Refactor/`) extends to handle **covariate shift** via likelihood-ratio reweighting (see `core/algorithm.py:AdaptedCAFHT` in the parent repo). The CAFHT γ-selection and ACI loop here are the same mechanism reimplemented (with extensions) in `AdaptedCAFHT`; the `Adaptive_Conformal_Inference`-style per-series α_t update is the direct ancestor of the ACI step described in the parent project's `CLAUDE.md`.
