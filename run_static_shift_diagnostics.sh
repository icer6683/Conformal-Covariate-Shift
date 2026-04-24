#!/usr/bin/env bash
# =============================================================================
# run_static_shift_diagnostics.sh
# =============================================================================
# Compares static shift signal strength under three parameter settings.
# Runs only the two most informative static-shift cells (LRandACI vs weightone)
# at 10 seeds each — fast enough to guide parameter selection.
#
# Variant A (baseline): covar_rate_shift=2.0, ar_coef=0.7  (current defaults)
# Variant B (larger shift): covar_rate_shift=4.0, ar_coef=0.7
# Variant C (alpha→1): covar_rate_shift=2.0, ar_coef=0.9
# =============================================================================

set -e
cd "$(dirname "$0")"

SEEDS=10
SAVE=results/synthetic

BASE="conda run -n boa python -m synthetic.multi_seed_experiments \
    --n_seeds $SEEDS --predictor algorithm --save_dir $SAVE \
    --with_shift --covariate_mode static"

echo ""; echo "=== Variant A: baseline (shift λ=2, α=0.7) ==="
$BASE --use_lr
$BASE

echo ""; echo "=== Variant B: larger shift (shift λ=4, α=0.7) ==="
$BASE --covar_rate_shift 4.0 --use_lr
$BASE --covar_rate_shift 4.0

echo ""; echo "=== Variant C: α closer to 1 (shift λ=2, α=0.9) ==="
$BASE --ar_coef 0.9 --use_lr
$BASE --ar_coef 0.9

echo ""; echo "Diagnostics complete. Results in $SAVE/"
