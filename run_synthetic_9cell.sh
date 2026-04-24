#!/usr/bin/env bash
# =============================================================================
# run_synthetic_9cell.sh
# =============================================================================
# 3 data conditions × 3 algorithm versions = 9 cells.
#
# Data conditions:
#   noshift      -- test data drawn from same distribution as train (no shift)
#   staticshift  -- test data has Poisson X with shifted rate (λ=1→2)
#   dynamicshift -- test data has dynamic X with shifted AR rate (ρ=0.6→0.9)
#
# Algorithm versions (all use AdaptedCAFHT):
#   LRandACI  -- LR weighting + ACI adaptive alpha  (full proposed method)
#   weightone -- uniform weights (w=1) + ACI         (ACI-only ablation)
#   LRonly    -- LR weighting, γ=0 (alpha fixed)    (LR-only ablation)
#
# Parameters: n_seeds=30, n_series=500, n_train=1000, n_cal=1000, T=40
# LR featurizer: rolling Y window of 5 steps, features = {mean, std, ar1}
# =============================================================================

set -e
cd "$(dirname "$0")"

SEEDS=30
SAVE=results/synthetic

CMD="conda run -n boa python -m synthetic.multi_seed_experiments \
    --n_seeds $SEEDS --predictor algorithm --save_dir $SAVE --ar_coef 0.9"

# ── 1. No-shift data ──────────────────────────────────────────────────────────
echo ""; echo "=== [1/9] noshift + LRandACI ==="; echo ""
$CMD --use_lr

echo ""; echo "=== [2/9] noshift + weightone ==="; echo ""
$CMD

echo ""; echo "=== [3/9] noshift + LRonly ==="; echo ""
$CMD --use_lr --aci_stepsize 0.0

# ── 2. Static-X shift ─────────────────────────────────────────────────────────
echo ""; echo "=== [4/9] staticshift + LRandACI ==="; echo ""
$CMD --with_shift --use_lr

echo ""; echo "=== [5/9] staticshift + weightone ==="; echo ""
$CMD --with_shift

echo ""; echo "=== [6/9] staticshift + LRonly ==="; echo ""
$CMD --with_shift --use_lr --aci_stepsize 0.0

# ── 3. Dynamic-X shift ────────────────────────────────────────────────────────
echo ""; echo "=== [7/9] dynamicshift + LRandACI ==="; echo ""
$CMD --with_shift --covariate_mode dynamic --x_rate 0.6 --x_rate_shift 0.9 --use_lr

echo ""; echo "=== [8/9] dynamicshift + weightone ==="; echo ""
$CMD --with_shift --covariate_mode dynamic --x_rate 0.6 --x_rate_shift 0.9

echo ""; echo "=== [9/9] dynamicshift + LRonly ==="; echo ""
$CMD --with_shift --covariate_mode dynamic --x_rate 0.6 --x_rate_shift 0.9 \
    --use_lr --aci_stepsize 0.0

echo ""
echo "All 9 cells complete. Results in $SAVE/"
