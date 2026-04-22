#!/usr/bin/env bash
# =============================================================================
# run_synthetic_experiments.sh
# =============================================================================
# Runs multi-seed synthetic experiments (Group C).
#
# Conditions:
#   C1: AdaptedCAFHT,           static-X,  with covariate shift
#   C2: AdaptedCAFHT,           static-X,  no shift
#   C3: OnlineConformalPredictor, static-X, with covariate shift
#   C4: OnlineConformalPredictor, static-X, no shift
#   C5: AdaptedCAFHT,           dynamic-X, with covariate shift
#   C6: AdaptedCAFHT,           dynamic-X, no shift
#
# Dynamic-X shift parameters: x_rate=0.6 (calibration), x_rate_shift=0.9 (test)
# All other parameters: package defaults (n_train=1000, n_cal=1000, n_test=500,
#   T=40, alpha=0.1, ar_coef=0.7, beta=1.0, noise_std=0.2)
#
# Outputs saved to: results/
#   results_{predictor}_{shift}_{timestamp}.json
#   plots_{predictor}_{shift}_{timestamp}.pdf    <-- NOTE: currently saves PNG;
#                                                    update plot code to use .pdf
# =============================================================================

set -e
RESULTS=results
SEEDS=100
mkdir -p "$RESULTS"

echo "============================================================"
echo "  C1: AdaptedCAFHT, static-X, WITH shift"
echo "============================================================"
python multi_seed_experiments.py \
    --predictor algorithm \
    --covariate_mode static \
    --with_shift \
    --n_seeds $SEEDS \
    --save_dir "$RESULTS"

echo ""
echo "============================================================"
echo "  C2: AdaptedCAFHT, static-X, NO shift"
echo "============================================================"
python multi_seed_experiments.py \
    --predictor algorithm \
    --covariate_mode static \
    --n_seeds $SEEDS \
    --save_dir "$RESULTS"

echo ""
echo "============================================================"
echo "  C3: OnlineConformalPredictor, static-X, WITH shift"
echo "============================================================"
python multi_seed_experiments.py \
    --predictor adaptive \
    --covariate_mode static \
    --with_shift \
    --n_seeds $SEEDS \
    --save_dir "$RESULTS"

echo ""
echo "============================================================"
echo "  C4: OnlineConformalPredictor, static-X, NO shift"
echo "============================================================"
python multi_seed_experiments.py \
    --predictor adaptive \
    --covariate_mode static \
    --n_seeds $SEEDS \
    --save_dir "$RESULTS"

echo ""
echo "============================================================"
echo "  C5: AdaptedCAFHT, dynamic-X, WITH shift"
echo "    x_rate=0.6 (calibration) -> x_rate_shift=0.9 (test)"
echo "============================================================"
python multi_seed_experiments.py \
    --predictor algorithm \
    --covariate_mode dynamic \
    --with_shift \
    --x_rate 0.6 \
    --x_rate_shift 0.9 \
    --n_seeds $SEEDS \
    --save_dir "$RESULTS"

echo ""
echo "============================================================"
echo "  C6: AdaptedCAFHT, dynamic-X, NO shift"
echo "    x_rate=0.6 (same for calibration and test)"
echo "============================================================"
python multi_seed_experiments.py \
    --predictor algorithm \
    --covariate_mode dynamic \
    --x_rate 0.6 \
    --n_seeds $SEEDS \
    --save_dir "$RESULTS"

echo ""
echo "All synthetic experiments complete. Results in $RESULTS/"
