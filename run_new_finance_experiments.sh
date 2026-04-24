#!/usr/bin/env bash
# Runs the new finance experiments requested 2026-04-23:
#   (1) Tech LR-only ablation (with_shift, gamma=0) across the 12 remaining
#       tech windows (window 20240102_20240229 already run as smoke test).
#   (2) Healthcare shift + noshift on 2 windows:
#         20240201_20240328 (adds a noshift to match existing shift file)
#         20240701_20240830 (fresh pair)
#
# Outputs:
#   results/finance/json/finance_tech_LRonly_DATES.json
#   results/finance/pdf/finance_tech_LRonly_DATES.pdf
#   results/finance/json/finance_healthcare_{shift,noshift}_DATES.json
#   results/finance/pdf/finance_healthcare_{shift,noshift}_DATES.pdf

set -e
PY=/Users/andrewlou/opt/anaconda3/envs/boa/bin/python
DATA=finance/data
RESULTS=results/finance
mkdir -p "$RESULTS/json" "$RESULTS/pdf"

TECH_NPZ_FILES=(
    $DATA/sp500_20240201_20240328.npz
    $DATA/sp500_20240301_20240430.npz
    $DATA/sp500_20240401_20240531.npz
    $DATA/sp500_20240501_20240628.npz
    $DATA/sp500_20240603_20240731.npz
    $DATA/sp500_20240701_20240830.npz
    $DATA/sp500_20240801_20240930.npz
    $DATA/sp500_20240903_20241031.npz
    $DATA/sp500_20241001_20241129.npz
    $DATA/sp500_20241101_20241231.npz
    $DATA/sp500_20241202_20250131.npz
    $DATA/sp500_20250102_20250228.npz
)

HC_NPZ_FILES=(
    $DATA/sp500_20240201_20240328.npz
    $DATA/sp500_20240701_20240830.npz
)

echo "### Tech LR-only (gamma=0) on 12 windows ###"
for NPZ in "${TECH_NPZ_FILES[@]}"; do
    if [ ! -f "$NPZ" ]; then
        echo "[SKIP] $NPZ not found"; continue
    fi
    DATES=$(basename "$NPZ" .npz); DATES="${DATES#sp500_}"
    echo ""
    echo "---- Tech LR-only  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" \
        --test_sector Technology \
        --with_shift \
        --gamma_grid 0.0 \
        --seed 42 \
        --save_json "$RESULTS/json/finance_tech_LRonly_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_tech_LRonly_${DATES}.pdf" \
        2>&1 | tail -4
done

echo ""
echo "### Healthcare shift + noshift on 2 windows ###"
for NPZ in "${HC_NPZ_FILES[@]}"; do
    if [ ! -f "$NPZ" ]; then
        echo "[SKIP] $NPZ not found"; continue
    fi
    DATES=$(basename "$NPZ" .npz); DATES="${DATES#sp500_}"

    echo ""
    echo "---- Healthcare shift  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" \
        --test_sector Healthcare \
        --with_shift \
        --seed 42 \
        --save_json "$RESULTS/json/finance_healthcare_shift_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_healthcare_shift_${DATES}.pdf" \
        2>&1 | tail -4

    echo ""
    echo "---- Healthcare noshift  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" \
        --test_sector Healthcare \
        --seed 42 \
        --save_json "$RESULTS/json/finance_healthcare_noshift_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_healthcare_noshift_${DATES}.pdf" \
        2>&1 | tail -4
done

echo ""
echo "All new finance experiments complete."
