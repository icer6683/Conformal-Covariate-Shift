#!/usr/bin/env bash
# Utilities shift + noshift with gamma grid {0.001, 0.005, 0.01, 0.05, 0.1}.
# Adds γ=0.1 as a larger-step option than the default grid.
# LRonly condition unchanged (forces γ=0).
#
# Outputs:
#   results/finance/json/finance_util_{shift,noshift}_g10_DATES.json
#   results/finance/pdf/finance_util_{shift,noshift}_g10_DATES.pdf

set -e
PY=/Users/andrewlou/opt/anaconda3/envs/boa/bin/python
DATA=finance/data
RESULTS=results/finance
GRID="0.001 0.005 0.01 0.05 0.1"

NPZ_FILES=(
    $DATA/sp500_20240102_20240229.npz
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

for NPZ in "${NPZ_FILES[@]}"; do
    if [ ! -f "$NPZ" ]; then
        echo "[SKIP] $NPZ"; continue
    fi
    DATES=$(basename "$NPZ" .npz); DATES="${DATES#sp500_}"
    echo ""
    echo "============================================================"
    echo "  Utilities (g10)  $DATES"
    echo "============================================================"

    echo "---- util shift g10  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" --test_sector Utilities --with_shift --gamma_grid $GRID --seed 42 \
        --save_json "$RESULTS/json/finance_util_shift_g10_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_util_shift_g10_${DATES}.pdf" \
        2>&1 | tail -3

    echo "---- util noshift g10  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" --test_sector Utilities --gamma_grid $GRID --seed 42 \
        --save_json "$RESULTS/json/finance_util_noshift_g10_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_util_noshift_g10_${DATES}.pdf" \
        2>&1 | tail -3
done

echo ""
echo "All utilities g10 experiments complete."
