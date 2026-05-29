#!/usr/bin/env bash
# Runs Utilities sector experiments on all 13 windows:
#   - shift   (Weighted CAFHT: LR + ACI)
#   - noshift (AdaptedCAFHT: uniform weights + ACI)
#   - LRonly  (AdaptedCAFHT: LR + gamma=0)
#
# Outputs:
#   results/finance/json/finance_util_{shift,noshift,LRonly}_DATES.json
#   results/finance/pdf/finance_util_{shift,noshift,LRonly}_DATES.pdf

set -e
PY=/Users/andrewlou/opt/anaconda3/envs/boa/bin/python
DATA=finance/data
RESULTS=results/finance
mkdir -p "$RESULTS/json" "$RESULTS/pdf"

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
        echo "[SKIP] $NPZ not found"; continue
    fi
    DATES=$(basename "$NPZ" .npz); DATES="${DATES#sp500_}"
    echo ""
    echo "============================================================"
    echo "  Utilities  $DATES"
    echo "============================================================"

    echo "---- util shift  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" --test_sector Utilities --with_shift --seed 42 \
        --save_json "$RESULTS/json/finance_util_shift_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_util_shift_${DATES}.pdf" \
        2>&1 | tail -3

    echo "---- util noshift  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" --test_sector Utilities --seed 42 \
        --save_json "$RESULTS/json/finance_util_noshift_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_util_noshift_${DATES}.pdf" \
        2>&1 | tail -3

    echo "---- util LRonly (gamma=0)  $DATES ----"
    $PY finance/finance_conformal.py \
        --npz "$NPZ" --test_sector Utilities --with_shift --gamma_grid 0.0 --seed 42 \
        --save_json "$RESULTS/json/finance_util_LRonly_${DATES}.json" \
        --save_plot "$RESULTS/pdf/finance_util_LRonly_${DATES}.pdf" \
        2>&1 | tail -3
done

echo ""
echo "All utilities experiments complete."
