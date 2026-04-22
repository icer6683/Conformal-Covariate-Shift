#!/usr/bin/env bash
# =============================================================================
# run_finance_experiments.sh
# =============================================================================
# Runs AdaptedCAFHT on S&P 500 data across all date windows:
#   - Technology sector as test, WITH LR shift correction
#   - Technology sector as test, WITHOUT shift correction (uniform weights)
#
# Mixed-sector null baseline (illustrative, one window only):
#   - Random 15% of all tickers, WITH LR weighting
#   - Random 15% of all tickers, WITHOUT LR weighting
#
# Excluded: sp500_20231004_20240328.npz (long overlapping window)
#
# Outputs saved to: results/
#   finance_tech_shift_DATES.json/.pdf
#   finance_tech_noshift_DATES.json/.pdf
#   finance_mixed_withweighting.json/.pdf
#   finance_mixed_noweighting.json/.pdf
# =============================================================================

set -e
DATA=data
RESULTS=results
mkdir -p "$RESULTS"

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

# ── Tech shift + no-shift across all date windows ─────────────────────────────
for NPZ in "${NPZ_FILES[@]}"; do

    if [ ! -f "$NPZ" ]; then
        echo "[SKIP] $NPZ not found, skipping."
        continue
    fi

    # Extract date suffix: data/sp500_20240102_20240229.npz -> 20240102_20240229
    BASENAME=$(basename "$NPZ" .npz)   # sp500_20240102_20240229
    DATES="${BASENAME#sp500_}"         # 20240102_20240229

    echo ""
    echo "============================================================"
    echo "  Window: $DATES"
    echo "============================================================"

    # Tech WITH shift correction
    echo "  [1/2] Technology test, WITH LR weighting..."
    python finance_conformal.py \
        --npz "$NPZ" \
        --test_sector Technology \
        --with_shift \
        --seed 42 \
        --save_json "$RESULTS/finance_tech_shift_${DATES}.json" \
        --save_plot "$RESULTS/finance_tech_shift_${DATES}.pdf"

    # Tech WITHOUT shift correction
    echo "  [2/2] Technology test, uniform weights (no shift correction)..."
    python finance_conformal.py \
        --npz "$NPZ" \
        --test_sector Technology \
        --seed 42 \
        --save_json "$RESULTS/finance_tech_noshift_${DATES}.json" \
        --save_plot "$RESULTS/finance_tech_noshift_${DATES}.pdf"

done

# ── Mixed-sector null baseline (one window, illustrative) ─────────────────────
MIXED_NPZ=$DATA/sp500_20240201_20240328.npz

echo ""
echo "============================================================"
echo "  Mixed-sector null baseline  ($MIXED_NPZ)"
echo "============================================================"

echo "  [1/2] Mixed test set, WITH LR weighting..."
python finance_conformal.py \
    --npz "$MIXED_NPZ" \
    --mixed \
    --with_shift \
    --seed 42 \
    --save_json "$RESULTS/finance_mixed_withweighting.json" \
    --save_plot "$RESULTS/finance_mixed_withweighting.pdf"

echo "  [2/2] Mixed test set, uniform weights..."
python finance_conformal.py \
    --npz "$MIXED_NPZ" \
    --mixed \
    --seed 42 \
    --save_json "$RESULTS/finance_mixed_noweighting.json" \
    --save_plot "$RESULTS/finance_mixed_noweighting.pdf"

echo ""
echo "All finance experiments complete. Results in $RESULTS/"
