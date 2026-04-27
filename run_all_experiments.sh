#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh
# =============================================================================
# Single entry point to (re-)run synthetic, finance, and medical experiments
# and rebuild the corresponding LaTeX tables.
#
# Usage:
#   ./run_all_experiments.sh --synthetic       # only synthetic (24 runs)
#   ./run_all_experiments.sh --finance         # only finance   (81 runs)
#   ./run_all_experiments.sh --medical         # only medical   (3 runs)
#   ./run_all_experiments.sh --all             # all three
#   ./run_all_experiments.sh --build-tables    # rebuild .tex from existing JSONs
#
# Flags can be combined, e.g. `--synthetic --build-tables`.
# Results JSONs are always re-written (no skip-if-exists). Tables land in
# results/<section>/tables/.
# =============================================================================

set -e
cd "$(dirname "$0")"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
RUN_SYN=0; RUN_FIN=0; RUN_MED=0; BUILD=0

if [ $# -eq 0 ]; then
    cat <<EOF
Usage: $0 [--synthetic] [--finance] [--medical] [--all] [--build-tables]
EOF
    exit 1
fi

for arg in "$@"; do
    case $arg in
        --synthetic)    RUN_SYN=1 ;;
        --finance)      RUN_FIN=1 ;;
        --medical)      RUN_MED=1 ;;
        --all)          RUN_SYN=1; RUN_FIN=1; RUN_MED=1 ;;
        --build-tables) BUILD=1 ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

PY="conda run -n boa python"

mkdir -p results/synthetic/json   results/synthetic/pdf   results/synthetic/tables
mkdir -p results/finance/json     results/finance/pdf     results/finance/tables
mkdir -p results/medical/json     results/medical/pdf     results/medical/tables

# -----------------------------------------------------------------------------
# Synthetic: 4 data conditions × 3 methods × 2 ar_coef = 24 runs (T=40)
# -----------------------------------------------------------------------------
run_synthetic_cell() {
    # $1 ar_tag (ar07|ar09)   $2 ar_coef
    # $3 data_tag (static_noshift|static_shift|dynamic_noshift|dynamic_shift)
    # $4 method_tag (full|uniform|zerog)
    # $5+ extra args
    local ar_tag=$1 ar_coef=$2 data_tag=$3 method_tag=$4
    shift 4
    local name="synth_${ar_tag}_${data_tag}_${method_tag}"
    local TMP
    TMP=$(mktemp -d)
    echo ""
    echo "============================================================"
    echo "  [synthetic] $name"
    echo "============================================================"
    $PY -m synthetic.multi_seed_experiments \
        --predictor algorithm \
        --n_seeds 30 \
        --T 40 \
        --ar_coef "$ar_coef" \
        --save_dir "$TMP" \
        "$@"
    mv "$TMP/json/"*.json "results/synthetic/json/${name}.json"
    mv "$TMP/pdf/"*.png   "results/synthetic/pdf/${name}.png"
    rm -rf "$TMP"
}

run_synthetic() {
    for ar_pair in "ar07 0.7" "ar09 0.9"; do
        ar_tag=${ar_pair%% *}
        ar_coef=${ar_pair##* }

        # static, no shift
        run_synthetic_cell "$ar_tag" "$ar_coef" static_noshift  full    --use_lr
        run_synthetic_cell "$ar_tag" "$ar_coef" static_noshift  uniform
        run_synthetic_cell "$ar_tag" "$ar_coef" static_noshift  zerog   --use_lr --aci_stepsize 0.0

        # static, shift
        run_synthetic_cell "$ar_tag" "$ar_coef" static_shift    full    --with_shift --use_lr
        run_synthetic_cell "$ar_tag" "$ar_coef" static_shift    uniform --with_shift
        run_synthetic_cell "$ar_tag" "$ar_coef" static_shift    zerog   --with_shift --use_lr --aci_stepsize 0.0

        # dynamic, no shift
        local DYN_NO="--covariate_mode dynamic --x_rate 0.6"
        run_synthetic_cell "$ar_tag" "$ar_coef" dynamic_noshift full    $DYN_NO --use_lr
        run_synthetic_cell "$ar_tag" "$ar_coef" dynamic_noshift uniform $DYN_NO
        run_synthetic_cell "$ar_tag" "$ar_coef" dynamic_noshift zerog   $DYN_NO --use_lr --aci_stepsize 0.0

        # dynamic, shift
        local DYN_SH="--covariate_mode dynamic --x_rate 0.6 --x_rate_shift 0.9"
        run_synthetic_cell "$ar_tag" "$ar_coef" dynamic_shift   full    $DYN_SH --with_shift --use_lr
        run_synthetic_cell "$ar_tag" "$ar_coef" dynamic_shift   uniform $DYN_SH --with_shift
        run_synthetic_cell "$ar_tag" "$ar_coef" dynamic_shift   zerog   $DYN_SH --with_shift --use_lr --aci_stepsize 0.0
    done
}

# -----------------------------------------------------------------------------
# Finance: 13 windows × {tech, util} × {full, uniform, zerog}  +  3 mixed = 81
#   Tech grid:  {0.001, 0.005, 0.01, 0.05}
#   Util grid:  {0.001, 0.005, 0.01, 0.05, 0.1}        (g10 grid)
#   Zero γ for either sector forces gamma_grid=[0.0]
# -----------------------------------------------------------------------------
TECH_GRID="0.001 0.005 0.01 0.05"
UTIL_GRID="0.001 0.005 0.01 0.05 0.1"

NPZ_FILES=(
    finance/data/sp500_20240102_20240229.npz
    finance/data/sp500_20240201_20240328.npz
    finance/data/sp500_20240301_20240430.npz
    finance/data/sp500_20240401_20240531.npz
    finance/data/sp500_20240501_20240628.npz
    finance/data/sp500_20240603_20240731.npz
    finance/data/sp500_20240701_20240830.npz
    finance/data/sp500_20240801_20240930.npz
    finance/data/sp500_20240903_20241031.npz
    finance/data/sp500_20241001_20241129.npz
    finance/data/sp500_20241101_20241231.npz
    finance/data/sp500_20241202_20250131.npz
    finance/data/sp500_20250102_20250228.npz
)
MIXED_NPZ=finance/data/sp500_20231004_20240328.npz   # long history for mixed mode

run_finance_window() {
    # $1 sector_tag (tech|util)  $2 sector_name  $3 grid  $4 method (full|uniform|zerog)
    # $5 npz file  $6 dates_tag  $7 file_suffix (e.g. "" for tech, "_g10" for util)
    local sector_tag=$1 sector_name=$2 grid=$3 method=$4 npz=$5 dates=$6 suffix=$7
    local stem="finance_${sector_tag}_${method}${suffix}_${dates}"
    local args=(--npz "$npz" --test_sector "$sector_name" --seed 42)
    case $method in
        full)    args+=(--with_shift --gamma_grid $grid) ;;
        uniform) args+=(--gamma_grid $grid) ;;
        zerog)   args+=(--with_shift --gamma_grid 0.0) ;;
    esac
    echo "  -- $stem --"
    $PY finance/finance_conformal.py "${args[@]}" \
        --save_json "results/finance/json/${stem}.json" \
        --save_plot "results/finance/pdf/${stem}.pdf" \
        2>&1 | tail -3
}

run_finance_mixed() {
    # $1 method (full|uniform|zerog)
    local method=$1
    local stem="finance_mixed_${method}"
    local args=(--npz "$MIXED_NPZ" --mixed --seed 42)
    case $method in
        full)    args+=(--with_shift --gamma_grid $TECH_GRID) ;;
        uniform) args+=(--gamma_grid $TECH_GRID) ;;
        zerog)   args+=(--with_shift --gamma_grid 0.0) ;;
    esac
    echo "  -- $stem --"
    $PY finance/finance_conformal.py "${args[@]}" \
        --save_json "results/finance/json/${stem}.json" \
        --save_plot "results/finance/pdf/${stem}.pdf" \
        2>&1 | tail -3
}

run_finance() {
    for npz in "${NPZ_FILES[@]}"; do
        if [ ! -f "$npz" ]; then
            echo "[SKIP missing npz] $npz"; continue
        fi
        local dates
        dates=$(basename "$npz" .npz); dates="${dates#sp500_}"
        echo ""
        echo "============================================================"
        echo "  [finance] window $dates"
        echo "============================================================"
        for method in full uniform zerog; do
            run_finance_window tech Technology "$TECH_GRID" "$method" "$npz" "$dates" ""
        done
        for method in full uniform zerog; do
            run_finance_window util Utilities  "$UTIL_GRID" "$method" "$npz" "$dates" "_g10"
        done
    done

    echo ""
    echo "============================================================"
    echo "  [finance] mixed (null baseline)"
    echo "============================================================"
    if [ ! -f "$MIXED_NPZ" ]; then
        echo "[SKIP missing npz] $MIXED_NPZ"
    else
        for method in full uniform zerog; do
            run_finance_mixed "$method"
        done
    fi
}

# -----------------------------------------------------------------------------
# Medical: 3 multi-seed runs (10 seeds, n_traincal=1000, n_test=500)
# -----------------------------------------------------------------------------
MED_PKL=medical/sepsis_experiment_data_nacl_target.pkl
MED_GAMMA="1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2"

run_medical() {
    if [ ! -f "$MED_PKL" ]; then
        echo "[medical] missing pickle: $MED_PKL"; return 1
    fi
    local common=(
        --pkl "$MED_PKL"
        --n_seeds 10
        --base_seed 1000
        --n_traincal 1000
        --n_test 500
    )

    echo ""
    echo "============================================================"
    echo "  [medical] full (LR + ACI)"
    echo "============================================================"
    $PY medical/multi_seed_medical.py "${common[@]}" \
        --with_shift --gamma_grid $MED_GAMMA \
        --save_json results/medical/json/medical_multi10_full.json \
        --save_plot results/medical/pdf/medical_multi10_full.pdf

    echo ""
    echo "============================================================"
    echo "  [medical] aci_only (uniform weights + ACI)"
    echo "============================================================"
    $PY medical/multi_seed_medical.py "${common[@]}" \
        --gamma_grid $MED_GAMMA \
        --save_json results/medical/json/medical_multi10_aci_only.json \
        --save_plot results/medical/pdf/medical_multi10_aci_only.pdf

    echo ""
    echo "============================================================"
    echo "  [medical] lr_only (γ=0)"
    echo "============================================================"
    $PY medical/multi_seed_medical.py "${common[@]}" \
        --with_shift --gamma_grid 0.0 \
        --save_json results/medical/json/medical_multi10_lr_only.json \
        --save_plot results/medical/pdf/medical_multi10_lr_only.pdf
}

# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------
[ $RUN_SYN -eq 1 ] && run_synthetic
[ $RUN_FIN -eq 1 ] && run_finance
[ $RUN_MED -eq 1 ] && run_medical

# Always rebuild tables for whatever sections were just run, plus on --build-tables.
if [ $BUILD -eq 1 ] || [ $RUN_SYN -eq 1 ] || [ $RUN_FIN -eq 1 ] || [ $RUN_MED -eq 1 ]; then
    echo ""
    echo "============================================================"
    echo "  Building LaTeX tables"
    echo "============================================================"
    SECTIONS=()
    [ $RUN_SYN -eq 1 ] || [ $BUILD -eq 1 ] && SECTIONS+=(synthetic)
    [ $RUN_FIN -eq 1 ] || [ $BUILD -eq 1 ] && SECTIONS+=(finance)
    [ $RUN_MED -eq 1 ] || [ $BUILD -eq 1 ] && SECTIONS+=(medical)
    $PY build_tex_tables.py --sections "${SECTIONS[@]}"
fi

echo ""
echo "Done."
