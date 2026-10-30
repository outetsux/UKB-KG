#!/usr/bin/env bash

# By default evaluate the same model as train.sh; ALL_MODELS=1 selects all six.

set -euo pipefail


# ====================== parameters ======================

DATA_TYPE="${DATA_TYPE:-ene}"
KGE="${KGE:-${UKB_KGE_MODEL:-TransE}}"
export DATA_TYPE UKB_KGE_MODEL="$KGE"
CONFIG_PATHS=$(python -m disease_prediction.config)
eval "$CONFIG_PATHS"
SELECTED_SAVE_DIR="$SAVE_DIR"
SELECTED_DATA_DIR="$DATA_DIR"

BINS=2
HIDDEN_SIZE=150
BATCH_SIZE=128
DEBUG=0

SAMPLE_IDX_DIR="${SAMPLE_IDX_DIR:-$RAW_DIR}"

MIN_VAL_POS=10

# Rare label fallback：
POOLED_MODE="rate"

SEED=1234

TEST_NAME="test_bin_${BINS}"


# ====================== common data ======================

VAL_FEATURE_PATH="${RAW_DIR}/features_val.npy"
TEST_FEATURE_PATH="${RAW_DIR}/features_test.npy"

TRAIN_LABEL_PATH="${RAW_DIR}/labels_train.npy"
VAL_LABEL_PATH="${RAW_DIR}/labels_val.npy"
TEST_LABEL_PATH="${RAW_DIR}/labels_test.npy"


# ====================== runs ======================

# Format：
# save_dir_name:data_type:KGE
#
# KGE = "-" indicates binary baseline

if [[ "${ALL_MODELS:-0}" == "1" ]]; then
    RUNS=(
        "nn5_binary:binary:-"
        "nn5_${TAG}_ComplEx_ene:ene:ComplEx"
        "nn5_${TAG}_RotatE_ene:ene:RotatE"
        "nn5_${TAG}_TransE_ene:ene:TransE"
        "nn5_${TAG}_HAKE_ene:ene:HAKE"
        "nn5_${TAG}_ModE_ene:ene:ModE"
    )
else
    RUNS=("$(basename "$SELECTED_SAVE_DIR"):${DATA_TYPE}:${KGE}")
fi


# ====================== helper functions ======================

check_file() {
    local file_path="$1"

    if [[ ! -f "$file_path" ]]; then
        echo "[ERROR] File not found: $file_path" >&2
        exit 1
    fi
}

check_dir() {
    local dir_path="$1"

    if [[ ! -d "$dir_path" ]]; then
        echo "[ERROR] Directory not found: $dir_path" >&2
        exit 1
    fi
}


# ====================== common input checks ======================

check_dir "$SAMPLE_IDX_DIR"
check_file "$TRAIN_LABEL_PATH"
check_file "$VAL_LABEL_PATH"
check_file "$TEST_LABEL_PATH"


# ====================== run all models ======================

for run in "${RUNS[@]}"; do

    IFS=':' read -r name data_type kge <<< "$run"

    SAVE_DIR="${SAVE_ROOT_DP}/${name}"
    CHECKPOINT="${SAVE_DIR}/best_classifier.pth.tar"
    LOG="${SAVE_DIR}/${TEST_NAME}.log"

    check_dir "$SAVE_DIR"
    check_file "$CHECKPOINT"

    if [[ "$data_type" == "binary" ]]; then

        check_file "$VAL_FEATURE_PATH"
        check_file "$TEST_FEATURE_PATH"

        FEATURE_ARGS=(
            --val_feature_path "$VAL_FEATURE_PATH"
            --test_feature_path "$TEST_FEATURE_PATH"
        )

    elif [[ "$data_type" == "ene" ]]; then

        if [[ "$kge" == "-" ]]; then
            echo "[ERROR] data_type='ene' requires a valid KGE name for ${name}" >&2
            exit 1
        fi

        DATA_DIR="$(dirname "$SELECTED_DATA_DIR")/data_${TAG}_${kge}"

        VAL_ENE_PATH="${DATA_DIR}/avr_ene_12r_val.npy"
        TEST_ENE_PATH="${DATA_DIR}/avr_ene_12r_test.npy"

        check_dir "$DATA_DIR"
        check_file "$VAL_ENE_PATH"
        check_file "$TEST_ENE_PATH"

        FEATURE_ARGS=(
            --val_ene_path "$VAL_ENE_PATH"
            --test_ene_path "$TEST_ENE_PATH"
        )

    else

        echo "[ERROR] Invalid data_type '${data_type}' for ${name}" >&2
        echo "[ERROR] Allowed values: binary, ene" >&2
        exit 1

    fi


    # ==================================================
    # Run information
    # ==================================================

    {
        echo "============================================================"
        echo "Start Time : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "Run        : ${name}"
        echo "Data Type  : ${data_type}"
        echo "KGE        : ${kge}"
        echo "Save Dir   : ${SAVE_DIR}"
        echo "============================================================"
    } | tee -a "$LOG"


    # Run test

    if python -u -m disease_prediction.test \
        --save_dir "$SAVE_DIR" \
        --test_name "$TEST_NAME" \
        --data_type "$data_type" \
        "${FEATURE_ARGS[@]}" \
        --train_label_path "$TRAIN_LABEL_PATH" \
        --val_label_path "$VAL_LABEL_PATH" \
        --test_label_path "$TEST_LABEL_PATH" \
        -hs "$HIDDEN_SIZE" \
        -bs "$BATCH_SIZE" \
        --bin_choice "$BINS" \
        --sample_idx_dir "$SAMPLE_IDX_DIR" \
        --min_val_pos "$MIN_VAL_POS" \
        --pooled_mode "$POOLED_MODE" \
        --seed "$SEED" \
        --debug "$DEBUG" \
        >> "$LOG" 2>&1
    then
        echo "-- ${name} done" | tee -a "$LOG"
    else
        status=$?
        echo "!! ${name} FAILED (exit ${status}), see ${LOG}" | tee -a "$LOG" >&2
        exit "$status"
    fi

done


echo "========== all runs finished =========="