#!/bin/bash

set -e
set -o pipefail

# -----------------------
# Configuration
# -----------------------
LIGHTGBM_BIN=../LightGBM/lightgbm

TRAIN_SET=data/mslr.train
VAL_SET=data/mslr.val
TEST_SET=data/mslr.test
RESULTS_FOLDER=results/sampling

SGB_FILE=configs/mslr/sgb.conf
LGB_FILE=configs/mslr/lightgbm.conf

mkdir -p "$RESULTS_FOLDER"

# Arrays must have the SAME length
SAMPLING_RATIOS=(0.1 0.15 0.2 0.25 0.3 0.35 0.4)
A_VALUES=(0.05 0.05 0.1 0.15 0.2 0.25 0.15)
B_VALUES=(0.05 0.1 0.1 0.1 0.1 0.1 0.25)

# Sanity check
if [ ${#SAMPLING_RATIOS[@]} -ne ${#A_VALUES[@]} ] || \
   [ ${#A_VALUES[@]} -ne ${#B_VALUES[@]} ]; then
  echo "ERROR: Array lengths do not match!"
  exit 1
fi

# -----------------------
# Run experiments
# -----------------------
for i in "${!SAMPLING_RATIOS[@]}"; do
  rate=${SAMPLING_RATIOS[$i]}
  A_val=${A_VALUES[$i]}
  B_val=${B_VALUES[$i]}

  tag=${rate//./_}

  sgb_train_log="$RESULTS_FOLDER/sgb_${tag}_train.log"
  sgb_test_log="$RESULTS_FOLDER/sgb_${tag}_test.log"
  sgb_model="$RESULTS_FOLDER/sgb_${tag}.txt"

  lgb_train_log="$RESULTS_FOLDER/lgb_${tag}_train.log"
  lgb_test_log="$RESULTS_FOLDER/lgb_${tag}_test.log"
  lgb_model="$RESULTS_FOLDER/lgb_${tag}.txt"

  echo "============================================"
  echo "Experiment $i"
  echo "  bagging_fraction = $rate"
  echo "  top_rate         = $A_val"
  echo "  other_rate       = $B_val"
  echo "============================================"

  # -------------------------------------------------
  # SGB — TRAIN
  # -------------------------------------------------
  "$LIGHTGBM_BIN" \
    config="$SGB_FILE" \
    data="$TRAIN_SET" \
    valid=$TEST_SET \
    objective=lambdarank \
    bagging_fraction="$rate" \
    metric=ndcg \
    ndcg_eval_at=1,3,5,10 \
    2>&1 | tee "$sgb_train_log"

  # -------------------------------------------------
  # GOSS — TRAIN
  # -------------------------------------------------
  "$LIGHTGBM_BIN" \
    config="$LGB_FILE" \
    data="$TRAIN_SET" \
    valid=$TEST_SET \
    objective=lambdarank \
    top_rate="$A_val" \
    other_rate="$B_val" \
    metric=ndcg \
    ndcg_eval_at=1,3,5,10 \
    2>&1 | tee "$lgb_train_log"

done

echo "All experiments completed!"
echo "Results saved in $RESULTS_FOLDER/"
