#!/bin/bash

set -e
set -o pipefail

# Configuration
LIGHTGBM_BIN=../LightGBM/lightgbm
DATASET=mslr
TRAIN_SET=data/${DATASET}.train
VAL_SET=data/${DATASET}.val
TEST_SET=data/${DATASET}.test
SPARSE=false
RESULTS_FOLDER=results/${DATASET}

# Common model config
OBJECTIVE=lambdarank
OBJECTIVE_XGB=rank:ndcg
NTHREADS=1

MIN_DATA=1
LR=0.1
MIN_HESSIAN=100
ROUNDS=100
LEARNER=serial

# Create results folder if it doesn't exist
mkdir -p "$RESULTS_FOLDER"

# Array of configurations to test
LGB_FILES=(
  # configs/${DATASET}/efb_only.conf
  configs/${DATASET}/lgb_baseline.conf
  # configs/${DATASET}/lightgbm.conf
  configs/${DATASET}/sgb.conf
)

XGB_FILES=(
  # configs/${DATASET}/xgb_exa.conf
  # configs/${DATASET}/xgb_his.conf
)

# Run LightGBM
echo "=== Running LightGBM experiments (train + test eval) ==="
for config in "${LGB_FILES[@]}"; do
  if [ ! -f "$config" ]; then
    echo "WARNING: Config file $config not found, skipping..."
    continue
  fi

  config_name=$(basename "$config" .conf)

  train_log="$RESULTS_FOLDER/${config_name}_train.log"
  test_log="$RESULTS_FOLDER/${config_name}_test.log"
  model_file="$RESULTS_FOLDER/${config_name}.txt"

  echo "============================================"
  echo "Training LightGBM with $config"
  echo "Model: $model_file"
  echo "============================================"

  # -----------------------
  # Phase A: Training
  # -----------------------
  if $LIGHTGBM_BIN \
    config=$config \
    data=$TRAIN_SET \
    valid=$TEST_SET \
    objective=$OBJECTIVE \
    num_threads=$NTHREADS \
    min_data_in_leaf=$MIN_DATA \
    learning_rate=$LR \
    min_child_weight=$MIN_HESSIAN \
    num_round=$ROUNDS \
    tree_learner=$LEARNER \
    is_sparse=$SPARSE \
    metric=ndcg \
    ndcg_eval_at=1,3,5,10 \
    output_model=$model_file \
    2>&1 | tee "$train_log"; then
    echo "✓ Training completed for $config_name"
  else
    echo "✗ Training FAILED for $config_name"
    continue
  fi

  # -----------------------
  # Phase B: Test evaluation
  # -----------------------
  echo "Evaluating on TEST set..."

  if $LIGHTGBM_BIN \
    task=prediction \
    input_model=$model_file \
    data=$TEST_SET \
    objective=$OBJECTIVE \
    metric=ndcg \
    ndcg_eval_at=10 \
    num_threads=$NTHREADS \
    verbosity=2 \
    2>&1 | tee "$test_log"; then
    echo "✓ Test evaluation completed for $config_name"
  else
    echo "✗ Test evaluation FAILED for $config_name"
  fi

  echo ""
done

# Run XGBoost
echo "=== Running XGBoost experiments ==="
for config in "${XGB_FILES[@]}"; do
  # Check if config file exists
  if [ ! -f "$config" ]; then
    echo "WARNING: Config file $config not found, skipping..."
    continue
  fi

  # Extract config name without path and .conf extension
  config_name=$(basename "$config" .conf)

  # Generate log filename
  log_file="$RESULTS_FOLDER/${config_name}.log"

  echo "Running XGBoost with $config..."
  echo "Logging to $log_file"

  # Run with error handling
  if python xgb_train.py \
    --config "$config" \
    --train "data/train.txt?format=libsvm" \
    --val "data/vali.txt?format=libsvm" \
    --test "data/test.txt?format=libsvm" \
    --objective "$OBJECTIVE_XGB" \
    --num_threads "$NTHREADS" \
    --learning_rate "$LR" \
    --min_child_weight "$MIN_HESSIAN" \
    --num_round "$ROUNDS" \
    --eval "ndcg@10" \
    2>&1 | tee "$log_file"; then
    echo "✓ $config_name completed successfully"
  else
    echo "✗ $config_name FAILED with exit code $?"
  fi
  echo ""
done

echo "All experiments completed! Results saved in $RESULTS_FOLDER/"
