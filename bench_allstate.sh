#!/bin/bash

set -e
set -o pipefail

# Configuration
LIGHTGBM_BIN=../LightGBM/lightgbm
DATASET=allstate
TRAIN_SET=data/${DATASET}.train
TEST_SET=data/${DATASET}.test
SPARSE=true
RESULTS_FOLDER=results/${DATASET}

# Common model config
OBJECTIVE=regression
OBJECTIVE_XGB=reg:squarederror
NTHREADS=8

MIN_DATA=1
LR=0.02
MIN_HESSIAN=100
ROUNDS=100
LEARNER=serial

# Create results folder if it doesn't exist
mkdir -p "$RESULTS_FOLDER"

# Array of configurations to test
LGB_FILES=(
  configs/${DATASET}/efb_only.conf
  configs/${DATASET}/lgb_baseline.conf
  configs/${DATASET}/lightgbm.conf
  configs/${DATASET}/sgb.conf
)

XGB_FILES=(
  configs/${DATASET}/xgb_exa.conf
  configs/${DATASET}/xgb_his.conf
)

# Run LightGBM
echo "=== Running LightGBM experiments ==="
for config in "${LGB_FILES[@]}"; do
  # Check if config file exists
  if [ ! -f "$config" ]; then
    echo "WARNING: Config file $config not found, skipping..."
    continue
  fi

  # Extract config name without path and .conf extension
  config_name=$(basename "$config" .conf)

  # Generate log filename
  log_file="$RESULTS_FOLDER/${config_name}.log"

  echo "Running LightGBM with $config..."
  echo "Logging to $log_file"

  # Run with error handling
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
    2>&1 | tee "$log_file"; then
    echo "✓ $config_name completed successfully"
  else
    echo "✗ $config_name FAILED with exit code $?"
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
    --train "$TRAIN_SET?format=libsvm" \
    --test "$TEST_SET?format=libsvm" \
    --objective "$OBJECTIVE_XGB" \
    --num_threads "$NTHREADS" \
    --learning_rate "$LR" \
    --min_child_weight "$MIN_HESSIAN" \
    --num_round "$ROUNDS" \
    2>&1 | tee "$log_file"; then
    echo "✓ $config_name completed successfully"
  else
    echo "✗ $config_name FAILED with exit code $?"
  fi
  echo ""
done

echo "All experiments completed! Results saved in $RESULTS_FOLDER/"
