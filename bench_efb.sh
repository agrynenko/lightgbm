#!/bin/bash

# Configuration
LIGHTGBM_BIN=../LightGBM/lightgbm
CONFIG_FILE=configs/efb_only.conf
TRAIN_SET=data/mslr.train
TEST_SET=data/mslr.test
RESULTS_FOLDER=results/efb

# Create results folder if it doesn't exist
mkdir -p "$RESULTS_FOLDER"

# Array of max_conflict_rate values to test
CONFLICT_RATES=(0 0.00001 0.0001 0.001 0.01 0.1)

# Run LightGBM with different conflict rates
for rate in "${CONFLICT_RATES[@]}"; do
  # Generate log filename
  log_file="$RESULTS_FOLDER/efb_only_${rate//./_}.log"
    
  echo "Running with max_conflict_rate=$rate..."
  
  $LIGHTGBM_BIN \
    config=$CONFIG_FILE \
    data=$TRAIN_SET \
    valid=$TEST_SET \
    max_conflict_rate=$rate \
    2>&1 | tee "$log_file"
done

echo "All experiments completed! Results saved in $RESULTS_FOLDER/"
