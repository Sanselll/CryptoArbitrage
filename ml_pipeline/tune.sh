#!/bin/bash

# Hyperparameter Tuning Script
# Usage: ./tune.sh [model_type] [n_trials] [data_path]

set -e

# Default values
MODEL_TYPE="${1:-success}"
N_TRIALS="${2:-50}"
DATA_PATH="${3:-../src/CryptoArbitrage.HistoricalCollector/data/training_data.csv}"

echo "========================================="
echo "ML Hyperparameter Tuning with Optuna"
echo "========================================="
echo "Model Type: $MODEL_TYPE"
echo "Trials: $N_TRIALS"
echo "Data Path: $DATA_PATH"
echo ""

# Run tuning
python3 -m src.training.tune_hyperparams \
    --data-path "$DATA_PATH" \
    --model-type "$MODEL_TYPE" \
    --n-trials "$N_TRIALS" \
    --output-config config/tuned_config.yaml \
    --test-size 0.2 \
    --val-size 0.2

echo ""
echo "✅ Tuning complete!"
echo "Best parameters saved to: config/tuned_config.yaml"
echo ""
echo "To train with tuned parameters:"
echo "  ./train.sh xgboost $DATA_PATH --config config/tuned_config.yaml"
