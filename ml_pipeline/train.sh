#!/bin/bash

# Training Script for ML Pipeline
# Usage: ./train.sh [model_type] [data_path]

set -e

# Default values
MODEL_TYPE="${1:-xgboost}"
DATA_PATH="${2:-../src/CryptoArbitrage.HistoricalCollector/data/simulations.csv}"

echo "========================================="
echo "ML Pipeline Training"
echo "========================================="
echo "Model Type: $MODEL_TYPE"
echo "Data Path: $DATA_PATH"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run training
python -m src.training.train \
    --data-path "$DATA_PATH" \
    --model-type "$MODEL_TYPE" \
    --config config/training_config.yaml \
    --output-dir models \
    --export-onnx \
    --test-size 0.2 \
    --val-size 0.2

echo ""
echo "✅ Training complete!"
echo "Models saved to: models/$MODEL_TYPE/"
