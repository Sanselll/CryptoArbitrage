#!/bin/bash

# Model Comparison Script
# Trains all models and runs backtest comparison
# Usage: ./compare.sh [data_path]

set -e

DATA_PATH="${1:-../src/CryptoArbitrage.HistoricalCollector/data/simulations.csv}"

echo "========================================="
echo "ML Model Comparison"
echo "========================================="
echo "Data Path: $DATA_PATH"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run comparison
python -m src.training.compare \
    --data-path "$DATA_PATH" \
    --config config/training_config.yaml \
    --scoring-config config/scoring_config.yaml \
    --output-dir results \
    --train-size 0.6 \
    --val-size 0.2 \
    --initial-capital 10000 \
    --selection-interval 24

echo ""
echo "✅ Comparison complete!"
echo "Results saved to: results/model_comparison.csv"
