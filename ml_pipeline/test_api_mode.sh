#!/bin/bash
# Test script to demonstrate API mode testing

echo "=========================================="
echo "ML API Mode Testing Demonstration"
echo "=========================================="
echo ""

# Check if ML API is running
echo "1. Checking if ML API is running..."
if curl -s http://localhost:5250/health > /dev/null 2>&1; then
    echo "   ✅ ML API is running"
else
    echo "   ❌ ML API is not running"
    echo ""
    echo "   Please start the ML API server first:"
    echo "   cd ml_pipeline/server && python app.py"
    echo ""
    exit 1
fi

echo ""
echo "2. Testing ML API endpoint directly..."
curl -s -X POST http://localhost:5250/health | head -5
echo ""

echo ""
echo "3. Running test_inference.py in API mode..."
echo "   (Testing 1 episode with 10-minute intervals, limited to 10 steps for demo)"
echo ""

python test_inference.py \
  --api-mode \
  --api-endpoint /rl/predict \
  --test-data-path data/rl_test.csv \
  --leverage 2.0 \
  --utilization 0.8 \
  --max-positions 2 \
  --num-episodes 1 \
  --no-full-test \
  --episode-length-days 1 \
  --step-minutes 10 \
  --seed 42 \
  --trades-output trades_api_test.csv

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ API Mode Test Completed Successfully"
    echo ""
    echo "Output files:"
    if [ -f "trades_api_test.csv" ]; then
        echo "  - trades_api_test.csv ($(wc -l < trades_api_test.csv) lines)"
    fi
    if [ -f "test_features.log" ]; then
        echo "  - test_features.log (feature breakdown)"
    fi
else
    echo "❌ API Mode Test Failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo ""

exit $EXIT_CODE
