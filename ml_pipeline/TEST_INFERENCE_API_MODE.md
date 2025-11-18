# Test Inference with API Mode

The `test_inference.py` script now supports two modes of operation:

## Modes

### 1. Direct Model Inference (Default)
Loads the model checkpoint directly and runs inference locally.

```bash
python test_inference.py \
  --checkpoint checkpoints/v3-4/checkpoint_ep1250.pt \
  --test-data-path data/rl_test.csv \
  --num-episodes 1
```

### 2. HTTP API Inference (New!)
Makes HTTP requests to the ML API server for predictions.

```bash
python test_inference.py \
  --api-mode \
  --api-url http://localhost:5250 \
  --api-endpoint /rl/predict \
  --test-data-path data/rl_test.csv \
  --num-episodes 1
```

## Command-Line Arguments

### API Mode Arguments (New)

- `--api-mode` - Enable HTTP API inference mode
- `--api-url URL` - ML API base URL (default: `http://localhost:5250`)
- `--api-endpoint PATH` - API endpoint path (default: `/rl/predict`)

### Existing Arguments

All existing arguments work in both modes:

- **Trading Configuration**:
  - `--leverage` - Max leverage (default: 2.0x)
  - `--utilization` - Capital utilization (default: 0.8)
  - `--max-positions` - Max concurrent positions (default: 2)

- **Test Configuration**:
  - `--num-episodes` - Number of episodes (default: 1)
  - `--full-test` / `--no-full-test` - Use entire dataset or episode length
  - `--episode-length-days` - Episode length in days (default: 5)
  - `--step-minutes` - Minutes per step (default: 5)
  - `--test-data-path` - Path to test data CSV
  - `--initial-capital` - Starting capital (default: 10000)
  - `--seed` - Random seed (default: 42)

- **Direct Mode Only**:
  - `--checkpoint` - Model checkpoint path (ignored in API mode)

- **Output**:
  - `--trades-output` - Output CSV file (default: trades_inference.csv)

## Usage Examples

### Example 1: Test ML API (UnifiedFeatureBuilder)

```bash
# Start ML API server (in separate terminal)
cd ml_pipeline/server
python app.py

# Run test with API mode
python test_inference.py \
  --api-mode \
  --api-endpoint /rl/predict \
  --test-data-path data/rl_test.csv \
  --leverage 2.0 \
  --utilization 0.8 \
  --max-positions 2 \
  --num-episodes 1 \
  --step-minutes 5
```

### Example 2: Compare Direct vs API Inference

**Direct inference:**
```bash
python test_inference.py \
  --checkpoint checkpoints/v3-4/checkpoint_ep1250.pt \
  --test-data-path data/rl_test.csv \
  --trades-output trades_direct.csv \
  --seed 42
```

**API inference:**
```bash
python test_inference.py \
  --api-mode \
  --api-endpoint /rl/predict \
  --test-data-path data/rl_test.csv \
  --trades-output trades_api.csv \
  --seed 42
```

Then compare results:
```bash
# Should produce identical results with same seed
diff trades_direct.csv trades_api.csv
```

### Example 3: Time Range Filtering (Works in Both Modes)

```bash
python test_inference.py \
  --api-mode \
  --test-data-path data/rl_test.csv \
  --start-time "2025-11-13 09:20:00" \
  --end-time "2025-11-13 09:30:00" \
  --step-minutes 1
```

## How It Works

### Direct Mode
1. Loads PyTorch model checkpoint
2. Creates environment from test data
3. Runs model inference locally
4. Executes actions and collects metrics

### API Mode
1. Starts HTTP client to ML API
2. Creates environment from test data
3. For each step:
   - Extracts raw data from environment
   - Sends HTTP POST request to ML API
   - Receives action prediction
   - Executes action in environment
4. Collects same metrics as direct mode

### Data Flow in API Mode

```
Test Script
    │
    ├─> Environment.reset()
    │
    └─> For each step:
         │
         ├─> MLAPIClient.build_raw_data_from_env()
         │    └─> Extracts: trading_config, portfolio, opportunities
         │
         ├─> HTTP POST to ML API
         │    {
         │      "trading_config": {...},
         │      "portfolio": {...},
         │      "opportunities": [...]
         │    }
         │
         ├─> ML API Response
         │    {
         │      "action_id": 5,
         │      "action": "ENTER",
         │      "confidence": 0.85,
         │      ...
         │    }
         │
         └─> Environment.step(action)
```

## Benefits of API Mode

1. **End-to-End Testing** - Tests the actual ML API server
2. **No Model Loading** - Faster startup (no checkpoint loading)
3. **Production Validation** - Same code path as production backend
4. **API Debugging** - Helps debug API issues
5. **Performance Testing** - Measures API latency under load
6. **Unified Architecture** - Tests the UnifiedFeatureBuilder integration

## Output

Both modes produce identical output:

- **Console**: Episode metrics, trading statistics, win rate, profit factor
- **CSV File**: Detailed trade records (same format in both modes)
- **Feature Log**: `test_features.log` (first 5 steps only)

## Error Handling

### API Mode Errors

**Server not running:**
```
RuntimeError: Failed to connect to ML API at http://localhost:5250/rl/predict.
Is the server running?
```
→ Start the ML API server first

**API timeout:**
```
RuntimeError: ML API timeout after 10 seconds
```
→ Check server logs, may be overloaded

**API error:**
```
RuntimeError: ML API error: 400 - Validation failed
```
→ Check request data format

## Performance

### Direct Mode
- Startup: ~5 seconds (load model)
- Per step: ~1-5ms (model inference)

### API Mode
- Startup: <1 second (no model loading)
- Per step: ~50-100ms (HTTP + inference)
  - HTTP overhead: ~30-50ms
  - Model inference: ~20-50ms

## Validation

To ensure both modes produce identical results:

1. Use same random seed (`--seed 42`)
2. Use same test data and time range
3. Use same trading configuration
4. Compare output CSVs

Expected: Actions should be identical (feature engineering is deterministic)

## Troubleshooting

### API returns different actions than direct mode

Possible causes:
1. Different random seeds
2. Different model versions (API using different checkpoint)
3. Feature engineering differences (should not happen - both use UnifiedFeatureBuilder)
4. Timing issues (market data changes between calls)

### API mode slower than expected

Possible causes:
1. Network latency (use localhost to minimize)
2. Server overloaded (check CPU usage)
3. Feature scaler loading delay (happens once on startup)
4. Logging enabled (disable verbose logs)

## Best Practices

1. **Always start ML API server first** when using `--api-mode`
2. **Use same seed** for reproducible comparisons
3. **Test small episodes first** (`--num-episodes 1 --episode-length-days 1`)
4. **Monitor server logs** while testing
5. **Check API health** before running long tests:
   ```bash
   curl http://localhost:5250/health
   ```

## Integration with CI/CD

API mode enables automated testing:

```bash
#!/bin/bash
# Start ML API
python ml_pipeline/server/app.py &
API_PID=$!

# Wait for API to be ready
sleep 5

# Run API tests
python ml_pipeline/test_inference.py \
  --api-mode \
  --test-data-path data/rl_test.csv \
  --num-episodes 1 \
  --step-minutes 60

# Check exit code
TEST_EXIT=$?

# Stop API
kill $API_PID

exit $TEST_EXIT
```
