# ML API Server - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Training Models](#training-models)
4. [Validation](#validation)
5. [Starting the Server](#starting-the-server)
6. [API Reference](#api-reference)
7. [Backend Integration](#backend-integration)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)

---

## Overview

The ML API Server is a Flask-based REST API that provides machine learning predictions for cryptocurrency arbitrage opportunities. It runs as an independent microservice and is called by the C# backend via HTTP.

### Why Flask API Instead of Python.NET?

**Previous Approach (Python.NET)**:
- Embedded Python runtime in C# process
- Required platform-specific Python DLL paths
- Complex debugging across language boundaries
- Deployment challenges (Python environment in .NET)

**Current Approach (Flask API)**:
- Simple HTTP REST API
- Independent microservice (can scale separately)
- Easy to debug (standard Flask logs)
- No platform-specific dependencies
- Can be deployed anywhere (Docker, cloud, etc.)

### Key Features

✅ **Three XGBoost Models**:
- **Profit Model**: Predicts expected profit percentage
- **Success Model**: Predicts probability of profitable outcome (0-1)
- **Duration Model**: Predicts optimal hold time in hours

✅ **Composite Scoring**: Combines predictions into single score (success_prob * profit)

✅ **Batch Processing**: Efficient batch predictions for multiple opportunities

✅ **Health Monitoring**: `/health` endpoint for service status

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React)                          │
│                    Port: 5173                                │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              C# Backend (.NET 8 API)                         │
│                    Port: 5052                                │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │  OpportunityEnricher                            │         │
│  │    └─> OpportunityMLScorer                     │         │
│  │          └─> PythonMLApiClient (HTTP)          │         │
│  └────────────────────────────────────────────────┘         │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTP POST
                       │ localhost:5250
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Python ML API Server (Flask)                       │
│                    Port: 5250                                │
│                                                              │
│  ┌────────────────────────────────────────────────┐         │
│  │  ml_api_server.py (Flask App)                  │         │
│  │    └─> MLPredictor (csharp_bridge.py)         │         │
│  │          ├─> XGBoost Models (*.pkl)            │         │
│  │          └─> FeaturePreprocessor               │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### ML Pipeline Components

```
ml_pipeline/
├── ml_api_server.py              # Flask API server (main entry point)
├── src/
│   ├── csharp_bridge.py          # ML prediction interface
│   ├── data/
│   │   └── preprocessor.py       # Feature engineering & normalization
│   ├── models/
│   │   └── xgboost_model.py      # XGBoost model implementations
│   └── training/
│       └── trainer.py            # Model training scripts
├── models/
│   └── xgboost/
│       ├── profit_model.pkl      # Trained profit prediction model
│       ├── success_model.pkl     # Trained success classification model
│       ├── duration_model.pkl    # Trained duration prediction model
│       └── preprocessor.pkl      # Fitted feature preprocessor
├── train.sh                      # Training script
└── validate_backend_predictions.py  # Validation script
```

---

## Training Models

### Prerequisites

1. **Install Dependencies**:
```bash
cd ml_pipeline
pip install -r requirements.txt
```

2. **Collect Training Data**:
```bash
cd ../src/CryptoArbitrage.HistoricalCollector
dotnet run -- full --start-date 2025-10-01 --end-date 2025-10-25
```

This creates `data/training_data.csv` with simulated arbitrage positions.

### Training Process

**Quick Start**:
```bash
cd ml_pipeline
./train.sh
```

**What It Does**:
1. Loads `../src/CryptoArbitrage.HistoricalCollector/data/training_data.csv`
2. Preprocesses features (54 engineered features)
3. Trains 3 XGBoost models (profit, success, duration)
4. Saves models to `models/xgboost/*.pkl`
5. Saves preprocessor to `models/xgboost/preprocessor.pkl`

**Manual Training**:
```python
from src.training.trainer import train_all_models

# Train with default hyperparameters
models = train_all_models(
    data_path='../src/CryptoArbitrage.HistoricalCollector/data/training_data.csv',
    output_dir='models/xgboost'
)

print(f"Models saved to models/xgboost/")
```

### Model Hyperparameters

Models are configured in `src/training/trainer.py`:

```python
xgboost_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',  # or 'binary:logistic' for success
    'random_state': 42
}
```

---

## Validation

### Validate Predictions Against Python Ground Truth

After the C# backend generates opportunity dumps, validate that ML API returns identical predictions:

```bash
python validate_backend_predictions.py \
  ../src/CryptoArbitrage.API/Data/backend_dumps/2025-10-26/opportunities_1603.json
```

**Output**:
```
Loaded 30 opportunities from backend JSON

PREDICTION COMPARISON:
========================
Opportunity: BTCUSDT (Binance -> Bybit)
  Python:   profit=1.234%, success=0.789, duration=12.5h
  C#:       profit=1.234%, success=0.789, duration=12.5h
  ✅ MATCH (diff < 0.001)
...

Summary: 30/30 predictions match (100.0%)
```

This ensures:
- Feature extraction is identical in C# and Python
- Preprocessor normalization matches
- Model predictions are deterministic

---

## Starting the Server

### Development Mode

**Start Server**:
```bash
cd ml_pipeline
python ml_api_server.py
```

**Expected Output**:
```
Initializing ML predictor...
✅ Preprocessor loaded from models/xgboost/preprocessor.pkl
✅ ML predictor initialized successfully

================================================================================
ML API Server
================================================================================
Starting server on http://localhost:5250
Endpoints:
  GET  /health           - Health check
  POST /predict          - Single prediction
  POST /predict/batch    - Batch prediction
================================================================================

 * Serving Flask app 'ml_api_server'
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5250
```

### Production Mode

For production, use Gunicorn (WSGI server):

```bash
# Install Gunicorn
pip install gunicorn

# Start with 4 workers
gunicorn --bind 0.0.0.0:5250 --workers 4 --timeout 60 ml_api_server:app
```

### Background Mode

**Linux/macOS**:
```bash
nohup python ml_api_server.py > ml_api.log 2>&1 &
```

**Using systemd** (see Production Deployment section)

---

## API Reference

### Base URL

```
http://localhost:5250
```

### Endpoints

#### 1. Health Check

**Request**:
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "service": "ml-api",
  "version": "1.0.0"
}
```

**Usage**: Monitor service availability, load balancer health checks

---

#### 2. Single Prediction

**Request**:
```http
POST /predict
Content-Type: application/json

{
  "symbol": "BTCUSDT",
  "longExchange": "Bybit",
  "shortExchange": "Binance",
  "longFundingRate": -0.001,
  "shortFundingRate": 0.002,
  "longFundingIntervalHours": 8,
  "shortFundingIntervalHours": 8,
  "longNextFundingTime": "2025-10-26T16:00:00Z",
  "shortNextFundingTime": "2025-10-26T16:00:00Z",
  "currentPriceSpreadPercent": 0.05,
  "volume24h": 1000000,
  "fundProfit8h": 2.5,
  "fundApr": 1095.0,
  "detectedAt": "2025-10-26T15:00:00Z"
}
```

**Response**:
```json
{
  "predicted_profit_percent": 1.234,
  "success_probability": 0.789,
  "predicted_duration_hours": 12.5,
  "composite_score": 0.974
}
```

**Fields**:
- `predicted_profit_percent`: Expected profit in percentage
- `success_probability`: Probability trade will be profitable (0-1)
- `predicted_duration_hours`: Optimal hold time
- `composite_score`: Combined score (success_prob × profit)

---

#### 3. Batch Prediction

**Request**:
```http
POST /predict/batch
Content-Type: application/json

[
  {
    "symbol": "BTCUSDT",
    ...
  },
  {
    "symbol": "ETHUSDT",
    ...
  }
]
```

**Response**:
```json
[
  {
    "predicted_profit_percent": 1.234,
    "success_probability": 0.789,
    "predicted_duration_hours": 12.5,
    "composite_score": 0.974
  },
  {
    "predicted_profit_percent": 0.856,
    "success_probability": 0.623,
    "predicted_duration_hours": 18.2,
    "composite_score": 0.533
  }
]
```

**Usage**: More efficient for scoring multiple opportunities at once

---

### cURL Examples

**Health Check**:
```bash
curl http://localhost:5250/health
```

**Single Prediction**:
```bash
curl -X POST http://localhost:5250/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "longExchange": "Bybit",
    "shortExchange": "Binance",
    "longFundingRate": -0.001,
    "shortFundingRate": 0.002,
    "longFundingIntervalHours": 8,
    "shortFundingIntervalHours": 8,
    "longNextFundingTime": "2025-10-26T16:00:00Z",
    "shortNextFundingTime": "2025-10-26T16:00:00Z",
    "currentPriceSpreadPercent": 0.05,
    "volume24h": 1000000,
    "fundProfit8h": 2.5,
    "fundApr": 1095.0,
    "detectedAt": "2025-10-26T15:00:00Z"
  }'
```

---

## Backend Integration

### C# Client Implementation

The C# backend uses `PythonMLApiClient` to call the ML API:

**File**: `src/CryptoArbitrage.API/Services/ML/PythonMLApiClient.cs`

```csharp
public class PythonMLApiClient
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl = "http://localhost:5250";

    public async Task<MLPredictionResult> ScoreOpportunityAsync(
        ArbitrageOpportunityDto opportunity)
    {
        var json = JsonSerializer.Serialize(opportunity, _jsonOptions);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("/predict", content);
        var responseJson = await response.Content.ReadAsStringAsync();

        return JsonSerializer.Deserialize<MLPredictionResult>(responseJson);
    }
}
```

### Integration Flow

1. **Startup**: C# backend checks ML API health
```csharp
var mlApiClient = serviceProvider.GetRequiredService<PythonMLApiClient>();
var isHealthy = await mlApiClient.HealthCheckAsync();
// ✅ Python ML API is available at http://localhost:5250
```

2. **Opportunity Detection**: OpportunityEnricher enriches opportunities
```csharp
public async Task<List<ArbitrageOpportunityDto>> EnrichOpportunitiesAsync(
    List<ArbitrageOpportunityDto> opportunities)
{
    // Phase 4: Apply ML scoring
    await _mlScorer.ScoreAndEnrichOpportunitiesAsync(opportunities);

    return opportunities;
}
```

3. **ML Scoring**: OpportunityMLScorer calls API
```csharp
var predictions = await _mlApiClient.ScoreOpportunitiesBatchAsync(opportunities);

for (int i = 0; i < opportunities.Count; i++)
{
    opportunities[i].MLPredictedProfitPercent = predictions[i].PredictedProfitPercent;
    opportunities[i].MLSuccessProbability = predictions[i].SuccessProbability;
    opportunities[i].MLPredictedDurationHours = predictions[i].PredictedDurationHours;
    opportunities[i].MLCompositeScore = predictions[i].CompositeScore;
}
```

4. **Frontend Display**: React displays ML scores in opportunities list

---

## Troubleshooting

### ML API Not Available

**Error**: `Failed to connect to Python ML API`

**Solutions**:
1. Check if server is running:
```bash
lsof -i :5250  # Should show python process
```

2. Start the server:
```bash
cd ml_pipeline
python ml_api_server.py
```

3. Check logs:
```bash
tail -f /tmp/ml_api_server.log
```

---

### Port Already in Use

**Error**: `Address already in use: 5250`

**Solutions**:
1. Find process using port:
```bash
lsof -i :5250
```

2. Kill process:
```bash
kill -9 <PID>
```

3. Or change port in `ml_api_server.py` and `appsettings.json`

---

### Models Not Found

**Error**: `FileNotFoundError: models/xgboost/profit_model.pkl`

**Solution**: Train models first:
```bash
./train.sh
```

---

### Prediction Errors

**Error**: `bad operand type for abs(): 'NoneType'`

**Cause**: Missing required fields in opportunity JSON

**Solution**: Ensure all required fields are present:
- `fundProfit8h`, `fundApr`, `volume24h`
- `longFundingRate`, `shortFundingRate`
- `detectedAt` (timestamp)

Optional fields default to 0:
- `fundProfit8h24hProj`, `fundApr24hProj`
- `priceSpread24hAvg`, `priceSpread3dAvg`

---

### Feature Dimension Mismatch

**Error**: `Expected 54 features, got 42`

**Cause**: Preprocessor and model mismatch

**Solution**: Retrain both models and preprocessor together:
```bash
rm -rf models/xgboost/*
./train.sh
```

---

## Production Deployment

### Using Systemd (Linux)

**Create Service File**: `/etc/systemd/system/ml-api.service`

```ini
[Unit]
Description=ML API Server for Crypto Arbitrage
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/CryptoArbitrage/ml_pipeline
ExecStart=/usr/bin/python3 ml_api_server.py
Restart=always
RestartSec=10
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
```

**Start Service**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ml-api
sudo systemctl start ml-api
sudo systemctl status ml-api
```

---

### Using Docker

**Dockerfile**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5250

CMD ["python", "ml_api_server.py"]
```

**Build and Run**:
```bash
docker build -t ml-api .
docker run -d -p 5250:5250 --name ml-api ml-api
```

---

### Using Gunicorn + Nginx

**Gunicorn Command**:
```bash
gunicorn --bind 127.0.0.1:5250 \
         --workers 4 \
         --timeout 60 \
         --access-logfile /var/log/ml-api/access.log \
         --error-logfile /var/log/ml-api/error.log \
         ml_api_server:app
```

**Nginx Reverse Proxy** (`/etc/nginx/sites-available/ml-api`):
```nginx
server {
    listen 80;
    server_name ml-api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:5250;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

### Monitoring

**Health Check Script**:
```bash
#!/bin/bash
# health-check.sh

RESPONSE=$(curl -s http://localhost:5250/health)
STATUS=$(echo $RESPONSE | jq -r '.status')

if [ "$STATUS" = "healthy" ]; then
    echo "✅ ML API is healthy"
    exit 0
else
    echo "❌ ML API is unhealthy"
    exit 1
fi
```

**Cron Job** (every 5 minutes):
```bash
*/5 * * * * /path/to/health-check.sh || systemctl restart ml-api
```

---

## Performance Optimization

### Batch Processing

Always use `/predict/batch` for multiple opportunities:

**Bad** (1000ms):
```csharp
foreach (var opp in opportunities)
{
    await ScoreOpportunityAsync(opp);  // 10 HTTP calls
}
```

**Good** (100ms):
```csharp
var predictions = await ScoreOpportunitiesBatchAsync(opportunities);  // 1 HTTP call
```

### Model Caching

Models are loaded once on server startup and kept in memory. No need to reload per request.

### Concurrent Requests

Gunicorn workers handle concurrent requests. Scale workers based on CPU cores:

```bash
gunicorn --workers $(nproc) ml_api_server:app
```

---

## Next Steps

1. ✅ Train models with historical data
2. ✅ Validate predictions match Python
3. ✅ Start ML API server
4. ✅ Verify C# backend connects successfully
5. 🔄 Monitor predictions in production
6. 🔄 Retrain models periodically with new data

For more details, see:
- [Training Guide](README.md)
- [Architecture Documentation](../docs/ARCHITECTURE.md)
- [ML Implementation Guide](../docs/ml-implementation-guide.md)
