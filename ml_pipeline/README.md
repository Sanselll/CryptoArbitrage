# ML Pipeline - Crypto Arbitrage Opportunity Scoring

Machine learning pipeline for scoring and ranking cryptocurrency arbitrage opportunities using historical data.

## Overview

This ML pipeline trains models to predict:
- **Profit Percentage**: Expected profit from an opportunity
- **Success Probability**: Likelihood the trade will be profitable
- **Hold Duration**: Optimal time to hold the position

These predictions are combined into a **composite score** (0-100) to rank opportunities and select the most profitable one.

## Features

✅ **Modular Architecture**: Easily swap between XGBoost, LightGBM, CatBoost, and Random Forest
✅ **Composite Scoring**: Rank 20+ opportunities and pick the best
✅ **Backtesting Framework**: Validate model performance on historical data
✅ **Walk-Forward Validation**: Train on period X, test on period Y
✅ **Performance Metrics**: Total return, win rate, Sharpe ratio, max drawdown
✅ **ONNX Export**: Export models for C# production integration
✅ **Apple Silicon Optimized**: Fast training on M1/M2/M3 chips

## Project Structure

```
ml_pipeline/
├── notebooks/              # Jupyter notebooks for exploration
├── src/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations (XGBoost, LightGBM, etc.)
│   ├── scoring/           # Opportunity scoring system
│   ├── backtesting/       # Backtesting framework
│   ├── training/          # Training scripts
│   └── export/            # ONNX export utilities
├── models/                # Saved models (gitignored)
├── results/               # Backtest reports and charts
├── config/                # Configuration files
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Installation

### 1. Create Virtual Environment

```bash
cd ml_pipeline
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: This will install XGBoost, LightGBM, CatBoost, scikit-learn, ONNX, and other ML libraries optimized for Apple Silicon.

## Quick Start

### 1. Load Data

```python
from src.data.loader import DataLoader

loader = DataLoader()
df = loader.load_csv('../src/CryptoArbitrage.HistoricalCollector/data/simulations.csv')
loader.print_summary()

train_df, test_df = loader.split_train_test(test_size=0.2, stratify_column='target_was_profitable')
```

### 2. Preprocess Features

```python
from src.data.preprocessor import FeaturePreprocessor

preprocessor = FeaturePreprocessor()
train_processed = preprocessor.fit_transform(train_df)
test_processed = preprocessor.transform(test_df)
```

### 3. Train Models

```python
from src.models.xgboost_model import XGBoostProfitPredictor, XGBoostSuccessClassifier, XGBoostDurationPredictor
from src.models.base_model import ModelEnsemble

# Train three models
profit_model = XGBoostProfitPredictor(config)
success_model = XGBoostSuccessClassifier(config)
duration_model = XGBoostDurationPredictor(config)

ensemble = ModelEnsemble(profit_model, success_model, duration_model)
ensemble.train_all(X_train, y_profit_train, y_success_train, y_duration_train)
```

### 4. Score Opportunities

```python
from src.scoring.opportunity_scorer import OpportunityScorer

scorer = OpportunityScorer(ensemble)

# Score multiple opportunities
scored_df = scorer.score_opportunities(opportunities_df)

# Select best opportunity
best_opp, score = scorer.select_best_opportunity(opportunities_df)
print(f"Best opportunity: {best_opp['symbol']} with score {score:.1f}/100")
```

### 5. Backtest

```python
from src.backtesting.backtester import Backtester

backtester = Backtester(scorer, initial_capital=10000)
result = backtester.run_backtest(
    historical_data=test_df,
    selection_interval_hours=24
)

print(f"Total Return: {result.metrics['total_return_pct']:.2f}%")
print(f"Win Rate: {result.metrics['win_rate_pct']:.1f}%")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
```

### 6. Compare Models

```python
from src.training.compare import compare_all_models

comparison_df, best_model_name = compare_all_models(historical_data)
print(f"Best model: {best_model_name}")
```

### 7. Start ML API Server

```bash
python ml_api_server.py
```

The Flask API server starts on port 5250 and provides:
- `GET /health` - Health check
- `POST /predict` - Single opportunity prediction
- `POST /predict/batch` - Batch predictions

## Configuration

### Training Configuration (`config/training_config.yaml`)

Configure hyperparameters for each model type:
- XGBoost parameters
- LightGBM parameters
- CatBoost parameters
- Random Forest parameters

### Scoring Configuration (`config/scoring_config.yaml`)

Configure scoring weights:
- `predicted_profit`: 40% (default)
- `success_probability`: 30%
- `risk_adjusted_return`: 20%
- `hold_duration`: 10%

Adjust weights based on your priorities.

## Backtesting Metrics

The backtester calculates:
- **Total Return %**: Cumulative profit across all trades
- **Win Rate %**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns (higher = better)
- **Max Drawdown %**: Largest peak-to-trough decline (lower = better)
- **Profit Factor**: Gross profit / Gross loss
- **Average Hold Time**: Mean position duration

## Model Comparison

Train all 4 model types and compare:

```bash
python src/training/compare.py --data-path data/simulations.csv
```

Output:
```
MODEL COMPARISON
=================================================================
Model                Return %    Win Rate %   Sharpe    Max DD %
-----------------------------------------------------------------
XGBoost                 15.23%        68.5%     2.45      -8.2%
LightGBM                14.87%        67.2%     2.38      -9.1%
CatBoost                14.12%        66.8%     2.21     -10.3%
RandomForest            12.45%        64.1%     1.95     -11.8%
=================================================================

✅ Best Model (by Sharpe Ratio): XGBoost
```

## Integration with C# API

The ML pipeline integrates with the C# backend via **Flask REST API**:

1. **Train Models**: Run `./train.sh` to train XGBoost models
2. **Start ML API**: Run `python ml_api_server.py` (port 5250)
3. **C# Integration**: Backend calls API via `PythonMLApiClient`
4. **Real-time Scoring**: Opportunities are enriched with ML predictions

### Architecture

```
C# Backend (Port 5052)              Python ML API (Port 5250)
┌─────────────────────┐             ┌────────────────────────┐
│ OpportunityEnricher │             │   ml_api_server.py     │
│        ↓            │             │          ↓             │
│ OpportunityMLScorer │  HTTP POST  │    MLPredictor         │
│        ↓            │────────────>│          ↓             │
│ PythonMLApiClient   │             │   XGBoost Models       │
└─────────────────────┘             └────────────────────────┘
```

### Why Flask API?

- ✅ **Simple Deployment**: No platform-specific Python DLL dependencies
- ✅ **Easy Debugging**: Standard Flask logs and error messages
- ✅ **Scalable**: Can run on separate server/container
- ✅ **Language Agnostic**: Any language can call HTTP API

For detailed integration steps, see [ML_API_GUIDE.md](ML_API_GUIDE.md)

## Development

### Run Jupyter Notebooks

```bash
jupyter notebook
# Open notebooks/01_data_exploration.ipynb
```

### Run Tests

```bash
pytest tests/
```

## Performance (Apple Silicon)

Expected training times on M1/M2:
- Data loading: 5-10 seconds
- Feature engineering: 10-30 seconds
- Model training (500K samples): 2-5 minutes
- Backtesting: 30-60 seconds
- ONNX export: 5-10 seconds

## Troubleshooting

### ImportError: No module named 'xgboost'

Make sure virtual environment is activated and dependencies are installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### ML API Server Won't Start

Check if port 5250 is available:
```bash
lsof -i :5250
# Kill any process using the port
kill -9 <PID>
```

### Models Not Found

Ensure models are trained:
```bash
./train.sh
# Creates models/xgboost/*.pkl files
```

## Quick Start Guide

### 1. Train Models
```bash
./train.sh
```

### 2. Start ML API Server
```bash
python ml_api_server.py
# Server runs on http://localhost:5250
```

### 3. Test API
```bash
curl http://localhost:5250/health
```

### 4. Start C# Backend
```bash
cd ../src/CryptoArbitrage.API
dotnet run
# Backend connects to ML API automatically
```

## Next Steps

1. Collect historical data with `CryptoArbitrage.HistoricalCollector`
2. Train models with `./train.sh`
3. Validate predictions with `validate_backend_predictions.py`
4. Start ML API server
5. Start C# backend (will connect to ML API)
6. Monitor ML predictions in frontend

## References

- [ML API Guide](ML_API_GUIDE.md) - Comprehensive API documentation
- [ML Implementation Guide](../docs/ml-implementation-guide.md) - System design
- [Architecture Documentation](../docs/ARCHITECTURE.md) - Overall system architecture
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Flask Documentation](https://flask.palletsprojects.com/)
