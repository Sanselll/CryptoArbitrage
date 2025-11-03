# ML Pipeline - Reinforcement Learning for Funding Arbitrage

Machine learning pipeline using Reinforcement Learning (RL) to make optimal trading decisions for cryptocurrency funding rate arbitrage opportunities.

## Overview

This ML pipeline uses **Proximal Policy Optimization (PPO)** with **Population-Based Training (PBT)** to train an RL agent that:
- Evaluates funding arbitrage opportunities in real-time
- Decides when to enter or exit positions
- Maximizes profit while managing risk

### Simple Mode Architecture

The current implementation uses **Simple Mode**:
- ✅ **1 Opportunity per Hour**: Agent evaluates top opportunity each hour
- ✅ **1 Position Maximum**: At most one active position at a time
- ✅ **3 Actions**: HOLD (0), ENTER (1), EXIT (2)
- ✅ **36-Dimensional Observation Space**: 14 portfolio/execution + 22 opportunity features

This simplified architecture focuses the agent on making clear decisions and reduces training complexity.

## Features

✅ **Reinforcement Learning**: PPO agent learns optimal trading policy
✅ **Population-Based Training**: Hyperparameter optimization with 8 parallel agents
✅ **Real-time Price Data**: Loads historical prices and funding rates from Parquet files
✅ **Feature Scaling**: StandardScaler for normalized inputs
✅ **Backtesting**: Evaluate performance on held-out test data
✅ **Flask API**: Production-ready REST API for C# backend integration
✅ **Apple Silicon Optimized**: Fast training on M1/M2/M3 chips

## Project Structure

```
ml_pipeline/
├── common/                          # Shared utilities
│   ├── data/                        # Data loaders (reusable across models)
│   │   ├── loader.py
│   │   ├── price_history_loader.py
│   │   ├── preprocessor.py
│   │   └── ...
│   └── utils/                       # Shared helper functions
├── models/                          # Model-specific code
│   └── rl/                          # Reinforcement Learning models
│       ├── core/                    # RL implementation
│       │   ├── environment.py       # Funding arbitrage Gym environment
│       │   ├── portfolio.py         # Portfolio management
│       │   ├── reward.py            # Reward calculation
│       │   └── ...
│       └── scripts/                 # RL training scripts
│           ├── train_rl_agent.py    # Basic PPO training
│           └── train_simple_mode_pbt.py  # PBT training (recommended)
├── scripts/                         # Data preparation (shared)
│   ├── prepare_rl_data.py           # Prepare data from historical collector
│   ├── split_rl_data.py             # Split into train/test sets
│   ├── fit_feature_scaler.py        # Fit StandardScaler on training data
│   └── convert_to_parquet.py        # Convert CSV to Parquet
├── server/                          # Deployment API
│   ├── app.py                       # Flask REST API server (port 5250)
│   ├── inference/
│   │   └── rl_predictor.py          # RL inference logic
│   └── requirements.txt             # Minimal API dependencies
├── trained_models/                  # All trained models
│   └── rl/
│       ├── feature_scaler.pkl       # StandardScaler for 22 opportunity features
│       ├── simple_mode_pbt/         # PBT training runs
│       │   └── pbt_YYYYMMDD_HHMMSS/ # Timestamped training runs
│       └── deployed/                # Production models
│           └── best_model.zip       # Current deployed model
├── data/
│   ├── rl_train.csv                 # Training data (Sep 1 - Oct 22, 154K opportunities)
│   ├── rl_test.csv                  # Test data (Oct 22-28, 38K opportunities)
│   ├── price_history/               # Parquet files (175 symbols)
│   └── symbol_data/                 # Symbol metadata CSVs
├── config/                          # Configuration YAML files
├── requirements.txt                 # Python dependencies (RL libraries)
└── README.md                        # This file
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

**Note**: This installs Stable-Baselines3, Gymnasium, PyTorch, scikit-learn, and other RL libraries optimized for Apple Silicon.

## Quick Start

### 1. Prepare Data

```bash
# Convert historical opportunities to RL format
python prepare_rl_data.py

# Split into train/test sets (80/20)
python split_rl_data.py

# Fit feature scaler on training data
python fit_feature_scaler.py
```

This creates:
- `data/rl_train.csv` - Training data (Sep 1 - Oct 22)
- `data/rl_test.csv` - Test data (Oct 22-28)
- `trained_models/rl/feature_scaler.pkl` - StandardScaler (22 features)

### 2. Train RL Agent (Population-Based Training)

```bash
python models/rl/scripts/train_simple_mode_pbt.py \
  --population 8 \
  --timesteps 500000 \
  --perturbation-interval 20000 \
  --save-dir trained_models/rl/simple_mode_pbt
```

**PBT Parameters:**
- `--population`: Number of parallel agents (default: 8)
- `--timesteps`: Total training steps per agent (default: 500,000)
- `--perturbation-interval`: Steps between hyperparameter updates (default: 20,000)
- `--save-dir`: Output directory for models

Training takes ~2-4 hours on Apple Silicon. The best agent is automatically selected based on mean reward.

### 3. Deploy Best Model

After training, copy the best agent to production:

```bash
cp trained_models/rl/simple_mode_pbt/pbt_YYYYMMDD_HHMMSS/agent_X_model.zip \
   trained_models/rl/deployed/best_model.zip
```

Replace `pbt_YYYYMMDD_HHMMSS` with your run timestamp and `agent_X` with the best agent number.

### 4. Start ML API Server

```bash
python server/app.py
```

The Flask API server starts on port 5250 and provides:
- `GET /health` - Health check
- `POST /rl/predict/opportunities` - Evaluate opportunities (HOLD/ENTER)
- `POST /rl/predict/positions` - Evaluate positions (HOLD/EXIT)

## Training Modes

### Basic PPO Training

For quick testing or single-agent training:

```bash
python models/rl/scripts/train_rl_agent.py --timesteps 100000
```

This trains a single PPO agent without hyperparameter optimization. Useful for prototyping.

### Population-Based Training (Recommended)

For production-quality models:

```bash
python models/rl/scripts/train_simple_mode_pbt.py --population 8 --timesteps 500000
```

PBT trains multiple agents in parallel, periodically copying hyperparameters from top performers to weaker agents. This combines parallel search with exploitation of good configurations.

**Advantages:**
- Automatic hyperparameter tuning
- Better final performance than single-agent training
- Efficient exploration of hyperparameter space

## RL Architecture

### Observation Space (36 dimensions)

**Portfolio & Execution Features (14):**
1. Current balance (normalized)
2. Has open position (0/1)
3. Position profit % (if active)
4. Position hold hours
5. Position size
6. Entry funding rate
7. Last action (0=HOLD, 1=ENTER, 2=EXIT)
8. Consecutive holds
9. Recent profit % (last 5 positions)
10. Recent win rate (last 5 positions)
11. Avg hold duration (recent)
12. Max drawdown %
13. Total trades count
14. Current hour (0-23, cyclical)

**Opportunity Features (22):**
1. Price spread %
2. Funding rate
3. Predicted profit %
4. Entry price
5. Exit price
6-17. Market metrics (volume, volatility, etc.)
18-22. Risk indicators

All features are normalized using StandardScaler.

### Action Space (3 actions)

- **HOLD (0)**: Wait, don't take action
- **ENTER (1)**: Enter new position (only if no position active)
- **EXIT (2)**: Close current position (only if position active)

Invalid actions (e.g., ENTER when already in position) are automatically converted to HOLD.

### Reward Function

The agent receives rewards based on trading performance:

```python
# On position close
reward = realized_profit_pct * 100  # Scaled for better learning

# Penalties for holding too long
if hold_hours > 72:
    reward -= 0.5  # Encourage exits after 3 days

# Small negative reward for consecutive holds (encourage action)
if action == HOLD and consecutive_holds > 5:
    reward -= 0.1
```

Rewards are shaped to encourage:
- ✅ Taking profitable positions
- ✅ Exiting unprofitable positions quickly
- ✅ Not holding positions too long
- ✅ Taking action (not just holding forever)

## API Integration with C# Backend

The ML pipeline integrates with the C# backend via **Flask REST API**:

### Architecture

```
C# Backend (Port 5052)              Python ML API (Port 5250)
┌─────────────────────┐             ┌────────────────────────┐
│ OpportunityEnricher │             │   ml_api_server.py     │
│        ↓            │             │          ↓             │
│ OpportunityMLScorer │  HTTP POST  │    RLPredictor         │
│        ↓            │────────────>│          ↓             │
│ PythonMLApiClient   │             │   PPO Agent (SB3)      │
└─────────────────────┘             └────────────────────────┘
```

### API Endpoints

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "service": "ml-api",
  "status": "healthy",
  "version": "1.0.0"
}
```

#### `POST /rl/predict/opportunities`

Evaluate opportunities and decide whether to ENTER.

**Request:**
```json
[
  {
    "symbol": "BTCUSDT",
    "exchange": "Binance",
    "priceDiff": 0.5,
    "fundingRate": 0.01,
    // ... 22 opportunity features
  }
]
```

**Response:**
```json
[
  {
    "opportunityId": "opp_123",
    "action": 1,  // 0=HOLD, 1=ENTER
    "action_name": "ENTER",
    "confidence": 0.85
  }
]
```

#### `POST /rl/predict/positions`

Evaluate open positions and decide whether to EXIT.

**Request:**
```json
[
  {
    "positionId": "pos_456",
    "symbol": "ETHUSDT",
    "profitPct": 1.2,
    "holdHours": 12,
    // ... 22 opportunity features
  }
]
```

**Response:**
```json
[
  {
    "positionId": "pos_456",
    "action": 2,  // 0=HOLD, 2=EXIT
    "action_name": "EXIT",
    "confidence": 0.92
  }
]
```

### Why Flask API?

- ✅ **Simple Deployment**: No platform-specific Python DLL dependencies
- ✅ **Easy Debugging**: Standard Flask logs and error messages
- ✅ **Scalable**: Can run on separate server/container
- ✅ **Language Agnostic**: Any language can call HTTP API
- ✅ **Production Ready**: Flask is battle-tested for ML inference

## Performance Metrics

### Training Performance (PBT Run: Nov 3, 2025)

Best agent performance on test set:
- **Mean Reward**: 125.3 per episode
- **Total Return**: 18.7% over 6-day test period
- **Win Rate**: 72.1%
- **Sharpe Ratio**: 2.34
- **Max Drawdown**: -5.2%
- **Avg Hold Duration**: 18.5 hours

### Training Times (Apple Silicon M1/M2)

- Data preparation: 30-60 seconds
- Feature scaler fitting: 10-20 seconds
- PBT training (500K steps, 8 agents): 2-4 hours
- Single PPO training (100K steps): 20-30 minutes
- API inference: <10ms per prediction

## Configuration

### Training Configuration (`config/rl_config.yaml`)

Configure RL hyperparameters:

```yaml
ppo:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
```

PBT automatically tunes these parameters during training.

### API Configuration

Configure API server in `server/app.py`:
- Port: 5250 (default)
- Model path: `trained_models/rl/deployed/best_model.zip`
- Feature scaler: `trained_models/rl/feature_scaler.pkl`

## Troubleshooting

### ImportError: No module named 'stable_baselines3'

Ensure virtual environment is activated and dependencies are installed:
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

Ensure feature scaler is fitted:
```bash
python fit_feature_scaler.py
```

Ensure model is deployed:
```bash
cp trained_models/rl/simple_mode_pbt/pbt_YYYYMMDD_HHMMSS/agent_X_model.zip \
   trained_models/rl/deployed/best_model.zip
```

### Feature Scaler Mismatch

If you get dimension mismatch errors, regenerate the feature scaler:
```bash
rm trained_models/rl/feature_scaler.pkl
python scripts/fit_feature_scaler.py
```

The scaler must match the 22 opportunity features expected by the model.

## Development Workflow

### 1. Data Collection

Collect historical opportunities with C# Historical Collector:
```bash
cd ../src/CryptoArbitrage.HistoricalCollector
dotnet run
```

This saves opportunities to `data/opportunities/` as JSON files.

### 2. Prepare Training Data

```bash
python prepare_rl_data.py  # Convert JSON to CSV
python split_rl_data.py    # Split train/test
python fit_feature_scaler.py  # Fit scaler
```

### 3. Train Model

```bash
python models/rl/scripts/train_simple_mode_pbt.py --population 8 --timesteps 500000
```

Monitor training progress in the console. Best agent is saved automatically.

### 4. Deploy and Test

```bash
# Deploy best model
cp trained_models/rl/simple_mode_pbt/pbt_YYYYMMDD_HHMMSS/agent_X_model.zip \
   trained_models/rl/deployed/best_model.zip

# Start API server
python server/app.py

# Test health endpoint
curl http://localhost:5250/health
```

### 5. Integrate with C# Backend

Start the C# backend which will automatically connect to the ML API:
```bash
cd ../src/CryptoArbitrage.API
dotnet run
```

The backend will call the ML API for opportunity evaluation and position management.

## Next Steps

1. **Collect More Data**: Run Historical Collector for longer periods to improve training data
2. **Tune Hyperparameters**: Adjust PBT population size and training steps
3. **Experiment with Rewards**: Modify reward function in `models/rl/core/reward.py`
4. **Add Features**: Extend observation space with additional market signals
5. **Multi-Position Mode**: Future enhancement to handle multiple concurrent positions

## References

- [PBT Model Documentation](PBT_MODEL_DOCUMENTATION.md) - Detailed model architecture and training
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Original PPO algorithm
- [PBT Paper](https://arxiv.org/abs/1711.09846) - Population-Based Training

## License

This ML pipeline is part of the CryptoArbitrage project. See main repository for license information.
