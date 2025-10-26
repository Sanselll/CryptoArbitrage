# ML-Based Opportunity Scoring - Complete Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Technology Stack Decision](#technology-stack-decision)
3. [Data Collection Strategy](#data-collection-strategy)
4. [Historical Data Collector (C# Console App)](#historical-data-collector)
5. [ML Training Pipeline (Python)](#ml-training-pipeline)
6. [Model Deployment & Integration](#model-deployment--integration)
7. [Implementation Phases](#implementation-phases)

---

## Overview

This guide covers the complete implementation of an ML-based scoring system to predict arbitrage opportunity profitability, optimal hold duration, and execution success probability.

### Goals:
- **Predict profitability**: Actual profit percentage from an opportunity
- **Predict success**: Binary classification (profitable vs not)
- **Predict hold duration**: Optimal time to hold position
- **Optimize position sizing**: Using predicted metrics + Kelly Criterion
- **Exit timing**: Condition-based exit signals

### Approach:
1. Collect historical market data (C# console app)
2. Simulate positions to generate training data
3. Train ML models (Python + XGBoost)
4. Deploy models for inference (.NET integration)
5. Continuous retraining with new data

---

## Technology Stack Decision

### **Python vs .NET ML - Comparison**

| Aspect | Python (Recommended) | .NET ML.NET |
|--------|---------------------|-------------|
| **ML Ecosystem** | ✅ Best-in-class (scikit-learn, XGBoost, LightGBM, PyTorch) | ⚠️ Limited (ML.NET is less mature) |
| **Apple Silicon (M1/M2)** | ✅ Excellent support (MLX, Metal acceleration) | ❌ No Metal support |
| **XGBoost** | ✅ Native, optimized | ⚠️ Wrapper, slower |
| **Feature Engineering** | ✅ Pandas, NumPy (powerful) | ⚠️ DataFrame API (limited) |
| **Model Export** | ✅ ONNX, pickle, joblib | ✅ ONNX, ML.NET format |
| **Deployment Options** | ✅ REST API, ONNX Runtime, gRPC | ✅ In-process (.NET native) |
| **Development Speed** | ✅ Faster iteration | ⚠️ Slower, more boilerplate |
| **Community/Resources** | ✅ Massive | ⚠️ Smaller |
| **Apple MLX Support** | ✅ Yes (cutting-edge) | ❌ No |

### **Recommended Architecture: Hybrid**

```
┌─────────────────────────────────────────────────────────────┐
│ Data Collection & Simulation (.NET/C#)                      │
│ - Historical data collector (console app)                   │
│ - Position simulator                                        │
│ - Uses existing services (OpportunityDetector, etc.)       │
└───────────────┬─────────────────────────────────────────────┘
                │
                │ CSV/Parquet Export
                ▼
┌─────────────────────────────────────────────────────────────┐
│ ML Training Pipeline (Python)                               │
│ - Feature engineering (Pandas)                              │
│ - Model training (XGBoost/LightGBM)                        │
│ - Hyperparameter tuning                                     │
│ - Model evaluation & validation                            │
│ - Export to ONNX                                            │
└───────────────┬─────────────────────────────────────────────┘
                │
                │ ONNX Model File
                ▼
┌─────────────────────────────────────────────────────────────┐
│ Model Deployment & Inference (.NET/C#)                      │
│ - ONNX Runtime in C# (Microsoft.ML.OnnxRuntime)            │
│ - Integrated into OpportunityEnricher                       │
│ - Real-time scoring in production                          │
└─────────────────────────────────────────────────────────────┘
```

### **Why This Hybrid Approach?**

✅ **Best of both worlds:**
- Python for ML (superior ecosystem, Apple Silicon optimization)
- C# for production (existing codebase, type safety, performance)

✅ **Apple Silicon Optimization:**
- Python with MLX framework uses Metal GPU acceleration
- XGBoost has optimized ARM builds
- Training is 2-5x faster on M1/M2

✅ **Production Deployment:**
- ONNX Runtime in C# is fast (native performance)
- No Python runtime needed in production
- Type-safe integration with existing code

✅ **Development Workflow:**
- Data scientists work in Jupyter notebooks (Python)
- Engineers integrate models seamlessly (C#)
- Clear separation of concerns

---

## Data Collection Strategy

### **What Historical Data Is Available?**

| Data Type | Availability | Coverage | Resolution | Source |
|-----------|-------------|----------|------------|--------|
| **Funding Rates** | ✅ Yes | 6-12 months | 8h intervals | Binance/Bybit API |
| **Price Klines** | ✅ Yes | Years | 1m, 5m, 1h | Binance/Bybit API |
| **Volume** | ✅ Yes | Years | 24h rolling | From klines |
| **Orderbook Depth** | ❌ No | Real-time only | N/A | WebSocket only |
| **Bid-Ask Spread** | ❌ No | Real-time only | N/A | WebSocket only |

### **Data Collection Plan**

**Phase 1: Backfill Historical (One-Time)**
- Fetch 6 months of funding rates (Binance: 1000 records, Bybit: 200 records)
- Fetch 6 months of 5-minute price klines
- Reconstruct market snapshots every 5 minutes
- Generate ~50,000 historical snapshots
- Estimate liquidity using volume as proxy

**Phase 2: Live Collection (Ongoing)**
- Collect complete snapshots every 5 minutes
- Include full liquidity data (orderbook, bid-ask spread)
- Run as background service
- Store in TimescaleDB or PostgreSQL

**Phase 3: Position Simulation**
- For each historical opportunity, simulate multiple hold durations
- Calculate realistic slippage, fees, funding payments
- Track peak profit and max drawdown
- Generate 500,000+ synthetic execution records

### **Data Schema**

```csharp
// Stored in database
public class HistoricalMarketSnapshot
{
    public DateTime Timestamp { get; set; }

    // All detected opportunities at this moment
    public List<ArbitrageOpportunityDto> Opportunities { get; set; }

    // Market prices
    public Dictionary<string, Dictionary<string, decimal>> PerpPrices { get; set; }

    // Funding rates
    public Dictionary<string, List<FundingRateDto>> FundingRates { get; set; }

    // Liquidity (only available in live mode)
    public Dictionary<string, LiquidityMetricsDto> Liquidity { get; set; }

    // Market context
    public decimal BtcPrice { get; set; }
    public decimal BtcVolume24h { get; set; }
    public string MarketRegime { get; set; } // "Bull", "Bear", "Ranging"
}

// Generated from simulation
public class SimulatedExecution
{
    // === INPUT FEATURES (X) ===
    public string OpportunitySnapshotJson { get; set; } // Full opportunity data
    public DateTime EntryTime { get; set; }
    public decimal BtcPriceAtEntry { get; set; }
    public string MarketRegimeAtEntry { get; set; }

    // Opportunity features (extracted for CSV)
    public string Symbol { get; set; }
    public string Strategy { get; set; }
    public decimal FundProfit8h { get; set; }
    public decimal FundProfit8h24hProj { get; set; }
    public decimal FundProfit8h3dProj { get; set; }
    public decimal SpreadVolatilityCv { get; set; }
    public decimal Volume24h { get; set; }
    public decimal BidAskSpreadPercent { get; set; }
    // ... 80+ more features

    // === TARGET VARIABLES (y) ===
    public decimal ActualHoldHours { get; set; }
    public decimal ActualProfitPercent { get; set; }
    public decimal ActualProfitUsd { get; set; }
    public bool WasProfitable { get; set; }

    // Performance metrics
    public decimal PeakUnrealizedProfitPercent { get; set; }
    public decimal MaxDrawdownPercent { get; set; }
    public int FundingPaymentsCount { get; set; }
    public decimal TotalFundingEarnedUsd { get; set; }

    // Execution quality
    public decimal TotalSlippagePercent { get; set; }
    public decimal TotalFeesUsd { get; set; }
}
```

---

## Historical Data Collector

### **Console Application Structure**

```
CryptoArbitrage.HistoricalCollector/
├── Program.cs                          # Entry point with CLI commands
├── Services/
│   ├── HistoricalDataFetcher.cs       # Fetch from exchange APIs
│   ├── SnapshotReconstructor.cs       # Reconstruct historical snapshots
│   ├── LiveDataCollector.cs           # Live 5min collection
│   └── PositionSimulator.cs           # Simulate positions
├── Models/
│   ├── HistoricalMarketSnapshot.cs
│   └── SimulatedExecution.cs
├── Exporters/
│   ├── CsvExporter.cs                 # Export to CSV for Python
│   └── ParquetExporter.cs             # Export to Parquet (optional)
├── appsettings.json
└── CryptoArbitrage.HistoricalCollector.csproj
```

### **Project Configuration**

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
  </PropertyGroup>

  <ItemGroup>
    <!-- Reference main API to reuse all services -->
    <ProjectReference Include="..\CryptoArbitrage.API\CryptoArbitrage.API.csproj" />

    <!-- CLI framework -->
    <PackageReference Include="System.CommandLine" Version="2.0.0-beta4.22272.1" />

    <!-- CSV export -->
    <PackageReference Include="CsvHelper" Version="30.0.1" />

    <!-- Parquet export (optional, for large datasets) -->
    <PackageReference Include="Parquet.Net" Version="4.15.0" />
  </ItemGroup>
</Project>
```

### **CLI Commands**

```bash
# 1. Backfill historical data (6 months)
dotnet run --project HistoricalCollector backfill \
    --start-date 2024-04-24 \
    --end-date 2024-10-24 \
    --interval 5

# 2. Run live collection (runs forever)
dotnet run --project HistoricalCollector live \
    --interval 5

# 3. Simulate positions from historical data
dotnet run --project HistoricalCollector simulate \
    --start-date 2024-09-01 \
    --end-date 2024-10-24 \
    --output training_data.csv

# 4. Export opportunities to CSV for analysis
dotnet run --project HistoricalCollector export \
    --format csv \
    --output opportunities.csv
```

### **Key Implementation Points**

**1. HistoricalDataFetcher.cs**
```csharp
public class HistoricalDataFetcher
{
    // Fetches historical funding rates from Binance/Bybit
    // Handles pagination (1000 records max per request)
    // Returns: Dictionary<Exchange, List<FundingRateDto>>

    public async Task<Dictionary<string, List<FundingRateDto>>> FetchAllFundingRates(
        DateTime startDate,
        DateTime endDate)
    {
        // Binance: /fapi/v1/fundingRate?symbol=BTCUSDT&startTime={unix}&limit=1000
        // Bybit: /v5/market/funding/history?symbol=BTCUSDT&startTime={unix}&limit=200
        // Loops through all symbols and exchanges
    }

    // Fetches 5-minute klines for price data
    public async Task<Dictionary<string, List<PriceDto>>> FetchAllPriceKlines(
        DateTime startDate,
        DateTime endDate,
        TimeSpan interval)
    {
        // Binance: /fapi/v1/klines?symbol=BTCUSDT&interval=5m&startTime={unix}&limit=1500
        // Returns close price + volume for each 5min candle
    }
}
```

**2. SnapshotReconstructor.cs**
```csharp
public class SnapshotReconstructor
{
    private readonly IOpportunityDetectionService _opportunityDetector; // REUSED from API!

    public async Task BackfillHistoricalData(
        DateTime startDate,
        DateTime endDate,
        TimeSpan interval)
    {
        // 1. Bulk download all funding rates and prices
        var fundingHistory = await _fetcher.FetchAllFundingRates(startDate, endDate);
        var priceHistory = await _fetcher.FetchAllPriceKlines(startDate, endDate, interval);

        // 2. Loop through each 5min interval
        var currentTime = startDate;
        while (currentTime <= endDate)
        {
            // 3. Reconstruct MarketDataSnapshot at this timestamp
            var marketData = BuildMarketDataSnapshot(currentTime, fundingHistory, priceHistory);

            // 4. Use EXISTING OpportunityDetectionService to detect opportunities
            var opportunities = await _opportunityDetector.DetectOpportunitiesAsync(marketData);

            // 5. Store snapshot
            await _repository.StoreAsync($"historical:{currentTime:yyyyMMddHHmmss}", snapshot);

            currentTime += interval;
        }
    }
}
```

**3. PositionSimulator.cs**
```csharp
public class PositionSimulator
{
    public async Task<List<SimulatedExecution>> SimulateHistoricalPositions(
        DateTime startDate,
        DateTime endDate)
    {
        var simulations = new List<SimulatedExecution>();
        var snapshots = await LoadSnapshots(startDate, endDate);

        // For each snapshot
        for (int i = 0; i < snapshots.Count - 336; i++) // Leave room for 7 days
        {
            var entrySnapshot = snapshots[i];

            // For each opportunity detected
            foreach (var opportunity in entrySnapshot.Opportunities)
            {
                // Simulate multiple hold durations
                var holdDurations = new[] { 1, 4, 8, 12, 24, 48, 72, 120, 168 }; // hours

                foreach (var holdHours in holdDurations)
                {
                    var simulation = SimulateSinglePosition(
                        opportunity,
                        snapshots,
                        i,
                        holdHours
                    );

                    if (simulation != null)
                        simulations.Add(simulation);
                }
            }
        }

        return simulations;
    }

    private SimulatedExecution SimulateSinglePosition(
        ArbitrageOpportunityDto opportunity,
        List<HistoricalMarketSnapshot> snapshots,
        int entryIndex,
        decimal holdHours)
    {
        // 1. Find exit snapshot
        var exitIndex = FindExitIndex(snapshots, entryIndex, holdHours);

        // 2. Calculate entry prices with slippage
        var entryPrices = CalculateEntryPrices(opportunity, snapshots[entryIndex]);

        // 3. Calculate exit prices with slippage
        var exitPrices = CalculateExitPrices(opportunity, snapshots[exitIndex]);

        // 4. Simulate funding payments during hold period
        var fundingPayments = SimulateFundingPayments(
            opportunity,
            snapshots,
            entryIndex,
            exitIndex
        );

        // 5. Calculate total PnL
        var pnl = CalculatePnL(entryPrices, exitPrices, fundingPayments);

        // 6. Track peak profit and max drawdown
        var (peakProfit, maxDrawdown) = CalculatePeakAndDrawdown(
            opportunity,
            snapshots,
            entryIndex,
            exitIndex,
            entryPrices
        );

        // 7. Return simulation result
        return new SimulatedExecution
        {
            // Input features
            OpportunitySnapshotJson = JsonSerializer.Serialize(opportunity),
            Symbol = opportunity.Symbol,
            Strategy = opportunity.SubType.ToString(),
            FundProfit8h = opportunity.FundProfit8h,
            // ... all other features

            // Target variables
            ActualHoldHours = (decimal)(exitTime - entryTime).TotalHours,
            ActualProfitPercent = pnl.TotalProfitPercent,
            WasProfitable = pnl.TotalProfitPercent > 0,
            PeakUnrealizedProfitPercent = peakProfit,
            MaxDrawdownPercent = maxDrawdown,
            // ... all other labels
        };
    }
}
```

**4. Realistic Slippage Modeling**
```csharp
private decimal EstimateSlippage(
    decimal positionSize,
    decimal orderbookDepth,
    decimal bidAskSpreadPercent)
{
    // Base slippage = half of bid-ask spread
    var baseSlippage = bidAskSpreadPercent / 2;

    // Additional slippage based on position size vs orderbook depth
    var depthRatio = positionSize / orderbookDepth;

    if (depthRatio < 0.01m) // <1% of depth
        return baseSlippage;
    else if (depthRatio < 0.05m) // 1-5%
        return baseSlippage * 1.5m;
    else if (depthRatio < 0.10m) // 5-10%
        return baseSlippage * 2.5m;
    else // >10% - significant market impact
        return baseSlippage * 5m;
}
```

### **CSV Export Format**

```csv
# training_data.csv (500,000+ rows)
timestamp,symbol,strategy,fund_profit_8h,fund_apr,fund_profit_24h_proj,fund_profit_3d_proj,break_even_hours,spread_volatility_cv,volume_24h,bid_ask_spread,orderbook_depth,liquidity_status,btc_price,market_regime,hour_of_day,day_of_week,...,actual_hold_hours,actual_profit_pct,was_profitable,peak_profit,max_drawdown
2024-09-01T00:00:00,BTCUSDT,CFFF,0.15,54.75,0.12,0.14,16.5,0.08,5000000,0.01,250000,Good,58000,Bull,0,1,...,48.2,2.35,True,3.1,-0.8
2024-09-01T00:00:00,ETHUSDT,CFFF,0.08,29.2,0.09,0.10,22.1,0.12,3000000,0.02,150000,Medium,2300,Bull,0,1,...,24.5,0.95,True,1.2,-0.3
...
```

---

## ML Training Pipeline (Python)

### **Python Environment Setup**

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (optimized for Apple Silicon)
pip install --upgrade pip

# Core ML libraries
pip install numpy pandas scikit-learn

# XGBoost (with Apple Silicon optimization)
pip install xgboost

# LightGBM (with Apple Silicon optimization)
pip install lightgbm

# For Apple MLX (cutting-edge, optional)
pip install mlx

# Model explainability
pip install shap

# Hyperparameter tuning
pip install optuna

# Model export
pip install onnx skl2onnx onnxruntime

# Data processing
pip install pyarrow  # For Parquet files
pip install matplotlib seaborn  # Visualization

# Jupyter for experimentation
pip install jupyter
```

### **Project Structure**

```
ml_pipeline/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py                  # Load CSV/Parquet data
│   │   └── preprocessor.py            # Feature engineering
│   ├── models/
│   │   ├── profit_predictor.py        # Profit regression model
│   │   ├── success_classifier.py      # Success classification
│   │   └── duration_predictor.py      # Hold duration model
│   ├── training/
│   │   ├── train.py                   # Main training script
│   │   └── evaluate.py                # Model evaluation
│   └── export/
│       └── onnx_exporter.py           # Export to ONNX
├── models/                             # Saved models
│   ├── profit_model.onnx
│   ├── success_model.onnx
│   └── duration_model.onnx
├── config/
│   └── training_config.yaml
├── requirements.txt
└── train.sh                            # Training script
```

### **Feature Engineering (Python)**

```python
# src/data/preprocessor.py
import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_names = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw opportunity data into ML features
        """
        df = df.copy()

        # === PROFITABILITY FEATURES ===
        df['rate_momentum_24h'] = (df['fund_profit_8h'] - df['fund_profit_24h_proj']) / df['fund_profit_24h_proj']
        df['rate_momentum_3d'] = (df['fund_profit_8h'] - df['fund_profit_3d_proj']) / df['fund_profit_3d_proj']
        df['rate_stability'] = np.abs(df['fund_profit_8h'] - df['fund_profit_3d_proj']) / df['fund_profit_3d_proj']
        df['rate_trend_score'] = np.where(
            (df['fund_profit_8h'] > df['fund_profit_24h_proj']) &
            (df['fund_profit_24h_proj'] > df['fund_profit_3d_proj']),
            1,  # Uptrend
            np.where(df['fund_profit_8h'] < df['fund_profit_3d_proj'], -1, 0)  # Downtrend vs Stable
        )

        # === RISK FEATURES ===
        df['volatility_risk_score'] = np.where(
            df['spread_volatility_cv'] < 0.2, 1,  # Low risk
            np.where(df['spread_volatility_cv'] < 0.5, 0.5, 0)  # High risk
        )

        # === LIQUIDITY FEATURES ===
        df['volume_adequacy'] = df['volume_24h'] / 1000  # Normalized by position size assumption
        df['liquidity_score'] = np.where(
            df['liquidity_status'] == 'Good', 1,
            np.where(df['liquidity_status'] == 'Medium', 0.6, 0.2)
        )

        # === TIMING FEATURES ===
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # === CATEGORICAL ENCODING ===
        df = pd.get_dummies(df, columns=['strategy', 'market_regime'], drop_first=True)

        # === INTERACTION FEATURES ===
        df['profit_x_liquidity'] = df['fund_profit_8h'] * df['liquidity_score']
        df['risk_adjusted_return'] = df['fund_apr'] / (1 + df['spread_volatility_cv'])

        self.feature_names = [col for col in df.columns if col not in self._get_target_columns()]

        return df

    def _get_target_columns(self):
        return ['actual_hold_hours', 'actual_profit_pct', 'was_profitable',
                'peak_profit', 'max_drawdown']
```

### **Model Training (XGBoost)**

```python
# src/models/profit_predictor.py
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

class ProfitPredictor:
    def __init__(self, use_gpu=False):
        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='reg:squarederror',
            eval_metric='rmse',
            tree_method='hist',  # Fast on CPU, 'gpu_hist' for CUDA
            random_state=42
        )

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train profit prediction model
        """
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=True
        )

        return self.model

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        """
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Profit Prediction Model:")
        print(f"  MAE: {mae:.4f}%")
        print(f"  RMSE: {rmse:.4f}%")
        print(f"  R²: {r2:.4f}")

        return {'mae': mae, 'rmse': rmse, 'r2': r2}

    def get_feature_importance(self):
        """
        Get feature importance for interpretability
        """
        return pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
```

### **Success Classification Model**

```python
# src/models/success_classifier.py
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class SuccessClassifier:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            scale_pos_weight=1,  # Adjust for class imbalance
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            random_state=42
        )

    def train(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=50,
            verbose=True
        )
        return self.model

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        print(f"Success Classification Model:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  AUC-ROC: {auc:.4f}")

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}
```

### **Main Training Script**

```python
# src/training/train.py
import pandas as pd
from src.data.loader import load_training_data
from src.data.preprocessor import FeatureEngineer
from src.models.profit_predictor import ProfitPredictor
from src.models.success_classifier import SuccessClassifier
from src.models.duration_predictor import DurationPredictor
from src.export.onnx_exporter import export_to_onnx

def main():
    print("Loading training data...")
    df = load_training_data('training_data.csv')

    print(f"Loaded {len(df)} records")
    print(f"Profitable: {df['was_profitable'].sum()} ({df['was_profitable'].mean()*100:.1f}%)")

    # Feature engineering
    print("Engineering features...")
    engineer = FeatureEngineer()
    df = engineer.create_features(df)

    # Split data
    feature_cols = engineer.feature_names

    # Train-test split (80/20)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]

    # === MODEL 1: Profit Prediction ===
    print("\n=== Training Profit Prediction Model ===")
    y_train_profit = train_df['actual_profit_pct']
    y_val_profit = val_df['actual_profit_pct']
    y_test_profit = test_df['actual_profit_pct']

    profit_model = ProfitPredictor()
    profit_model.train(X_train, y_train_profit, X_val, y_val_profit)
    profit_metrics = profit_model.evaluate(X_test, y_test_profit)

    # Save feature importance
    importance = profit_model.get_feature_importance()
    importance.to_csv('models/profit_feature_importance.csv', index=False)

    # === MODEL 2: Success Classification ===
    print("\n=== Training Success Classification Model ===")
    y_train_success = train_df['was_profitable']
    y_val_success = val_df['was_profitable']
    y_test_success = test_df['was_profitable']

    success_model = SuccessClassifier()
    success_model.train(X_train, y_train_success, X_val, y_val_success)
    success_metrics = success_model.evaluate(X_test, y_test_success)

    # === MODEL 3: Hold Duration Prediction ===
    print("\n=== Training Hold Duration Model ===")
    y_train_duration = train_df['actual_hold_hours']
    y_val_duration = val_df['actual_hold_hours']
    y_test_duration = test_df['actual_hold_hours']

    duration_model = DurationPredictor()
    duration_model.train(X_train, y_train_duration, X_val, y_val_duration)
    duration_metrics = duration_model.evaluate(X_test, y_test_duration)

    # === EXPORT TO ONNX ===
    print("\n=== Exporting models to ONNX ===")
    export_to_onnx(profit_model.model, feature_cols, 'models/profit_model.onnx')
    export_to_onnx(success_model.model, feature_cols, 'models/success_model.onnx')
    export_to_onnx(duration_model.model, feature_cols, 'models/duration_model.onnx')

    print("\n✅ Training complete! Models exported to models/")

    return {
        'profit': profit_metrics,
        'success': success_metrics,
        'duration': duration_metrics
    }

if __name__ == '__main__':
    main()
```

### **ONNX Export**

```python
# src/export/onnx_exporter.py
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def export_to_onnx(model, feature_names, output_path):
    """
    Export XGBoost model to ONNX format for C# inference
    """
    # Define input shape (number of features)
    initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]

    # Convert to ONNX
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=12  # Compatible with ONNX Runtime
    )

    # Save to file
    onnx.save_model(onnx_model, output_path)

    print(f"✅ Model exported to {output_path}")

    # Validate ONNX model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model validated")
```

---

## Model Deployment & Integration

### **Flask API Microservice Architecture**

The ML models are deployed as an independent Flask REST API that runs alongside the C# backend. This approach provides:

- ✅ **Simple Deployment**: No platform-specific Python DLL dependencies
- ✅ **Easy Debugging**: Standard Flask logs and error messages
- ✅ **Scalable**: Can run on separate server/container
- ✅ **Language Agnostic**: Any language can call HTTP API

### **Architecture Overview**

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

### **Step 1: Train Models**

```bash
cd /Users/sansel/Projects/CryptoArbitrage/ml_pipeline

# Activate virtual environment
source venv/bin/activate

# Train XGBoost models (creates .pkl files)
./train.sh

# Output:
# models/xgboost/profit_model.pkl
# models/xgboost/success_model.pkl
# models/xgboost/duration_model.pkl
# models/xgboost/scaler.pkl
```

### **Step 2: Start Flask ML API Server**

```bash
# Start server (runs on http://localhost:5250)
python ml_api_server.py

# Output:
# Loading XGBoost models from models/xgboost...
# ✅ Models loaded successfully
#  * Running on http://0.0.0.0:5250
```

**ml_api_server.py** (already implemented):

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from src.csharp_bridge import MLPredictor

app = Flask(__name__)
CORS(app)

predictor = MLPredictor(model_dir='models/xgboost')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'ml-api',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    opportunity = request.get_json()
    result = predictor.predict_single(opportunity)
    return jsonify(result)

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    opportunities = request.get_json()
    results = predictor.predict_batch(opportunities)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5250, debug=False)
```

### **Step 3: Test ML API**

```bash
# Health check
curl http://localhost:5250/health

# Single prediction
curl -X POST http://localhost:5250/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "fundProfit8h": -0.0242,
    "fundApr": -10.63,
    "volume24h": 1500000000,
    ...
  }'

# Response:
{
  "predicted_profit_pct": -2.42,
  "success_probability": 0.049,
  "predicted_hold_hours": 16.54,
  "composite_score": 12.3,
  "model_version": "xgboost_1.0.0"
}
```

### **Step 4: C# HTTP Client Integration**

**Services/ML/PythonMLApiClient.cs** (already implemented):

```csharp
public class PythonMLApiClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly ILogger<PythonMLApiClient> _logger;

    public PythonMLApiClient(
        IConfiguration configuration,
        ILogger<PythonMLApiClient> logger)
    {
        _logger = logger;
        var host = configuration["MLApi:Host"] ?? "localhost";
        var port = configuration["MLApi:Port"] ?? "5250";
        _baseUrl = $"http://{host}:{port}";

        _httpClient = new HttpClient
        {
            BaseAddress = new Uri(_baseUrl),
            Timeout = TimeSpan.FromSeconds(30)
        };
    }

    public async Task<bool> HealthCheckAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/health");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ML API health check failed");
            return false;
        }
    }

    public async Task<MLPredictionResult> ScoreOpportunityAsync(
        ArbitrageOpportunityDto opportunity)
    {
        var json = JsonSerializer.Serialize(opportunity, _jsonOptions);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("/predict", content);
        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<MLPredictionResult>(responseJson, _jsonOptions);
    }

    public async Task<List<MLPredictionResult>> ScoreOpportunitiesBatchAsync(
        IEnumerable<ArbitrageOpportunityDto> opportunities)
    {
        var json = JsonSerializer.Serialize(opportunities, _jsonOptions);
        var content = new StringContent(json, Encoding.UTF8, "application/json");

        var response = await _httpClient.PostAsync("/predict/batch", content);
        response.EnsureSuccessStatusCode();

        var responseJson = await response.Content.ReadAsStringAsync();
        return JsonSerializer.Deserialize<List<MLPredictionResult>>(responseJson, _jsonOptions);
    }
}
```

### **Step 5: ML Scoring Service**

**Services/ML/OpportunityMLScorer.cs** (already implemented):

```csharp
public class OpportunityMLScorer
{
    private readonly PythonMLApiClient _mlApiClient;
    private readonly ILogger<OpportunityMLScorer> _logger;

    public async Task<MLPredictionResult> ScoreOpportunityAsync(
        ArbitrageOpportunityDto opportunity)
    {
        try
        {
            return await _mlApiClient.ScoreOpportunityAsync(opportunity);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "ML scoring failed for {Symbol}", opportunity.Symbol);
            return GetDefaultPrediction();
        }
    }

    public async Task<List<MLPredictionResult>> ScoreOpportunitiesBatchAsync(
        IEnumerable<ArbitrageOpportunityDto> opportunities)
    {
        try
        {
            return await _mlApiClient.ScoreOpportunitiesBatchAsync(opportunities);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Batch ML scoring failed");
            return opportunities.Select(_ => GetDefaultPrediction()).ToList();
        }
    }

    public async Task ScoreAndEnrichOpportunitiesAsync(
        IEnumerable<ArbitrageOpportunityDto> opportunities)
    {
        var results = await ScoreOpportunitiesBatchAsync(opportunities);

        var opportunitiesList = opportunities.ToList();
        for (int i = 0; i < opportunitiesList.Count; i++)
        {
            var opp = opportunitiesList[i];
            var result = results[i];

            opp.MLPredictedProfitPercent = result.PredictedProfitPct;
            opp.MLSuccessProbability = result.SuccessProbability;
            opp.MLPredictedHoldHours = result.PredictedHoldHours;
            opp.MLCompositeScore = result.CompositeScore;
            opp.MLModelVersion = result.ModelVersion;
        }
    }
}
```

### **Step 6: Integration into OpportunityEnricher**

**Services/Arbitrage/Detection/OpportunityEnricher.cs** (already implemented):

```csharp
public class OpportunityEnricher : BackgroundService
{
    private readonly OpportunityMLScorer _mlScorer;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        while (!stoppingToken.IsCancellationRequested)
        {
            var opportunities = await _opportunityAggregator.GetOpportunitiesAsync();

            // Enrich with market data
            await EnrichWithMarketDataAsync(opportunities);

            // Enrich with ML predictions
            await _mlScorer.ScoreAndEnrichOpportunitiesAsync(opportunities);

            // Broadcast enriched opportunities
            await _broadcaster.BroadcastOpportunitiesAsync(opportunities);

            await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken);
        }
    }
}
```

### **Step 7: Configuration**

**appsettings.json**:

```json
{
  "MLApi": {
    "Host": "localhost",
    "Port": "5250"
  }
}
```

**Program.cs** (already configured):

```csharp
// Register ML services
builder.Services.AddSingleton<PythonMLApiClient>();
builder.Services.AddSingleton<OpportunityMLScorer>();

// Health check on startup
var mlApiClient = scope.ServiceProvider.GetRequiredService<PythonMLApiClient>();
var isHealthy = await mlApiClient.HealthCheckAsync();
if (isHealthy)
{
    logger.LogInformation("✅ Python ML API is available at http://localhost:5250");
}
else
{
    logger.LogWarning("⚠️ Python ML API is not available. ML predictions will be disabled.");
}
```

### **Production Deployment Options**

#### **Option 1: Systemd Service (Linux)**

```ini
# /etc/systemd/system/ml-api.service
[Unit]
Description=Crypto Arbitrage ML API
After=network.target

[Service]
Type=simple
User=appuser
WorkingDirectory=/opt/crypto-arbitrage/ml_pipeline
Environment="PATH=/opt/crypto-arbitrage/ml_pipeline/venv/bin"
ExecStart=/opt/crypto-arbitrage/ml_pipeline/venv/bin/python ml_api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ml-api
sudo systemctl start ml-api
sudo systemctl status ml-api
```

#### **Option 2: Docker Container**

```dockerfile
# Dockerfile.ml-api
FROM python:3.11-slim

WORKDIR /app
COPY ml_pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_pipeline/ .

EXPOSE 5250
CMD ["python", "ml_api_server.py"]
```

```bash
docker build -f Dockerfile.ml-api -t crypto-arbitrage-ml-api .
docker run -d -p 5250:5250 --name ml-api crypto-arbitrage-ml-api
```

#### **Option 3: Gunicorn (Production)**

```bash
# Install Gunicorn
pip install gunicorn

# Start with multiple workers
gunicorn -w 4 -b 0.0.0.0:5250 ml_api_server:app

# With logging
gunicorn -w 4 -b 0.0.0.0:5250 \
  --access-logfile /var/log/ml-api-access.log \
  --error-logfile /var/log/ml-api-error.log \
  ml_api_server:app
```

### **Monitoring & Troubleshooting**

```bash
# Check ML API health
curl http://localhost:5250/health

# View Flask logs
tail -f /tmp/ml_api_server.log

# Test single prediction
curl -X POST http://localhost:5250/predict -H "Content-Type: application/json" -d @test_opportunity.json

# Check if port is available
lsof -i :5250

# Kill process on port
kill -9 $(lsof -t -i:5250)
```

---

## Implementation Phases

### **Phase 1: Data Collection (Week 1)**

**Goals:**
- Backfill 6 months of historical data
- Generate 500,000+ simulated positions
- Start live collection

**Tasks:**
1. ✅ Create `CryptoArbitrage.HistoricalCollector` console project
2. ✅ Implement `HistoricalDataFetcher`
3. ✅ Implement `SnapshotReconstructor`
4. ✅ Implement `PositionSimulator`
5. ✅ Run backfill: `dotnet run backfill --start-date 2024-04-24 --end-date 2024-10-24`
6. ✅ Export to CSV: `dotnet run simulate --output training_data.csv`
7. ✅ Start live collection: `dotnet run live --interval 5`

**Deliverables:**
- 50,000 historical snapshots
- 500,000 simulated executions (CSV)
- Live data collection running

---

### **Phase 2: ML Model Training (Week 2)**

**Goals:**
- Train XGBoost models
- Validate model performance
- Export to ONNX

**Tasks:**
1. ✅ Setup Python environment with XGBoost, MLX
2. ✅ Implement feature engineering pipeline
3. ✅ Train profit prediction model (regression)
4. ✅ Train success classification model (binary)
5. ✅ Train hold duration model (regression)
6. ✅ Evaluate models (RMSE, AUC, MAE)
7. ✅ Generate SHAP feature importance plots
8. ✅ Export models to ONNX format
9. ✅ Validate ONNX models work in Python ONNX Runtime

**Deliverables:**
- `profit_model.onnx`
- `success_model.onnx`
- `duration_model.onnx`
- Model performance report (metrics, feature importance)

---

### **Phase 3: .NET Integration (Week 3)**

**Goals:**
- Integrate ONNX models into C# API
- Real-time ML scoring in production

**Tasks:**
1. ✅ Add `Microsoft.ML.OnnxRuntime` NuGet package
2. ✅ Implement `OpportunityMLScorer` service
3. ✅ Implement feature extraction (match Python exactly)
4. ✅ Integrate into `OpportunityEnricher`
5. ✅ Add ML fields to `ArbitrageOpportunityDto`
6. ✅ Update database schema
7. ✅ Test inference performance (latency)
8. ✅ Add logging and monitoring

**Deliverables:**
- ML scoring integrated into production API
- Real-time predictions on every opportunity
- Performance metrics (inference time < 10ms)

---

### **Phase 4: Validation & Monitoring (Week 4)**

**Goals:**
- A/B test ML vs rule-based scoring
- Monitor prediction accuracy
- Setup retraining pipeline

**Tasks:**
1. ✅ Implement prediction tracking (store predictions + actual outcomes)
2. ✅ Build A/B testing framework
3. ✅ Create dashboard for model performance monitoring
4. ✅ Setup weekly retraining cron job
5. ✅ Implement drift detection
6. ✅ Paper trade both strategies in parallel
7. ✅ Compare performance (profit, win rate, Sharpe ratio)

**Deliverables:**
- Prediction accuracy report
- A/B test results
- Automated retraining pipeline

---

## Apple Silicon Optimization

### **Why Apple Silicon (M1/M2/M3) is Perfect for This**

✅ **XGBoost is highly optimized** for ARM architecture
✅ **MLX framework** (Apple's ML library) uses Metal GPU acceleration
✅ **Training speed:** 2-5x faster than comparable x86 CPUs
✅ **Unified memory:** Fast data transfer between CPU and GPU
✅ **Energy efficient:** Train models without massive electricity costs

### **Optimized Python Setup for Apple Silicon**

```bash
# Use Homebrew Python (optimized for ARM)
brew install python@3.11

# Install XGBoost with Apple Silicon optimization
pip install xgboost

# Install LightGBM with Apple Silicon support
brew install cmake libomp
pip install lightgbm

# Optional: Apple MLX (cutting-edge, experimental)
pip install mlx mlx-data

# Verify optimization
python -c "import xgboost; print(xgboost.__version__)"
```

### **XGBoost Training with Apple Metal GPU**

```python
# Use 'hist' tree method (CPU-optimized for Apple Silicon)
model = xgb.XGBRegressor(
    tree_method='hist',  # Optimized histogram algorithm
    n_jobs=-1,           # Use all CPU cores
    # ...
)

# Alternative: Experimental GPU support (requires specific XGBoost build)
# tree_method='gpu_hist'  # Requires Metal backend (not widely available yet)
```

### **Performance Expectations (M1/M2)**

- Training 500,000 samples: **2-5 minutes**
- Inference (1 prediction): **< 1ms** (ONNX Runtime in C#)
- Feature engineering: **10-30 seconds**

---

## Cost & Resources

### **Storage Requirements**

| Data | Size | Notes |
|------|------|-------|
| Historical snapshots (6 months) | ~500 MB | Compressed JSON in PostgreSQL |
| Simulated executions CSV | ~200 MB | 500,000 rows × ~100 features |
| ONNX models | ~10 MB | All 3 models combined |
| Training data (Parquet) | ~100 MB | Compressed format |

**Total: ~1 GB** (very manageable)

### **Compute Requirements**

| Task | Time | Hardware |
|------|------|----------|
| Backfill historical data | 30-60 min | M1 Mac / Cloud VM |
| Position simulation | 10-30 min | M1 Mac / Cloud VM |
| Model training | 2-5 min | M1 Mac |
| Inference (per prediction) | < 1 ms | Any modern CPU |

### **Cloud Costs (if not using local Mac)**

- **AWS EC2 t3.medium** (backfill): ~$0.50/day
- **Python ML training**: Can run locally on Mac (free)
- **Production inference**: Negligible (< 1ms CPU time)

---

## Summary

### **Technology Choices**

✅ **Data Collection:** C# console app (reuses existing services)
✅ **ML Training:** Python + XGBoost (best ecosystem, Apple Silicon optimized)
✅ **Model Format:** ONNX (interoperable, production-ready)
✅ **Production Inference:** C# + ONNX Runtime (native performance)

### **Timeline**

- **Week 1:** Collect data (50K snapshots, 500K simulations)
- **Week 2:** Train ML models (XGBoost, export ONNX)
- **Week 3:** Integrate into .NET API
- **Week 4:** Validate, A/B test, deploy

### **Expected Results**

- **Prediction accuracy:** 70-85% (success classification)
- **Profit prediction:** MAE < 0.5% (within 0.5% of actual)
- **Inference latency:** < 10ms per opportunity
- **Improved profitability:** 10-30% vs rule-based scoring

---

## Next Steps

1. Create `CryptoArbitrage.HistoricalCollector` project
2. Implement historical data fetching
3. Run backfill for 6 months
4. Generate simulation dataset
5. Setup Python ML environment
6. Train initial models
7. Export to ONNX
8. Integrate into .NET API
