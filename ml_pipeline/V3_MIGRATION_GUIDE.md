# RL Model V3 Migration Guide

## Overview

The V3 refactoring streamlines the RL model from **301→203 dimensions** (32% reduction) by removing redundant features and adding velocity-based temporal features.

**Key Changes:**
- Portfolio: 6→3 features (removed historical metrics)
- Execution: 100→85 features (5×20→5×17, added velocities)
- Opportunity: 190→110 features (10×19→10×11, removed market quality)

---

## Breaking Changes

### 1. **Model Checkpoints Not Compatible**

❌ **V2 models cannot be loaded into V3 architecture**

The network dimensions have changed:
```
Portfolio Encoder:  6→3 input dimensions
Execution Encoder:  20→17 features per slot
Opportunity Encoder: 19→11 features per slot
```

**Action Required:**
- Retrain models from scratch with V3 architecture
- Old checkpoints in `trained_models/rl/` are incompatible

### 2. **Feature Scaler Updated**

The opportunity feature scaler expects **11 features** (was 19).

**Files Changed:**
- `trained_models/rl/feature_scaler_v2.pkl` (retrained)
- Training scripts now use `--feature-scaler-path trained_models/rl/feature_scaler_v2.pkl`

**Action Required:**
- Ensure you're using the new `feature_scaler_v2.pkl`
- Retrain if needed: `python scripts/fit_feature_scaler.py`

---

## Feature Changes

### Portfolio Features (6→3)

**Removed:**
- `avg_position_pnl_pct` - Violates Markov property (depends on closed positions)
- `total_pnl_pct` - Violates Markov property (cumulative metric)
- `max_drawdown_pct` - Violates Markov property (historical peak)

**Kept:**
```python
1. num_positions_ratio     # Current positions / max_positions
2. min_liquidation_distance # Closest position to liquidation
3. capital_utilization     # Total notional / capital
```

**Rationale:** Historical portfolio metrics don't help the agent make better decisions in the current timestep.

---

### Execution Features (20→17 per slot)

**Removed (9 features):**
- `net_funding_ratio` - Redundant with estimated_funding_8h_pct
- `net_funding_rate` - Confusing (who pays whom unclear)
- `funding_efficiency` - Not actionable
- `entry_spread_pct` - Entry conditions don't affect exit decisions
- `long_pnl_pct` / `short_pnl_pct` - Redundant (net_pnl_pct captures combined P&L)
- `pnl_velocity` (old) - Replaced with better velocity features
- `peak_drawdown` - Violates Markov property
- `is_old_loser` - Binary feature, poor signal

**Added (6 NEW features):**
```python
4. estimated_pnl_pct         # Price P&L only (no fees, no funding)
5. estimated_pnl_velocity    # Change in price P&L (trend signal)
6. estimated_funding_8h_pct  # Expected funding profit in next 8h
7. funding_velocity          # Change in funding estimate (rate trend)
8. spread_pct                # Current price spread
9. spread_velocity           # Change in spread (converging/diverging)
17. pnl_imbalance            # (long_pnl - short_pnl) / 200 (directional exposure)
```

**Updated:**
- `hours_held`: Linear /72 → **log normalization** `log(hours+1) / log(73)`
- `APR normalization`: ±100% → **±5000%** (handles extreme crypto rates)

**New Feature Calculations:**

```python
# estimated_funding_8h_pct
long_payments_8h = 8.0 / long_funding_interval_hours
short_payments_8h = 8.0 / short_funding_interval_hours
long_funding_8h = -long_funding_rate * long_payments_8h
short_funding_8h = short_funding_rate * short_payments_8h
estimated_funding_8h_pct = (long_funding_8h + short_funding_8h) * 100

# Velocity features (calculated at end of each timestep)
estimated_pnl_velocity = (current_estimated_pnl_pct - prev_estimated_pnl_pct) / 100
funding_velocity = (current_funding_8h_pct - prev_funding_8h_pct) / 100
spread_velocity = current_spread_pct - prev_spread_pct
```

---

### Opportunity Features (19→11 per slot)

**Removed (9 features):**
- `long_funding_rate` / `short_funding_rate` - Raw rates (already in projections)
- `long_funding_interval_hours` / `short_funding_interval_hours` - Not needed
- `volume_24h` - Market quality (pre-filtered upstream)
- `bidAskSpreadPercent` - Market quality
- `orderbookDepthUsd` - Market quality
- `estimatedProfitPercentage` - Redundant with fund_profit_8h
- `positionCostPercent` - Redundant with leverage

**Kept (10 features):**
```python
1-6. Profit/APR projections:
     - fund_profit_8h
     - fundProfit8h24hProj
     - fundProfit8h3dProj
     - fund_apr
     - fundApr24hProj
     - fundApr3dProj

7-10. Spread metrics:
      - spread30SampleAvg
      - priceSpread24hAvg
      - priceSpread3dAvg
      - spread_volatility_stddev
```

**Added (1 feature):**
```python
11. apr_velocity = fund_profit_8h - fundProfit8h24hProj  # Funding rate trend
```

**Rationale:** Assumes market quality is pre-filtered upstream (only good opportunities reach the agent).

---

## Code Migration

### Updated Files

**Core RL:**
- ✅ `models/rl/core/portfolio.py` - Added velocity tracking fields and methods
- ✅ `models/rl/core/environment.py` - Updated observation space (301→203)
- ✅ `models/rl/networks/modular_ppo.py` - Updated encoder dimensions

**Training:**
- ✅ `train_ppo.py` - Uses `feature_scaler_v2.pkl`
- ✅ `scripts/fit_feature_scaler.py` - Extracts 11 features (was 19)

**Inference:**
- ✅ `test_inference.py` - Updated for V3 dimensions
- ✅ `server/inference/rl_predictor.py` - Complete V3 feature extraction

**Backend (C#):**
- ✅ `Services/ML/RLPredictionService.cs` - Comments updated (no code changes needed)

**Tests:**
- ✅ `tests/test_v3_features.py` - Comprehensive V3 unit tests

---

## Training New Models

### 1. Retrain Feature Scaler (if needed)

```bash
cd ml_pipeline
python scripts/fit_feature_scaler.py
# Output: trained_models/rl/feature_scaler_v2.pkl
```

### 2. Train V3 Model

```bash
# Standard training
python train_ppo.py \
  --num-episodes 3000 \
  --feature-scaler-path trained_models/rl/feature_scaler_v2.pkl \
  --checkpoint-dir checkpoints/v3_model

# Quick smoke test (20 episodes)
python train_ppo.py \
  --num-episodes 20 \
  --episode-length-days 3 \
  --checkpoint-dir checkpoints/v3_smoke_test
```

### 3. Test Inference

```bash
python test_inference.py \
  --checkpoint checkpoints/v3_model/final_model.pt \
  --feature-scaler-path trained_models/rl/feature_scaler_v2.pkl
```

---

## Verification Checklist

### ✅ V3 Implementation Complete

- [x] Position class has velocity tracking fields (`prev_estimated_pnl_pct`, etc.)
- [x] Environment observation space is 203 dimensions
- [x] ModularPPO network accepts 203-dim input
- [x] Feature scaler expects 11 features
- [x] ML API (rl_predictor.py) implements V3 features
- [x] Unit tests pass (`tests/test_v3_features.py`)

### 🧪 Testing

```bash
# Run V3 unit tests
pytest tests/test_v3_features.py -v

# Test velocity tracking
pytest tests/test_v3_features.py::TestV3VelocityTracking -v

# Test network architecture
pytest tests/test_v3_features.py::TestV3NetworkArchitecture -v

# Test feature dimensions
pytest tests/test_v3_features.py::TestV3ObservationSpace -v
```

---

## Performance Impact

### Model Size

- **Network Parameters:** ~791K (unchanged - same hidden dimensions)
- **Observation Size:** 301→203 dims (32% reduction)
- **Memory:** Reduced batch memory usage (~30% less)

### Training Speed

- **Faster forward pass** due to smaller input vectors
- **Faster data loading** (fewer features to scale)
- **Expected:** ~15-20% faster training

### Quality

- **Better Markov compliance** (removed historical metrics)
- **Richer temporal information** (velocity features)
- **Clearer funding signals** (estimated_funding_8h_pct replaces confusing net_funding_rate)

---

## Troubleshooting

### "RuntimeError: size mismatch for portfolio_encoder"

**Cause:** Trying to load V2 checkpoint into V3 model

**Solution:** Retrain model with V3 architecture (checkpoints not compatible)

### "KeyError: 'long_funding_rate'" in opportunity features

**Cause:** Using old test data without required columns

**Solution:** Ensure test data has all required columns (see `test_v3_features.py` fixture)

### Feature scaler expects 19 features, got 11

**Cause:** Using old `feature_scaler.pkl` instead of `feature_scaler_v2.pkl`

**Solution:** Retrain scaler with `python scripts/fit_feature_scaler.py`

---

## Rollback Procedure

If you need to revert to V2:

1. **Checkout V2 code:**
   ```bash
   git checkout <commit-before-v3>
   ```

2. **Use old feature scaler:**
   ```bash
   # V2 uses feature_scaler.pkl (19 features)
   python train_ppo.py --feature-scaler-path trained_models/rl/feature_scaler.pkl
   ```

3. **Load V2 checkpoints:**
   ```bash
   python test_inference.py --checkpoint <v2_checkpoint>.pt
   ```

---

## Questions?

For implementation details, see:
- `RL_FEATURE_REFACTORING_PLAN.txt` - Detailed V3 specification
- `tests/test_v3_features.py` - Unit tests with examples
- Git commit history for specific changes

---

**Migration completed:** 2025-11-13
**V3 Refactoring:** Portfolio(6→3), Execution(100→85), Opportunity(190→110) = 301→203 dims
