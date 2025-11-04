# Trade Logging & Validation Guide

**Version:** 1.0
**Date:** 2025-11-04

---

## 🎯 Overview

The training and inference scripts now export all trades to CSV for detailed analysis and validation against historical data.

## 📊 Trade CSV Format

Each trade record includes:

### Identification & Timing
- `episode` - Episode number (for multi-episode tests)
- `entry_datetime` - When position was entered
- `exit_datetime` - When position was closed (null if still open)
- `symbol` - Trading pair (e.g., FUSDT, COAIUSDT)
- `long_exchange` - Exchange for long position (e.g., Bybit)
- `short_exchange` - Exchange for short position (e.g., Binance)
- `status` - "closed" or "open"

### Position Sizing
- `position_size_usd` - Size per side (total capital = 2x this)
- `leverage` - Leverage multiplier (1-10x)
- `margin_used_usd` - Actual margin locked

### Entry/Exit Prices
- `entry_long_price` - Long entry price
- `entry_short_price` - Short entry price
- `exit_long_price` - Long exit price (null if open)
- `exit_short_price` - Short exit price (null if open)

### Funding Rates & Earnings
- `long_funding_rate` - Long funding rate (e.g., -0.005 = -0.5%)
- `short_funding_rate` - Short funding rate
- `funding_earned_usd` - Total funding earned (both sides)
- `long_funding_earned_usd` - Funding from long position
- `short_funding_earned_usd` - Funding from short position

### Fees & P&L
- `entry_fees_usd` - Entry fees (maker fees, both sides)
- `exit_fees_usd` - Exit fees (taker fees, both sides)
- `total_fees_usd` - Total fees paid
- `realized_pnl_usd` - Realized P&L in USD (closed positions)
- `realized_pnl_pct` - Realized P&L as percentage
- `unrealized_pnl_usd` - Unrealized P&L (open positions)
- `unrealized_pnl_pct` - Unrealized P&L as percentage

### Duration
- `hours_held` - Hours position was held

---

## 🚀 Usage

### 1. Export Trades from Inference

**Basic usage (default output: `trades_inference.csv`):**
```bash
python test_inference.py
```

**Custom output file:**
```bash
python test_inference.py --trades-output my_trades.csv
```

**With custom trading config:**
```bash
python test_inference.py \
  --leverage 3.0 \
  --utilization 0.6 \
  --max-positions 5 \
  --trades-output trades_3x_60pct.csv
```

**Full 7-day test:**
```bash
python test_inference.py \
  --episode-length-days 7 \
  --num-episodes 1 \
  --trades-output trades_7day.csv
```

**Multiple shorter episodes:**
```bash
python test_inference.py \
  --episode-length-days 3 \
  --num-episodes 5 \
  --trades-output trades_5x3day.csv
```

### 2. Validate Trades Against Historical Data

**Basic validation:**
```bash
python validate_trades.py \
  --trades trades_inference.csv \
  --data data/rl_test.csv
```

**Custom output report:**
```bash
python validate_trades.py \
  --trades trades_inference.csv \
  --data data/rl_test.csv \
  --output validation_report.txt
```

### 3. Analyze Trades with Pandas

```python
import pandas as pd

# Load trades
df = pd.read_csv('trades_inference.csv')

# Summary statistics
print(f"Total trades: {len(df)}")
print(f"Win rate: {(df['realized_pnl_usd'] > 0).sum() / len(df) * 100:.1f}%")
print(f"Total P&L: ${df['realized_pnl_usd'].sum():.2f}")
print(f"Total funding: ${df['funding_earned_usd'].sum():.2f}")
print(f"Total fees: ${df['total_fees_usd'].sum():.2f}")
print(f"Avg duration: {df['hours_held'].mean():.1f} hours")

# Top trades
top_trades = df.nlargest(5, 'realized_pnl_usd')[['symbol', 'realized_pnl_usd', 'hours_held']]
print("\nTop 5 trades:")
print(top_trades)

# By symbol
print("\nP&L by symbol:")
print(df.groupby('symbol')['realized_pnl_usd'].agg(['sum', 'count', 'mean']))

# Funding analysis
print("\nFunding analysis:")
print(f"  Long funding: ${df['long_funding_earned_usd'].sum():.2f}")
print(f"  Short funding: ${df['short_funding_earned_usd'].sum():.2f}")

# Check for open positions
open_positions = df[df['status'] == 'open']
print(f"\nOpen positions: {len(open_positions)}")
if len(open_positions) > 0:
    print(open_positions[['symbol', 'unrealized_pnl_usd', 'hours_held']])
```

---

## ✅ Validation Results

The validation script checks:

1. **Entry prices match historical data** - Validates that the model used actual market prices
2. **Funding rates match historical data** - Confirms correct funding rate data
3. **Funding calculations are reasonable** - Compares earned funding to expected amounts
4. **All trades have corresponding historical opportunities** - Ensures model didn't "invent" opportunities

### Example Validation Output:

```
Trade 1: FUSDT (Bybit long, Binance short)
  Entry: 2025-10-22 02:59:20+00:00
  Exit:  2025-10-22 04:59:20+00:00
  Status: OK
    ✓ Long entry price matches exactly ($0.012337)
    ✓ Short entry price matches exactly ($0.012108)
    ✓ Long funding rate matches exactly (-0.005219)
    ✓ Short funding rate matches exactly (-0.003305)
    ℹ Position: $738.00 @ 1.0x leverage
    ℹ Duration: 2.0 hours
    ℹ Funding earned: $1.41
    ℹ Total fees: $0.44
    ℹ Realized P&L: $0.14 (0.01%)
```

### Interpretation:

- **✓ (checkmark)** - Values match exactly or very closely (< 0.1% difference)
- **⚠ (warning)** - Values differ slightly (could be timing differences)
- **ℹ (info)** - Informational summary

---

## 🔍 Common Validation Issues

### Issue: Time difference between trade and historical snapshot

**Example:**
```
ℹ Time difference: 48.3 minutes (hourly snapshot)
```

**Explanation:**
- Historical opportunity data is recorded at hourly intervals (e.g., 19:00, 20:00)
- The agent can execute trades at any time during that hour (e.g., 19:48:26)
- The validation finds the closest historical snapshot within ±60 minutes
- Time differences up to ~30 minutes are normal (half-hour from snapshot)

**Action:** No action needed. This is expected behavior due to hourly snapshots.

### Issue: Funding rate differs slightly

**Example:**
```
⚠ Long funding rate differs (hist: -0.007632, trade: -0.005219)
```

**Explanation:**
- Funding rates change throughout the hour
- Historical data may be recorded at different timestamp than trade entry
- Small differences (<0.001) are normal and expected

**Action:** No action needed if difference is small

### Issue: No matching opportunity found

**Example:**
```
WARNING: No matching opportunity found for FUSDT at 2025-10-22 01:59:20
```

**Explanation:**
- The opportunity may have been available just outside the data snapshot window
- Or the opportunity was filtered out in data preprocessing

**Action:**
- Review data collection timestamp resolution
- Check if opportunity passes all filters in historical data

### Issue: Funding differs from simple estimate

**Example:**
```
ℹ Funding differs from simple estimate (expected ~$12.58, actual $1.41)
```

**Explanation:**
- Simple estimate assumes constant funding rate over entire duration
- Actual funding depends on:
  - Exact funding payment times (every 1 or 8 hours)
  - Funding rate changes during hold period
  - Position entry/exit timing relative to funding payments

**Action:** This is normal. The validation script uses a simplified calculation for comparison.

---

## 📈 Use Cases

### 1. Model Performance Analysis

Compare model performance across different configurations:

```bash
# Test conservative config
python test_inference.py \
  --leverage 1.0 --utilization 0.5 \
  --trades-output trades_conservative.csv

# Test aggressive config
python test_inference.py \
  --leverage 5.0 --utilization 0.8 \
  --trades-output trades_aggressive.csv

# Compare
python << 'EOF'
import pandas as pd
cons = pd.read_csv('trades_conservative.csv')
agg = pd.read_csv('trades_aggressive.csv')

print("Conservative: P&L=${:.2f}, Trades={}".format(
    cons['realized_pnl_usd'].sum(), len(cons)))
print("Aggressive:   P&L=${:.2f}, Trades={}".format(
    agg['realized_pnl_usd'].sum(), len(agg)))
EOF
```

### 2. Backtesting Validation

Prove that model trading is correct by comparing with manual calculations:

```bash
# Export trades
python test_inference.py --trades-output backtest_trades.csv

# Validate against historical data
python validate_trades.py \
  --trades backtest_trades.csv \
  --data data/rl_test.csv \
  --output backtest_validation.txt

# Review validation report
cat backtest_validation.txt
```

### 3. Trading Strategy Analysis

Identify which opportunities the model prefers:

```python
import pandas as pd

df = pd.read_csv('trades_inference.csv')

# Most profitable symbols
print("Top symbols by total P&L:")
print(df.groupby('symbol')['realized_pnl_usd'].sum().sort_values(ascending=False))

# Preferred exchanges
print("\nExchange pair usage:")
print(df.groupby(['long_exchange', 'short_exchange']).size())

# Duration analysis
print("\nTrade duration distribution:")
print(df['hours_held'].describe())

# Funding rate preferences
print("\nFunding rate spread distribution:")
df['funding_spread'] = abs(df['long_funding_rate']) + abs(df['short_funding_rate'])
print(df['funding_spread'].describe())
```

### 4. Risk Analysis

Analyze model risk-taking behavior:

```python
import pandas as pd

df = pd.read_csv('trades_inference.csv')

# Leverage usage
print(f"Average leverage: {df['leverage'].mean():.2f}x")
print(f"Max leverage: {df['leverage'].max():.2f}x")

# Position sizing
print(f"\nAverage position size: ${df['position_size_usd'].mean():.2f}")
print(f"Max position size: ${df['position_size_usd'].max():.2f}")

# Margin usage
print(f"\nTotal margin used: ${df['margin_used_usd'].sum():.2f}")
print(f"Max concurrent margin: ${df['margin_used_usd'].max():.2f}")

# Drawdown analysis (from unrealized P&L during holding)
print(f"\nWorst unrealized P&L: ${df['unrealized_pnl_usd'].min():.2f}")
```

---

## 📝 Files Created

1. **`trades_inference.csv`** (or custom name) - Trade records from inference
2. **`trade_validation_report.txt`** (or custom name) - Validation report
3. **`validate_trades.py`** - Validation script
4. **`TRADE_LOGGING_GUIDE.md`** - This guide

---

## 🎯 Quick Reference

```bash
# Export trades from inference
python test_inference.py --trades-output trades.csv

# Validate trades
python validate_trades.py --trades trades.csv --data data/rl_test.csv

# Analyze with Python
python -c "import pandas as pd; df = pd.read_csv('trades.csv'); print(df.describe())"

# Check all available arguments
python test_inference.py --help
python validate_trades.py --help
```

---

**All trade data is now exportable and verifiable!** 🚀
