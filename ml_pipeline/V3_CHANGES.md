# V3 Changes: Pure P&L Learning

**Date:** 2025-10-31
**Goal:** Remove all bonuses, let agent learn naturally from actual P&L

---

## V2 Problems (Identified from Trade Analysis)

1. **Churning**: 78.5% of trades exit <2h (too early)
   - Trades <2h: -$1,657 loss
   - Trades 2-8h: +$795 profit
   - Exit bonus incentivized premature exits

2. **Symbol Inconsistency**: MYXUSDT traded 40 times, lost -$237
   - Quality bonus rewarded high APR regardless of actual outcomes
   - Agent didn't learn which symbols actually profit

3. **COAIUSDT Timing**: Best symbol (+$217) but 7/10 worst trades
   - Agent found right symbol but entered at wrong times
   - Quality bonus gave reward before knowing outcome

4. **Reward-P&L Misalignment**:
   - Example: MYXUSDT trade gets +10 quality bonus, loses -$6 → net +4 reward (positive!)
   - Agent thought it was profitable because of bonuses

---

## V3 Fixes Implemented

### 1. Remove Exit Bonus ✅
**File**: `src/rl/environment.py:451-455`

**Before (V2)**:
```python
exit_bonus = max(0, (pnl_pct - 0.5)) * 2.0
return exit_bonus
```

**After (V3)**:
```python
# NO EXIT BONUS - let agent learn optimal exit timing from cumulative P&L
# Holding winners longer accumulates more P&L → agent naturally learns to hold
return 0.0
```

**Expected Impact**:
- Agent receives more cumulative P&L reward by holding 8h vs 2h
- Natural incentive to hold winners longer
- No artificial incentive to "take profit early"

---

### 2. Remove Quality Bonus + Add Immediate Entry Fee Penalty ✅
**File**: `src/rl/environment.py:385-399`

**Before (V2)**:
```python
entry_cost = 0.0  # No entry fee penalty

# Quality bonus based on APR
expected_profit_usd = (apr / 365 / 3 / 100) * position_size * 2
if apr > 150 and spread_pct < 0.4:
    quality_reward = expected_profit_usd * pnl_reward_scale * 0.5
elif apr < 75 or spread_pct > 0.5:
    quality_reward = expected_profit_usd * pnl_reward_scale * -0.5

return entry_cost + quality_reward
```

**After (V3)**:
```python
# IMMEDIATE ENTRY FEE PENALTY for better credit assignment
entry_fee_penalty = -position.entry_fees_paid_usd * pnl_reward_scale

# NO QUALITY BONUS - pure P&L learning
# - Good symbols → positive P&L → agent learns to enter more
# - Bad symbols → negative P&L → agent learns to avoid

return entry_fee_penalty
```

**Expected Impact**:
- Agent feels entry cost ($0.66) immediately
- Learns to be selective: only enter when expected P&L > $0.66
- COAIUSDT: -0.66 entry + 40 P&L = +39.34 total (very positive!) → agent learns this is good
- MYXUSDT: -0.66 entry - 6 P&L = -6.66 total (very negative!) → agent learns to avoid
- No forward-looking rewards, only backward-looking outcomes

---

### 3. Increase Gamma to 0.99 ✅
**File**: `train_rl_agent.py:131`

**Before (V2)**:
```python
gamma: float = 0.96
```

**After (V3)**:
```python
gamma: float = 0.99  # Value future rewards more (hold longer)
```

**Decision-Making Impact**:
```
V2 (γ=0.96):
- Exit now (+10): 10 × 1.0 = 10
- Hold 8h (+40): 40 × 0.96^8 = 40 × 0.72 = 28.8
- Ratio: 2.88x incentive

V3 (γ=0.99):
- Exit now (+10): 10 × 1.0 = 10
- Hold 8h (+40): 40 × 0.99^8 = 40 × 0.92 = 36.8
- Ratio: 3.68x incentive (27% stronger!)
```

**Expected Impact**:
- Agent values long-term rewards 27% more
- Stronger incentive to hold positions longer
- Reduced temporal discounting = better credit assignment over 8h window

---

### 4. Adjust GAE Lambda to 0.98 ✅
**File**: `train_rl_agent.py:132`

**Before (V2)**:
```python
gae_lambda: float = 0.9888
```

**After (V3)**:
```python
gae_lambda: float = 0.98  # Faster credit assignment
```

**Expected Impact**:
- Slightly more TD-like (vs Monte Carlo)
- Faster learning of which actions led to outcomes
- Better balance between bias and variance in advantage estimation

---

## Training Configuration

**Model**: V3 Pure P&L Learning
**Timesteps**: 200,000 (quick validation)
**Parallel workers**: 8
**Save directory**: `models/rl_v3/`

**Key Hyperparameters**:
- Learning rate: 5.989e-05 (unchanged)
- Gamma: **0.99** (↑ from 0.96)
- GAE Lambda: **0.98** (↓ from 0.9888)
- P&L reward scale: 1.0 (unchanged)
- Entry bonus: **0.0** (removed)
- Exit bonus: **0.0** (removed)
- Quality bonus: **0.0** (removed)

---

## Expected V3 Results

### Behavioral Changes:

1. **Longer Hold Duration**:
   - V2: 1.88h avg → V3: **4-6h avg**
   - Less churning (55.7% → <30% of trades <2h)

2. **Symbol Selection**:
   - Agent naturally learns: COAIUSDT profitable → enter more
   - Agent naturally learns: MYXUSDT unprofitable → avoid
   - No need for manual filtering!

3. **Entry Selectivity**:
   - Entry fee penalty encourages selectivity
   - Only enters when confident P&L will exceed $0.66 cost
   - Fewer total trades but higher quality

4. **Exit Timing**:
   - No exit bonus → agent optimizes cumulative P&L
   - Holds winners longer (accumulates more funding)
   - Exits losers appropriately (stop-loss still active)

### Performance Targets:

| Metric | V2 Actual | V3 Target | Improvement |
|--------|-----------|-----------|-------------|
| **Win Rate** | 10% | 30-40% | 3-4× better |
| **Avg P&L** | -0.86% | +0.3-0.8% | Positive! |
| **Avg Duration** | 1.88h | 4-6h | 2-3× longer |
| **Trades/Episode** | 14.9 | 6-10 | Less churning |
| **MYXUSDT trades** | 40 (27%) | <5 (<5%) | Natural avoidance |
| **COAIUSDT trades** | 30 (20%) | 35-40 (40%) | Natural preference |

---

## Validation Plan

After 200k timesteps training completes (~10 minutes):

1. **Evaluate on test set** (10 episodes)
   - Compare win rate to V2 (10%)
   - Compare avg P&L to V2 (-0.86%)
   - Check avg hold duration vs V2 (1.88h)

2. **Analyze trades**:
   - Run `analyze_trades.py` on V3 CSV
   - Check hold duration distribution
   - Check symbol selection patterns
   - Compare to V2 analysis

3. **Decision**:
   - If V3 ≥ +0.3% P&L: ✅ **CONTINUE TO 1M TIMESTEPS**
   - If V3 < +0.3% but better than V2: 🔄 **ADJUST & RETRY**
   - If V3 worse than V2: ❌ **INVESTIGATE BUGS**

---

## Key Insights from V2 Analysis

1. **Data is good** (32.1% HIGH QUALITY opportunities, realistic max +348% P&L)
2. **Agent found right symbol** (COAIUSDT)
3. **Agent just exited too early** (78.5% trades <2h)
4. **Bonuses broke learning** (rewarded expectations, not outcomes)

**V3 Solution**: Remove all bonuses, let reality teach the agent!

---

## TensorBoard Monitoring

With new `TradingMetricsCallback`, we can now monitor in real-time:
- `trading/episode_pnl_pct` - Actual P&L per episode
- `trading/episode_reward` - Episode reward
- `trading/reward_per_pnl` - Alignment metric (should be stable)
- `trading/trades_count` - Number of trades per episode
- `trading/win_rate` - Win rate per episode

**What to watch for**:
- P&L should trend upward (not stay negative like V2)
- Reward and P&L should move together (alignment)
- Trades per episode should decrease (less churning)
- Win rate should increase above 10%

---

**Status**: Training in progress (200k timesteps, ~10 min ETA)
**Next**: Evaluate and compare to V2 baseline
