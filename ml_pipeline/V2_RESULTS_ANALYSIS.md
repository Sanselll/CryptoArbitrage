# V2 Training Results Analysis

**Date:** 2025-10-31
**Model:** V2 with reward fixes (double-counting removed, proportional quality signals)
**Status:** ❌ FAILED TARGETS (Win rate: 10%, P&L: -1.39%)

---

## Training Summary

### Training Metrics (1M timesteps, 42 minutes)
- **Final Training Reward**: +919 (rolling mean)
- **Final Eval Reward**: +2,448 ± 222
- **Explained Variance**: 0.854 (excellent value function)
- **Learning**: Stable, consistent improvement throughout training

### Test Set Evaluation (10 episodes)
| Metric | V2 Result | Target | Status |
|--------|-----------|--------|--------|
| **Average Reward** | +8,112 ± 521 | N/A | ✅ HIGH |
| **Average P&L** | **-1.39% ± 1.31%** | >0.5% | ❌ FAILED |
| **Win Rate** | **10%** | >50% | ❌ FAILED |
| Episodes | 10 | 10 | ✅ |
| Avg Length | 72 steps | 72 | ✅ |

---

## Critical Problem: Reward-P&L Misalignment

### The Issue
**Agent receives +8,112 reward but loses -1.39% P&L**

This is a **SEVERE misalignment** between what the agent is optimizing (reward) and what we care about (P&L).

### Comparison to V1
| Metric | V1 (Before Fixes) | V2 (After Fixes) | Change |
|--------|-------------------|------------------|--------|
| Avg Reward | ~+50 | +8,112 | **+16,124%** 🎉 |
| Avg P&L | -0.24% | -1.39% | **-479% worse** ❌ |
| Win Rate | 30% | 10% | **-67% worse** ❌ |

**The fixes made rewards better but P&L WORSE!**

---

## Root Cause Analysis

### Problem 1: Quality Bonuses TOO STRONG

**Quality bonus calculation** (environment.py:390-400):
```python
# For APR=329 (avg HIGH QUALITY), position_size=$3,333:
expected_profit_8h_pct = (329 / 365 / 3) = 0.30%
expected_profit_usd = (0.30 / 100) * 3333 * 2 = $20.00
quality_reward = 20.00 * 1.0 * 0.5 = +10.00 per entry
```

**P&L signal** (hourly during hold):
```python
# Actual P&L change per hour (varies, but typically small):
pnl_change_usd = +$5 to +$10 per hour (for HIGH QUALITY)
pnl_reward = pnl_change * 1.0 = +5 to +10 per hour
```

**Analysis**:
- Quality bonus: **+10 reward at entry** (one-time, EXPECTED profit)
- P&L reward: **+5-10 per hour** (ongoing, ACTUAL profit)
- For 8h hold: Total P&L reward = +40 to +80
- **Quality bonus = 12.5-25% of total reward**

**The problem**: Quality bonus rewards EXPECTED profit based on APR, but:
1. APR is just a **prediction** (funding rate extrapolated)
2. Actual profit depends on **price movement**, **liquidity**, **execution**
3. If actual profit ≠ expected profit, agent learns wrong signal

### Problem 2: Agent Overfitting to ONE Symbol

**Top 5 trades - ALL COAIUSDT**:
```
1. COAIUSDT | +$152.50 (+2.29%) | 2.0h
2. COAIUSDT | +$92.43 (+1.42%) | 1.0h
3. COAIUSDT | +$76.02 (+1.16%) | 1.0h
4. COAIUSDT | +$69.33 (+1.04%) | 3.0h
5. COAIUSDT | +$68.86 (+1.04%) | 3.0h
```

**Analysis**:
- Agent found COAIUSDT has high APR
- Quality bonus strongly rewards entering COAIUSDT
- But COAIUSDT might have:
  - High spread (execution slippage)
  - Poor liquidity (can't fill orders)
  - Volatile price (funding profit erased by price loss)
  - Funding rate not realized (rate changes before payment)

**Result**: Agent spams COAIUSDT entries for quality bonuses, ignores actual P&L

### Problem 3: Expected vs Actual Profit Gap

**Theory** (from analyze_max_profit.py):
- HIGH QUALITY opportunities (APR>150, spread<0.4%): 32.1% of dataset
- Average APR: 328.83
- Expected profit: +348.44% per episode (realistic max)

**Reality** (test set):
- Agent selects opportunities meeting HIGH QUALITY criteria
- But achieves -1.39% P&L
- **Gap: 349.83% between theory and reality**

**Why?**:
1. **APR ≠ Realized Funding**: Funding rate changes during hold period
2. **Spread matters more**: 0.4% spread can erase 8h of 329 APR funding
3. **Price movement risk**: Long-short hedge not perfect, net exposure exists
4. **Execution issues**: Test data may not reflect actual fills
5. **COAIUSDT specific**: This symbol may be anomalous (illiquid, manipulated)

---

## What Went Wrong with V2 Fixes?

### Fix #3 Was a Mistake: Proportional Quality Signals

**Intent**: Scale quality signals to match P&L magnitude
**Implementation**: Bonus = 50% of expected 8h profit
**Result**: Quality signals TOO influential, drowning out actual P&L

**The error**:
- We calculated expected profit based on APR
- We rewarded 50% of that expected profit at entry
- But APR is often **wrong** (rates change, spreads matter, execution varies)
- Agent learned to optimize expected profit (quality bonus), not actual profit (P&L)

**Better approach**:
- ❌ Don't reward expected profit (forward-looking, uncertain)
- ✅ Only reward actual profit (backward-looking, certain)
- ✅ Use quality signals as WEAK hints, not strong incentives
- ✅ Let P&L dominate (it's the ground truth)

---

## Solution: V3 Fixes

### Fix 1: Remove or Drastically Reduce Quality Bonuses

**Option A - Remove Completely**:
```python
# Delete lines 379-406 entirely
# Let agent learn purely from P&L (ground truth)
return 0.0  # No entry reward
```

**Option B - Make VERY Weak** (recommended):
```python
# Fixed tiny bonuses (not proportional)
if apr > 150 and spread_pct < 0.4:
    quality_reward = +0.5  # Weak hint
elif apr < 75 or spread_pct > 0.5:
    quality_reward = -0.5  # Weak warning
return quality_reward
```

**Rationale**:
- P&L signals are ±50-200 per episode
- Quality hints should be ±0.5 (1% of P&L)
- Agent learns 99% from actual profit, 1% from quality hints

### Fix 2: Increase P&L Reward Scale Back to 3.0

We reduced it from 3.0 → 1.0 to reduce variance, but this made P&L signals too weak relative to quality bonuses.

**Change**:
```python
pnl_reward_scale: float = 3.0  # Back to original
```

**Rationale**:
- With weak quality signals (±0.5), P&L needs to dominate
- Scale of 3.0 makes ±150-600 reward range
- Quality signals now 0.08-0.33% of total (negligible)

### Fix 3: Add Diversity Penalty

Prevent agent from spamming one symbol:

```python
# After entering position
if len(set([p.symbol for p in self.portfolio.positions])) < len(self.portfolio.positions):
    # Multiple positions on same symbol
    diversity_penalty = -1.0
    reward += diversity_penalty
```

### Fix 4: Stronger Spread Penalty

Current quality check: `spread_pct < 0.4%`
But COAIUSDT trades show spread may be the real killer.

**Change**:
```python
# In quality check
if apr > 150 and spread_pct < 0.2:  # STRICTER: 0.4 → 0.2
    quality_reward = +0.5
```

---

## Expected V3 Results

With fixes:
- **Win Rate**: 10% → 40-50% (P&L-driven learning)
- **Avg P&L**: -1.39% → +0.3-0.8% (below realistic max, but positive)
- **Reward-P&L alignment**: Strong positive correlation
- **Symbol diversity**: Multiple symbols, not just COAIUSDT

If still fails:
- Remove quality bonuses entirely (pure P&L learning)
- Filter training data (remove low-liquidity symbols like COAIUSDT)
- Longer episodes (5-7 days for more funding accumulation)
- Add price movement features (predict execution quality)

---

## Key Learnings from V2

1. **High rewards ≠ Good performance**: Agent optimized wrong objective
2. **Expected profit ≠ Actual profit**: APR predictions are unreliable
3. **Quality signals can be TOO strong**: Drowning out ground truth (P&L)
4. **Proportional rewards backfired**: Scaled quality signals too aggressively
5. **Symbol concentration is a red flag**: Agent gaming one anomalous symbol
6. **Reward engineering is hard**: Small changes have massive unintended effects

---

## Next Steps

1. ✅ Implement V3 fixes (weak quality signals, higher P&L scale)
2. ✅ Add diversity penalty and stricter spread threshold
3. ✅ Retrain with V3 config
4. ✅ Evaluate on test set
5. If V3 succeeds (>50% win rate, >0.5% P&L):
   - Deploy to paper trading
   - Monitor live performance
6. If V3 fails:
   - Remove quality bonuses entirely
   - Filter training data (symbol liquidity, spread)
   - Consider supervised pre-training (learn from best historical trades)

---

**Author:** Claude Code
**Status:** V2 complete but failed targets, V3 fixes ready
**Time spent:** 45 min training, 2 hours total analysis and fixes
