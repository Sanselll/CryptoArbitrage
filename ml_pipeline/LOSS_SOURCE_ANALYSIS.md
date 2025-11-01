# Loss Source Analysis - V2 Model

**Date:** 2025-10-31
**Model:** V2 (with double-counting fix, proportional quality signals)
**Result:** -0.86% avg P&L, 10% win rate, 149 total trades

---

## Executive Summary

**ROOT CAUSE IDENTIFIED: Agent is CHURNING + Trading Wrong Symbols**

1. **Churning (55.7% of trades <2h)**: Agent exits too quickly, pays fees but misses profit
2. **Symbol Inconsistency**: COAIUSDT wins overall but has 7/10 worst trades (agent enters at wrong times)
3. **MYXUSDT catastrophe**: 40 trades, -$237 total loss (16% of all trades, 27% of losses)
4. **Duration mismatch**: Holds <2h lose -$1,657, holds 2-8h profit +$795

---

## Detailed Findings

### 1. Trade Statistics

| Metric | Value |
|--------|-------|
| Total trades | 149 |
| Win rate | 35.6% (53 wins, 96 losses) |
| Total P&L | **-$862** |
| Avg P&L per trade | -$5.79 |
| Avg duration | 1.88 hours |
| Total fees paid | $254 |

### 2. Performance by Duration

| Duration | Trades | Total P&L | Avg P&L | Win Rate |
|----------|--------|-----------|---------|----------|
| **0-2h** | 117 (78.5%) | **-$1,657** | -$14.17 | 35.9% |
| **2-8h** | 32 (21.5%) | **+$795** | +$24.85 | 34.4% |

**KEY INSIGHT**: Trades held 2-8h are PROFITABLE (+$795), but agent exits 78.5% of trades before 2h!

### 3. Performance by Symbol

#### Top 5 Profitable Symbols:
1. **COAIUSDT**: +$217 (30 trades, 46.7% win rate) ✅ **BEST SYMBOL**
2. MIRAUSDT: +$21 (1 trade, 100% win rate)
3. EDENUSDT: +$19 (1 trade, 100% win rate)
4. LINEAUSDT: +$10 (1 trade, 100% win rate)
5. IPUSDT: +$9 (1 trade, 100% win rate)

#### Top 5 Losing Symbols:
1. **MYXUSDT**: **-$237** (40 trades, 37.5% win rate) 💀 **WORST SYMBOL**
2. FFUSDT: -$232 (6 trades, 16.7% win rate)
3. DOODUSDT: -$130 (5 trades, 40.0% win rate)
4. MUSDT: -$115 (22 trades, 36.4% win rate)
5. SUPERUSDT: -$99 (3 trades, 0% win rate)

**KEY INSIGHT**: Agent correctly identifies COAIUSDT as good symbol BUT also trades MYXUSDT heavily (40 trades!) which loses -$237.

### 4. Catastrophic Episodes

#### Episode 2: -$401 loss (26 trades)
- **Problem**: 7 COAIUSDT trades, 3 big losers (-$110, -$106, -$104)
- **Duration**: All losers were 1-3h holds
- **Interpretation**: Agent entered COAIUSDT at WRONG TIMES (bad price movements)

#### Episode 4: -$445 loss (22 trades)
- **Problem**: MUSDT (-$164), FFUSDT (-$163), MYXUSDT (-$70)
- **Duration**: All losers were 1-2h holds
- **Interpretation**: Agent entered low-quality symbols and exited too early

### 5. Worst 10 Trades (All Short Duration!)

| Rank | Symbol | P&L | Duration | Episode |
|------|--------|-----|----------|---------|
| 1 | MUSDT | -$164 | **2h** | 4 |
| 2 | FFUSDT | -$163 | **2h** | 4 |
| 3 | COAIUSDT | -$156 | **1h** | 8 |
| 4 | COAIUSDT | -$122 | **1h** | 8 |
| 5 | DOODUSDT | -$120 | **2h** | 3 |
| 6 | COAIUSDT | -$111 | **1h** | 2 |
| 7 | COAIUSDT | -$106 | **3h** | 2 |
| 8 | COAIUSDT | -$104 | **1h** | 2 |
| 9 | COAIUSDT | -$85 | **2h** | 2 |
| 10 | FFUSDT | -$73 | **2h** | 9 |

**PATTERN**: All worst trades are 1-3h holds. 7/10 are COAIUSDT (supposedly the best symbol!)

### 6. Fee Impact Analysis

- **Gross P&L (before fees)**: -$609
- **Total fees**: $254
- **Net P&L (after fees)**: -$862
- **Fee impact**: 41.7% of gross P&L

**Short trades (<2h):**
- Count: 83 (55.7% of all trades)
- Total fees: $162
- Net P&L: -$814

**💀 AGENT IS CHURNING**: Exiting too early, paying fees, missing profit accumulation.

---

## Root Cause Analysis

### Why is the agent losing money?

#### 1. **Exit Bonus Incentivizes Premature Exits**

**From `environment.py:406-441`:**
```python
def _exit_position(self, position_idx: int) -> float:
    pnl_pct = position.unrealized_pnl_pct
    # Exit bonus for profitable exits
    exit_bonus = max(0, (pnl_pct - 0.5)) * 2.0
    return exit_bonus
```

**Problem**:
- Agent gets **exit bonus** for exiting at +0.6% P&L
- But if it held for 8h, it could get +2% P&L
- Exit bonus creates incentive to "take profit early"
- Data shows: 78.5% of trades exit <2h (too early!)

**Evidence**:
- Trades held 2-8h: **+$795 profit**
- Trades held <2h: **-$1,657 loss**
- If agent held ALL trades for 2-8h instead of <2h, total P&L could be +$795!

#### 2. **Quality Bonuses Reward Wrong Opportunities**

**From `environment.py:374-407`:**
```python
# Entry quality bonus
if apr > 150 and spread_pct < 0.4:
    quality_reward = expected_profit_usd * pnl_reward_scale * 0.5
```

**Problem**:
- COAIUSDT likely has APR>150 (gets quality bonus)
- MYXUSDT also has high APR (gets quality bonus)
- But COAIUSDT has inconsistent execution (7/10 worst trades)
- MYXUSDT is consistently losing (-$237 total)
- Quality bonus rewards APR but ignores:
  - Price volatility risk
  - Liquidity issues
  - Execution slippage
  - Optimal entry timing

**Evidence**:
- COAIUSDT: 46.7% win rate (inconsistent - timing matters!)
- MYXUSDT: 37.5% win rate but agent trades it 40 times!
- Agent doesn't learn which symbols work, just which have high APR

#### 3. **Agent Hasn't Learned to Hold Positions**

**The reward structure doesn't incentivize holding:**
- Hourly P&L reward: +$5-10 per hour
- Quality entry bonus: +$10 one-time
- Exit bonus: +2-5 for profitable exit

**Optimal strategy (from agent's perspective):**
1. Enter high APR opportunity → get +$10 quality bonus
2. Hold 1-2h until slightly profitable → get +$10-20 P&L reward
3. Exit with +0.6% P&L → get +$2 exit bonus
4. Total reward: ~+$22-32

**Better strategy (ignored by agent):**
1. Enter high APR opportunity → get +$10 quality bonus
2. Hold 8h until funding accumulates → get +$40-80 P&L reward
3. Exit with +2% P&L → get +$8 exit bonus
4. Total reward: ~+$58-98

**Why agent doesn't do this:**
- No explicit "hold duration bonus" for 8h+
- Exit bonus available immediately at +0.5% P&L
- Impatience encouraged by reward structure

---

## Why COAIUSDT Works Sometimes But Not Always

COAIUSDT is the best symbol (+$217 profit, 46.7% win rate) BUT:
- **7/10 worst trades are COAIUSDT** (-$798 total from worst 7)
- **Remaining 23 COAIUSDT trades: +$1,015 profit**

**Interpretation**:
- COAIUSDT is a GREAT symbol when entered at right time
- COAIUSDT is TERRIBLE when entered at wrong time
- Agent hasn't learned **WHEN** to enter COAIUSDT
- Quality bonus just says "APR>150" but doesn't check:
  - Current price trend
  - Recent funding rate changes
  - Spread stability
  - Liquidity conditions

---

## Comparison to Theoretical Maximum

From `analyze_max_profit.py`:
- **Realistic maximum P&L**: +348.44% per episode
- **V2 actual P&L**: -0.86% per episode
- **Gap**: 349.30%

**Why such a huge gap?**
1. Agent exits 78.5% of trades <2h (misses funding accumulation)
2. Agent trades MYXUSDT heavily (40 trades, -$237 loss)
3. Agent enters COAIUSDT at wrong times (7 catastrophic trades)
4. Fees erode 41.7% of gross P&L (churning)

---

## Recommendations for V3

### Fix 1: Remove Exit Bonus (CRITICAL)

**Problem**: Exit bonus incentivizes premature exits
**Solution**: Remove exit bonus entirely

```python
def _exit_position(self, position_idx: int) -> float:
    # Close position (updates portfolio)
    realized_pnl = self.portfolio.close_position(...)
    # NO EXIT BONUS - let agent learn optimal exit timing from P&L alone
    return 0.0
```

**Expected impact**: Agent will hold positions longer (2-8h instead of <2h)

### Fix 2: Add Hold Duration Bonus

**Problem**: No incentive to hold for 8h (funding payment interval)
**Solution**: Add bonus for holding positions 6-8h

```python
# In step() method, when updating positions hourly
if position.duration_hours >= 6 and position.unrealized_pnl_pct > 0:
    hold_duration_bonus = +2.0  # Encourage holding winners
    reward += hold_duration_bonus
```

**Expected impact**: Agent learns to hold profitable positions longer

### Fix 3: Filter Out Bad Symbols

**Problem**: MYXUSDT causes 27% of losses
**Solution**: Remove MYXUSDT from training data or add strong penalty

```python
# Option A: Filter data
df = df[~df['symbol'].isin(['MYXUSDT', 'FFUSDT', 'DOODUSDT'])]

# Option B: Add penalty
if symbol in ['MYXUSDT', 'FFUSDT', 'DOODUSDT']:
    quality_reward = -5.0  # Strong penalty for bad symbols
```

**Expected impact**: Eliminate 40 losing MYXUSDT trades (-$237 loss)

### Fix 4: Make Quality Signals Weaker (Already Tried, But Go Further)

**Problem**: Quality bonuses still too strong, reward wrong behavior
**Solution**: Remove quality bonuses entirely, let P&L be the ONLY signal

```python
# Delete lines 379-406 in environment.py
# Pure P&L learning - agent learns what actually works, not what's "expected"
return 0.0  # No entry bonus
```

**Expected impact**: Agent learns from actual outcomes, not predictions

### Fix 5: Increase Minimum Hold Time

**Problem**: 55.7% of trades <2h (churning)
**Solution**: Add penalty for exiting <2h

```python
def _exit_position(self, position_idx: int) -> float:
    duration_hours = (self.current_time - position.entry_time).total_seconds() / 3600

    if duration_hours < 2.0:
        # Penalty for exiting too early (before funding payment)
        early_exit_penalty = -3.0
        return early_exit_penalty
    else:
        return 0.0
```

**Expected impact**: Reduce churning from 55.7% to <20%

---

## Expected V3 Results

**With fixes applied:**
- Win rate: 10% → **40-50%** (better hold duration)
- Avg P&L: -0.86% → **+0.5-1.0%** (eliminate MYXUSDT, hold longer)
- Avg duration: 1.88h → **4-6h** (optimal for funding accumulation)
- Trades per episode: 14.9 → **8-10** (less churning)
- Reward-P&L alignment: Strong positive correlation

**Key improvements:**
1. **No more churning**: Agent holds 2-8h (profitable zone)
2. **No MYXUSDT trades**: Eliminate -$237 loss source
3. **Better COAIUSDT timing**: P&L-driven learning finds optimal entry
4. **Less variance**: Fewer trades, longer holds, less fee impact

---

## Summary

**V2 failed because:**
1. ✅ Agent CORRECTLY identified COAIUSDT as best symbol
2. ❌ Agent exits TOO EARLY (78.5% of trades <2h)
3. ❌ Agent trades MYXUSDT heavily (40 trades, -$237 loss)
4. ❌ Agent enters COAIUSDT at WRONG TIMES (7 catastrophic trades)
5. ❌ Exit bonus incentivizes premature exits
6. ❌ Quality bonuses reward APR, not actual profit

**V3 fixes:**
1. Remove exit bonus
2. Add hold duration bonus (6-8h)
3. Filter out MYXUSDT/FFUSDT/DOODUSDT
4. Remove quality bonuses (pure P&L learning)
5. Penalty for exiting <2h

**Expected outcome**: 40-50% win rate, +0.5-1.0% avg P&L

---

**Next Steps:**
1. Implement V3 fixes
2. Retrain for 1M timesteps
3. Evaluate on test set
4. Compare to V2 and theoretical maximum
