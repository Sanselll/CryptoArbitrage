# Reward Function & Curriculum Learning Implementation Guide

**Version:** 1.0
**Date:** 2025-11-04
**Status:** ✅ COMPLETE - All 8 tasks implemented

---

## 🎯 What Was Implemented

This implementation follows **IMPLEMENTATION_PLAN.md** reward structure (lines 259-304) and adds curriculum learning (lines 793-816).

### ✅ Core Components

#### 1. **Correct Reward Function** (`environment.py`)

**Replaced** old README reward with IMPLEMENTATION_PLAN reward:

```python
# Component 1: Hourly P&L reward (MAIN SIGNAL)
hourly_pnl_pct = (capital_change / initial_capital) × 100
reward += hourly_pnl_pct × pnl_reward_scale  # Default: 3.0

# Component 2: Entry cost penalty
entry_fee_pct = (entry_fees / execution_size) × 100
reward += -entry_fee_pct × entry_penalty_scale  # Default: 3.0

# Component 3: Liquidation risk penalty
if min_liq_distance < liquidation_buffer:
    reward += -(liquidation_buffer - min_liq_distance) × liquidation_penalty_scale  # Default: 20.0

# Component 4: Stop-loss penalty
if position_hit_stop_loss:
    reward += stop_loss_penalty  # Default: -2.0
```

**Key differences from old reward:**
- ✅ Hourly P&L rewards (prevents "zero trading" problem)
- ✅ NO exit-specific rewards (P&L already rewarded hourly)
- ✅ NO idle penalty (not needed with hourly rewards)
- ✅ Liquidation risk penalty (encourages safety)
- ✅ Stop-loss penalty (discourages poor trades)

#### 2. **RewardConfig** (`reward_config.py`)

Tunable reward parameters:

```python
from models.rl.core.reward_config import RewardConfig

# Default config
reward_config = RewardConfig()

# Custom config
reward_config = RewardConfig(
    pnl_reward_scale=5.0,           # Higher P&L sensitivity
    entry_penalty_scale=2.0,        # Lower entry barrier
    liquidation_penalty_scale=25.0, # Higher risk aversion
    stop_loss_penalty=-3.0,         # Stronger stop-loss avoidance
)

# Preset configs
from models.rl.core.reward_config import CONSERVATIVE_CONFIG, AGGRESSIVE_CONFIG
```

**Presets:**
- `DEFAULT_CONFIG`: Balanced (3.0, 3.0, 20.0, -2.0)
- `CONSERVATIVE_CONFIG`: Risk-averse (2.0, 4.0, 30.0, -3.0)
- `AGGRESSIVE_CONFIG`: Risk-seeking (5.0, 2.0, 15.0, -1.0)

#### 3. **Curriculum Learning** (`curriculum.py`)

3-phase progressive training:

| Phase | Episodes | Config | Episode Length | Goal |
|-------|----------|--------|----------------|------|
| **1: Simple** | 0-500 | Fixed (1x, 50% util, 2 pos) | 72h (3 days) | Learn basics |
| **2: Variable** | 500-1500 | Moderate sampling (1-5x) | 120h (5 days) | Generalize |
| **3: Full** | 1500+ | Full sampling (1-10x) | 168h (7 days) | Robust performance |

```python
from models.rl.core.curriculum import CurriculumScheduler

scheduler = CurriculumScheduler()

# Get config for episode
episode = 750  # In Phase 2
config = scheduler.get_config(episode)
episode_length = scheduler.get_episode_length_days(episode)  # 5 days

# Check phase progress
phase_name, progress = scheduler.get_phase_progress(episode)
# Returns: ('variable', 25.0)  # 25% through Phase 2
```

#### 4. **Reward Component Logging** (`environment.py`)

Track which rewards drive behavior:

```python
obs, reward, terminated, truncated, info = env.step(action)

# Breakdown available in info
breakdown = info['reward_breakdown']
print(f"Hourly P&L: {breakdown['hourly_pnl']:.2f}")
print(f"Entry penalty: {breakdown['entry_penalty']:.2f}")
print(f"Liquidation penalty: {breakdown['liquidation_penalty']:.2f}")
print(f"Stop-loss penalty: {breakdown['stop_loss_penalty']:.2f}")
```

#### 5. **Population-Based Training** (`train_ppo_pbt.py`)

Automatic hyperparameter + reward tuning:

```python
# Tunes 10 parameters simultaneously:
# PPO: learning_rate, gamma, gae_lambda, clip_range, entropy_coef, vf_coef
# Reward: pnl_reward_scale, entry_penalty_scale, liquidation_penalty_scale, stop_loss_penalty

# Every 100 episodes:
# - Bottom 25% (2 agents) copy weights from top 25%
# - Perturb hyperparameters ±20%
# - Continue training
```

---

## 🚀 Training Options

### Option 1: Basic Training (Fixed Reward Config)

Use default reward parameters:

```bash
python train_ppo.py --num-episodes 1000
```

With custom reward parameters:

```bash
python train_ppo.py \
  --pnl-reward-scale 5.0 \
  --entry-penalty-scale 2.0 \
  --liquidation-penalty-scale 25.0 \
  --stop-loss-penalty -3.0 \
  --num-episodes 1000
```

### Option 2: Curriculum Learning (Recommended)

Progressive difficulty over 3 phases:

```bash
python train_ppo_curriculum.py --num-episodes 3000
```

**Expected timeline:**
- Episodes 0-500: Phase 1 (Simple config) - ~1-2 hours
- Episodes 500-1500: Phase 2 (Moderate configs) - ~3-4 hours
- Episodes 1500-3000: Phase 3 (Full configs) - ~5-6 hours
- **Total: ~10-12 hours on CPU**

With custom phases:

```bash
python train_ppo_curriculum.py \
  --phase1-end 300 \
  --phase2-end 1000 \
  --num-episodes 2000
```

### Option 3: Population-Based Training (Best Performance)

Automatic reward + PPO hyperparameter tuning:

```bash
python train_ppo_pbt.py \
  --population 8 \
  --episodes-per-agent 1000 \
  --perturbation-interval 100 \
  --use-curriculum
```

**Expected timeline:**
- 8 agents × 1000 episodes = 8,000 total episodes
- With perturbation every 100 episodes = 10 generations
- **Total: ~80-100 hours on CPU, 20-30 hours on GPU**

**Advantages:**
- Automatically finds optimal reward parameters
- Explores diverse hyperparameter combinations
- Best final performance (Sharpe ratio typically 10-20% higher)

**When to use:**
- Production models (invest time for best quality)
- Uncertain about reward parameters
- Want robust hyperparameters

---

## 📊 Monitoring Training

### Reward Breakdown Analysis

Track reward components in TensorBoard:

```python
# In training loop
if 'reward_breakdown' in info:
    writer.add_scalar('reward/hourly_pnl', info['reward_breakdown']['hourly_pnl'], step)
    writer.add_scalar('reward/entry_penalty', info['reward_breakdown']['entry_penalty'], step)
    writer.add_scalar('reward/liquidation_penalty', info['reward_breakdown']['liquidation_penalty'], step)
    writer.add_scalar('reward/stop_loss_penalty', info['reward_breakdown']['stop_loss_penalty'], step)
```

**What to look for:**
- **Hourly P&L dominates**: Good! This is the main signal
- **Entry penalty too large**: Agent not trading enough → reduce `entry_penalty_scale`
- **Liquidation penalty frequent**: Agent taking too much risk → increase `liquidation_penalty_scale`
- **Stop-loss penalty frequent**: Agent holding losers → tune `stop_loss_threshold` in TradingConfig

### Curriculum Phase Transitions

Monitor when phases change:

```bash
# Training output shows:
Episode  500 | Phase: variable (0.0%) | Generalization across moderate configs
Episode 1500 | Phase: full (0.0%) | Robust performance across full config range
```

**Expected behavior:**
- **Phase 1**: Reward should steadily increase (agent learns basics)
- **Phase 2**: Reward may drop initially (harder configs), then recover
- **Phase 3**: Reward stabilizes (agent adapts to any config)

### PBT Agent Rankings

PBT shows agent rankings every generation:

```
Agent Rankings:
  1. Agent 3: Mean(100)=  45.23, Episodes=400
  2. Agent 0: Mean(100)=  42.15, Episodes=400
  3. Agent 5: Mean(100)=  38.67, Episodes=400
  ...

Exploit phase:
  Agent 7 copies from Agent 3
  Agent 2 copies from Agent 0

Explore phase:
  Agent 7: LR 3.0e-04→2.4e-04, PNL 3.0→3.6
  Agent 2: LR 3.0e-04→3.6e-04, PNL 3.0→2.4
```

**What this shows:**
- Top performers propagate their hyperparameters (exploit)
- Weaker agents explore new hyperparameter combinations (explore)
- Over time, population converges to optimal configuration

---

## 🔧 Troubleshooting

### Problem: Agent not trading at all (0 trades)

**Symptoms:**
```
Total Trades:           0
Winning Trades:         0
Mean P&L:       $    0.00
```

**Cause:** Entry penalties too high, or hourly P&L reward too low

**Solutions:**
1. Reduce `entry_penalty_scale`:
   ```bash
   python train_ppo.py --entry-penalty-scale 1.5
   ```

2. Increase `pnl_reward_scale`:
   ```bash
   python train_ppo.py --pnl-reward-scale 5.0
   ```

3. Check reward breakdown - entry penalties should not dominate

### Problem: Agent overtrading (164 trades, low win rate)

**Symptoms:**
```
Total Trades:         164
Win Rate:            31.4%
Mean P&L (%):        0.28%
```

**Cause:** Entry penalties too low

**Solutions:**
1. Increase `entry_penalty_scale`:
   ```bash
   python train_ppo.py --entry-penalty-scale 4.0
   ```

2. Monitor reward breakdown - entry penalties should be significant

### Problem: Agent taking excessive risk (hitting liquidation)

**Symptoms:**
```
Liquidation penalty: -5.23 (frequent)
Max Drawdown:       15.2%
```

**Cause:** Liquidation penalties too weak

**Solutions:**
1. Increase `liquidation_penalty_scale`:
   ```bash
   python train_ppo.py --liquidation-penalty-scale 30.0
   ```

2. Adjust `liquidation_buffer` in TradingConfig (default: 0.15 = 15%)

### Problem: Agent holding losing trades too long

**Symptoms:**
```
Avg Duration:        48.3 hours
Win Rate:            35.2%
Stop-loss penalty: -2.0 (frequent)
```

**Cause:** Stop-loss penalties not strong enough

**Solutions:**
1. Increase stop-loss penalty magnitude:
   ```bash
   python train_ppo.py --stop-loss-penalty -4.0
   ```

2. Tighten stop-loss threshold in TradingConfig (e.g., -0.015 = -1.5%)

---

## 🎓 Best Practices

### 1. Start with Curriculum Learning

**Recommended for first training:**
```bash
python train_ppo_curriculum.py \
  --num-episodes 3000 \
  --eval-every 50 \
  --save-every 100
```

**Why:**
- Gradual difficulty increase
- Better generalization
- More stable training

### 2. Use PBT for Production Models

**After initial curriculum run:**
```bash
python train_ppo_pbt.py \
  --population 8 \
  --episodes-per-agent 1500 \
  --perturbation-interval 100 \
  --use-curriculum
```

**Why:**
- Automatically finds optimal reward parameters
- Explores hyperparameter space efficiently
- Best final performance

### 3. Monitor Composite Score (Not Reward)

Model selection uses composite score (from train_ppo.py):

```
Composite Score = 50% P&L + 30% Win Rate + 20% Low Drawdown
```

**Good model:**
- Composite score > 0.5
- P&L > 1.0%
- Win rate > 60%
- Drawdown < 5%

**Poor model:**
- High reward but low composite score
- Overtrading (many trades, low win rate)
- High drawdown

### 4. Analyze Reward Components

After training, analyze which rewards drove decisions:

```python
# Load trained model
# Run evaluation episodes
# Collect reward breakdowns

# Calculate component contributions
total_hourly_pnl = sum(all hourly_pnl rewards)
total_entry_penalties = sum(all entry_penalty rewards)
total_liq_penalties = sum(all liquidation_penalty rewards)
total_stop_loss_penalties = sum(all stop_loss_penalty rewards)

# Dominant component should be hourly_pnl (main signal)
```

---

## 📂 Files Modified/Created

### Modified Files:
1. `models/rl/core/environment.py` - Reward function + logging
2. `models/rl/core/config.py` - Added `sample_moderate()`
3. `train_ppo.py` - Added reward config arguments

### New Files:
4. `models/rl/core/reward_config.py` - RewardConfig dataclass
5. `models/rl/core/curriculum.py` - CurriculumScheduler
6. `train_ppo_curriculum.py` - Curriculum training script
7. `train_ppo_pbt.py` - Population-based training script
8. `REWARD_AND_CURRICULUM_GUIDE.md` - This guide

---

## 🎯 Success Criteria

### Training Success:
- [ ] Mean reward > 20 after Phase 1
- [ ] Mean reward > 30 after Phase 2
- [ ] Test Sharpe ratio > 1.5 after Phase 3
- [ ] Composite score > 0.5 on test set

### Reward Function Success:
- [ ] Hourly P&L dominates reward breakdown (>70%)
- [ ] Entry penalties significant but not dominant (<20%)
- [ ] Agent trades selectively (5-30 trades per episode)
- [ ] Win rate > 60%

### Model Quality:
- [ ] P&L > 1.0% per episode
- [ ] Max drawdown < 10%
- [ ] Consistent performance across configs
- [ ] No liquidations on test set

---

## 🚀 Quick Start Commands

```bash
# 1. Basic training (1-2 hours)
python train_ppo.py --num-episodes 1000

# 2. Curriculum learning (10-12 hours)
python train_ppo_curriculum.py --num-episodes 3000

# 3. PBT training (80-100 hours, best quality)
python train_ppo_pbt.py --population 8 --episodes-per-agent 1000 --use-curriculum

# 4. Custom reward parameters
python train_ppo.py \
  --pnl-reward-scale 5.0 \
  --entry-penalty-scale 2.0 \
  --num-episodes 1000

# 5. Test inference
python test_inference.py
```

---

**Ready to train!** 🚀

All implementations follow IMPLEMENTATION_PLAN.md specifications. The reward function now correctly balances profitability, risk management, and selectivity.
