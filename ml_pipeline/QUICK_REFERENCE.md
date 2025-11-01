# PBT Model - Quick Reference Card

## 🚀 Quick Start

### Train PBT Model
```bash
python train_pbt_simple.py \
    --population 8 \
    --timesteps 500000 \
    --perturbation-interval 20000
```

### Evaluate Best Model
```bash
python train_rl_agent.py \
    --eval-only models/pbt_TIMESTAMP/agent_X_model.zip \
    --eval-data-path data/rl_test.csv \
    --n-eval-episodes 1
```

---

## 📊 Current Best Model

**Location:** `models/pbt_20251101_083701/agent_2_model.zip`

**Performance:**
- P&L: **+2.91%** (target: >0.5%) ✅
- Win Rate: **45.1%** (target: >50%) ⚠️
- Trades: 51 over 7 days
- Avg Profit: $5.71/trade
- Win/Loss Ratio: 4.2x

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `train_pbt_simple.py` | Main PBT training script |
| `train_rl_agent.py` | Evaluation script |
| `src/rl/environment.py` | Trading environment |
| `PBT_MODEL_DOCUMENTATION.md` | Full documentation |
| `PBT_GUIDE.md` | User guide |

---

## 🎯 Key Parameters

### Training
- `--population`: Number of agents (default: 8)
- `--timesteps`: Training steps per agent (default: 500k)
- `--perturbation-interval`: PBT interval (default: 20k)

### Environment
- Initial capital: $10,000
- Max positions: 3
- Episode length: 3 days (training), full range (eval)
- Max opportunities/hour: 5

---

## 🔧 Common Commands

### Quick Test
```bash
python train_pbt_simple.py --population 2 --timesteps 10000
```

### View Agent Hyperparams
```bash
cat models/pbt_TIMESTAMP/agent_X_hyperparams.json
```

### Analyze Trades
```python
import pandas as pd
df = pd.read_csv('evaluation_trades_TIMESTAMP.csv')
print(df.describe())
```

---

## 📈 Metrics to Watch

- **P&L %**: Primary profitability metric
- **Win Rate**: % of profitable trades
- **Avg Trade Duration**: Capital efficiency
- **Win/Loss Ratio**: Risk/reward profile

---

## 🐛 Quick Fixes

**Slow training?**
```bash
--population 4  # Reduce agents
```

**Out of memory?**
```bash
# Edit line 243 in train_pbt_simple.py
with mp.Pool(processes=2) as pool:
```

**Different eval results?**
- Ensure using updated code (bug fixed 2025-11-01)

---

**For detailed documentation, see:** `PBT_MODEL_DOCUMENTATION.md`
