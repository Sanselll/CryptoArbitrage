# PPO Training Guide

## ⚠️ Data Leakage Prevention

**IMPORTANT**: The training script uses **separate train and test data** to prevent data leakage:
- **Training**: Learns from `data/rl_train.csv` (154K samples, Sep-Oct)
- **Evaluation**: Tests on `data/rl_test.csv` (39K samples, Oct 22-28)

The data is split **chronologically** (by time), which is critical for time-series data.

## Quick Start

### Basic Training (Recommended)

Train with default settings (uses pre-split train/test data):

```bash
python train_ppo.py --num-episodes 1000
```

### Training with Random Config Sampling

For robust config-aware training (recommended):

```bash
python train_ppo.py \
    --sample-random-config \
    --num-episodes 2000 \
    --eval-every 100 \
    --save-every 200
```

### Training with Fixed Config

Train for a specific risk profile:

```bash
# Conservative (1x leverage, 40% utilization, 2 positions)
python train_ppo.py \
    --max-leverage 1.0 \
    --target-utilization 0.4 \
    --max-positions 2 \
    --num-episodes 1000

# Moderate (3x leverage, 60% utilization, 3 positions)
python train_ppo.py \
    --max-leverage 3.0 \
    --target-utilization 0.6 \
    --max-positions 3 \
    --num-episodes 1000

# Aggressive (5x leverage, 75% utilization, 5 positions)
python train_ppo.py \
    --max-leverage 5.0 \
    --target-utilization 0.75 \
    --max-positions 5 \
    --num-episodes 1000
```

### Using Custom Train/Test Data

If you have your own split data:

```bash
python train_ppo.py \
    --train-data-path path/to/train.csv \
    --test-data-path path/to/test.csv \
    --num-episodes 1000
```

## Advanced Options

### Hyperparameter Tuning

```bash
python train_ppo.py \
    --data-path path/to/data.csv \
    --learning-rate 1e-4 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-range 0.2 \
    --value-coef 0.5 \
    --entropy-coef 0.01 \
    --n-epochs 10 \
    --batch-size 128 \
    --num-episodes 5000
```

### Resume Training

```bash
python train_ppo.py \
    --data-path path/to/data.csv \
    --resume-from checkpoints/checkpoint_ep1000.pt \
    --num-episodes 2000
```

### GPU Training

**Auto-detect best device (recommended):**
```bash
python train_ppo.py --num-episodes 2000
# Automatically uses MPS on Apple Silicon, CUDA on NVIDIA, or CPU
```

**Apple Silicon (M1/M2/M3) - MPS:**
```bash
python train_ppo.py \
    --device mps \
    --batch-size 256 \
    --num-episodes 10000
```

**NVIDIA GPU - CUDA:**
```bash
python train_ppo.py \
    --device cuda \
    --batch-size 256 \
    --num-episodes 10000
```

## Configuration Parameters

### Data Parameters

- `--train-data-path`: Path to training CSV (default: data/rl_train.csv)
- `--test-data-path`: Path to test CSV for evaluation (default: data/rl_test.csv)
- `--price-history-path`: Path to price history directory (optional)

### Environment Parameters

- `--initial-capital`: Starting capital in USD (default: 10000)
- `--episode-length-days`: Episode length in days (default: 3)
- `--sample-random-config`: Sample random TradingConfig each episode

### Trading Configuration (when not sampling random)

- `--max-leverage`: Maximum leverage 1-10x (default: 3.0)
- `--target-utilization`: Target capital utilization 0-1 (default: 0.6)
- `--max-positions`: Max concurrent positions 1-5 (default: 3)

### PPO Hyperparameters

- `--learning-rate`: Learning rate (default: 3e-4)
- `--gamma`: Discount factor (default: 0.99)
- `--gae-lambda`: GAE lambda (default: 0.95)
- `--clip-range`: PPO clip range (default: 0.2)
- `--value-coef`: Value loss coefficient (default: 0.5)
- `--entropy-coef`: Entropy coefficient (default: 0.01)
- `--n-epochs`: Epochs per update (default: 4)
- `--batch-size`: Mini-batch size (default: 64)

### Training Parameters

- `--num-episodes`: Number of training episodes (default: 1000)
- `--eval-every`: Evaluate every N episodes (default: 50)
- `--save-every`: Save checkpoint every N episodes (default: 100)
- `--device`: Device to use: auto, cpu, cuda, or mps (default: auto)
  - `auto`: Automatically selects best device (MPS → CUDA → CPU)
  - `mps`: Apple Silicon GPU (M1/M2/M3)
  - `cuda`: NVIDIA GPU
  - `cpu`: CPU only

### Checkpointing

- `--checkpoint-dir`: Directory for checkpoints (default: checkpoints)
- `--resume-from`: Path to checkpoint to resume from

### Logging

- `--log-interval`: Log every N episodes (default: 10)

## Output Files

Training produces the following files:

- `checkpoints/best_model.pt`: Best model based on evaluation reward
- `checkpoints/final_model.pt`: Final model after training
- `checkpoints/checkpoint_epN.pt`: Periodic checkpoints every N episodes

## Monitoring Training

Watch for these metrics during training:

- **Episode Reward**: Total reward per episode (higher is better)
- **Mean(100)**: Rolling mean reward over last 100 episodes
- **Policy Loss**: Actor loss (should stabilize)
- **Value Loss**: Critic loss (should decrease)
- **Entropy**: Action distribution entropy (should be > 0)
- **KL**: Approximate KL divergence (should be small, < 0.05)
- **Clipfrac**: Fraction of clipped ratios (should be 0.1-0.3)

## Expected Training Time

- **CPU**: ~30-40 episodes/min (slower, but works everywhere)
- **Apple MPS (M1/M2/M3)**: ~40-60 episodes/min (2-3x faster than CPU)
- **NVIDIA GPU (CUDA)**: ~100-200 episodes/min (fastest)

**Estimated time for 1000 episodes:**
- CPU: ~30-40 minutes
- MPS: ~20-30 minutes
- CUDA: ~5-10 minutes

**Note**: First episode on MPS/CUDA may be slower due to kernel compilation, then speeds up significantly.

## Recommended Training Schedule

### Phase 1: Initial Training (Episodes 1-1000)
- Use `--sample-random-config` for diversity
- Monitor for stable learning (Mean(100) trending up)
- Evaluate every 50 episodes

### Phase 2: Fine-tuning (Episodes 1000-2000)
- Resume from best checkpoint
- Reduce learning rate to 1e-4
- Increase batch size to 128

### Phase 3: Final Optimization (Episodes 2000-3000)
- Further reduce learning rate to 3e-5
- Focus on specific configs if needed

## Troubleshooting

### Training is unstable (rewards oscillating)
- Reduce learning rate: `--learning-rate 1e-4`
- Increase batch size: `--batch-size 128`
- Reduce clip range: `--clip-range 0.1`

### Agent not learning (flat rewards)
- Increase entropy coefficient: `--entropy-coef 0.05`
- Check data quality
- Verify action masking is working

### Memory issues
- Reduce batch size: `--batch-size 32`
- Reduce episode length: `--episode-length-days 1`

### Slow training
- Use GPU: `--device cuda`
- Reduce number of epochs: `--n-epochs 2`
- Increase batch size: `--batch-size 256`

## Next Steps

After training:

1. **Evaluate**: Run evaluation on held-out data
2. **Backtest**: Test on different time periods
3. **Deploy**: Use trained model for live trading decisions
4. **Monitor**: Track real-world performance
5. **Retrain**: Periodically update with new data
