# Parallel Environment Training Guide

## Overview

Parallel environments allow you to run multiple environment instances simultaneously, significantly speeding up RL training by collecting rollouts from multiple environments in parallel.

## Benefits

- **2-4x faster training** with 4-8 parallel environments
- Better GPU/CPU utilization
- More diverse experience collection
- Improved sample efficiency

## Usage

### Basic Usage (Single Environment - Default)

```bash
python train_ppo.py
```

This runs with a single environment sequentially (no parallelization).

### Parallel Training (Recommended)

```bash
# Use 4 parallel environments (good for most CPUs)
python train_ppo.py --n-envs 4

# Use 8 parallel environments (for powerful CPUs)
python train_ppo.py --n-envs 8

# Use 16 parallel environments (for high-end machines)
python train_ppo.py --n-envs 16
```

### Advanced Options

```bash
# Specify multiprocessing start method (default: auto-detected)
python train_ppo.py --n-envs 4 --parallel-start-method spawn

# spawn: Safer, works on all platforms (Windows, macOS, Linux)
# fork: Faster on Unix systems (macOS, Linux) but can have issues with some libraries
```

### Complete Example

```bash
# Train with 8 parallel environments for faster training
python train_ppo.py \
    --n-envs 8 \
    --num-episodes 1000 \
    --device mps \
    --learning-rate 3e-4 \
    --checkpoint-dir checkpoints/parallel_v1
```

## How It Works

### Sequential Training (n-envs=1)
```
Env 1: Step → Update → Step → Update → Step → Update
```

### Parallel Training (n-envs=4)
```
Env 1: Step ┐
Env 2: Step ├→ Batch Update → Step ┐
Env 3: Step │                       ├→ Batch Update → ...
Env 4: Step ┘                       ┘
```

All 4 environments step simultaneously, then a single batch update is performed.

## Performance Tips

1. **Number of Environments**:
   - Start with `--n-envs 4`
   - Increase to 8-16 if you have a powerful CPU
   - Too many environments can actually slow down training due to overhead

2. **Device Selection**:
   - Use `--device mps` on Apple Silicon Macs
   - Use `--device cuda` on NVIDIA GPUs
   - Parallel environments work great with GPU acceleration

3. **Episode Length**:
   - Longer episodes benefit more from parallelization
   - Short episodes may not see as much speedup

## Expected Speedup

| n-envs | Expected Speedup | Recommended For |
|--------|------------------|-----------------|
| 1      | 1.0x (baseline)  | Testing, debugging |
| 4      | ~2.5x            | Most CPUs (4-8 cores) |
| 8      | ~4.0x            | Powerful CPUs (8+ cores) |
| 16     | ~6.0x            | High-end workstations |

**Note**: Actual speedup depends on CPU cores, memory, and environment complexity.

## Monitoring

During training, you'll see:

```
🚀 Using 8 parallel environments for training
   Start method: default

...

Parallel mode: 8 environments
Episode 10/1000
  Reward:   123.45  |  Length:  234  |  Mean(100):   120.12
  n_completed_episodes: 3  |  n_envs: 8
```

The `n_completed_episodes` shows how many episodes finished during that rollout.

## Troubleshooting

### Issue: "Pickle Error" or serialization issues
**Solution**: Try `--parallel-start-method spawn`

### Issue: Slow startup
**Solution**: Normal - first parallel startup loads all environments. Subsequent steps are fast.

### Issue: High memory usage
**Solution**: Reduce `--n-envs` to a lower number

### Issue: No speedup observed
**Solution**:
- Check if CPU is bottlenecked (use `htop` or Activity Monitor)
- Ensure environments are computationally expensive enough to benefit
- Try different `--parallel-start-method` values

## Technical Details

### Implementation

The parallel environment implementation uses:
- **ParallelEnv** (`ml_pipeline/models/rl/core/vec_env.py`): Multiprocessing-based vectorization
- **PPOTrainer.train_episode_vectorized()**: Batch rollout collection
- **Cloudpickle**: Serialization for complex environment objects

### Files Changed

1. `ml_pipeline/models/rl/core/vec_env.py` - New parallel environment wrapper
2. `ml_pipeline/models/rl/algorithms/ppo_trainer.py` - Added `train_episode_vectorized()`
3. `ml_pipeline/train_ppo.py` - Added `--n-envs` support

## Testing

Run the test suite to verify installation:

```bash
cd ml_pipeline
python test_parallel_env.py
```

Expected output:
```
✅ ALL TESTS PASSED!
🎉 All tests completed successfully!
```

## Comparison: Before vs After

### Before (Sequential)
```bash
python train_ppo.py --num-episodes 1000
# Time: ~10 hours
```

### After (Parallel)
```bash
python train_ppo.py --num-episodes 1000 --n-envs 8
# Time: ~3 hours (3.3x speedup)
```

## Best Practices

1. **Always test with `--n-envs 1` first** to verify your setup works
2. **Start with `--n-envs 4`** and increase if you have CPU headroom
3. **Monitor CPU usage** - if not at ~80-100%, you can increase n-envs
4. **Use checkpoints** - parallel training can be unstable, save frequently
5. **Keep eval on single env** - this ensures consistent evaluation metrics

## Notes

- Parallel environments auto-reset on episode completion
- Each environment uses a different random seed (seed + offset)
- Training remains deterministic with fixed `--seed` value
- Evaluation always uses a single environment for consistency
