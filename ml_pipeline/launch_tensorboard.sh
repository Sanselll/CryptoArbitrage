#!/bin/bash

# Launch TensorBoard to monitor V3 1M training

echo "="*80
echo "LAUNCHING TENSORBOARD"
echo "="*80
echo ""
echo "TensorBoard Directory: models/rl_v3_1m/ppo_20251031_120100/tensorboard"
echo ""
echo "Starting TensorBoard on http://localhost:6006"
echo ""
echo "📊 METRICS TO WATCH:"
echo "  • trading/episode_pnl_pct - Actual P&L (should trend upward)"
echo "  • trading/win_rate - Win rate per episode (target: 50%+)"
echo "  • trading/trades_count - Number of trades per episode"
echo "  • trading/reward_per_pnl - Alignment metric (should be stable)"
echo "  • rollout/ep_rew_mean - Episode reward"
echo "  • train/explained_variance - Value function quality"
echo ""
echo "Press Ctrl+C to stop TensorBoard"
echo ""

tensorboard --logdir=models/rl_v3_1m/ppo_20251031_120100/tensorboard --port=6006
