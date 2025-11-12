"""
Run a full episode to understand actual reward scale
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig

# Create environment with Pure RL-v2 settings
env = FundingArbitrageEnv(
    data_path="data/rl_train.csv",
    initial_capital=10000,
    trading_config=TradingConfig(),
    reward_config=RewardConfig(
        funding_reward_scale=5.0,
        price_reward_scale=1.0,
        liquidation_penalty_scale=1000.0,
    ),
    episode_length_days=7,
    step_hours=1,
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    verbose=False,
)

# Run full episode with random actions
obs, info = env.reset()
print("="*80)
print("FULL EPISODE DIAGNOSTIC")
print("="*80)
print(f"Initial capital: ${env.initial_capital:,.2f}")
print(f"Episode length: {env.episode_length_hours} hours")
print(f"Step hours: {env.step_hours}")
print(f"Max steps: ~{env.episode_length_hours // env.step_hours}")
print()

total_reward = 0
step_count = 0
step_rewards = []

while True:
    # Random action
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_rewards.append(reward)
    step_count += 1

    # Print progress every 50 steps
    if step_count % 50 == 0:
        print(f"Step {step_count}: Total reward = {total_reward:.2f}, "
              f"Portfolio value = ${info['portfolio_value']:,.2f}")

    if terminated or truncated:
        print(f"\n{'='*80}")
        print(f"Episode ended at step {step_count}")
        print(f"Reason: {'terminated' if terminated else 'truncated'}")
        break

print(f"{'='*80}")
print("FINAL RESULTS")
print(f"{'='*80}")
print(f"Total steps: {step_count}")
print(f"Total episode reward: {total_reward:.2f}")
print(f"Mean reward per step: {np.mean(step_rewards):.4f}")
print(f"Final portfolio value: ${info['portfolio_value']:,.2f}")
print(f"Total P&L: {info['total_pnl_pct']:.2f}%")
print()
print(f"Reward distribution:")
print(f"  Min: {np.min(step_rewards):.2f}")
print(f"  25%: {np.percentile(step_rewards, 25):.2f}")
print(f"  50%: {np.median(step_rewards):.2f}")
print(f"  75%: {np.percentile(step_rewards, 75):.2f}")
print(f"  Max: {np.max(step_rewards):.2f}")
