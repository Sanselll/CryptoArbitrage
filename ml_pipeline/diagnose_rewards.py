"""
Quick diagnostic to understand reward scale in practice
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

# Reset and take random actions
obs, info = env.reset()
print("=" * 80)
print("REWARD DIAGNOSTIC - 20 Random Steps")
print("=" * 80)

total_reward = 0
step_rewards = []

for step in range(20):
    # Random action
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    step_rewards.append(reward)
    total_reward += reward

    # Print every 5 steps
    if (step + 1) % 5 == 0:
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Total reward so far: {total_reward:.2f}")
        print(f"  Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"  Num positions: {info['num_positions']}")

        # Show reward breakdown
        breakdown = info.get('reward_breakdown', {})
        print(f"  Breakdown: Funding={breakdown.get('funding_reward', 0):.2f}, "
              f"Price={breakdown.get('price_reward', 0):.2f}, "
              f"Liq Penalty={breakdown.get('liquidation_penalty', 0):.2f}")

    if terminated or truncated:
        print(f"\nEpisode ended at step {step + 1}")
        break

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total steps: {len(step_rewards)}")
print(f"Total reward: {total_reward:.2f}")
print(f"Mean reward per step: {np.mean(step_rewards):.2f}")
print(f"Median reward per step: {np.median(step_rewards):.2f}")
print(f"Min reward: {np.min(step_rewards):.2f}")
print(f"Max reward: {np.max(step_rewards):.2f}")
print(f"\nIf full episode (168 steps at 7 days), projected total: {np.mean(step_rewards) * 168:.2f}")
