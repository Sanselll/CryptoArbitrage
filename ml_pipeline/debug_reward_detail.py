"""
Debug reward calculation step-by-step
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig

# Create environment
env = FundingArbitrageEnv(
    data_path="data/rl_train.csv",
    initial_capital=10000,
    trading_config=TradingConfig(),
    reward_config=RewardConfig(funding_reward_scale=1.0, price_reward_scale=1.0),
    episode_length_days=5,
    step_hours=5.0 / 60.0,
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    verbose=False,
)

obs, info = env.reset()

print("Taking 20 random steps and logging reward breakdown...")
print()

total_reward = 0
funding_rewards = []
price_rewards = []
liq_penalties = []

for step in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    breakdown = info.get('reward_breakdown', {})
    funding_r = breakdown.get('funding_reward', 0.0)
    price_r = breakdown.get('price_reward', 0.0)
    liq_p = breakdown.get('liquidation_penalty', 0.0)

    funding_rewards.append(funding_r)
    price_rewards.append(price_r)
    liq_penalties.append(liq_p)

    if step < 5 or step % 5 == 0:  # Print first 5, then every 5th
        print(f"Step {step+1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Funding reward: {funding_r:.4f}")
        print(f"  Price reward: {price_r:.4f}")
        print(f"  Liquidation penalty: {liq_p:.4f}")
        print(f"  Portfolio value: ${info['portfolio_value']:,.2f}")
        print(f"  Total P&L: {info['total_pnl_pct']:.2f}%")

    if terminated or truncated:
        break

print()
print("=" * 80)
print("BREAKDOWN SUMMARY (20 steps)")
print("=" * 80)
print(f"Total reward: {total_reward:.2f}")
print()
print("Component rewards:")
print(f"  Sum of funding rewards: {sum(funding_rewards):.2f}")
print(f"  Sum of price rewards: {sum(price_rewards):.2f}")
print(f"  Sum of liquidation penalties: {sum(liq_penalties):.2f}")
print(f"  Total: {sum(funding_rewards) + sum(price_rewards) + sum(liq_penalties):.2f}")
print()
print("Average per step:")
print(f"  Funding reward: {np.mean(funding_rewards):.4f}")
print(f"  Price reward: {np.mean(price_rewards):.4f}")
print(f"  Liquidation penalty: {np.mean(liq_penalties):.4f}")
print()
print(f"If this continues for 1440 steps (5 days @ 5min):")
print(f"  Projected total reward: {(sum(funding_rewards) + sum(price_rewards) + sum(liq_penalties)) / 20 * 1440:.2f}")
