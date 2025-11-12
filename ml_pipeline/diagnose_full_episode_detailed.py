"""
Track rewards throughout a full episode to find where they diverge
"""
import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent))

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig

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

total_reward = 0
funding_total = 0
price_total = 0
liq_total = 0
step_count = 0

print("Running full episode and tracking reward divergence...")
print()

while True:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    breakdown = info.get('reward_breakdown', {})
    funding_r = breakdown.get('funding_reward', 0.0)
    price_r = breakdown.get('price_reward', 0.0)
    liq_p = breakdown.get('liquidation_penalty', 0.0)

    funding_total += funding_r
    price_total += price_r
    liq_total += liq_p

    components_sum = funding_total + price_total + liq_total
    divergence = total_reward - components_sum

    # Print when divergence starts appearing
    if abs(divergence) > 0.1 and step_count % 10 == 0:
        print(f"Step {step_count}: DIVERGENCE = {divergence:.2f}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Components sum: {components_sum:.2f}")
        print(f"  Latest step reward: {reward:.4f}")
        print(f"  Latest breakdown: F={funding_r:.4f}, P={price_r:.4f}, L={liq_p:.4f}")
        print()

    if terminated or truncated:
        break

print("=" * 80)
print(f"FINAL: Step {step_count}")
print("=" * 80)
print(f"Total reward: {total_reward:.2f}")
print(f"Components sum: {components_sum:.2f}")
print(f"  Funding: {funding_total:.2f}")
print(f"  Price: {price_total:.2f}")
print(f"  Liquidation: {liq_total:.2f}")
print()
print(f"DIVERGENCE: {total_reward - components_sum:.2f}")
print(f"Final P&L: {info['total_pnl_pct']:.2f}%")
print(f"Portfolio value: ${info['portfolio_value']:,.2f}")
