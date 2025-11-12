"""Diagnose observation differences at step 23 when position opens."""

import numpy as np
import torch
from pathlib import Path

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.algorithms.ppo_trainer import PPOTrainer
from models.rl.networks.modular_ppo import ModularPPONetwork
from test_server_vs_inference_comparison import convert_portfolio_to_server_format
from server.inference.rl_predictor import ModularRLPredictor

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Create environment
trading_config = TradingConfig(max_leverage=2.0, target_utilization=0.8, max_positions=2)
reward_config = RewardConfig()

env = FundingArbitrageEnv(
    data_path="data/rl_test.csv",
    initial_capital=10000.0,
    trading_config=trading_config,
    reward_config=reward_config,
    episode_length_days=7,
    step_hours=5/60,
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    use_full_range_episodes=True,
    verbose=False,
)

# Load models
network = ModularPPONetwork()
trainer = PPOTrainer(network=network, learning_rate=3e-4, device='cpu')
trainer.load("checkpoints/best_model.pt")
predictor = ModularRLPredictor("checkpoints/best_model.pt", device='cpu')

# Run to step 23
obs_env, _ = env.reset(seed=seed)

for step in range(23):
    action_mask = env._get_action_mask()
    action, _, _ = trainer.select_action(obs_env, action_mask, deterministic=True)
    obs_env, _, _, _, _ = env.step(action)
    print(f"Step {step}: Action={action}, Positions={len(env.portfolio.positions)}")

# Now at step 23 - position should be open
print(f"\n{'='*80}")
print(f"STEP 23 - DETAILED COMPARISON")
print(f"{'='*80}\n")

# Get environment observation
print("Environment observation...")
obs_env_step23 = env._get_observation()

# Get server observation
opportunities = [dict(opp) for opp in env.current_opportunities[:10]]
price_data = env._get_current_prices()
portfolio_dict = convert_portfolio_to_server_format(env.portfolio, price_data=price_data)
config_dict = {
    'max_leverage': env.current_config.max_leverage,
    'target_utilization': env.current_config.target_utilization,
    'max_positions': env.current_config.max_positions,
    'stop_loss_threshold': env.current_config.stop_loss_threshold,
    'liquidation_buffer': env.current_config.liquidation_buffer,
}

print("Server observation...")
obs_server_step23 = predictor._build_observation(config_dict, portfolio_dict, opportunities)

# Compare
diff = np.abs(obs_env_step23 - obs_server_step23)
max_diff_idx = np.argmax(diff)
max_diff = diff[max_diff_idx]

print(f"\nMax difference: {max_diff:.6f} at index {max_diff_idx}")
print(f"  ENV value: {obs_env_step23[max_diff_idx]:.6f}")
print(f"  SERVER value: {obs_server_step23[max_diff_idx]:.6f}")

# Show all differences > 0.01
large_diffs = np.where(diff > 0.01)[0]
if len(large_diffs) > 0:
    print(f"\n📊 All differences > 0.01:")
    for idx in large_diffs[:30]:
        print(f"  [{idx:3d}]: ENV={obs_env_step23[idx]:10.6f}  SERVER={obs_server_step23[idx]:10.6f}  diff={diff[idx]:10.6f}")

# Check portfolio
print(f"\nPortfolio state:")
print(f"  Positions: {len(env.portfolio.positions)}")
if env.portfolio.positions:
    pos = env.portfolio.positions[0]
    print(f"  Position 0:")
    print(f"    Symbol: {pos.symbol}")
    print(f"    Size: ${pos.position_size_usd}")
    print(f"    PnL%: {pos.unrealized_pnl_pct}")
    print(f"    Hours held: {pos.hours_held}")
    print(f"    PnL history len: {len(pos.pnl_history)}")
    print(f"    PnL history: {pos.pnl_history}")
