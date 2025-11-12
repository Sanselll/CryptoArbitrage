"""
Debug Observation Differences Between Server and Environment

This script compares observations step-by-step to identify EXACTLY which
features differ and by how much.
"""

import numpy as np
import torch
from pathlib import Path
import sys

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer
from server.inference.rl_predictor import ModularRLPredictor

def convert_portfolio_to_server_format(portfolio, price_data):
    """Convert environment portfolio to server format."""
    positions_list = []

    for pos in portfolio.positions:
        # Get current prices
        current_long_price = pos.entry_long_price
        current_short_price = pos.entry_short_price
        if price_data and pos.symbol in price_data:
            current_long_price = price_data[pos.symbol]['long_price']
            current_short_price = price_data[pos.symbol]['short_price']

        pos_dict = {
            'symbol': pos.symbol,
            'position_size_usd': pos.position_size_usd,
            'leverage': pos.leverage,
            'current_long_price': current_long_price,
            'current_short_price': current_short_price,
            'entry_long_price': pos.entry_long_price,
            'entry_short_price': pos.entry_short_price,
            'long_funding_rate': pos.long_funding_rate,
            'short_funding_rate': pos.short_funding_rate,
            'long_net_funding_usd': pos.long_net_funding_usd,
            'short_net_funding_usd': pos.short_net_funding_usd,
            'unrealized_pnl_pct': pos.unrealized_pnl_pct,
            'hours_held': pos.hours_held,
            'position_age_hours': pos.hours_held,
            'entry_fees_paid_usd': pos.entry_fees_paid_usd,
            'long_pnl_pct': pos.long_pnl_pct,
            'short_pnl_pct': pos.short_pnl_pct,
            'is_active': True,
            'position_is_active': 1.0,
            'long_funding_interval_hours': pos.long_funding_interval_hours,
            'short_funding_interval_hours': pos.short_funding_interval_hours,
            'pnl_history': pos.pnl_history.copy(),
            'peak_pnl_pct': pos.peak_pnl_pct,
            'entry_apr': pos.entry_apr,
        }

        # Calculate spreads
        if current_long_price > 0 and current_short_price > 0:
            avg_price = (current_long_price + current_short_price) / 2
            pos_dict['current_spread_pct'] = abs(current_long_price - current_short_price) / avg_price
        else:
            pos_dict['current_spread_pct'] = 0.0

        if pos.entry_long_price > 0 and pos.entry_short_price > 0:
            avg_entry_price = (pos.entry_long_price + pos.entry_short_price) / 2
            pos_dict['entry_spread_pct'] = abs(pos.entry_long_price - pos.entry_short_price) / avg_entry_price
        else:
            pos_dict['entry_spread_pct'] = 0.0

        # Calculate liquidation distance
        if pos.leverage > 0:
            long_liq = pos.entry_long_price * (1 - 0.9 / pos.leverage)
            short_liq = pos.entry_short_price * (1 + 0.9 / pos.leverage)
            long_dist = abs(current_long_price - long_liq) / current_long_price if current_long_price > 0 else 1.0
            short_dist = abs(short_liq - current_short_price) / current_short_price if current_short_price > 0 else 1.0
            pos_dict['liquidation_distance'] = min(long_dist, short_dist)
        else:
            pos_dict['liquidation_distance'] = 1.0

        positions_list.append(pos_dict)

    # Pad to 5 positions
    while len(positions_list) < 5:
        empty_pos = {
            'symbol': '', 'position_size_usd': 0.0, 'unrealized_pnl_pct': 0.0,
            'hours_held': 0.0, 'position_age_hours': 0.0, 'is_active': False,
            'position_is_active': 0.0, 'long_net_funding_usd': 0.0, 'short_net_funding_usd': 0.0,
            'long_funding_rate': 0.0, 'short_funding_rate': 0.0, 'current_long_price': 0.0,
            'current_short_price': 0.0, 'entry_long_price': 0.0, 'entry_short_price': 0.0,
            'current_spread_pct': 0.0, 'entry_spread_pct': 0.0, 'long_pnl_pct': 0.0,
            'short_pnl_pct': 0.0, 'liquidation_distance': 1.0, 'entry_fees_paid_usd': 0.0,
            'long_funding_interval_hours': 8, 'short_funding_interval_hours': 8,
            'pnl_history': [], 'peak_pnl_pct': 0.0, 'entry_apr': 0.0,
        }
        positions_list.append(empty_pos)

    return {
        'total_capital': portfolio.total_capital,
        'utilization': portfolio.capital_utilization,
        'total_pnl_pct': portfolio.total_pnl_pct,
        'avg_pnl_pct': portfolio.get_execution_avg_pnl_pct(),
        'max_drawdown': portfolio.max_drawdown_pct,
        'positions': positions_list,
    }

print("="*80)
print("DETAILED OBSERVATION COMPARISON - DEBUG")
print("="*80)

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
print(f"\n🎲 Random seed: {seed}")

# Create config (match test_inference.py defaults: 2.0x, 0.8, 2)
trading_config = TradingConfig(
    max_leverage=2.0,
    target_utilization=0.8,
    max_positions=2,
)

reward_config = RewardConfig(
    funding_reward_scale=1.0,
    price_reward_scale=1.0,
    liquidation_penalty_scale=10.0,
    opportunity_cost_scale=0.0,
)

# Create environment
print("\n1. Creating environment...")
np.random.seed(seed)
torch.manual_seed(seed)
env = FundingArbitrageEnv(
    data_path="data/rl_test.csv",
    initial_capital=10000.0,
    trading_config=trading_config,
    reward_config=reward_config,
    episode_length_days=5,
    step_hours=5/60,
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    use_full_range_episodes=True,
    verbose=False,
)
print("✅ Environment created")

# Load models
print("\n2. Loading models...")
env_network = ModularPPONetwork()
env_trainer = PPOTrainer(network=env_network, learning_rate=3e-4, device='cpu')
env_trainer.load("checkpoints/best_model.pt")

server_predictor = ModularRLPredictor('checkpoints/best_model.pt', device='cpu')
print("✅ Models loaded")

# Reset
print("\n3. Running first 10 steps and comparing observations...\n")
np.random.seed(seed)
obs_env, info_env = env.reset(seed=seed)

# Feature ranges for labeling
feature_labels = [
    # Config (5)
    "max_leverage", "target_utilization", "max_positions", "stop_loss", "liq_buffer",
    # Portfolio (6)
    "num_positions", "avg_pnl", "total_pnl", "max_drawdown", "min_liq_dist", "utilization",
    # Executions (100 = 5 slots × 20 features)
] + [f"exec{i//20+1}_f{i%20+1}" for i in range(100)] + [
    # Opportunities (190 = 10 slots × 19 features)
] + [f"opp{i//19+1}_f{i%19+1}" for i in range(190)]

for step in range(10):
    print(f"{'='*80}")
    print(f"STEP {step}")
    print(f"{'='*80}\n")

    # Get server observation
    opportunities = [dict(opp) for opp in env.current_opportunities[:10]]
    price_data = env._get_current_prices()
    portfolio_dict = convert_portfolio_to_server_format(env.portfolio, price_data)
    config_dict = {
        'max_leverage': env.current_config.max_leverage,
        'target_utilization': env.current_config.target_utilization,
        'max_positions': env.current_config.max_positions,
        'stop_loss_threshold': env.current_config.stop_loss_threshold,
        'liquidation_buffer': env.current_config.liquidation_buffer,
    }

    obs_server = server_predictor._build_observation(config_dict, portfolio_dict, opportunities)

    # Compare observations
    diff = np.abs(obs_env - obs_server)
    max_diff = np.max(diff)
    max_diff_idx = np.argmax(diff)

    print(f"Max difference: {max_diff:.10f} at index {max_diff_idx} ({feature_labels[max_diff_idx]})")
    print(f"  ENV value: {obs_env[max_diff_idx]:.10f}")
    print(f"  SERVER value: {obs_server[max_diff_idx]:.10f}")

    # Show all differences > 0.001
    large_diffs = np.where(diff > 0.001)[0]
    if len(large_diffs) > 0:
        print(f"\n📊 All differences > 0.001:")
        for idx in large_diffs[:20]:  # Show first 20
            label = feature_labels[idx] if idx < len(feature_labels) else f"idx{idx}"
            print(f"  [{idx:3d}] {label:20s}: ENV={obs_env[idx]:10.6f}  SERVER={obs_server[idx]:10.6f}  diff={diff[idx]:10.6f}")
    else:
        print(f"\n✅ All feature differences < 0.001")

    # Portfolio debug info
    if step == 0:
        print(f"\n📦 Portfolio Debug:")
        print(f"  capital_utilization: {env.portfolio.capital_utilization:.6f} (0-100 scale)")
        print(f"  capital_utilization/100: {env.portfolio.capital_utilization/100:.6f} (normalized)")
        print(f"  utilization from dict: {portfolio_dict['utilization']:.6f}")
        print(f"  total_capital: {env.portfolio.total_capital:.2f}")
        print(f"  num positions: {len(env.portfolio.positions)}")

    # Step environment
    action_mask = env._get_action_mask()
    action, _, _ = env_trainer.select_action(obs_env, action_mask, deterministic=True)
    print(f"Environment action: {action}")
    obs_env, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

    print()

print("\n" + "="*80)
print("✅ DEBUG COMPLETE")
print("="*80)
