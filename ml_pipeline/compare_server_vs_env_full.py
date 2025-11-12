"""
Full Comparison: Server vs Environment Trading Results

Runs a complete episode using both:
1. Environment (test_inference.py approach)
2. Server predictor (ML API approach)

Compares final P&L, trades, win rate to verify identical behavior.
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

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
print("FULL SERVER VS ENVIRONMENT COMPARISON")
print("="*80)

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
print(f"\n🎲 Random seed: {seed}")

# Create config
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
env = FundingArbitrageEnv(
    data_path="data/rl_test.csv",
    initial_capital=10000.0,
    trading_config=trading_config,
    reward_config=reward_config,
    episode_length_days=5,
    step_hours=5/60,
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    use_full_range_episodes=True,  # Use entire test dataset (matches test_inference.py)
    verbose=False,
)
print("✅ Environment created")

# Create trainers
print("\n2. Loading models...")
env_network = ModularPPONetwork()
env_trainer = PPOTrainer(network=env_network, learning_rate=3e-4, device='cpu')
env_trainer.load("checkpoints/best_model.pt")

server_predictor = ModularRLPredictor('checkpoints/best_model.pt', device='cpu')
print("✅ Models loaded")

# Reset
print("\n3. Running episode...")
obs, info = env.reset(seed=seed)

action_mismatches = 0
max_steps = 3088  # Full episode
step = 0

# Suppress verbose output from predictor
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

while True:
    # Environment action
    action_mask = env._get_action_mask()
    env_action, _, _ = env_trainer.select_action(obs, action_mask, deterministic=True)

    # Server action
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

    server_result = server_predictor.predict_opportunities(opportunities, portfolio_dict, config_dict)
    server_action = server_result['action_id']

    # Compare
    if env_action != server_action:
        action_mismatches += 1

    # Step environment
    obs, reward, terminated, truncated, info = env.step(env_action)
    step += 1

    if terminated or truncated:
        break

    if step % 500 == 0:
        sys.stdout = old_stdout
        print(f"   Step {step}/{max_steps}")
        sys.stdout = open(os.devnull, 'w')

# Restore stdout
sys.stdout = old_stdout

# Get final metrics
final_pnl_pct = env.portfolio.total_pnl_pct
final_capital = env.portfolio.total_capital
closed_trades = env.portfolio.closed_positions
winning_trades = sum(1 for t in closed_trades if t.realized_pnl_pct > 0)
losing_trades = sum(1 for t in closed_trades if t.realized_pnl_pct <= 0)
win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\n📊 Environment (test_inference.py path):")
print(f"   P&L: ${final_capital - 10000:.2f} ({final_pnl_pct:.2f}%)")
print(f"   Trades: {len(closed_trades)}")
print(f"   Win Rate: {win_rate:.1f}% ({winning_trades}W / {losing_trades}L)")
print(f"   Final Capital: ${final_capital:.2f}")

print(f"\n🔄 Server (ML API path):")
print(f"   Actions match: {step - action_mismatches}/{step} ({(step - action_mismatches)/step*100:.1f}%)")
print(f"   Action mismatches: {action_mismatches}")

if action_mismatches == 0:
    print(f"\n✅ SUCCESS: Server produces IDENTICAL trading results!")
    print(f"   Same P&L: ${final_capital - 10000:.2f} ({final_pnl_pct:.2f}%)")
    print(f"   Same Trades: {len(closed_trades)}")
    print(f"   Same Win Rate: {win_rate:.1f}%")
else:
    print(f"\n⚠️  Server has {action_mismatches} different actions")
    print(f"   Trading results may differ slightly")

print("\n" + "="*80)
