"""
Test Server Trading Results

Runs a full episode using the server predictor (same as backend would use)
and collects trading metrics for comparison with test_inference.py.
"""

import numpy as np
import torch
from pathlib import Path

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from server.inference.rl_predictor import ModularRLPredictor


def convert_portfolio_to_server_format(portfolio, price_data):
    """Convert environment portfolio to server format."""
    positions_list = []

    for pos in portfolio.positions:
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


print("=" * 80)
print("SERVER TRADING TEST")
print("=" * 80)

# Set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
print(f"\n🎲 Random seed: {seed}")

# Create config (EXACT match to test_inference.py defaults)
trading_config = TradingConfig(
    max_leverage=2.0,
    target_utilization=0.8,
    max_positions=2,
    stop_loss_threshold=-0.02,
    liquidation_buffer=0.15,
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
    episode_length_days=7,
    step_hours=5/60,
    price_history_path="data/symbol_data",
    feature_scaler_path=None,
    use_full_range_episodes=True,  # TRUE to use entire data range (matches test_inference.py --full-test default)
    verbose=False,
)
print("✅ Environment created")

# Load server predictor (suppress verbose output by redirecting)
print("\n2. Loading server predictor...")
import sys
import os
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')
server_predictor = ModularRLPredictor('checkpoints/best_model.pt', device='cpu')
sys.stdout = old_stdout
print("✅ Server predictor loaded")

# Reset
print("\n3. Running full episode with server predictor...\n")
np.random.seed(seed)
obs, info = env.reset(seed=seed)

total_reward = 0.0
steps = 0
trades_executed = 0

while True:
    # Get opportunities and portfolio state
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

    # Get action from server predictor
    result = server_predictor.predict_opportunities(
        opportunities,
        portfolio_dict,
        config_dict
    )

    # Extract action integer from result
    action = result['action_id']

    # Log first 10 actions for debugging
    if steps < 10:
        print(f"  Step {steps}: Action={action}, Confidence={result['confidence']:.4f}")

    # Track trades
    if action > 0:  # Not HOLD
        trades_executed += 1

    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1

    if steps % 500 == 0:
        print(f"  Step {steps}: Reward={total_reward:.2f}, P&L={env.portfolio.total_pnl_pct:.2f}%, Trades={trades_executed}, Last action={action}")

    if terminated or truncated:
        break

# Calculate metrics
final_pnl_pct = env.portfolio.total_pnl_pct
final_pnl_usd = env.portfolio.total_capital - env.portfolio.initial_capital
winning_trades = sum(1 for pos in env.portfolio.closed_positions if pos.unrealized_pnl_pct > 0)
losing_trades = sum(1 for pos in env.portfolio.closed_positions if pos.unrealized_pnl_pct <= 0)
total_closed = len(env.portfolio.closed_positions)
win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0

print("\n" + "=" * 80)
print("SERVER TRADING RESULTS")
print("=" * 80)
print(f"Mean Reward:        {total_reward:.2f}")
print(f"Mean P&L (USD):  $ {final_pnl_usd:.2f}")
print(f"Mean P&L (%):       {final_pnl_pct:.2f}%")
print(f"Total Trades:         {total_closed}")
print(f"Winning Trades:        {winning_trades}")
print(f"Losing Trades:         {losing_trades}")
print(f"Win Rate:            {win_rate:.1f}%")
print(f"Total Steps:          {steps}")
print("=" * 80)
