"""
Server Trading Results: Run episode using SERVER predictions

This runs a complete episode using the SERVER's predictions to see:
- What P&L the server achieves
- How many trades the server makes
- What win rate the server gets

Compares to test_inference.py baseline results.
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
print("SERVER TRADING RESULTS COMPARISON")
print("="*80)

# Set seed
seed = 42
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

# Suppress verbose output from predictor
old_stdout = sys.stdout
sys.stdout = open(os.devnull, 'w')

# Create environment for ENV path
print("\n1. Creating environment for ENV path...")
np.random.seed(seed)  # Seed before creating env
torch.manual_seed(seed)
env_path = FundingArbitrageEnv(
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

# Create environment for SERVER path
env_server = FundingArbitrageEnv(
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

sys.stdout = old_stdout
print("✅ Environments created")

# Load models
print("\n2. Loading models...")
sys.stdout = open(os.devnull, 'w')

env_network = ModularPPONetwork()
env_trainer = PPOTrainer(network=env_network, learning_rate=3e-4, device='cpu')
env_trainer.load("checkpoints/best_model.pt")

server_predictor = ModularRLPredictor('checkpoints/best_model.pt', device='cpu')

sys.stdout = old_stdout
print("✅ Models loaded")

# Run ENV path
print("\n3. Running ENV path episode...")
np.random.seed(seed)  # Re-seed to ensure deterministic episode selection
obs_env, info_env = env_path.reset(seed=seed)
step_env = 0

sys.stdout = open(os.devnull, 'w')
while True:
    action_mask = env_path._get_action_mask()
    action, _, _ = env_trainer.select_action(obs_env, action_mask, deterministic=True)
    obs_env, reward_env, terminated_env, truncated_env, info_env = env_path.step(action)
    step_env += 1

    if terminated_env or truncated_env:
        break

    if step_env % 500 == 0:
        sys.stdout = old_stdout
        print(f"   ENV Step {step_env}")
        sys.stdout = open(os.devnull, 'w')

sys.stdout = old_stdout

# Get ENV results
env_final_pnl_pct = env_path.portfolio.total_pnl_pct
env_final_capital = env_path.portfolio.total_capital
env_closed_trades = env_path.portfolio.closed_positions
env_winning = sum(1 for t in env_closed_trades if t.realized_pnl_pct > 0)
env_losing = sum(1 for t in env_closed_trades if t.realized_pnl_pct <= 0)
env_win_rate = (env_winning / len(env_closed_trades) * 100) if env_closed_trades else 0

print(f"✅ ENV path completed: {step_env} steps")

# Run SERVER path
print("\n4. Running SERVER path episode...")
np.random.seed(seed)  # Re-seed to ensure deterministic episode selection
obs_server, info_server = env_server.reset(seed=seed)
step_server = 0

sys.stdout = open(os.devnull, 'w')
while True:
    # Get server action
    opportunities = [dict(opp) for opp in env_server.current_opportunities[:10]]
    price_data = env_server._get_current_prices()
    portfolio_dict = convert_portfolio_to_server_format(env_server.portfolio, price_data)
    config_dict = {
        'max_leverage': env_server.current_config.max_leverage,
        'target_utilization': env_server.current_config.target_utilization,
        'max_positions': env_server.current_config.max_positions,
        'stop_loss_threshold': env_server.current_config.stop_loss_threshold,
        'liquidation_buffer': env_server.current_config.liquidation_buffer,
    }

    server_result = server_predictor.predict_opportunities(opportunities, portfolio_dict, config_dict)
    server_action = server_result['action_id']

    obs_server, reward_server, terminated_server, truncated_server, info_server = env_server.step(server_action)
    step_server += 1

    if terminated_server or truncated_server:
        break

    if step_server % 500 == 0:
        sys.stdout = old_stdout
        print(f"   SERVER Step {step_server}")
        sys.stdout = open(os.devnull, 'w')

sys.stdout = old_stdout

# Get SERVER results
server_final_pnl_pct = env_server.portfolio.total_pnl_pct
server_final_capital = env_server.portfolio.total_capital
server_closed_trades = env_server.portfolio.closed_positions
server_winning = sum(1 for t in server_closed_trades if t.realized_pnl_pct > 0)
server_losing = sum(1 for t in server_closed_trades if t.realized_pnl_pct <= 0)
server_win_rate = (server_winning / len(server_closed_trades) * 100) if server_closed_trades else 0

print(f"✅ SERVER path completed: {step_server} steps")

print("\n" + "="*80)
print("RESULTS COMPARISON")
print("="*80)

print(f"\n📊 ENV PATH (test_inference.py):")
print(f"   P&L: ${env_final_capital - 10000:.2f} ({env_final_pnl_pct:.2f}%)")
print(f"   Trades: {len(env_closed_trades)}")
print(f"   Win Rate: {env_win_rate:.1f}% ({env_winning}W / {env_losing}L)")
print(f"   Final Capital: ${env_final_capital:.2f}")

print(f"\n🔄 SERVER PATH (ML API):")
print(f"   P&L: ${server_final_capital - 10000:.2f} ({server_final_pnl_pct:.2f}%)")
print(f"   Trades: {len(server_closed_trades)}")
print(f"   Win Rate: {server_win_rate:.1f}% ({server_winning}W / {server_losing}L)")
print(f"   Final Capital: ${server_final_capital:.2f}")

print(f"\n📈 DIFFERENCES:")
pnl_diff = server_final_pnl_pct - env_final_pnl_pct
trades_diff = len(server_closed_trades) - len(env_closed_trades)
win_rate_diff = server_win_rate - env_win_rate
print(f"   P&L Difference: {pnl_diff:+.2f}%")
print(f"   Trade Count Difference: {trades_diff:+d}")
print(f"   Win Rate Difference: {win_rate_diff:+.1f}%")

if abs(pnl_diff) < 0.1 and abs(trades_diff) <= 1:
    print(f"\n✅ SUCCESS: Server produces NEARLY IDENTICAL results!")
elif abs(pnl_diff) < 5.0:
    print(f"\n⚠️  Server results differ slightly (within 5%)")
else:
    print(f"\n❌ SIGNIFICANT DIFFERENCE: Server results diverge substantially")

print("\n" + "="*80)
