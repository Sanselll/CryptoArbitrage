"""
Test Server vs Inference Comparison

Compares predictions from:
1. test_inference.py approach (using FundingArbitrageEnv)
2. ML server approach (using ModularRLPredictor directly)

Goal: Verify both produce identical predictions given the same data.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import timedelta

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer
from server.inference.rl_predictor import ModularRLPredictor


def parse_args():
    parser = argparse.ArgumentParser(description='Compare server vs inference predictions')

    parser.add_argument('--num-steps', type=int, default=288,
                        help='Number of steps to test (default: 288 = 1 day at 5-min intervals)')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv',
                        help='Path to test data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--feature-scaler-path', type=str, default=None,
                        help='Path to feature scaler')
    parser.add_argument('--output-csv', type=str, default='comparison_results.csv',
                        help='Output CSV for comparison results')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed step-by-step output')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')

    # Trading config
    parser.add_argument('--leverage', type=float, default=1.0,
                        help='Max leverage')
    parser.add_argument('--utilization', type=float, default=0.9,
                        help='Target utilization')
    parser.add_argument('--max-positions', type=int, default=3,
                        help='Max positions')

    return parser.parse_args()


def convert_portfolio_to_server_format(portfolio, positions_list=None, price_data=None) -> Dict:
    """
    Convert environment portfolio to server JSON format.

    Args:
        portfolio: Portfolio object from environment
        positions_list: Optional list of position dicts (if already converted)
        price_data: Dict mapping symbol to {'long_price': float, 'short_price': float}

    Returns:
        Dict in server format
    """
    # Convert positions
    if positions_list is None:
        positions_list = []

        for pos in portfolio.positions:
            # Get current prices from price_data
            current_long_price = pos.entry_long_price
            current_short_price = pos.entry_short_price

            if price_data and pos.symbol in price_data:
                current_long_price = price_data[pos.symbol]['long_price']
                current_short_price = price_data[pos.symbol]['short_price']

            pos_dict = {
                'symbol': pos.symbol,
                'long_exchange': pos.long_exchange,
                'short_exchange': pos.short_exchange,
                'position_size_usd': pos.position_size_usd,
                'leverage': pos.leverage,
                'margin_used_usd': pos.margin_used_usd,
                'entry_long_price': pos.entry_long_price,
                'entry_short_price': pos.entry_short_price,
                'current_long_price': current_long_price,
                'current_short_price': current_short_price,
                'long_funding_rate': pos.long_funding_rate,
                'short_funding_rate': pos.short_funding_rate,
                'long_net_funding_usd': pos.long_net_funding_usd,
                'short_net_funding_usd': pos.short_net_funding_usd,
                'unrealized_pnl_usd': pos.unrealized_pnl_usd,
                'unrealized_pnl_pct': pos.unrealized_pnl_pct,
                'hours_held': pos.hours_held,
                'position_age_hours': pos.hours_held,
                'entry_fees_paid_usd': pos.entry_fees_paid_usd,
                'long_pnl_pct': pos.long_pnl_pct,
                'short_pnl_pct': pos.short_pnl_pct,
                'is_active': True,
                'position_is_active': 1.0,
                # Funding intervals (needed for Phase 2 APR calculation)
                'long_funding_interval_hours': pos.long_funding_interval_hours,
                'short_funding_interval_hours': pos.short_funding_interval_hours,
                # Phase 1 features
                'pnl_history': pos.pnl_history.copy() if hasattr(pos, 'pnl_history') else [],
                'peak_pnl_pct': pos.peak_pnl_pct if hasattr(pos, 'peak_pnl_pct') else 0.0,
                'entry_apr': pos.entry_apr if hasattr(pos, 'entry_apr') else 0.0,
            }

            # Calculate current spread
            if current_long_price > 0 and current_short_price > 0:
                avg_price = (current_long_price + current_short_price) / 2
                pos_dict['current_spread_pct'] = abs(current_long_price - current_short_price) / avg_price
            else:
                pos_dict['current_spread_pct'] = 0.0

            # Calculate entry spread
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

    # Pad to 5 positions if needed (environment uses max 5 position slots)
    while len(positions_list) < 5:
        # Add empty position slot
        empty_pos = {
            'symbol': '',
            'position_size_usd': 0.0,
            'unrealized_pnl_pct': 0.0,
            'hours_held': 0.0,
            'position_age_hours': 0.0,
            'is_active': False,
            'position_is_active': 0.0,
            'long_net_funding_usd': 0.0,
            'short_net_funding_usd': 0.0,
            'long_funding_rate': 0.0,
            'short_funding_rate': 0.0,
            'current_long_price': 0.0,
            'current_short_price': 0.0,
            'entry_long_price': 0.0,
            'entry_short_price': 0.0,
            'current_spread_pct': 0.0,
            'entry_spread_pct': 0.0,
            'long_pnl_pct': 0.0,
            'short_pnl_pct': 0.0,
            'liquidation_distance': 1.0,
            'entry_fees_paid_usd': 0.0,
            # Phase 1 features
            'pnl_history': [],
            'peak_pnl_pct': 0.0,
            'entry_apr': 0.0,
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


def compare_predictions(
    env_action: int,
    env_value: float,
    env_obs: np.ndarray,
    server_result: Dict,
    server_obs: np.ndarray,
    step_num: int,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    """
    Compare environment prediction vs server prediction.

    Returns:
        (match: bool, details: Dict)
    """
    server_action = server_result['action_id']
    server_value = server_result['state_value']
    server_confidence = server_result['confidence']

    # Check if actions match
    actions_match = (env_action == server_action)

    # Check if observations match (allow small floating point differences)
    obs_diff = np.abs(env_obs - server_obs)
    max_obs_diff = np.max(obs_diff)
    obs_match = max_obs_diff < 1e-5

    # Check if values match
    value_diff = abs(env_value - server_value)
    values_match = value_diff < 1e-5

    all_match = actions_match and obs_match and values_match

    details = {
        'step': step_num,
        'actions_match': actions_match,
        'env_action': env_action,
        'server_action': server_action,
        'obs_match': obs_match,
        'max_obs_diff': max_obs_diff,
        'values_match': values_match,
        'env_value': env_value,
        'server_value': server_value,
        'value_diff': value_diff,
        'server_confidence': server_confidence,
        'all_match': all_match,
    }

    if verbose or not all_match:
        print(f"\n{'='*80}")
        print(f"Step {step_num} Comparison:")
        print(f"{'='*80}")
        print(f"  Actions: Env={env_action:3d} | Server={server_action:3d} | Match={actions_match}")
        print(f"  Values:  Env={env_value:8.4f} | Server={server_value:8.4f} | Diff={value_diff:.6f} | Match={values_match}")
        print(f"  Obs:     Max diff={max_obs_diff:.6e} | Match={obs_match}")
        print(f"  Server Confidence: {server_confidence:.4f}")

        if not all_match:
            print(f"\n  ⚠️  MISMATCH DETECTED!")
            if not actions_match:
                print(f"     Action mismatch: {env_action} != {server_action}")
            if not obs_match:
                print(f"     Observation mismatch: max diff = {max_obs_diff:.6e}")
                # Find which feature differs most
                max_diff_idx = np.argmax(obs_diff)
                print(f"     Largest diff at index {max_diff_idx}: {env_obs[max_diff_idx]:.6f} vs {server_obs[max_diff_idx]:.6f}")
            if not values_match:
                print(f"     Value mismatch: {env_value:.6f} vs {server_value:.6f}")

    return all_match, details


def main():
    args = parse_args()

    print("="*80)
    print("SERVER VS INFERENCE COMPARISON TEST")
    print("="*80)
    print(f"Test data: {args.test_data_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Feature scaler: {args.feature_scaler_path}")
    print(f"Num steps: {args.num_steps}")
    print(f"Random seed: {args.seed}")
    print(f"Trading config: leverage={args.leverage}x, util={args.utilization}, max_pos={args.max_positions}")
    print("="*80)

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"\n🎲 Random seed set to {args.seed}")

    # Create trading config
    trading_config = TradingConfig(
        max_leverage=args.leverage,
        target_utilization=args.utilization,
        max_positions=args.max_positions,
        stop_loss_threshold=-0.02,
        liquidation_buffer=0.15,
    )

    print("\n1. Initializing Environment (test_inference.py path)...")
    env = FundingArbitrageEnv(
        data_path=args.test_data_path,
        initial_capital=10000.0,
        trading_config=trading_config,
        episode_length_days=7,  # Doesn't matter, we'll control steps
        step_hours=5/60,  # 5 minutes
        feature_scaler_path=args.feature_scaler_path,
        verbose=False,
    )
    print("✅ Environment created")

    # Create trainer for environment
    print("\n2. Initializing Environment Trainer...")
    env_network = ModularPPONetwork()
    env_trainer = PPOTrainer(
        network=env_network,
        learning_rate=3e-4,
        device='cpu',
    )
    env_trainer.load(args.checkpoint)
    print("✅ Environment trainer loaded")

    print("\n3. Initializing Server Predictor (ML server path)...")
    server_predictor = ModularRLPredictor(
        model_path=args.checkpoint,
        feature_scaler_path=args.feature_scaler_path,
        device='cpu',
    )
    print("✅ Server predictor loaded")

    # Reset environment
    print("\n4. Resetting environment with seed...")
    obs, info = env.reset(seed=args.seed)
    print(f"✅ Environment reset (episode: {info['episode_start']} to {info['episode_end']})")

    # Comparison tracking
    comparison_results = []
    total_steps = 0
    matching_steps = 0

    print(f"\n5. Running {args.num_steps} step comparison...")
    print("="*80)

    for step in range(args.num_steps):
        total_steps += 1

        # === ENVIRONMENT PATH ===
        # Get action mask
        if hasattr(env, '_get_action_mask'):
            action_mask = env._get_action_mask()
        else:
            action_mask = None

        # Get environment observation (already built)
        env_obs = obs.copy()

        # Get environment prediction
        env_action, env_value, env_log_prob = env_trainer.select_action(
            env_obs,
            action_mask,
            deterministic=True
        )

        # === SERVER PATH ===
        # Build server request from environment state
        opportunities = []
        for opp in env.current_opportunities[:10]:  # Server processes up to 10
            opportunities.append(dict(opp))  # Convert to dict if needed

        # Get current prices from environment (needed for accurate position features)
        price_data = env._get_current_prices()

        portfolio_dict = convert_portfolio_to_server_format(env.portfolio, price_data=price_data)

        # Add episode_progress (calculated from environment's step_count and episode_length)
        portfolio_dict['episode_progress'] = env.step_count / env.episode_length_hours if env.episode_length_hours > 0 else 0.0

        trading_config_dict = {
            'max_leverage': env.current_config.max_leverage,
            'target_utilization': env.current_config.target_utilization,
            'max_positions': env.current_config.max_positions,
            'stop_loss_threshold': env.current_config.stop_loss_threshold,
            'liquidation_buffer': env.current_config.liquidation_buffer,
        }

        # Get server prediction
        server_result = server_predictor.predict_opportunities(
            opportunities,
            portfolio_dict,
            trading_config_dict
        )

        # Build server observation (for comparison)
        server_obs = server_predictor._build_observation(
            trading_config_dict,
            portfolio_dict,
            opportunities
        )

        # === COMPARE ===
        match, details = compare_predictions(
            env_action,
            env_value,
            env_obs,
            server_result,
            server_obs,
            step,
            verbose=args.verbose
        )

        if match:
            matching_steps += 1

        comparison_results.append(details)

        # Step environment forward for next iteration
        obs, reward, terminated, truncated, info = env.step(env_action)

        if terminated or truncated:
            print(f"\n⚠️  Episode ended at step {step+1}")
            break

    # === RESULTS ===
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"Total steps tested: {total_steps}")
    print(f"Matching steps: {matching_steps}")
    print(f"Mismatched steps: {total_steps - matching_steps}")
    print(f"Match rate: {matching_steps / total_steps * 100:.2f}%")

    if matching_steps == total_steps:
        print("\n✅ SUCCESS: All predictions match!")
        print("   Server and test_inference.py produce identical results.")
    else:
        print("\n❌ FAILURE: Some predictions differ!")
        print("   Server and test_inference.py produce different results.")
        print("\nMismatched steps:")
        for result in comparison_results:
            if not result['all_match']:
                print(f"  Step {result['step']}: "
                      f"Actions={result['actions_match']}, "
                      f"Obs={result['obs_match']}, "
                      f"Values={result['values_match']}")

    # Save results to CSV
    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv(args.output_csv, index=False)
    print(f"\n💾 Detailed results saved to: {args.output_csv}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

    # Exit with appropriate code
    if matching_steps == total_steps:
        return 0
    else:
        return 1


if __name__ == '__main__':
    exit(main())
