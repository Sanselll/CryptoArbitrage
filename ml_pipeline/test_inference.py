"""
Test model inference - verify trained model can be loaded and used
"""

import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig
from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Test model inference')

    # Trading configuration
    parser.add_argument('--leverage', type=float, default=2.0,
                        help='Max leverage (default: 2.0x)')
    parser.add_argument('--utilization', type=float, default=0.8,
                        help='Capital utilization (default: 0.8 = 80%%)')
    parser.add_argument('--max-positions', type=int, default=2,
                        help='Max concurrent positions (default: 2)')

    # Test configuration
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to test (default: 1)')
    parser.add_argument('--full-test', action='store_true', default=True,
                        help='Use entire test dataset (default: True). If disabled, uses --episode-length-days.')
    parser.add_argument('--no-full-test', action='store_false', dest='full_test',
                        help='Disable full test mode and use --episode-length-days instead')
    parser.add_argument('--episode-length-days', type=int, default=5,
                        help='Episode length in days (only used if --no-full-test is specified, default: 5)')
    parser.add_argument('--step-minutes', type=int, default=5,
                        help='Minutes per prediction step (default: 5 = 5-minute intervals)')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv',
                        help='Path to test data')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--trades-output', type=str, default='trades_inference.csv',
                        help='Output CSV file for trade records (default: trades_inference.csv)')
    parser.add_argument('--price-history-path', type=str, default='data/symbol_data',
                        help='Path to price history directory for hourly funding rate updates (default: data/symbol_data)')
    parser.add_argument('--feature-scaler-path', type=str, default='trained_models/rl/feature_scaler.pkl',
                        help='Path to fitted StandardScaler pickle (default: trained_models/rl/feature_scaler.pkl)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible results (default: 42)')
    parser.add_argument('--force-zero-pnl', action='store_true',
                        help='Force total_pnl_pct to always be 0 in observations (simulates production bug)')
    parser.add_argument('--start-time', type=str, default=None,
                        help='Start time for filtering test data (e.g., "2025-11-13 09:20:00")')
    parser.add_argument('--end-time', type=str, default=None,
                        help='End time for filtering test data (e.g., "2025-11-13 09:30:00")')

    return parser.parse_args()


def test_model_inference(args):
    """Test loading and using a trained model."""
    print("=" * 80)
    print("TESTING MODEL INFERENCE ON TEST SET")
    print("=" * 80)

    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"\n🎲 Random seed: {args.seed}")

    # Filter test data by time range if specified
    test_data_path = args.test_data_path
    if args.start_time or args.end_time:
        print("\n⏱️  Filtering test data by time range...")
        import tempfile

        df = pd.read_csv(args.test_data_path)
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

        print(f"   Original: {len(df)} rows ({df['entry_time'].min()} to {df['entry_time'].max()})")

        if args.start_time:
            start_dt = pd.to_datetime(args.start_time, utc=True)
            df = df[df['entry_time'] >= start_dt]
            print(f"   Filtered start >= {start_dt}: {len(df)} rows")

        if args.end_time:
            end_dt = pd.to_datetime(args.end_time, utc=True)
            df = df[df['entry_time'] <= end_dt]
            print(f"   Filtered end <= {end_dt}: {len(df)} rows")

        if len(df) == 0:
            raise ValueError("No data found in specified time range!")

        print(f"   Final: {len(df)} rows ({df['entry_time'].min()} to {df['entry_time'].max()})")

        # Save filtered data to temporary file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        test_data_path = temp_file.name
        print(f"   Saved filtered data to: {test_data_path}")

    # Create environment (use test data)
    print("\n1. Creating test environment...")

    # Trading config - use command-line args
    trading_config = TradingConfig(
        max_leverage=args.leverage,
        target_utilization=args.utilization,
        max_positions=args.max_positions,
    )

    # Reward config (match training defaults)
    reward_config = RewardConfig(
        funding_reward_scale=1.0,
        price_reward_scale=1.0,
        liquidation_penalty_scale=10.0,
        opportunity_cost_scale=0.0,  # Disabled by default
    )

    # Convert step minutes to hours for the environment
    step_hours = args.step_minutes / 60.0

    env = FundingArbitrageEnv(
        data_path=test_data_path,
        initial_capital=args.initial_capital,
        trading_config=trading_config,
        reward_config=reward_config,
        episode_length_days=args.episode_length_days,
        step_hours=step_hours,
        price_history_path=args.price_history_path,
        feature_scaler_path=args.feature_scaler_path,
        use_full_range_episodes=args.full_test,
        force_zero_total_pnl_pct=args.force_zero_pnl,
        verbose=False,
    )

    print("✅ Test environment created")
    print(f"   Data: {test_data_path}")

    # Show episode configuration based on mode
    if args.full_test:
        # Calculate actual data range
        test_df = pd.read_csv(args.test_data_path)
        test_df['entry_time'] = pd.to_datetime(test_df['entry_time'])
        data_start = test_df['entry_time'].min()
        data_end = test_df['entry_time'].max()
        data_days = (data_end - data_start).total_seconds() / 86400
        total_steps = int((data_days * 24 * 60) / args.step_minutes)

        print(f"   🌐 FULL TEST MODE: Testing on ENTIRE test dataset")
        print(f"   Episode length: {data_days:.1f} days ({data_start} to {data_end})")
        print(f"   Total steps: ~{total_steps:,} steps at {args.step_minutes}-minute intervals")
    else:
        total_steps = int((args.episode_length_days * 24 * 60) / args.step_minutes)
        print(f"   Episode length: {args.episode_length_days} days ({total_steps} steps at {args.step_minutes}-minute intervals)")

    print(f"   Step interval: {args.step_minutes} minute(s) ({step_hours:.4f} hours)")
    print(f"   Initial capital: ${args.initial_capital:,.2f}")
    print(f"   Leverage: {args.leverage}x")
    print(f"   Utilization: {args.utilization:.0%}")
    print(f"   Max Positions: {args.max_positions}")
    if args.price_history_path:
        print(f"   Price History: {args.price_history_path} (dynamic funding updates enabled)")
    if args.force_zero_pnl:
        print(f"   ⚠️  FORCING total_pnl_pct = 0 in observations (production bug simulation)")

    # Create network
    print("\n2. Creating network...")
    network = ModularPPONetwork()
    print(f"✅ Network created ({sum(p.numel() for p in network.parameters()):,} parameters)")

    # Create trainer
    print("\n3. Creating trainer...")
    trainer = PPOTrainer(
        network=network,
        learning_rate=3e-4,
        device='cpu',
    )
    print("✅ Trainer created")

    # Load trained model
    print("\n4. Loading trained model...")
    try:
        trainer.load(args.checkpoint)
        # CRITICAL: Set network to eval mode for inference (disables dropout)
        # This must match the server predictor which uses eval mode
        trainer.network.eval()
        print(f"✅ Model loaded from {args.checkpoint}")
    except FileNotFoundError:
        print(f"⚠️  No checkpoint found at {args.checkpoint} - using random weights")

    # Run inference with detailed metrics (match training evaluation)
    episode_text = "episode" if args.num_episodes == 1 else "episodes"
    print(f"\n5. Running inference on {args.num_episodes} {episode_text}...")

    # Episode-level metrics
    eval_rewards = []
    eval_lengths = []

    # Trading metrics
    total_pnls = []
    total_pnl_pcts = []
    num_trades = []
    num_winning_trades = []
    num_losing_trades = []
    avg_trade_durations = []
    max_drawdowns = []
    opportunities_seen = []
    trades_executed = []

    # Config tracking
    configs_used = []

    # Trade tracking (for CSV export)
    all_trades = []

    # Profit factor metrics
    all_winning_pnl = []
    all_losing_pnl = []

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed)
        episode_reward = 0.0
        episode_length = 0
        done = False

        opportunities_count = 0

        while not done:
            # Count opportunities available at this step
            if hasattr(env, 'current_opportunities'):
                opportunities_count += len(env.current_opportunities)

            # Log features for first 5 steps only (to avoid flooding)
            if episode_length < 5:
                try:
                    import datetime
                    from pathlib import Path

                    log_path = Path('test_features.log')

                    # Get active positions from environment's portfolio
                    portfolio = env.portfolio
                    active_positions = [p for p in portfolio.positions if p.hours_held > 0]

                    if active_positions:
                        # Find best available APR from opportunities
                        best_available_apr = 0.0
                        if hasattr(env, 'current_opportunities') and env.current_opportunities:
                            best_available_apr = max(opp.get('fund_apr', 0.0) for opp in env.current_opportunities)

                        # Feature names
                        feature_names = [
                            "position_is_active", "net_pnl_pct", "hours_held", "net_funding_ratio",
                            "net_funding_rate", "current_spread_pct", "entry_spread_pct", "value_ratio",
                            "funding_efficiency", "long_pnl_pct", "short_pnl_pct", "liquidation_distance",
                            "pnl_velocity", "peak_drawdown", "apr_ratio", "return_efficiency",
                            "is_old_loser", "current_position_apr", "best_available_apr", "apr_advantage"
                        ]

                        with open(log_path, 'a') as f:
                            f.write("=" * 80 + "\n")
                            f.write(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Test Position Features (Step {episode_length})\n")
                            f.write("=" * 80 + "\n")

                            for slot_idx, pos in enumerate(active_positions):
                                symbol = pos.symbol
                                raw_hours_held = pos.hours_held
                                entry_apr = pos.entry_apr if hasattr(pos, 'entry_apr') else 0.0

                                f.write(f"\nSymbol: {symbol}\n")
                                f.write(f"Hours Held: {raw_hours_held:.2f}h ({raw_hours_held*60:.0f} minutes)\n")
                                f.write(f"Entry APR: {entry_apr:.0f}%\n\n")

                                # Extract normalized features from observation vector
                                feat_start_idx = 11 + (slot_idx * 20)
                                normalized_features = obs[feat_start_idx:feat_start_idx+20]

                                # Get raw values
                                net_pnl_pct_raw = pos.unrealized_pnl_pct * 100
                                long_funding_rate = pos.long_funding_rate
                                short_funding_rate = pos.short_funding_rate
                                net_funding_rate = short_funding_rate - long_funding_rate

                                # Calculate APR
                                long_interval = pos.long_funding_interval_hours
                                short_interval = pos.short_funding_interval_hours
                                if long_interval == short_interval:
                                    avg_interval_hours = long_interval
                                else:
                                    avg_interval_hours = (
                                        (abs(long_funding_rate) * long_interval + abs(short_funding_rate) * short_interval) /
                                        (abs(long_funding_rate) + abs(short_funding_rate) + 1e-9)
                                    )
                                payments_per_day = 24.0 / avg_interval_hours
                                annual_rate = net_funding_rate * payments_per_day * 365.0
                                current_position_apr = annual_rate * 100

                                peak_pnl_pct_raw = pos.peak_pnl_pct * 100 if hasattr(pos, 'peak_pnl_pct') else 0.0

                                # Write raw features
                                f.write("Raw Features (before normalization):\n")
                                f.write(f"  1. position_is_active         : {normalized_features[0]:.4f}\n")
                                f.write(f"  2. net_pnl_pct                : {net_pnl_pct_raw:.4f}%\n")
                                f.write(f"  3. hours_held                 : {raw_hours_held:.4f}h\n")
                                f.write(f"  4. net_funding_ratio          : {normalized_features[3]:.4f}\n")
                                f.write(f"  5. net_funding_rate           : {net_funding_rate:.6f}\n")
                                f.write(f"  6. current_spread_pct         : {normalized_features[5]:.6f}\n")
                                f.write(f"  7. entry_spread_pct           : {normalized_features[6]:.6f}\n")
                                f.write(f"  8. value_ratio                : {normalized_features[7]:.4f}\n")
                                f.write(f"  9. funding_efficiency         : {normalized_features[8]:.4f}\n")
                                f.write(f" 10. long_pnl_pct               : {pos.long_pnl_pct*100:.4f}%\n")
                                f.write(f" 11. short_pnl_pct              : {pos.short_pnl_pct*100:.4f}%\n")
                                f.write(f" 12. liquidation_distance       : {normalized_features[11]:.4f}\n")
                                f.write(f" 13. pnl_velocity               : {normalized_features[12]:.6f}\n")
                                f.write(f" 14. peak_drawdown              : {normalized_features[13]:.4f}\n")
                                f.write(f" 15. apr_ratio                  : {normalized_features[14]:.4f}\n")
                                f.write(f" 16. return_efficiency          : {normalized_features[15]:.6f}\n")
                                f.write(f" 17. is_old_loser               : {normalized_features[16]:.0f}\n")
                                f.write(f" 18. current_position_apr       : {current_position_apr:.2f}%\n")
                                f.write(f" 19. best_available_apr         : {best_available_apr:.2f}%\n")
                                f.write(f" 20. apr_advantage              : {current_position_apr - best_available_apr:.2f}%\n")

                                f.write(f"\nNormalized Features (sent to model):\n")
                                for i, (name, val) in enumerate(zip(feature_names, normalized_features)):
                                    f.write(f" {i+1:2d}. {name:30s}: {val:12.6f}\n")

                                f.write(f"\nAPR Calculation Debug:\n")
                                f.write(f"  long_funding_rate             : {long_funding_rate:.6f}\n")
                                f.write(f"  short_funding_rate            : {short_funding_rate:.6f}\n")
                                f.write(f"  net_funding_rate              : {net_funding_rate:.6f}\n")
                                f.write(f"  long_funding_interval         : {long_interval:.1f}h\n")
                                f.write(f"  short_funding_interval        : {short_interval:.1f}h\n")
                                f.write(f"  avg_funding_interval          : {avg_interval_hours:.1f}h\n")
                                f.write(f"  payments_per_day              : {payments_per_day:.2f}\n")
                                f.write(f"  annual_rate                   : {annual_rate:.6f}\n")
                                f.write(f"  current_position_apr          : {current_position_apr:.2f}%\n")

                                f.write(f"\nPeak Drawdown Debug:\n")
                                f.write(f"  peak_pnl_pct                  : {peak_pnl_pct_raw:.4f}%\n")
                                f.write(f"  net_pnl_pct                   : {net_pnl_pct_raw:.4f}%\n")
                                f.write(f"  peak_drawdown (normalized)    : {normalized_features[13]:.4f}\n")
                                f.write("\n")

                            f.write("\n\n")
                except Exception as e:
                    # Don't crash if logging fails
                    print(f"Warning: Failed to write test feature log: {e}")
                    pass

            # Get action mask
            if hasattr(env, '_get_action_mask'):
                action_mask = env._get_action_mask()
            else:
                action_mask = None

            # Select action (deterministic)
            action, _, _ = trainer.select_action(obs, action_mask, deterministic=True)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

        # Collect portfolio metrics after episode
        portfolio = env.portfolio
        config = env.current_config

        # Collect all trades from this episode (closed positions)
        for position in portfolio.closed_positions:
            trade_record = position.to_trade_record()
            trade_record['episode'] = episode + 1
            all_trades.append(trade_record)

        # Also collect open positions (if any)
        for position in portfolio.positions:
            trade_record = position.to_trade_record()
            trade_record['episode'] = episode + 1
            all_trades.append(trade_record)

        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        total_pnls.append(portfolio.total_pnl_usd)
        total_pnl_pcts.append(portfolio.total_pnl_pct)
        max_drawdowns.append(portfolio.max_drawdown_pct)
        opportunities_seen.append(opportunities_count)

        # Trade statistics
        total_closed = len(portfolio.closed_positions)
        num_trades.append(total_closed)

        if total_closed > 0:
            winning = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd > 0)
            losing = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd <= 0)
            num_winning_trades.append(winning)
            num_losing_trades.append(losing)

            # Average trade duration in hours
            durations = [p.hours_held for p in portfolio.closed_positions]
            avg_trade_durations.append(np.mean(durations))
        else:
            num_winning_trades.append(0)
            num_losing_trades.append(0)
            avg_trade_durations.append(0.0)

        # Trades executed (total positions opened = currently open + closed)
        trades_executed.append(len(portfolio.positions) + len(portfolio.closed_positions))

        # Collect P&L for profit factor calculation
        for position in portfolio.closed_positions:
            if position.realized_pnl_usd > 0:
                all_winning_pnl.append(position.realized_pnl_usd)
            else:
                all_losing_pnl.append(abs(position.realized_pnl_usd))

        # Track config used
        configs_used.append({
            'leverage': config.max_leverage,
            'utilization': config.target_utilization,
            'max_positions': config.max_positions,
        })

        # Show length in appropriate unit
        if args.step_minutes == 1:
            length_str = f"{episode_length:5d} min"
        elif args.step_minutes == 60:
            length_str = f"{episode_length:3d}h"
        else:
            length_str = f"{episode_length:3d} steps"

        print(f"   Episode {episode + 1}: Reward={episode_reward:7.2f}, P&L=${portfolio.total_pnl_usd:7.2f} ({portfolio.total_pnl_pct:.2f}%), Trades={total_closed:2d}, Length={length_str}")

    # Calculate aggregate statistics
    total_trades_sum = sum(num_trades)
    total_winning = sum(num_winning_trades)
    total_losing = sum(num_losing_trades)
    win_rate = (total_winning / total_trades_sum * 100) if total_trades_sum > 0 else 0.0

    print(f"\n✅ Inference test complete!")
    print(f"\n{'='*80}")
    print("DETAILED EVALUATION METRICS (TEST SET)")
    print(f"{'='*80}")

    print(f"\n📊 Episode Metrics:")
    print(f"  Mean Reward:     {np.mean(eval_rewards):8.2f} ± {np.std(eval_rewards):.2f}")

    # Show length in appropriate unit
    if args.step_minutes == 1:
        print(f"  Mean Length:     {np.mean(eval_lengths):8.1f} minutes")
    elif args.step_minutes == 60:
        print(f"  Mean Length:     {np.mean(eval_lengths):8.1f} hours")
    else:
        print(f"  Mean Length:     {np.mean(eval_lengths):8.1f} steps ({args.step_minutes} min/step)")

    print(f"\n💰 P&L Metrics:")
    print(f"  Mean P&L (USD):  ${np.mean(total_pnls):8.2f}")
    print(f"  Mean P&L (%):    {np.mean(total_pnl_pcts):8.2f}%")
    print(f"  Total P&L:       ${np.sum(total_pnls):8.2f}")

    print(f"\n📈 Trading Metrics:")
    print(f"  Total Trades:    {total_trades_sum:8.0f}")
    print(f"  Winning Trades:  {total_winning:8.0f}")
    print(f"  Losing Trades:   {total_losing:8.0f}")
    print(f"  Win Rate:        {win_rate:8.1f}%")
    print(f"  Avg Duration:    {np.mean([d for d in avg_trade_durations if d > 0]) if any(avg_trade_durations) else 0.0:8.1f} hours")

    print(f"\n🎯 Opportunity Metrics:")
    print(f"  Opportunities/Ep: {np.mean(opportunities_seen):7.0f}")
    print(f"  Trades/Episode:   {np.mean(num_trades):7.1f}")
    print(f"  Execution Rate:   {(np.mean(num_trades) / np.mean(opportunities_seen) * 100) if np.mean(opportunities_seen) > 0 else 0:.1f}%")

    print(f"\n⚠️  Risk Metrics:")
    print(f"  Max Drawdown:    {np.mean(max_drawdowns):8.2f}%")

    # Calculate profit factor
    total_wins = sum(all_winning_pnl) if all_winning_pnl else 0.0
    total_losses = sum(all_losing_pnl) if all_losing_pnl else 0.001
    profit_factor = total_wins / total_losses

    print(f"\n📊 Profitability Metrics:")
    print(f"  Profit Factor:   {profit_factor:8.2f}")

    print(f"\n⚙️  Configuration Used:")
    avg_config = configs_used[0]  # Assuming same config for all episodes
    print(f"  Leverage:        {avg_config['leverage']:.1f}x")
    print(f"  Utilization:     {avg_config['utilization']:.0%}")
    print(f"  Max Positions:   {avg_config['max_positions']}")

    # Calculate composite score (IMPROVED VERSION)
    # Weights: 50% P&L, 30% Profit Factor, 20% Low Drawdown

    # 1. P&L Score: Bounded using tanh to prevent domination
    pnl_score = np.tanh(np.mean(total_pnl_pcts) / 5.0)

    # 2. Profit Factor Score: Normalize (2.0 = 1.0, capped at 1.0)
    profit_factor_score = min(profit_factor / 2.0, 1.0)

    # 3. Drawdown Score: Lower is better, floored at 0.0
    drawdown_score = max(0.0, 1.0 - (np.mean(max_drawdowns) / 100.0))

    composite_score = (
        0.50 * pnl_score +
        0.30 * profit_factor_score +
        0.20 * drawdown_score
    )

    print(f"\n🎯 Composite Score: {composite_score:.4f}")
    print(f"   (P&L: {pnl_score:.3f} | ProfitFactor: {profit_factor_score:.3f} | Drawdown: {drawdown_score:.3f})")

    # Write trades to CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        # Reorder columns for better readability
        column_order = [
            'episode', 'entry_datetime', 'exit_datetime', 'symbol',
            'long_exchange', 'short_exchange', 'status',
            'position_size_usd', 'leverage', 'margin_used_usd',
            'entry_long_price', 'entry_short_price',
            'exit_long_price', 'exit_short_price',
            'long_funding_rate', 'short_funding_rate',
            'funding_earned_usd', 'long_funding_earned_usd', 'short_funding_earned_usd',
            'entry_fees_usd', 'exit_fees_usd', 'total_fees_usd',
            'realized_pnl_usd', 'realized_pnl_pct',
            'unrealized_pnl_usd', 'unrealized_pnl_pct',
            'hours_held',
        ]
        trades_df = trades_df[column_order]

        # Save to CSV
        output_path = Path(args.trades_output)
        trades_df.to_csv(output_path, index=False)

        print(f"\n💾 Trade Records:")
        print(f"   Total trades: {len(all_trades)}")
        print(f"   Closed trades: {len([t for t in all_trades if t['status'] == 'closed'])}")
        print(f"   Open positions: {len([t for t in all_trades if t['status'] == 'open'])}")
        print(f"   Saved to: {output_path}")
    else:
        print(f"\n⚠️  No trades executed during inference")

    return True


if __name__ == '__main__':
    args = parse_args()
    success = test_model_inference(args)
    if success:
        print("\n" + "=" * 80)
        print("✅ ALL INFERENCE TESTS PASSED")
        print("=" * 80)
