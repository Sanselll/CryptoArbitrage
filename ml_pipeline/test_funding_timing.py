"""
Funding Timing Baseline Strategy Test

Simple timing-based strategy (no ML) that:
- ENTERS positions 2 minutes before funding payment (e.g., 07:58, 15:58, 23:58 UTC)
- EXITS positions 2 minutes after funding payment (e.g., 08:02, 16:02, 00:02 UTC)

This serves as a baseline to compare ML model performance against a naive funding-capture strategy.

Usage:
    python test_funding_timing.py --test-data-path data/rl_test.csv --leverage 2 --step-minutes 1
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Any, Optional, Tuple

from models.rl.core.environment import FundingArbitrageEnv
from models.rl.core.config import TradingConfig
from models.rl.core.reward_config import RewardConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Test funding timing baseline strategy')

    # Trading configuration
    parser.add_argument('--leverage', type=float, default=2.0,
                        help='Max leverage (default: 2.0x)')
    parser.add_argument('--utilization', type=float, default=0.8,
                        help='Capital utilization (default: 0.8 = 80%%)')
    parser.add_argument('--max-positions', type=int, default=2,
                        help='Max concurrent positions (default: 2)')

    # Timing configuration
    parser.add_argument('--enter-minutes-before', type=int, default=2,
                        help='Minutes before funding to enter (default: 2)')
    parser.add_argument('--exit-minutes-after', type=int, default=2,
                        help='Minutes after funding to exit (default: 2)')
    parser.add_argument('--min-apr', type=float, default=50.0,
                        help='Minimum APR to enter position (default: 50%% annualized)')

    # Test configuration
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to test (default: 1)')
    parser.add_argument('--full-test', action='store_true', default=True,
                        help='Use entire test dataset (default: True)')
    parser.add_argument('--no-full-test', action='store_false', dest='full_test',
                        help='Disable full test mode')
    parser.add_argument('--episode-length-days', type=int, default=5,
                        help='Episode length in days (only if --no-full-test)')
    parser.add_argument('--step-minutes', type=int, default=5,
                        help='Minutes per step (default: 5 to match data granularity)')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv',
                        help='Path to test data')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--trades-output', type=str, default='trades_funding_timing.csv',
                        help='Output CSV file for trade records')
    parser.add_argument('--price-history-path', type=str, default='data/symbol_data',
                        help='Path to price history directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--start-time', type=str, default=None,
                        help='Start time filter (e.g., "2025-12-12 07:35:00")')
    parser.add_argument('--end-time', type=str, default=None,
                        help='End time filter (e.g., "2025-12-12 10:00:00")')

    return parser.parse_args()


def is_near_funding_time(current_time: pd.Timestamp, funding_interval_hours: int = 8) -> Tuple[bool, bool]:
    """
    Check if current time is near a funding payment time.

    Funding schedules:
    - 8h: 00:00, 08:00, 16:00 UTC
    - 4h: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
    - 1h: Every hour at :00

    Args:
        current_time: Current timestamp
        funding_interval_hours: Funding interval (1, 4, or 8 hours)

    Returns:
        Tuple of (is_entry_window, is_exit_window)
    """
    hour = current_time.hour
    minute = current_time.minute

    # Determine funding hours based on interval
    if funding_interval_hours == 1:
        # Every hour is a funding hour
        is_funding_hour = True
        is_pre_funding = True  # Previous hour is always before funding
    elif funding_interval_hours == 4:
        funding_hours = [0, 4, 8, 12, 16, 20]
        is_funding_hour = hour in funding_hours
        pre_funding_hours = [(h - 1) % 24 for h in funding_hours]  # 23, 3, 7, 11, 15, 19
        is_pre_funding = hour in pre_funding_hours
    else:  # 8h default
        funding_hours = [0, 8, 16]
        is_funding_hour = hour in funding_hours
        pre_funding_hours = [23, 7, 15]
        is_pre_funding = hour in pre_funding_hours

    # Entry: 5 min before funding (minute == 55)
    is_entry_window = is_pre_funding and minute == 55

    # Exit: 5 min after funding (minute == 5)
    is_exit_window = is_funding_hour and minute == 5

    return is_entry_window, is_exit_window


def get_action_by_timing(
    current_time: pd.Timestamp,
    positions: List,
    opportunities: List[Dict],
    max_positions: int,
    enter_minutes_before: int = 2,
    exit_minutes_after: int = 2,
    entered_this_window: set = None
) -> Tuple[str, Optional[int], Optional[str]]:
    """
    Simple timing-based strategy - no ML.

    Enters positions 5 min before funding, exits 5 min after.
    Supports multiple funding intervals (1h, 4h, 8h).
    Selects best opportunity by fund_apr.

    Args:
        current_time: Current timestamp
        positions: List of active positions
        opportunities: List of available opportunities
        max_positions: Maximum number of positions allowed
        enter_minutes_before: Minutes before funding to enter (unused, kept for API)
        exit_minutes_after: Minutes after funding to exit (unused, kept for API)
        entered_this_window: Set of symbols already entered this window

    Returns:
        Tuple of (action_type, index, size)
    """
    if entered_this_window is None:
        entered_this_window = set()

    minute = current_time.minute

    # Count active positions
    num_positions = len([p for p in positions if hasattr(p, 'symbol')])

    # Check EXIT first - exit at :05 of any hour (covers all funding intervals)
    if minute == 5 and num_positions > 0:
        return 'EXIT', 0, None

    # Check ENTER - at :55 or :00 of any hour (covers all funding intervals)
    # :55 = 5 min before funding, :00 = at funding time
    # This allows entering 2 positions: one at :55, one at :00
    if minute in [55, 0] and num_positions < max_positions and opportunities:
        # Get symbols we already have positions in
        existing_symbols = set()
        for p in positions:
            if hasattr(p, 'symbol'):
                existing_symbols.add(p.symbol)

        # Filter opportunities:
        # 1. No existing position for this symbol (check our own tracking, not just has_existing_position)
        # 2. Not already entered this window
        # 3. Has positive APR (profitable)
        available_opps = [
            (i, opp) for i, opp in enumerate(opportunities)
            if opp.get('symbol') not in existing_symbols
            and opp.get('symbol') not in entered_this_window
            and opp.get('fund_apr', 0) > 0  # Only positive APR
        ]

        if available_opps:
            # Enter best opportunity by fund_apr
            # Use LARGE size (33% of available capital) to maximize capital usage
            best_idx, best_opp = max(available_opps, key=lambda x: x[1].get('fund_apr', 0))
            return 'ENTER', best_idx, 'LARGE'

    return 'HOLD', None, None


def action_to_env_action(action_type: str, index: Optional[int], size: Optional[str]) -> int:
    """
    Convert strategy action to environment action ID.

    Action space (36 total):
    - 0: HOLD
    - 1-10: ENTER SMALL (opportunity 0-9)
    - 11-20: ENTER MEDIUM (opportunity 0-9)
    - 21-30: ENTER LARGE (opportunity 0-9)
    - 31-35: EXIT (position 0-4)
    """
    if action_type == 'HOLD':
        return 0
    elif action_type == 'ENTER':
        if index is None:
            return 0
        size_offset = {'SMALL': 1, 'MEDIUM': 11, 'LARGE': 21}
        return size_offset.get(size, 11) + index
    elif action_type == 'EXIT':
        if index is None:
            return 0
        return 31 + index
    return 0


def execute_multiple_actions(env, actions: List[int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
    """
    Execute multiple actions at the same timestep.

    This allows entering/exiting multiple positions simultaneously
    without advancing time between actions.

    Args:
        env: FundingArbitrageEnv instance
        actions: List of action IDs to execute

    Returns:
        Final (observation, reward, terminated, truncated, info) after all actions
    """
    total_reward = 0.0
    info = {}

    for i, action in enumerate(actions):
        if i < len(actions) - 1:
            # For all but the last action, execute without advancing time
            # We do this by calling _execute_action directly
            reward, action_info = env._execute_action(action)
            total_reward += reward
            info.update(action_info)
        else:
            # Last action uses normal step() which advances time
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            info.update(step_info)

    return obs, total_reward, terminated, truncated, info


def test_funding_timing_strategy(args):
    """Run the funding timing baseline strategy test."""

    print("=" * 80)
    print("FUNDING TIMING BASELINE STRATEGY TEST")
    print("=" * 80)

    # Set random seed
    np.random.seed(args.seed)

    print(f"\n1. Configuration:")
    print(f"   Strategy: Enter {args.enter_minutes_before} min before funding, "
          f"Exit {args.exit_minutes_after} min after")
    print(f"   Funding times: 00:00, 08:00, 16:00 UTC")
    print(f"   Entry windows: 23:58, 07:58, 15:58 UTC")
    print(f"   Exit windows: 00:01-00:02, 08:01-08:02, 16:01-16:02 UTC")

    # Create trading config
    trading_config = TradingConfig(
        max_leverage=args.leverage,
        target_utilization=args.utilization,
        max_positions=args.max_positions,
        stop_loss_threshold=-0.02,
        liquidation_buffer=0.15,
    )

    # Create reward config
    reward_config = RewardConfig()

    # Calculate step size
    step_hours = args.step_minutes / 60.0

    # Filter data by time range if specified
    data_path_to_use = args.test_data_path
    if args.start_time or args.end_time:
        print(f"\n⏱️  Filtering test data by time range...")
        test_df = pd.read_csv(args.test_data_path)
        test_df['entry_time'] = pd.to_datetime(test_df['entry_time'])
        print(f"   Original: {len(test_df)} rows ({test_df['entry_time'].min()} to {test_df['entry_time'].max()})")

        if args.start_time:
            start_ts = pd.Timestamp(args.start_time, tz='UTC')
            test_df = test_df[test_df['entry_time'] >= start_ts]
            print(f"   Filtered start >= {start_ts}: {len(test_df)} rows")

        if args.end_time:
            end_ts = pd.Timestamp(args.end_time, tz='UTC')
            test_df = test_df[test_df['entry_time'] <= end_ts]
            print(f"   Filtered end <= {end_ts}: {len(test_df)} rows")

        print(f"   Final: {len(test_df)} rows ({test_df['entry_time'].min()} to {test_df['entry_time'].max()})")

        # Save filtered data to temp file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        test_df.to_csv(temp_file.name, index=False)
        data_path_to_use = temp_file.name
        print(f"   Saved filtered data to: {data_path_to_use}")

    # Create environment
    print(f"\n2. Creating environment...")
    env = FundingArbitrageEnv(
        data_path=data_path_to_use,
        trading_config=trading_config,
        reward_config=reward_config,
        initial_capital=args.initial_capital,
        price_history_path=args.price_history_path if args.price_history_path else None,
        step_hours=step_hours,
        use_full_range_episodes=args.full_test,
        episode_length_days=args.episode_length_days if not args.full_test else 5,
    )
    print(f"✅ Environment created")

    # Calculate data range
    if args.full_test:
        test_df = pd.read_csv(args.test_data_path)
        test_df['entry_time'] = pd.to_datetime(test_df['entry_time'])
        data_start = test_df['entry_time'].min()
        data_end = test_df['entry_time'].max()
        data_days = (data_end - data_start).total_seconds() / 86400
        total_steps = int((data_days * 24 * 60) / args.step_minutes)

        print(f"\n   FULL TEST MODE: {data_days:.1f} days ({data_start} to {data_end})")
        print(f"   Total steps: ~{total_steps:,} at {args.step_minutes}-minute intervals")
    else:
        print(f"   Episode length: {args.episode_length_days} days")

    print(f"   Initial capital: ${args.initial_capital:,.2f}")
    print(f"   Leverage: {args.leverage}x")
    print(f"   Max Positions: {args.max_positions}")

    # Tracking variables
    all_trades = []
    all_funding_details = []
    action_counts = {'HOLD': 0, 'ENTER': 0, 'EXIT': 0}

    print(f"\n3. Running strategy...")

    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed)
        done = False
        step_count = 0
        episode_enters = 0
        episode_exits = 0

        while not done:
            # Get current state
            current_time = env.current_time
            opportunities = env.current_opportunities

            # Collect ALL actions for this timestep before executing
            actions_to_execute = []
            entered_symbols = set()

            # Get symbols we already have positions in
            existing_symbols = set()
            for p in env.portfolio.positions:
                if hasattr(p, 'symbol'):
                    existing_symbols.add(p.symbol)

            minute = current_time.minute
            num_positions = len(existing_symbols)

            # Debug: show position tracking (disabled for cleaner output)
            # if step_count < 500 and num_positions > 0:
            #     print(f"   [Step {step_count}] Time={current_time.strftime('%H:%M')}, min={minute}, pos={num_positions}")

            # Check EXIT - at :05, exit ALL positions
            if minute == 5 and num_positions > 0:
                # Add exit action for each position
                # Always exit position 0 because after each exit, positions shift down
                for i in range(num_positions):
                    actions_to_execute.append(('EXIT', 0, None))

            # Check ENTER - at :55, enter up to max_positions
            elif minute == 55:
                positions_to_enter = args.max_positions - num_positions

                # Debug: log opportunities at entry windows (disabled for cleaner output)
                # if step_count < 500 or step_count % 5000 == 0:
                #     print(f"   DEBUG [{current_time}]: {len(opportunities)} opps, {num_positions} pos, slots={positions_to_enter}")

                # Get best opportunities by ACTUAL expected funding in next hour
                # Algorithm:
                # 1. Check which legs have funding at the next :00 (within 5-10 min)
                # 2. Calculate expected funding ONLY from those legs
                # 3. Sort by expected funding profit (not APR)
                available_opps = []
                for i, opp in enumerate(opportunities):
                    if opp.get('symbol') in existing_symbols:
                        continue

                    # Get funding rates
                    # Long: negative rate = we receive (profit), positive rate = we pay (cost)
                    # Short: positive rate = we receive (profit), negative rate = we pay (cost)
                    long_rate = opp.get('long_funding_rate', 0)
                    short_rate = opp.get('short_funding_rate', 0)

                    # Check if funding happens within our hold window
                    # Entry at :55, exit at :05 of next hour
                    # We capture funding that happens at :00 (between entry and exit)
                    long_next_funding = pd.to_datetime(opp.get('long_next_funding_time'))
                    short_next_funding = pd.to_datetime(opp.get('short_next_funding_time'))

                    # Time to next funding for each leg
                    time_to_long_funding = (long_next_funding - current_time).total_seconds() / 60  # minutes
                    time_to_short_funding = (short_next_funding - current_time).total_seconds() / 60  # minutes

                    # Funding is captured if it happens between now and exit (10 min)
                    # We enter at :55, funding at :00 (5 min), exit at :05 (10 min)
                    long_funding_captured = 0 < time_to_long_funding <= 10
                    short_funding_captured = 0 < time_to_short_funding <= 10

                    # Calculate expected funding from legs that will be captured
                    expected_funding_pct = 0
                    if long_funding_captured:
                        # Long position: we RECEIVE when rate is negative, PAY when positive
                        # So profit = -rate (negative rate becomes positive profit)
                        expected_funding_pct += -long_rate
                    if short_funding_captured:
                        # Short position: we RECEIVE when rate is positive, PAY when negative
                        # So profit = +rate (positive rate becomes positive profit)
                        expected_funding_pct += short_rate

                    # Only enter if we capture at least one funding AND it's profitable
                    if (long_funding_captured or short_funding_captured) and expected_funding_pct > 0:
                        available_opps.append((i, opp, expected_funding_pct))

                # Debug: show available opportunities count (disabled for cleaner output)
                # if step_count < 500 or step_count % 5000 == 0:
                #     print(f"         Available opps meeting criteria: {len(available_opps)}")

                # Sort by expected funding profit (descending)
                available_opps.sort(key=lambda x: x[2], reverse=True)

                # Take top N opportunities with highest expected funding
                for i, (opp_idx, opp, exp_funding_pct) in enumerate(available_opps[:positions_to_enter]):
                    symbol = opp.get('symbol')
                    if symbol not in entered_symbols:
                        actions_to_execute.append(('ENTER', opp_idx, 'LARGE'))
                        entered_symbols.add(symbol)
                        # Debug (disabled for cleaner output)
                        # if step_count < 500:
                        #     print(f"         -> ENTER {symbol} at idx {opp_idx}, ExpFund={exp_funding_pct*100:.4f}%")

            # Execute all collected actions
            if actions_to_execute:
                # Convert to env action IDs
                action_ids = [action_to_env_action(a[0], a[1], a[2]) for a in actions_to_execute]

                # Debug (disabled for cleaner output)
                # if step_count < 500:
                #     print(f"         Executing actions: {action_ids}")

                # Track actions
                for action_type, index, size in actions_to_execute:
                    action_counts[action_type] += 1
                    if action_type == 'ENTER':
                        episode_enters += 1
                    elif action_type == 'EXIT':
                        episode_exits += 1

                # Execute all actions at same timestep
                obs, reward, terminated, truncated, info = execute_multiple_actions(env, action_ids)
                done = terminated or truncated

                # Debug (disabled for cleaner output)
                # if step_count < 500:
                #     print(f"         After exec: {len(env.portfolio.positions)} positions, reward={reward:.4f}")
            else:
                # No actions - just HOLD and advance time
                obs, reward, terminated, truncated, info = env.step(0)
                done = terminated or truncated
                action_counts['HOLD'] += 1

            step_count += 1

            # Progress indicator
            if step_count % 1000 == 0:
                portfolio = env.portfolio
                print(f"   Step {step_count}: Time={current_time}, "
                      f"Positions={len(portfolio.positions)}, "
                      f"P&L=${portfolio.total_pnl_usd:.2f}")

        # Collect trades from episode
        portfolio = env.portfolio
        for position in portfolio.closed_positions:
            trade_record = position.to_trade_record()
            trade_record['episode'] = episode + 1
            all_trades.append(trade_record)

            funding_summary = position.get_funding_summary()
            funding_summary['episode'] = episode + 1
            all_funding_details.append(funding_summary)

        # Also collect open positions
        for position in portfolio.positions:
            trade_record = position.to_trade_record()
            trade_record['episode'] = episode + 1
            all_trades.append(trade_record)

        print(f"\n   Episode {episode + 1} Complete:")
        print(f"   Steps: {step_count}")
        print(f"   Enters: {episode_enters}, Exits: {episode_exits}")
        print(f"   Total P&L: ${portfolio.total_pnl_usd:.2f} ({portfolio.total_pnl_pct:.2f}%)")

    # Final results
    print("\n" + "=" * 80)
    print("FUNDING TIMING STRATEGY RESULTS")
    print("=" * 80)

    portfolio = env.portfolio

    # Calculate metrics
    total_pnl_usd = portfolio.total_pnl_usd
    total_pnl_pct = portfolio.total_pnl_pct
    num_closed = len(portfolio.closed_positions)

    winning_trades = [p for p in portfolio.closed_positions if p.realized_pnl_usd > 0]
    losing_trades = [p for p in portfolio.closed_positions if p.realized_pnl_usd <= 0]
    win_rate = len(winning_trades) / num_closed * 100 if num_closed > 0 else 0

    avg_duration = np.mean([p.hours_held for p in portfolio.closed_positions]) if num_closed > 0 else 0

    # Funding breakdown
    total_long_funding = sum(p.long_net_funding_usd for p in portfolio.closed_positions)
    total_short_funding = sum(p.short_net_funding_usd for p in portfolio.closed_positions)
    total_funding = total_long_funding + total_short_funding

    # Fees breakdown
    total_entry_fees = sum(p.entry_fees_paid_usd for p in portfolio.closed_positions)
    total_exit_fees = sum(p.exit_fees_paid_usd for p in portfolio.closed_positions)
    total_fees = total_entry_fees + total_exit_fees

    # Price P&L calculation (for closed positions)
    total_price_pnl = sum(p.realized_pnl_usd + p.entry_fees_paid_usd + p.exit_fees_paid_usd - p.long_net_funding_usd - p.short_net_funding_usd
                          for p in portfolio.closed_positions)

    # Funding payment counts
    total_long_payments = sum(p.long_funding_payment_count for p in portfolio.closed_positions)
    total_short_payments = sum(p.short_funding_payment_count for p in portfolio.closed_positions)

    print(f"\n📊 Performance Summary:")
    print(f"   Total P&L: ${total_pnl_usd:.2f} ({total_pnl_pct:.2f}%)")
    print(f"   Total Trades: {num_closed}")
    print(f"   Win Rate: {win_rate:.1f}% ({len(winning_trades)}W / {len(losing_trades)}L)")
    print(f"   Avg Trade Duration: {avg_duration:.2f}h ({avg_duration*60:.0f} min)")

    print(f"\n💵 P&L Breakdown:")
    print(f"   Funding (long):  ${total_long_funding:+.2f} ({total_long_payments} payments)")
    print(f"   Funding (short): ${total_short_funding:+.2f} ({total_short_payments} payments)")
    print(f"   Funding (total): ${total_funding:+.2f}")
    print(f"   Entry Fees:      ${-total_entry_fees:.2f}")
    print(f"   Exit Fees:       ${-total_exit_fees:.2f}")
    print(f"   Total Fees:      ${-total_fees:.2f}")
    print(f"   Price P&L:       ${total_price_pnl:+.2f}")
    print(f"   Net P&L:         ${total_pnl_usd:+.2f}")

    print(f"\n📈 Action Distribution:")
    print(f"   HOLD: {action_counts['HOLD']:,}")
    print(f"   ENTER: {action_counts['ENTER']:,}")
    print(f"   EXIT: {action_counts['EXIT']:,}")

    # Save trades to CSV
    if all_trades:
        trades_df = pd.DataFrame(all_trades)
        trades_df.to_csv(args.trades_output, index=False)
        print(f"\n💾 Trade details saved to: {args.trades_output}")

    # Show top trades by funding
    if all_funding_details:
        funding_df = pd.DataFrame(all_funding_details)

        print(f"\n💰 Funding Details:")
        print(f"   Total positions: {len(all_funding_details)}")
        print(f"   Total funding earned: ${funding_df['net_funding_usd'].sum():.2f}")
        print(f"   Avg funding per position: ${funding_df['net_funding_usd'].mean():.2f}")

        # Count trades with zero funding
        zero_funding_trades = len(funding_df[funding_df['net_funding_usd'] == 0])
        neg_funding_trades = len(funding_df[funding_df['net_funding_usd'] < 0])
        pos_funding_trades = len(funding_df[funding_df['net_funding_usd'] > 0])
        print(f"   Positive funding: {pos_funding_trades} trades")
        print(f"   Zero funding: {zero_funding_trades} trades")
        print(f"   Negative funding: {neg_funding_trades} trades")

        # Top 5 by funding
        top_funding = funding_df.nlargest(5, 'net_funding_usd')[
            ['symbol', 'hours_held', 'net_funding_usd', 'net_funding_pct', 'realized_pnl_usd', 'entry_time', 'exit_time']
        ]
        print(f"\n   Top 5 by Funding Earned:")
        for idx, row in top_funding.iterrows():
            entry_str = row['entry_time'].strftime('%m-%d %H:%M') if pd.notna(row['entry_time']) else 'N/A'
            exit_str = row['exit_time'].strftime('%m-%d %H:%M') if pd.notna(row['exit_time']) else 'N/A'
            print(f"     {row['symbol']:12s}: ${row['net_funding_usd']:6.2f} ({row['net_funding_pct']:.2f}%) "
                  f"over {row['hours_held']:.1f}h, P&L: ${row['realized_pnl_usd']:6.2f}  "
                  f"[{entry_str} → {exit_str}]")

        # Show worst trades (zero or negative funding) for debugging
        worst_funding = funding_df.nsmallest(5, 'net_funding_usd')[
            ['symbol', 'hours_held', 'net_funding_usd', 'net_funding_pct', 'realized_pnl_usd', 'entry_time', 'exit_time']
        ]
        print(f"\n   Bottom 5 (debugging - zero/negative funding):")
        for idx, row in worst_funding.iterrows():
            entry_str = row['entry_time'].strftime('%m-%d %H:%M') if pd.notna(row['entry_time']) else 'N/A'
            exit_str = row['exit_time'].strftime('%m-%d %H:%M') if pd.notna(row['exit_time']) else 'N/A'
            print(f"     {row['symbol']:12s}: ${row['net_funding_usd']:6.2f} ({row['net_funding_pct']:.2f}%) "
                  f"over {row['hours_held']:.1f}h, P&L: ${row['realized_pnl_usd']:6.2f}  "
                  f"[{entry_str} → {exit_str}]")

    return True


if __name__ == '__main__':
    args = parse_args()
    success = test_funding_timing_strategy(args)
    if success:
        print("\n" + "=" * 80)
        print("✅ FUNDING TIMING TEST COMPLETE")
        print("=" * 80)
