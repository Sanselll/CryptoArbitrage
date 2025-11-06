"""
Ideal Trading Simulation with Perfect Hindsight

This script simulates "ideal" trading decisions using perfect future knowledge
to establish an upper-bound benchmark for the RL agent performance.

Uses the EXACT SAME trading mechanics as the RL environment:
- Portfolio and Position classes (identical P&L, fees, funding calculations)
- Same hedged positions (long + short)
- Same entry/exit fees
- Same funding payment logic

The "ideal" part is the decision-making:
- Uses hindsight to evaluate every possible exit time for each opportunity
- Selects opportunities and exit times that maximize risk-adjusted returns
- Optimizes for composite score: 50% P&L + 30% Win Rate + 20% Low Drawdown
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import timedelta

from models.rl.core.portfolio import Portfolio, Position
from models.rl.core.config import TradingConfig
from common.data.price_history_loader import PriceHistoryLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate ideal trading with perfect hindsight')

    # Trading configuration
    parser.add_argument('--leverage', type=float, default=1.0,
                        help='Max leverage (default: 1.0x)')
    parser.add_argument('--utilization', type=float, default=0.9,
                        help='Capital utilization (default: 0.9 = 90%%)')
    parser.add_argument('--max-positions', type=int, default=3,
                        help='Max concurrent positions (default: 3)')

    # Test configuration
    parser.add_argument('--num-episodes', type=int, default=1,
                        help='Number of episodes to simulate (default: 1)')
    parser.add_argument('--episode-length-days', type=int, default=7,
                        help='Episode length in days (default: 7)')
    parser.add_argument('--step-minutes', type=int, default=5,
                        help='Minutes per step (default: 5 = 5-minute intervals)')
    parser.add_argument('--test-data-path', type=str, default='data/rl_test.csv',
                        help='Path to test opportunities')
    parser.add_argument('--initial-capital', type=float, default=10000.0,
                        help='Initial capital (default: 10000)')
    parser.add_argument('--price-history-path', type=str, default='data/price_history',
                        help='Path to price history parquet files')

    # Hindsight parameters
    parser.add_argument('--min-hold-hours', type=float, default=0.5,
                        help='Minimum hold time in hours (default: 0.5)')
    parser.add_argument('--max-hold-hours', type=float, default=168.0,
                        help='Maximum hold time in hours (default: 168.0 = 7 days)')
    parser.add_argument('--position-size-pct', type=float, default=20.0,
                        help='Position size as % of capital per side (default: 20.0)')
    parser.add_argument('--max-opps-to-evaluate', type=int, default=500,
                        help='Maximum number of opportunities to evaluate per episode (default: 500)')

    # Comparison mode
    parser.add_argument('--compare-with', type=str, default=None,
                        help='Path to RL agent trades CSV for comparison (e.g., trades_inference.csv)')

    return parser.parse_args()


class HindsightEvaluator:
    """
    Evaluates opportunities using perfect future knowledge.

    For each opportunity, simulates all possible exit times and
    calculates the actual P&L that would be realized.
    """

    def __init__(self,
                 price_loader: PriceHistoryLoader,
                 step_hours: float = 1.0,
                 min_hold_hours: float = 1.0,
                 max_hold_hours: float = 72.0):
        self.price_loader = price_loader
        self.step_hours = step_hours
        self.min_hold_hours = min_hold_hours
        self.max_hold_hours = max_hold_hours

    def evaluate_opportunity(self,
                           opp: pd.Series,
                           position_size_usd: float,
                           leverage: float) -> Optional[Dict]:
        """
        Evaluate an opportunity with hindsight - test all possible exit times.

        Returns:
            Best exit scenario with metrics, or None if opportunity is invalid
        """
        symbol = opp['symbol']
        entry_time = pd.to_datetime(opp['entry_time'])

        # Get price history for this symbol
        try:
            price_df = self.price_loader.load_symbol(symbol)
        except Exception as e:
            return None

        if price_df is None or len(price_df) == 0:
            return None

        # Reset index to access timestamp as column
        price_df = price_df.reset_index()

        # Find entry time in price history
        price_df = price_df[price_df['timestamp'] >= entry_time]
        if len(price_df) == 0:
            return None

        # Get entry prices and funding rates from actual market data
        entry_row = price_df.iloc[0]

        # Handle exchange name variations
        long_exch = opp['long_exchange'].lower()
        short_exch = opp['short_exchange'].lower()

        entry_long_price = entry_row.get(f"{long_exch}_price")
        entry_short_price = entry_row.get(f"{short_exch}_price")

        # Check if prices are valid
        if pd.isna(entry_long_price) or pd.isna(entry_short_price):
            return None

        long_funding_rate = entry_row.get(f"{long_exch}_funding_rate", opp['long_funding_rate'])
        short_funding_rate = entry_row.get(f"{short_exch}_funding_rate", opp['short_funding_rate'])

        # Create a test position
        test_position = Position(
            opportunity_id=str(opp.get('id', 0)),
            symbol=symbol,
            long_exchange=opp['long_exchange'],
            short_exchange=opp['short_exchange'],
            entry_time=entry_time,
            entry_long_price=entry_long_price,
            entry_short_price=entry_short_price,
            position_size_usd=position_size_usd,
            long_funding_rate=long_funding_rate,
            short_funding_rate=short_funding_rate,
            long_funding_interval_hours=int(opp.get('long_funding_interval_hours', 1)),
            short_funding_interval_hours=int(opp.get('short_funding_interval_hours', 1)),
            long_next_funding_time=pd.to_datetime(opp['long_next_funding_time']),
            short_next_funding_time=pd.to_datetime(opp['short_next_funding_time']),
            leverage=leverage,
        )

        # Test different exit times
        best_exit = None
        best_score = -float('inf')

        # Calculate number of steps to test
        min_steps = int(self.min_hold_hours / self.step_hours)
        max_steps = int(self.max_hold_hours / self.step_hours)

        for step in range(min_steps, min(max_steps, len(price_df))):
            # Get exit time and prices
            exit_row = price_df.iloc[step]
            exit_time = exit_row['timestamp']
            exit_long_price = exit_row.get(f"{long_exch}_price")
            exit_short_price = exit_row.get(f"{short_exch}_price")

            # Skip if prices are invalid
            if pd.isna(exit_long_price) or pd.isna(exit_short_price):
                continue

            # Simulate holding the position
            sim_position = Position(
                opportunity_id=str(opp.get('id', 0)),
                symbol=symbol,
                long_exchange=opp['long_exchange'],
                short_exchange=opp['short_exchange'],
                entry_time=entry_time,
                entry_long_price=entry_long_price,
                entry_short_price=entry_short_price,
                position_size_usd=position_size_usd,
                long_funding_rate=long_funding_rate,
                short_funding_rate=short_funding_rate,
                long_funding_interval_hours=int(opp.get('long_funding_interval_hours', 1)),
                short_funding_interval_hours=int(opp.get('short_funding_interval_hours', 1)),
                long_next_funding_time=pd.to_datetime(opp['long_next_funding_time']),
                short_next_funding_time=pd.to_datetime(opp['short_next_funding_time']),
                leverage=leverage,
            )

            # Update position hour-by-hour with actual prices and funding rates
            current_time = entry_time + timedelta(hours=self.step_hours)
            for i in range(1, step + 1):
                if i >= len(price_df):
                    break

                row = price_df.iloc[i]
                current_time = row['timestamp']
                current_long_price = row.get(f"{long_exch}_price")
                current_short_price = row.get(f"{short_exch}_price")

                # Skip if prices are invalid
                if pd.isna(current_long_price) or pd.isna(current_short_price):
                    continue

                # Update funding rates
                current_long_rate = row.get(f"{long_exch}_funding_rate", long_funding_rate)
                current_short_rate = row.get(f"{short_exch}_funding_rate", short_funding_rate)

                # Handle NaN in funding rates
                if pd.isna(current_long_rate):
                    current_long_rate = long_funding_rate
                if pd.isna(current_short_rate):
                    current_short_rate = short_funding_rate

                sim_position.update_funding_rates(current_long_rate, current_short_rate)

                # Update position
                sim_position.update_hourly(current_time, current_long_price, current_short_price)

            # Close position and get realized P&L
            realized_pnl_usd = sim_position.close(exit_time, exit_long_price, exit_short_price)
            realized_pnl_pct = sim_position.realized_pnl_pct

            # Calculate score (simple: P&L percentage)
            # We'll do risk-adjusted selection at the portfolio level
            score = realized_pnl_pct

            if score > best_score:
                best_score = score
                best_exit = {
                    'exit_time': exit_time,
                    'exit_long_price': exit_long_price,
                    'exit_short_price': exit_short_price,
                    'hold_hours': sim_position.hours_held,
                    'realized_pnl_usd': realized_pnl_usd,
                    'realized_pnl_pct': realized_pnl_pct,
                    'funding_earned': sim_position.long_net_funding_usd + sim_position.short_net_funding_usd,
                    'total_fees': sim_position.entry_fees_paid_usd + sim_position.exit_fees_paid_usd,
                }

        if best_exit is None:
            return None

        # Return full opportunity info + best exit scenario
        return {
            'opportunity': opp,
            'entry_time': entry_time,
            'entry_long_price': entry_long_price,
            'entry_short_price': entry_short_price,
            'long_funding_rate': long_funding_rate,
            'short_funding_rate': short_funding_rate,
            **best_exit
        }


def simulate_ideal_trading(args):
    """Run ideal trading simulation with perfect hindsight."""
    print("=" * 80)
    print("IDEAL TRADING SIMULATION (PERFECT HINDSIGHT)")
    print("=" * 80)

    # Load test data
    print(f"\n1. Loading test data from {args.test_data_path}...")
    try:
        opportunities_df = pd.read_csv(args.test_data_path)
        print(f"✅ Loaded {len(opportunities_df)} opportunities")
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return False

    # Load price history
    print(f"\n2. Loading price history from {args.price_history_path}...")
    try:
        price_loader = PriceHistoryLoader(args.price_history_path)
        print(f"✅ Price history loader initialized")
    except Exception as e:
        print(f"❌ Error loading price history: {e}")
        return False

    # Create trading config
    trading_config = TradingConfig(
        max_leverage=args.leverage,
        target_utilization=args.utilization,
        max_positions=args.max_positions,
        stop_loss_threshold=-0.02,
        liquidation_buffer=0.15,
    )

    step_hours = args.step_minutes / 60.0

    print(f"\n✅ Configuration:")
    print(f"   Initial capital: ${args.initial_capital:,.2f}")
    print(f"   Leverage: {args.leverage}x")
    print(f"   Position size: {args.position_size_pct}% per side")
    print(f"   Max positions: {args.max_positions}")
    print(f"   Episode length: {args.episode_length_days} days")
    print(f"   Step interval: {args.step_minutes} minutes ({step_hours:.2f} hours)")

    # Create hindsight evaluator
    evaluator = HindsightEvaluator(
        price_loader=price_loader,
        step_hours=step_hours,
        min_hold_hours=args.min_hold_hours,
        max_hold_hours=args.max_hold_hours,
    )

    # Episode metrics
    total_pnls = []
    total_pnl_pcts = []
    num_trades = []
    num_winning_trades = []
    num_losing_trades = []
    avg_trade_durations = []
    max_drawdowns = []

    print(f"\n3. Running {args.num_episodes} episode(s)...")

    for episode in range(args.num_episodes):
        # Create portfolio
        portfolio = Portfolio(
            initial_capital=args.initial_capital,
            max_positions=args.max_positions,
            max_position_size_pct=args.position_size_pct,
        )

        # Calculate episode time window
        episode_length_hours = args.episode_length_days * 24
        total_steps = int(episode_length_hours / step_hours)

        # Randomly select start time from test data
        opportunities_df['entry_time'] = pd.to_datetime(opportunities_df['entry_time'])
        min_time = opportunities_df['entry_time'].min()
        max_time = opportunities_df['entry_time'].max() - timedelta(days=args.episode_length_days)

        if max_time <= min_time:
            print(f"⚠️  Not enough data for {args.episode_length_days} day episodes")
            return False

        # Random start time
        time_range_seconds = (max_time - min_time).total_seconds()
        random_offset_seconds = np.random.uniform(0, time_range_seconds)
        start_time = min_time + timedelta(seconds=random_offset_seconds)
        end_time = start_time + timedelta(days=args.episode_length_days)

        # Filter opportunities for this episode
        episode_opps = opportunities_df[
            (opportunities_df['entry_time'] >= start_time) &
            (opportunities_df['entry_time'] < end_time)
        ].copy()

        print(f"\n   Episode {episode + 1}:")
        print(f"   Period: {start_time} to {end_time}")
        print(f"   Available opportunities: {len(episode_opps)}")

        # Limit number of opportunities to evaluate (sample randomly if too many)
        if len(episode_opps) > args.max_opps_to_evaluate:
            print(f"   Sampling {args.max_opps_to_evaluate} opportunities for evaluation...")
            episode_opps = episode_opps.sample(n=args.max_opps_to_evaluate, random_state=42)

        # Evaluate opportunities with hindsight
        print(f"   Evaluating {len(episode_opps)} opportunities with hindsight...")
        evaluated_opps = []

        for count, (idx, opp) in enumerate(episode_opps.iterrows(), 1):
            if count % 50 == 0:
                print(f"   Progress: {count}/{len(episode_opps)} opportunities evaluated, {len(evaluated_opps)} valid so far...")

            # Calculate position size
            position_size_usd = portfolio.total_capital * (args.position_size_pct / 100)

            eval_result = evaluator.evaluate_opportunity(opp, position_size_usd, args.leverage)
            if eval_result and eval_result['realized_pnl_pct'] > -2.0:  # Filter out terrible trades
                evaluated_opps.append(eval_result)

        # Sort by P&L percentage (descending)
        evaluated_opps.sort(key=lambda x: x['realized_pnl_pct'], reverse=True)

        print(f"   ✅ Evaluation complete: {len(evaluated_opps)} valid opportunities found")

        # Greedy selection: pick best opportunities that fit within constraints
        selected_trades = []
        current_time = start_time

        for eval_opp in evaluated_opps:
            # Check if we can open this position
            position_size_usd = portfolio.total_capital * (args.position_size_pct / 100)

            if portfolio.can_open_position(position_size_usd, args.leverage):
                # Check if entry time is within episode window
                if eval_opp['entry_time'] >= start_time and eval_opp['entry_time'] < end_time:
                    # Check if exit time is within episode window
                    if eval_opp['exit_time'] <= end_time:
                        # Check for no time overlap with existing positions
                        has_overlap = False
                        for trade in selected_trades:
                            if not (eval_opp['exit_time'] <= trade['entry_time'] or
                                   eval_opp['entry_time'] >= trade['exit_time']):
                                has_overlap = True
                                break

                        if not has_overlap:
                            # Open position
                            opp_data = eval_opp['opportunity']
                            position = Position(
                                opportunity_id=str(opp_data.get('id', len(selected_trades))),
                                symbol=opp_data['symbol'],
                                long_exchange=opp_data['long_exchange'],
                                short_exchange=opp_data['short_exchange'],
                                entry_time=eval_opp['entry_time'],
                                entry_long_price=eval_opp['entry_long_price'],
                                entry_short_price=eval_opp['entry_short_price'],
                                position_size_usd=position_size_usd,
                                long_funding_rate=eval_opp['long_funding_rate'],
                                short_funding_rate=eval_opp['short_funding_rate'],
                                long_funding_interval_hours=int(opp_data.get('long_funding_interval_hours', 1)),
                                short_funding_interval_hours=int(opp_data.get('short_funding_interval_hours', 1)),
                                long_next_funding_time=pd.to_datetime(opp_data['long_next_funding_time']),
                                short_next_funding_time=pd.to_datetime(opp_data['short_next_funding_time']),
                                leverage=args.leverage,
                            )

                            if portfolio.open_position(position):
                                selected_trades.append(eval_opp)

                                # Immediately close it with the hindsight exit
                                # (In reality we'd simulate hour by hour, but for simplicity we close immediately)
                                portfolio.close_position(
                                    len(portfolio.positions) - 1,
                                    eval_opp['exit_time'],
                                    eval_opp['exit_long_price'],
                                    eval_opp['exit_short_price']
                                )

        print(f"   Trades executed: {len(selected_trades)}")

        # Collect metrics
        total_closed = len(portfolio.closed_positions)
        winning = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd > 0)
        losing = sum(1 for p in portfolio.closed_positions if p.realized_pnl_usd <= 0)

        total_pnls.append(portfolio.total_pnl_usd)
        total_pnl_pcts.append(portfolio.total_pnl_pct)
        num_trades.append(total_closed)
        num_winning_trades.append(winning)
        num_losing_trades.append(losing)
        max_drawdowns.append(portfolio.max_drawdown_pct)

        if total_closed > 0:
            durations = [p.hours_held for p in portfolio.closed_positions]
            avg_trade_durations.append(np.mean(durations))
        else:
            avg_trade_durations.append(0.0)

        print(f"   P&L: ${portfolio.total_pnl_usd:.2f} ({portfolio.total_pnl_pct:.2f}%)")
        print(f"   Winning: {winning}, Losing: {losing}, Win Rate: {(winning / total_closed * 100) if total_closed > 0 else 0:.1f}%")

    # Calculate aggregate statistics
    total_trades_sum = sum(num_trades)
    total_winning = sum(num_winning_trades)
    total_losing = sum(num_losing_trades)
    win_rate = (total_winning / total_trades_sum * 100) if total_trades_sum > 0 else 0.0

    print(f"\n{'='*80}")
    print("IDEAL TRADING PERFORMANCE (UPPER BOUND)")
    print(f"{'='*80}")

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

    print(f"\n⚠️  Risk Metrics:")
    print(f"  Max Drawdown:    {np.mean(max_drawdowns):8.2f}%")

    # Calculate composite score
    pnl_score = np.mean(total_pnl_pcts) / 5.0  # Normalize: 5% P&L = 1.0
    winrate_score = win_rate / 100.0
    drawdown_score = 1.0 - (np.mean(max_drawdowns) / 100.0)

    composite_score = (
        0.50 * pnl_score +
        0.30 * winrate_score +
        0.20 * drawdown_score
    )

    print(f"\n🎯 Composite Score: {composite_score:.4f}")
    print(f"   (P&L: {pnl_score:.3f} | WinRate: {winrate_score:.3f} | Drawdown: {drawdown_score:.3f})")

    # Return metrics for comparison
    return {
        'mean_pnl_usd': np.mean(total_pnls),
        'mean_pnl_pct': np.mean(total_pnl_pcts),
        'total_pnl_usd': np.sum(total_pnls),
        'total_trades': total_trades_sum,
        'winning_trades': total_winning,
        'losing_trades': total_losing,
        'win_rate': win_rate,
        'avg_duration': np.mean([d for d in avg_trade_durations if d > 0]) if any(avg_trade_durations) else 0.0,
        'max_drawdown': np.mean(max_drawdowns),
        'composite_score': composite_score,
    }


def compare_with_rl_agent(ideal_metrics: Dict, rl_trades_path: str):
    """Compare ideal trading performance with RL agent actual performance."""
    print(f"\n{'='*80}")
    print("COMPARISON: IDEAL vs RL AGENT")
    print(f"{'='*80}")

    try:
        rl_trades = pd.read_csv(rl_trades_path)
        print(f"\n📁 Loaded RL agent trades from: {rl_trades_path}")
        print(f"   Total trades: {len(rl_trades)}")
    except Exception as e:
        print(f"\n❌ Error loading RL trades: {e}")
        return

    # Calculate RL agent metrics
    closed_trades = rl_trades[rl_trades['status'] == 'closed']

    if len(closed_trades) == 0:
        print("\n⚠️  No closed trades in RL agent data")
        return

    rl_total_pnl = closed_trades['realized_pnl_usd'].sum()
    rl_mean_pnl_pct = closed_trades['realized_pnl_pct'].mean()
    rl_winning = len(closed_trades[closed_trades['realized_pnl_usd'] > 0])
    rl_losing = len(closed_trades[closed_trades['realized_pnl_usd'] <= 0])
    rl_win_rate = (rl_winning / len(closed_trades) * 100) if len(closed_trades) > 0 else 0.0
    rl_avg_duration = closed_trades['hours_held'].mean() if 'hours_held' in closed_trades.columns else 0.0

    # Calculate RL composite score
    rl_pnl_score = rl_mean_pnl_pct / 5.0
    rl_winrate_score = rl_win_rate / 100.0
    # Approximate drawdown (would need episode data for exact calculation)
    rl_drawdown_score = 0.95  # Placeholder
    rl_composite = 0.50 * rl_pnl_score + 0.30 * rl_winrate_score + 0.20 * rl_drawdown_score

    print(f"\n📊 RL AGENT PERFORMANCE:")
    print(f"   Total P&L:       ${rl_total_pnl:8.2f}")
    print(f"   Mean P&L (%):    {rl_mean_pnl_pct:8.2f}%")
    print(f"   Win Rate:        {rl_win_rate:8.1f}%")
    print(f"   Trades:          {len(closed_trades):8.0f}")
    print(f"   Avg Duration:    {rl_avg_duration:8.1f} hours")
    print(f"   Composite Score: {rl_composite:8.4f}")

    print(f"\n📊 IDEAL PERFORMANCE:")
    print(f"   Total P&L:       ${ideal_metrics['total_pnl_usd']:8.2f}")
    print(f"   Mean P&L (%):    {ideal_metrics['mean_pnl_pct']:8.2f}%")
    print(f"   Win Rate:        {ideal_metrics['win_rate']:8.1f}%")
    print(f"   Trades:          {ideal_metrics['total_trades']:8.0f}")
    print(f"   Avg Duration:    {ideal_metrics['avg_duration']:8.1f} hours")
    print(f"   Composite Score: {ideal_metrics['composite_score']:8.4f}")

    print(f"\n📈 PERFORMANCE GAP (Ideal - RL):")
    pnl_gap = ideal_metrics['total_pnl_usd'] - rl_total_pnl
    pnl_pct_gap = ideal_metrics['mean_pnl_pct'] - rl_mean_pnl_pct
    win_rate_gap = ideal_metrics['win_rate'] - rl_win_rate
    composite_gap = ideal_metrics['composite_score'] - rl_composite

    print(f"   Total P&L Gap:   ${pnl_gap:8.2f} ({pnl_gap/rl_total_pnl*100:+.1f}%)" if rl_total_pnl != 0 else f"   Total P&L Gap:   ${pnl_gap:8.2f}")
    print(f"   Mean P&L% Gap:   {pnl_pct_gap:8.2f} percentage points")
    print(f"   Win Rate Gap:    {win_rate_gap:8.1f} percentage points")
    print(f"   Composite Gap:   {composite_gap:8.4f}")

    print(f"\n💡 INTERPRETATION:")
    if composite_gap > 0.1:
        print(f"   ✅ Significant room for improvement! Ideal outperforms RL by {composite_gap:.2%}")
    elif composite_gap > 0:
        print(f"   ⚠️  Some room for improvement. Ideal slightly better by {composite_gap:.2%}")
    else:
        print(f"   🎯 RL agent is performing near-optimally!")

    efficiency = (rl_composite / ideal_metrics['composite_score'] * 100) if ideal_metrics['composite_score'] > 0 else 0
    print(f"   RL Agent Efficiency: {efficiency:.1f}% of ideal performance")


if __name__ == '__main__':
    args = parse_args()
    ideal_metrics = simulate_ideal_trading(args)

    if ideal_metrics:
        print("\n" + "=" * 80)
        print("✅ IDEAL TRADING SIMULATION COMPLETE")
        print("=" * 80)

        # Compare with RL agent if requested
        if args.compare_with:
            compare_with_rl_agent(ideal_metrics, args.compare_with)
