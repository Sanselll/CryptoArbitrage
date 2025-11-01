"""
Analyze Trade Data from RL Agent Evaluation

This script analyzes the detailed trade CSV exported by evaluate_agent()
to identify where and why the agent is losing money.
"""

import pandas as pd
import numpy as np
import sys


def load_trades(csv_path: str) -> pd.DataFrame:
    """Load trade data from CSV."""
    print("="*80)
    print("LOADING TRADE DATA")
    print("="*80)
    df = pd.read_csv(csv_path)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    df['exit_time'] = pd.to_datetime(df['exit_time'])

    print(f"Loaded {len(df):,} trades from {csv_path}")
    print(f"Date range: {df['entry_time'].min()} to {df['exit_time'].max()}")
    print(f"Episodes: {df['episode'].min()} to {df['episode'].max()}")
    print()
    return df


def analyze_overall_stats(df: pd.DataFrame):
    """Print overall trade statistics."""
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80)

    total_trades = len(df)
    winning_trades = len(df[df['pnl_usd'] > 0])
    losing_trades = len(df[df['pnl_usd'] <= 0])

    total_profit = df['pnl_usd'].sum()
    avg_profit = df['pnl_usd'].mean()
    avg_duration = df['duration_hours'].mean()

    total_fees = df['entry_fees_usd'].sum() + df['exit_fees_usd'].sum()

    print(f"Total Trades: {total_trades:,}")
    print(f"  Winning: {winning_trades:,} ({winning_trades/total_trades*100:.1f}%)")
    print(f"  Losing: {losing_trades:,} ({losing_trades/total_trades*100:.1f}%)")
    print()
    print(f"Total P&L: ${total_profit:+,.2f}")
    print(f"Average P&L per trade: ${avg_profit:+,.2f}")
    print(f"Average Duration: {avg_duration:.2f} hours")
    print(f"Total Fees Paid: ${total_fees:,.2f}")
    print()


def analyze_by_symbol(df: pd.DataFrame):
    """Analyze performance by symbol."""
    print("="*80)
    print("PERFORMANCE BY SYMBOL")
    print("="*80)

    symbol_stats = df.groupby('symbol').agg({
        'pnl_usd': ['count', 'sum', 'mean'],
        'pnl_pct': 'mean',
        'duration_hours': 'mean'
    }).round(2)

    symbol_stats.columns = ['trades', 'total_pnl_usd', 'avg_pnl_usd', 'avg_pnl_pct', 'avg_duration_h']

    # Add win rate
    symbol_win_rates = df.groupby('symbol').apply(
        lambda x: (x['pnl_usd'] > 0).sum() / len(x) * 100
    ).round(1)
    symbol_stats['win_rate'] = symbol_win_rates

    # Sort by total P&L
    symbol_stats = symbol_stats.sort_values('total_pnl_usd', ascending=False)

    print(symbol_stats.to_string())
    print()

    # Highlight best and worst
    best_symbol = symbol_stats.index[0]
    worst_symbol = symbol_stats.index[-1]

    print(f"🏆 BEST SYMBOL: {best_symbol}")
    print(f"   Total P&L: ${symbol_stats.loc[best_symbol, 'total_pnl_usd']:+,.2f}")
    print(f"   Win Rate: {symbol_stats.loc[best_symbol, 'win_rate']:.1f}%")
    print()

    print(f"💀 WORST SYMBOL: {worst_symbol}")
    print(f"   Total P&L: ${symbol_stats.loc[worst_symbol, 'total_pnl_usd']:+,.2f}")
    print(f"   Win Rate: {symbol_stats.loc[worst_symbol, 'win_rate']:.1f}%")
    print()


def analyze_by_duration(df: pd.DataFrame):
    """Analyze performance by hold duration."""
    print("="*80)
    print("PERFORMANCE BY HOLD DURATION")
    print("="*80)

    # Create duration bins
    bins = [0, 2, 8, 24, 72, 1000]
    labels = ['0-2h', '2-8h', '8-24h', '24-72h', '72h+']
    df['duration_bin'] = pd.cut(df['duration_hours'], bins=bins, labels=labels)

    duration_stats = df.groupby('duration_bin', observed=True).agg({
        'pnl_usd': ['count', 'sum', 'mean'],
        'pnl_pct': 'mean'
    }).round(2)

    duration_stats.columns = ['trades', 'total_pnl_usd', 'avg_pnl_usd', 'avg_pnl_pct']

    # Add win rate
    duration_win_rates = df.groupby('duration_bin', observed=True).apply(
        lambda x: (x['pnl_usd'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).round(1)
    duration_stats['win_rate'] = duration_win_rates

    print(duration_stats.to_string())
    print()


def analyze_by_episode(df: pd.DataFrame):
    """Analyze performance by episode."""
    print("="*80)
    print("PERFORMANCE BY EPISODE")
    print("="*80)

    episode_stats = df.groupby('episode').agg({
        'pnl_usd': ['count', 'sum', 'mean'],
        'symbol': lambda x: x.nunique()
    }).round(2)

    episode_stats.columns = ['trades', 'total_pnl_usd', 'avg_pnl_usd', 'unique_symbols']

    # Add win rate
    episode_win_rates = df.groupby('episode').apply(
        lambda x: (x['pnl_usd'] > 0).sum() / len(x) * 100 if len(x) > 0 else 0
    ).round(1)
    episode_stats['win_rate'] = episode_win_rates

    print(episode_stats.to_string())
    print()

    # Identify catastrophic episodes (< -2% total P&L)
    catastrophic_episodes = episode_stats[episode_stats['total_pnl_usd'] < -200]

    if len(catastrophic_episodes) > 0:
        print(f"⚠️  CATASTROPHIC EPISODES (total loss > $200):")
        for ep in catastrophic_episodes.index:
            print(f"\n   Episode {ep}:")
            print(f"   - Total P&L: ${episode_stats.loc[ep, 'total_pnl_usd']:+,.2f}")
            print(f"   - Trades: {int(episode_stats.loc[ep, 'trades'])}")
            print(f"   - Win Rate: {episode_stats.loc[ep, 'win_rate']:.1f}%")

            # Show trades from this episode
            ep_trades = df[df['episode'] == ep].sort_values('pnl_usd')
            print(f"   - Top losers:")
            for _, trade in ep_trades.head(3).iterrows():
                print(f"     • {trade['symbol']}: ${trade['pnl_usd']:+.2f} ({trade['duration_hours']:.1f}h)")
        print()


def analyze_worst_trades(df: pd.DataFrame, n: int = 10):
    """Show worst trades."""
    print("="*80)
    print(f"WORST {n} TRADES")
    print("="*80)

    worst_trades = df.nsmallest(n, 'pnl_usd')

    for i, (_, trade) in enumerate(worst_trades.iterrows(), 1):
        print(f"{i}. {trade['symbol']} (Episode {trade['episode']}):")
        print(f"   Entry: {trade['entry_time'].strftime('%Y-%m-%d %H:%M')}")
        print(f"   Exit:  {trade['exit_time'].strftime('%Y-%m-%d %H:%M')}")
        print(f"   P&L: ${trade['pnl_usd']:+.2f} ({trade['pnl_pct']:+.2f}%)")
        print(f"   Duration: {trade['duration_hours']:.1f}h")
        print(f"   Fees: ${trade['entry_fees_usd'] + trade['exit_fees_usd']:.2f}")
        print()


def analyze_fees_impact(df: pd.DataFrame):
    """Analyze fee impact on profitability."""
    print("="*80)
    print("FEE IMPACT ANALYSIS")
    print("="*80)

    df['total_fees'] = df['entry_fees_usd'] + df['exit_fees_usd']
    df['gross_pnl'] = df['pnl_usd'] + df['total_fees']

    total_fees = df['total_fees'].sum()
    total_gross = df['gross_pnl'].sum()
    total_net = df['pnl_usd'].sum()

    print(f"Gross P&L (before fees): ${total_gross:+,.2f}")
    print(f"Total Fees Paid: ${total_fees:,.2f}")
    print(f"Net P&L (after fees): ${total_net:+,.2f}")
    print(f"Fee Impact: {(total_fees / abs(total_gross) * 100):.1f}% of gross P&L")
    print()

    # Check if churning (many short trades)
    short_trades = df[df['duration_hours'] < 2]
    if len(short_trades) > 0:
        short_fees = short_trades['total_fees'].sum()
        short_pnl = short_trades['pnl_usd'].sum()
        print(f"⚠️  SHORT TRADES (<2h):")
        print(f"   Count: {len(short_trades)} ({len(short_trades)/len(df)*100:.1f}% of all trades)")
        print(f"   Total Fees: ${short_fees:,.2f}")
        print(f"   Net P&L: ${short_pnl:+,.2f}")
        if short_pnl < -short_fees * 0.5:
            print(f"   💀 Agent is CHURNING (short trades losing money to fees)")
        print()


def generate_recommendations(df: pd.DataFrame):
    """Generate recommendations based on analysis."""
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Analyze winning vs losing patterns
    winners = df[df['pnl_usd'] > 0]
    losers = df[df['pnl_usd'] <= 0]

    if len(winners) > 0 and len(losers) > 0:
        # Duration patterns
        winner_avg_duration = winners['duration_hours'].mean()
        loser_avg_duration = losers['duration_hours'].mean()

        print(f"1. HOLD DURATION:")
        print(f"   Winners avg: {winner_avg_duration:.1f}h")
        print(f"   Losers avg: {loser_avg_duration:.1f}h")
        if winner_avg_duration < loser_avg_duration:
            print(f"   ✅ Winners exit faster - good!")
        else:
            print(f"   ⚠️  Losers exit faster - agent may be exiting winners too early")
        print()

        # Symbol concentration
        winner_symbols = winners['symbol'].value_counts().head(3)
        loser_symbols = losers['symbol'].value_counts().head(3)

        print(f"2. SYMBOL SELECTION:")
        print(f"   Top winning symbols: {', '.join(winner_symbols.index.tolist())}")
        print(f"   Top losing symbols: {', '.join(loser_symbols.index.tolist())}")

        # Check if any overlap
        overlap = set(winner_symbols.index) & set(loser_symbols.index)
        if overlap:
            print(f"   ⚠️  Symbols in both lists: {', '.join(overlap)}")
            print(f"      Agent is inconsistent on these symbols")
        print()

        # Fee ratio
        avg_fee = (df['entry_fees_usd'] + df['exit_fees_usd']).mean()
        avg_winner_profit = winners['pnl_usd'].mean()

        print(f"3. FEE EFFICIENCY:")
        print(f"   Avg fees per trade: ${avg_fee:.2f}")
        print(f"   Avg winner profit: ${avg_winner_profit:.2f}")
        print(f"   Profit/Fee ratio: {avg_winner_profit / avg_fee:.1f}x")
        if avg_winner_profit < avg_fee * 3:
            print(f"   ⚠️  Winners too small relative to fees - increase quality threshold")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trades.py <path_to_evaluation_trades.csv>")
        print("\nExample:")
        print("  python analyze_trades.py evaluation_trades_20251031_120000.csv")
        sys.exit(1)

    csv_path = sys.argv[1]

    # Load data
    df = load_trades(csv_path)

    # Run analyses
    analyze_overall_stats(df)
    analyze_by_symbol(df)
    analyze_by_duration(df)
    analyze_by_episode(df)
    analyze_worst_trades(df)
    analyze_fees_impact(df)
    generate_recommendations(df)

    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
