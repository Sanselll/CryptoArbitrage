"""
Analyze Test Dataset for Maximum Achievable Profit

This script calculates the theoretical maximum profit a perfect RL agent
could achieve given the test dataset and trading constraints.
"""

import pandas as pd
import numpy as np
from datetime import timedelta

# Episode configuration (matches RL environment)
INITIAL_CAPITAL = 10000.0
MAX_POSITIONS = 3
MAX_POSITION_SIZE_PCT = 33.3
POSITION_SIZE = INITIAL_CAPITAL * (MAX_POSITION_SIZE_PCT / 100)  # $3,333
EPISODE_LENGTH_HOURS = 72
MAKER_FEE_PCT = 0.01  # 0.01% per side
ROUND_TRIP_FEE_PCT = MAKER_FEE_PCT * 2  # 0.02%

# Quality thresholds
HIGH_QUALITY_APR = 150
HIGH_QUALITY_SPREAD = 0.4
LOW_QUALITY_APR = 75
LOW_QUALITY_SPREAD = 0.5


def load_test_data(path='data/rl_test.csv'):
    """Load and prepare test data."""
    print("="*80)
    print("LOADING TEST DATA")
    print("="*80)
    df = pd.read_csv(path, low_memory=False)
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df = df.sort_values('entry_time').reset_index(drop=True)

    print(f"Loaded {len(df):,} opportunities")
    print(f"Date range: {df['entry_time'].min()} to {df['entry_time'].max()}")
    print(f"Duration: {(df['entry_time'].max() - df['entry_time'].min()).days} days")
    print()

    return df


def classify_opportunities(df):
    """Classify opportunities by quality."""
    print("="*80)
    print("CLASSIFYING OPPORTUNITIES BY QUALITY")
    print("="*80)

    # Get spread (handle both column names)
    if 'priceSpreadPct' in df.columns:
        spread = df['priceSpreadPct'].abs()
    elif 'currentPriceSpreadPercent' in df.columns:
        spread = df['currentPriceSpreadPercent'].abs()
    else:
        spread = pd.Series([0.0] * len(df))
        print("WARNING: No spread column found, using 0.0")

    apr = df['fund_apr']

    # Classify
    high_quality = (apr > HIGH_QUALITY_APR) & (spread < HIGH_QUALITY_SPREAD)
    medium_quality = ((apr >= LOW_QUALITY_APR) & (apr <= HIGH_QUALITY_APR)) & ~high_quality
    low_quality = ((apr < LOW_QUALITY_APR) | (spread > LOW_QUALITY_SPREAD)) & ~high_quality

    df['quality'] = 'UNKNOWN'
    df.loc[high_quality, 'quality'] = 'HIGH'
    df.loc[medium_quality, 'quality'] = 'MEDIUM'
    df.loc[low_quality, 'quality'] = 'LOW'

    # Statistics
    total = len(df)
    high_count = high_quality.sum()
    medium_count = medium_quality.sum()
    low_count = low_quality.sum()

    print(f"Total opportunities: {total:,}")
    print()
    print(f"HIGH QUALITY (APR>{HIGH_QUALITY_APR}, spread<{HIGH_QUALITY_SPREAD}%):")
    print(f"  Count: {high_count:,} ({high_count/total*100:.1f}%)")
    if high_count > 0:
        print(f"  Avg APR: {apr[high_quality].mean():.2f}")
        print(f"  Avg Spread: {spread[high_quality].mean():.3f}%")
        print(f"  Min APR: {apr[high_quality].min():.2f}")
        print(f"  Max APR: {apr[high_quality].max():.2f}")
    print()

    print(f"MEDIUM QUALITY (APR {LOW_QUALITY_APR}-{HIGH_QUALITY_APR}):")
    print(f"  Count: {medium_count:,} ({medium_count/total*100:.1f}%)")
    if medium_count > 0:
        print(f"  Avg APR: {apr[medium_quality].mean():.2f}")
        print(f"  Avg Spread: {spread[medium_quality].mean():.3f}%")
    print()

    print(f"LOW QUALITY (APR<{LOW_QUALITY_APR} or spread>{LOW_QUALITY_SPREAD}%):")
    print(f"  Count: {low_count:,} ({low_count/total*100:.1f}%)")
    if low_count > 0:
        print(f"  Avg APR: {apr[low_quality].mean():.2f}")
        print(f"  Avg Spread: {spread[low_quality].mean():.3f}%")
    print()

    return df


def calculate_profit(apr, hours_held, position_size=POSITION_SIZE):
    """Calculate profit from a single position."""
    # Funding profit = (APR / 365 / 24) * hours * position_size
    # Position is 2x (long + short)
    funding_profit_pct = (apr / 365 / 24) * hours_held
    funding_profit_usd = (funding_profit_pct / 100) * position_size * 2

    # Subtract fees
    fees_usd = (ROUND_TRIP_FEE_PCT / 100) * position_size * 2

    net_profit_usd = funding_profit_usd - fees_usd
    net_profit_pct = (net_profit_usd / (position_size * 2)) * 100

    return net_profit_usd, net_profit_pct


def simulate_perfect_agent(df, hold_hours=24):
    """Simulate perfect agent with perfect foresight."""
    print("="*80)
    print("SIMULATING PERFECT AGENT (Theoretical Maximum)")
    print("="*80)
    print(f"Strategy: Always select top {MAX_POSITIONS} highest-APR opportunities")
    print(f"Hold duration: {hold_hours} hours")
    print(f"Position size: ${POSITION_SIZE:,.2f} per position")
    print()

    # Group by hour
    df['hour'] = df['entry_time'].dt.floor('H')
    hourly_groups = df.groupby('hour')

    total_profit = 0
    total_trades = 0

    for hour, group in hourly_groups:
        # Select top MAX_POSITIONS by APR
        top_opps = group.nlargest(MAX_POSITIONS, 'fund_apr')

        for _, opp in top_opps.iterrows():
            profit_usd, profit_pct = calculate_profit(opp['fund_apr'], hold_hours)
            total_profit += profit_usd
            total_trades += 1

    # Calculate episode metrics
    total_hours = (df['entry_time'].max() - df['entry_time'].min()).total_seconds() / 3600
    num_episodes = total_hours / EPISODE_LENGTH_HOURS

    profit_per_episode = total_profit / num_episodes if num_episodes > 0 else 0
    profit_pct_per_episode = (profit_per_episode / INITIAL_CAPITAL) * 100

    trades_per_episode = total_trades / num_episodes if num_episodes > 0 else 0

    print(f"Total trades simulated: {total_trades:,}")
    print(f"Total profit: ${total_profit:,.2f}")
    print(f"Number of episodes: {num_episodes:.1f}")
    print()
    print(f"📊 PERFECT AGENT PERFORMANCE:")
    print(f"  Avg trades per episode: {trades_per_episode:.1f}")
    print(f"  Avg profit per episode: ${profit_per_episode:,.2f}")
    print(f"  Avg P&L per episode: +{profit_pct_per_episode:.2f}%")
    print()

    return profit_pct_per_episode


def simulate_realistic_perfect_agent(df, hold_hours=24):
    """Simulate realistic perfect agent (only HIGH QUALITY, no foresight)."""
    print("="*80)
    print("SIMULATING REALISTIC PERFECT AGENT")
    print("="*80)
    print(f"Strategy: Only enter HIGH QUALITY opportunities (APR>{HIGH_QUALITY_APR}, spread<{HIGH_QUALITY_SPREAD}%)")
    print(f"Maintain {MAX_POSITIONS} positions when possible")
    print(f"Hold duration: {hold_hours} hours")
    print()

    # Filter to HIGH QUALITY only
    high_quality = df[df['quality'] == 'HIGH'].copy()

    if len(high_quality) == 0:
        print("ERROR: No HIGH QUALITY opportunities found!")
        return 0.0

    print(f"HIGH QUALITY opportunities available: {len(high_quality):,}")
    print(f"Avg APR of HIGH QUALITY: {high_quality['fund_apr'].mean():.2f}")
    print()

    # Group by hour
    high_quality['hour'] = high_quality['entry_time'].dt.floor('H')
    hourly_groups = high_quality.groupby('hour')

    total_profit = 0
    total_trades = 0
    hours_with_opps = 0

    for hour, group in hourly_groups:
        hours_with_opps += 1
        # Take up to MAX_POSITIONS from HIGH QUALITY
        opps_to_take = min(MAX_POSITIONS, len(group))
        top_opps = group.nlargest(opps_to_take, 'fund_apr')

        for _, opp in top_opps.iterrows():
            profit_usd, profit_pct = calculate_profit(opp['fund_apr'], hold_hours)
            total_profit += profit_usd
            total_trades += 1

    # Calculate episode metrics
    total_hours = (df['entry_time'].max() - df['entry_time'].min()).total_seconds() / 3600
    num_episodes = total_hours / EPISODE_LENGTH_HOURS

    profit_per_episode = total_profit / num_episodes if num_episodes > 0 else 0
    profit_pct_per_episode = (profit_per_episode / INITIAL_CAPITAL) * 100

    trades_per_episode = total_trades / num_episodes if num_episodes > 0 else 0
    opps_per_hour = len(high_quality) / total_hours if total_hours > 0 else 0

    print(f"Total trades simulated: {total_trades:,}")
    print(f"Total profit: ${total_profit:,.2f}")
    print(f"Hours with HIGH QUALITY opps: {hours_with_opps:,} ({hours_with_opps/total_hours*100:.1f}%)")
    print(f"Avg HIGH QUALITY opps per hour: {opps_per_hour:.2f}")
    print()
    print(f"📊 REALISTIC PERFECT AGENT PERFORMANCE:")
    print(f"  Avg trades per episode: {trades_per_episode:.1f}")
    print(f"  Avg profit per episode: ${profit_per_episode:,.2f}")
    print(f"  Avg P&L per episode: +{profit_pct_per_episode:.2f}%")
    print()

    return profit_pct_per_episode


def main():
    # Load data
    df = load_test_data()

    # Classify opportunities
    df = classify_opportunities(df)

    # Simulate perfect agent (theoretical max)
    perfect_pnl = simulate_perfect_agent(df, hold_hours=24)

    # Simulate realistic perfect agent
    realistic_pnl = simulate_realistic_perfect_agent(df, hold_hours=24)

    # Summary
    print("="*80)
    print("SUMMARY & COMPARISON")
    print("="*80)
    print(f"Current Agent Performance:")
    print(f"  Win Rate: 30%")
    print(f"  Avg P&L: -0.24%")
    print()
    print(f"Theoretical Maximum (Perfect Foresight):")
    print(f"  Avg P&L: +{perfect_pnl:.2f}%")
    print()
    print(f"Realistic Maximum (HIGH QUALITY Only):")
    print(f"  Avg P&L: +{realistic_pnl:.2f}%")
    print()
    print(f"Gap Analysis:")
    print(f"  Current vs Realistic Max: {realistic_pnl - (-0.24):.2f}% gap")
    print(f"  Improvement potential: {(realistic_pnl - (-0.24)) / abs(-0.24) * 100:.0f}%")
    print()

    if realistic_pnl > 0.5:
        print(f"✅ Target of >0.5% P&L is ACHIEVABLE")
        print(f"   Agent needs to learn to select HIGH QUALITY opportunities")
    else:
        print(f"⚠️  Target of >0.5% P&L may be challenging")
        print(f"   Realistic max is +{realistic_pnl:.2f}%")
        print(f"   May need longer hold times or better data filtering")
    print("="*80)


if __name__ == "__main__":
    main()
