"""
Validate trades CSV against historical data

This script helps verify that model trading is correct by comparing
trade records against the actual historical market data.

Usage:
    python validate_trades.py --trades trades_test.csv --data data/rl_test.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Validate trades against historical data')
    parser.add_argument('--trades', type=str, required=True,
                        help='Path to trades CSV file')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historical data CSV file')
    parser.add_argument('--output', type=str, default='trade_validation_report.txt',
                        help='Output report file (default: trade_validation_report.txt)')
    return parser.parse_args()


def load_data(trades_path: str, data_path: str):
    """Load trades and historical data."""
    print(f"Loading trades from: {trades_path}")
    trades_df = pd.read_csv(trades_path)
    trades_df['entry_datetime'] = pd.to_datetime(trades_df['entry_datetime'])
    trades_df['exit_datetime'] = pd.to_datetime(trades_df['exit_datetime'])

    print(f"Loading historical data from: {data_path}")
    data_df = pd.read_csv(data_path)

    # Determine timestamp column (different data formats)
    if 'entry_time' in data_df.columns:
        data_df['timestamp'] = pd.to_datetime(data_df['entry_time'])
    elif 'detectedAt' in data_df.columns:
        data_df['timestamp'] = pd.to_datetime(data_df['detectedAt'])
    elif 'RecordedAt' in data_df.columns:
        data_df['timestamp'] = pd.to_datetime(data_df['RecordedAt'])
    else:
        raise ValueError("Could not find timestamp column in historical data")

    return trades_df, data_df


def validate_trade(trade, historical_data):
    """Validate a single trade against historical data."""
    symbol = trade['symbol']
    entry_time = trade['entry_datetime']
    exit_time = trade['exit_datetime']
    long_exchange = trade['long_exchange']
    short_exchange = trade['short_exchange']

    # Find matching historical opportunities
    # Since historical data is recorded hourly, but trades can happen anytime during that hour,
    # we need to check within ±60 minutes to cover the full hour window
    time_tolerance = pd.Timedelta(minutes=60)
    mask = (
        (historical_data['symbol'] == symbol) &
        (historical_data['long_exchange'] == long_exchange) &
        (historical_data['short_exchange'] == short_exchange) &
        (historical_data['timestamp'] >= entry_time - time_tolerance) &
        (historical_data['timestamp'] <= entry_time + time_tolerance)
    )
    matching_opportunities = historical_data[mask]

    if len(matching_opportunities) == 0:
        return {
            'status': 'WARNING',
            'message': f'No matching opportunity found for {symbol} ({long_exchange}/{short_exchange}) at {entry_time}'
        }

    # Find the closest opportunity to the trade entry time
    matching_opportunities['time_diff'] = abs(matching_opportunities['timestamp'] - entry_time)
    matching_opportunities = matching_opportunities.sort_values('time_diff')

    # Use the closest match
    closest_match = matching_opportunities.iloc[0]
    time_diff_minutes = closest_match['time_diff'].total_seconds() / 60

    validations = []

    # Show time difference for context
    if time_diff_minutes < 1:
        validations.append(f"✓ Exact time match with historical data")
    elif time_diff_minutes < 30:
        validations.append(f"✓ Close time match (±{time_diff_minutes:.1f} minutes)")
    else:
        validations.append(f"ℹ Time difference: {time_diff_minutes:.1f} minutes (hourly snapshot)")

    # Validate entry prices
    historical_long_price = closest_match['entry_long_price']
    trade_long_price = trade['entry_long_price']
    long_price_diff_pct = abs(historical_long_price - trade_long_price) / historical_long_price * 100

    if long_price_diff_pct < 0.01:  # Less than 0.01% difference
        validations.append(f"✓ Long entry price matches exactly (${trade_long_price:.6f})")
    elif long_price_diff_pct < 0.1:
        validations.append(f"✓ Long entry price close (diff: {long_price_diff_pct:.4f}%)")
    else:
        validations.append(f"⚠ Long entry price differs by {long_price_diff_pct:.2f}% (hist: ${historical_long_price:.6f}, trade: ${trade_long_price:.6f})")

    historical_short_price = closest_match['entry_short_price']
    trade_short_price = trade['entry_short_price']
    short_price_diff_pct = abs(historical_short_price - trade_short_price) / historical_short_price * 100

    if short_price_diff_pct < 0.01:
        validations.append(f"✓ Short entry price matches exactly (${trade_short_price:.6f})")
    elif short_price_diff_pct < 0.1:
        validations.append(f"✓ Short entry price close (diff: {short_price_diff_pct:.4f}%)")
    else:
        validations.append(f"⚠ Short entry price differs by {short_price_diff_pct:.2f}% (hist: ${historical_short_price:.6f}, trade: ${trade_short_price:.6f})")

    # Validate funding rates
    historical_long_funding = closest_match['long_funding_rate']
    trade_long_funding = trade['long_funding_rate']
    long_funding_diff = abs(historical_long_funding - trade_long_funding)

    if long_funding_diff < 0.000001:  # Very small difference
        validations.append(f"✓ Long funding rate matches exactly ({trade_long_funding:.6f})")
    elif long_funding_diff < 0.00001:
        validations.append(f"✓ Long funding rate close (diff: {long_funding_diff:.6f})")
    else:
        validations.append(f"⚠ Long funding rate differs (hist: {historical_long_funding:.6f}, trade: {trade_long_funding:.6f})")

    historical_short_funding = closest_match['short_funding_rate']
    trade_short_funding = trade['short_funding_rate']
    short_funding_diff = abs(historical_short_funding - trade_short_funding)

    if short_funding_diff < 0.000001:
        validations.append(f"✓ Short funding rate matches exactly ({trade_short_funding:.6f})")
    elif short_funding_diff < 0.00001:
        validations.append(f"✓ Short funding rate close (diff: {short_funding_diff:.6f})")
    else:
        validations.append(f"⚠ Short funding rate differs (hist: {historical_short_funding:.6f}, trade: {trade_short_funding:.6f})")

    # Trade summary
    hours_held = trade['hours_held']
    position_size = trade['position_size_usd']
    funding_earned = trade['funding_earned_usd']
    pnl = trade['realized_pnl_usd']
    pnl_pct = trade['realized_pnl_pct']

    validations.append(f"ℹ Position: ${position_size:.2f} @ {trade['leverage']:.1f}x leverage")
    validations.append(f"ℹ Duration: {hours_held:.1f} hours")
    validations.append(f"ℹ Funding earned: ${funding_earned:.2f}")
    validations.append(f"ℹ Total fees: ${trade['total_fees_usd']:.2f}")
    validations.append(f"ℹ Realized P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")

    # Calculate expected funding (rough estimate)
    funding_rate_diff = abs(trade['long_funding_rate']) + abs(trade['short_funding_rate'])
    expected_funding_per_hour = position_size * funding_rate_diff
    expected_total_funding = expected_funding_per_hour * hours_held

    funding_accuracy = abs(funding_earned - expected_total_funding) / max(expected_total_funding, 0.01) * 100
    if funding_accuracy < 20:  # Within 20% of simple estimate
        validations.append(f"✓ Funding calculation appears reasonable (expected ~${expected_total_funding:.2f})")
    else:
        validations.append(f"ℹ Funding differs from simple estimate (expected ~${expected_total_funding:.2f})")

    return {
        'status': 'OK',
        'message': '\n    '.join(validations)
    }


def main():
    args = parse_args()

    # Load data
    trades_df, data_df = load_data(args.trades, args.data)

    print(f"\n{'='*80}")
    print("TRADE VALIDATION REPORT")
    print(f"{'='*80}")
    print(f"Total trades: {len(trades_df)}")
    print(f"Historical data records: {len(data_df)}")
    print()

    # Validate each trade
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("TRADE VALIDATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Total trades: {len(trades_df)}")
    report_lines.append(f"Historical data records: {len(data_df)}")
    report_lines.append("")

    for idx, trade in trades_df.iterrows():
        print(f"\nTrade {idx + 1}/{len(trades_df)}: {trade['symbol']} ({trade['long_exchange']} long, {trade['short_exchange']} short)")
        print(f"  Entry: {trade['entry_datetime']}")
        print(f"  Exit:  {trade['exit_datetime']}")

        report_lines.append(f"\nTrade {idx + 1}: {trade['symbol']} ({trade['long_exchange']} long, {trade['short_exchange']} short)")
        report_lines.append(f"  Entry: {trade['entry_datetime']}")
        report_lines.append(f"  Exit:  {trade['exit_datetime']}")

        result = validate_trade(trade, data_df)

        if result['status'] == 'OK':
            print(f"  Status: {result['status']}")
            print(f"    {result['message']}")
            report_lines.append(f"  Status: {result['status']}")
            report_lines.append(f"    {result['message']}")
        else:
            print(f"  Status: {result['status']}")
            print(f"  {result['message']}")
            report_lines.append(f"  Status: {result['status']}")
            report_lines.append(f"  {result['message']}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    report_lines.append(f"\n{'='*80}")
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append(f"{'='*80}")

    total_pnl = trades_df['realized_pnl_usd'].sum()
    total_funding = trades_df['funding_earned_usd'].sum()
    total_fees = trades_df['total_fees_usd'].sum()
    avg_duration = trades_df['hours_held'].mean()
    win_rate = (trades_df['realized_pnl_usd'] > 0).sum() / len(trades_df) * 100

    stats = [
        f"Total P&L: ${total_pnl:.2f}",
        f"Total funding earned: ${total_funding:.2f}",
        f"Total fees paid: ${total_fees:.2f}",
        f"Net profit: ${total_pnl:.2f}",
        f"Average trade duration: {avg_duration:.1f} hours",
        f"Win rate: {win_rate:.1f}%",
        f"Average position size: ${trades_df['position_size_usd'].mean():.2f}",
    ]

    for stat in stats:
        print(f"  {stat}")
        report_lines.append(f"  {stat}")

    # Write report to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n{'='*80}")
    print(f"Validation report saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
