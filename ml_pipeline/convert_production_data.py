#!/usr/bin/env python3
"""
Convert production data snapshots (JSON) to ML training format

Outputs:
    data/production/symbol_data/*.csv - Price and funding rate history (CSV format)
    data/production/price_history/*.parquet - Price and funding rate history (Parquet format)
    data/production/rl_opportunities.csv - Detected opportunities

Usage:
    python convert_production_data.py --date 2025-11-13
    python convert_production_data.py --start-date 2025-11-01 --end-date 2025-11-30
"""

import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import os


def load_snapshots(production_data_dir: Path, date_str: str):
    """Load all snapshots for a given date"""
    date_dir = production_data_dir / date_str

    if not date_dir.exists():
        print(f"No data found for date: {date_str}")
        return []

    snapshots = []

    # Load all JSON files (except manifest.json)
    for json_file in sorted(date_dir.glob("*.json")):
        if json_file.name == "manifest.json":
            continue

        try:
            with open(json_file, 'r') as f:
                snapshot = json.load(f)
                snapshots.append(snapshot)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return snapshots


def convert_snapshots_to_symbol_data(snapshots):
    """
    Convert snapshots to per-symbol data structure

    Returns:
        dict: {symbol: list of (timestamp, binance_price, bybit_price, binance_funding, bybit_funding)}
    """
    symbol_data = defaultdict(list)

    for snapshot in snapshots:
        timestamp = pd.to_datetime(snapshot['timestamp'])

        # Group market data by symbol
        symbol_exchange_data = defaultdict(dict)

        for market_entry in snapshot['marketData']:
            symbol = market_entry['symbol']
            exchange = market_entry['exchange']

            symbol_exchange_data[symbol][exchange] = {
                'price': market_entry.get('price'),
                'funding_rate': market_entry.get('fundingRate', 0.0),
            }

        # Create rows for symbols that have data from both exchanges
        for symbol, exchange_data in symbol_exchange_data.items():
            binance_data = exchange_data.get('Binance', {})
            bybit_data = exchange_data.get('Bybit', {})

            # Only include if we have price data from at least one exchange
            if binance_data or bybit_data:
                row = {
                    'timestamp': timestamp,
                    'binance_price': binance_data.get('price'),
                    'bybit_price': bybit_data.get('price'),
                    'binance_funding_rate': binance_data.get('funding_rate', 0.0),
                    'bybit_funding_rate': bybit_data.get('funding_rate', 0.0),
                }

                symbol_data[symbol].append(row)

    return symbol_data


def is_valid_funding_time(time_str):
    """Check if a funding time string is valid (not year 1 or empty)"""
    if not time_str:
        return False
    try:
        parsed_time = pd.to_datetime(time_str)
        # Check if datetime is realistic (year 1900 or later)
        return parsed_time.year >= 1900
    except:
        return False


def calculate_next_funding_time(current_time, funding_interval_hours):
    """Calculate next funding time based on interval"""
    # Round up to next funding interval
    hours_since_epoch = (current_time - pd.Timestamp('1970-01-01', tz='UTC')).total_seconds() / 3600
    next_funding_hours = int(hours_since_epoch // funding_interval_hours + 1) * funding_interval_hours
    next_time = pd.Timestamp('1970-01-01', tz='UTC') + pd.Timedelta(hours=next_funding_hours)
    return next_time.isoformat().replace('+00:00', 'Z')


def convert_opportunities_to_csv(snapshots):
    """
    Convert opportunities from snapshots to CSV format matching rl_opportunities.csv

    Filters out opportunities with zero APR values (data quality issue).

    Returns:
        list of dict: Opportunities in CSV format
    """
    opportunities = []
    invalid_funding_count = 0
    zero_apr_count = 0

    for snapshot in snapshots:
        timestamp = pd.to_datetime(snapshot['timestamp'])

        for opp in snapshot['opportunities']:
            # Map JSON camelCase to CSV snake_case
            row = {
                'symbol': opp.get('symbol', ''),
                'strategy': opp.get('strategy', 0),
                'subType': opp.get('subType', 1),
                'long_exchange': opp.get('longExchange', ''),
                'short_exchange': opp.get('shortExchange', ''),
                'long_funding_rate': opp.get('longFundingRate', 0.0),
                'short_funding_rate': opp.get('shortFundingRate', 0.0),
                'long_funding_interval_hours': opp.get('longFundingIntervalHours', 0),
                'short_funding_interval_hours': opp.get('shortFundingIntervalHours', 0),
                'long_next_funding_time': opp.get('longNextFundingTime', ''),
                'short_next_funding_time': opp.get('shortNextFundingTime', ''),
                'entry_long_price': opp.get('longExchangePrice', 0.0),
                'entry_short_price': opp.get('shortExchangePrice', 0.0),
                'currentPriceSpreadPercent': opp.get('currentPriceSpreadPercent', 0.0),
                'exchange': opp.get('exchange', ''),
                'spotPrice': opp.get('spotPrice', 0.0),
                'perpetualPrice': opp.get('perpetualPrice', 0.0),
                'spreadRate': opp.get('spreadRate', 0.0),
                'annualizedSpread': opp.get('annualizedSpread', 0.0),
                'estimatedProfitPercentage': opp.get('estimatedProfitPercentage', 0.0),
                'positionCostPercent': opp.get('positionCostPercent', 0.2),
                'breakEvenTimeHours': opp.get('breakEvenTimeHours', 0.0),
                'volume_24h': opp.get('volume24h', 0.0),
                'fund_profit_8h': opp.get('fundProfit8h', 0.0),
                'fund_apr': opp.get('fundApr', 0.0),
                'fund_profit_8h_24h_proj': opp.get('fund_profit_8h_24h_proj', 0.0),
                'fund_apr_24h_proj': opp.get('fund_apr_24h_proj', 0.0),
                'fundBreakEvenTime24hProj': opp.get('fundBreakEvenTime24hProj', 0.0),
                'fund_profit_8h_3d_proj': opp.get('fund_profit_8h_3d_proj', 0.0),
                'fund_apr_3d_proj': opp.get('fund_apr_3d_proj', 0.0),
                'fundBreakEvenTime3dProj': opp.get('fundBreakEvenTime3dProj', 0.0),
                'price_spread_24h_avg': opp.get('price_spread_24h_avg', 0.0),
                'price_spread_3d_avg': opp.get('price_spread_3d_avg', 0.0),
                'spread_30_sample_avg': opp.get('spread_30_sample_avg', 0.0),
                'spread_volatility_stddev': opp.get('spreadVolatilityStdDev', 0.0),
                'spreadVolatilityCv': opp.get('spreadVolatilityCv', 0.0),
                'longVolume24h': opp.get('longVolume24h', 0.0),
                'shortVolume24h': opp.get('shortVolume24h', 0.0),
                'bidAskSpreadPercent': opp.get('bidAskSpreadPercent', None),
                'orderbookDepthUsd': opp.get('orderbookDepthUsd', None),
                'liquidityStatus': opp.get('liquidityStatus', ''),
                'liquidityWarning': opp.get('liquidityWarning', ''),
                'status': opp.get('status', 0),
                'detectedAt': opp.get('detectedAt', timestamp.isoformat()),
                'isExistingPosition': opp.get('isExistingPosition', False),
                'uniqueKey': opp.get('uniqueKey', ''),
                'hourly_timestamp': timestamp.floor('h').isoformat() + 'Z',
                'entry_time': timestamp.isoformat()
            }

            # Validate and fix funding times
            long_funding_time = row['long_next_funding_time']
            short_funding_time = row['short_next_funding_time']

            if not is_valid_funding_time(long_funding_time):
                # Calculate reasonable next funding time
                row['long_next_funding_time'] = calculate_next_funding_time(
                    timestamp,
                    row['long_funding_interval_hours']
                )
                invalid_funding_count += 1

            if not is_valid_funding_time(short_funding_time):
                # Calculate reasonable next funding time
                row['short_next_funding_time'] = calculate_next_funding_time(
                    timestamp,
                    row['short_funding_interval_hours']
                )
                invalid_funding_count += 1

            # Filter out opportunities with zero APR (data quality issue)
            # Skip if fund_apr is 0 - this is the primary APR used for decisions
            fund_apr = row.get('fund_apr', 0.0)

            if fund_apr == 0.0:
                zero_apr_count += 1
                continue  # Skip this opportunity

            opportunities.append(row)

    if invalid_funding_count > 0:
        print(f"  ⚠️  Fixed {invalid_funding_count} invalid funding times")

    if zero_apr_count > 0:
        print(f"  ⚠️  Filtered out {zero_apr_count} opportunities with zero APR")

    return opportunities


def save_symbol_csv(symbol, data, output_dir: Path):
    """Save symbol data to CSV file"""
    if not data:
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by timestamp (keep as column, not index)
    df = df.sort_values('timestamp')

    # Save to CSV (timestamp remains as column)
    output_file = output_dir / f"{symbol}.csv"
    df.to_csv(output_file, index=False)

    print(f"  Saved {symbol}: {len(df)} rows")


def save_symbol_parquet(symbol, data, output_dir: Path):
    """Save symbol data to Parquet file"""
    if not data:
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by timestamp (keep as column, not index)
    df = df.sort_values('timestamp')

    # Save to Parquet (timestamp remains as column)
    output_file = output_dir / f"{symbol}.parquet"
    df.to_parquet(output_file, index=False)

    print(f"  Saved {symbol}: {len(df)} rows")


def main():
    parser = argparse.ArgumentParser(description='Convert production data to ML training format')
    parser.add_argument('--date', help='Single date to convert (YYYY-MM-DD)')
    parser.add_argument('--start-date', help='Start date for range (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for range (YYYY-MM-DD)')
    parser.add_argument('--production-data-dir',
                       default='../src/CryptoArbitrage.API/data/production',
                       help='Production data directory')

    args = parser.parse_args()

    # Determine date range
    if args.date:
        dates = [args.date]
    elif args.start_date and args.end_date:
        start = datetime.strptime(args.start_date, '%Y-%m-%d')
        end = datetime.strptime(args.end_date, '%Y-%m-%d')
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
    else:
        print("Error: Must specify either --date or both --start-date and --end-date")
        return

    # Setup paths - mirror data/ structure
    production_data_dir = Path(args.production_data_dir)
    output_base_dir = Path('data/production')
    symbol_data_dir = output_base_dir / 'symbol_data'
    price_history_dir = output_base_dir / 'price_history'
    opportunities_file = output_base_dir / 'rl_opportunities.csv'

    # Create directories
    output_base_dir.mkdir(parents=True, exist_ok=True)
    symbol_data_dir.mkdir(parents=True, exist_ok=True)
    price_history_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting production data to ML format...")
    print(f"Production source: {production_data_dir}")
    print(f"Output: {output_base_dir}")
    print(f"Date range: {dates[0]} to {dates[-1]} ({len(dates)} days)")
    print()

    # Collect all data across dates
    all_symbol_data = defaultdict(list)
    all_snapshots = []
    total_snapshots = 0

    for date_str in dates:
        print(f"Loading {date_str}...")
        snapshots = load_snapshots(production_data_dir, date_str)

        if not snapshots:
            continue

        total_snapshots += len(snapshots)
        all_snapshots.extend(snapshots)

        # Convert snapshots to symbol data
        symbol_data = convert_snapshots_to_symbol_data(snapshots)

        # Merge into all_symbol_data
        for symbol, rows in symbol_data.items():
            all_symbol_data[symbol].extend(rows)

    print(f"\nProcessed {total_snapshots} snapshots")
    print(f"Found {len(all_symbol_data)} unique symbols")
    print()

    # Save symbol data to both CSV and Parquet
    print("Saving symbol data (CSV + Parquet)...")
    for symbol in sorted(all_symbol_data.keys()):
        save_symbol_csv(symbol, all_symbol_data[symbol], symbol_data_dir)
        save_symbol_parquet(symbol, all_symbol_data[symbol], price_history_dir)

    # Convert and save opportunities
    print("\nExtracting opportunities...")
    opportunities = convert_opportunities_to_csv(all_snapshots)

    if opportunities:
        df_opportunities = pd.DataFrame(opportunities)

        # Save to CSV
        df_opportunities.to_csv(opportunities_file, index=False)
        print(f"  Saved {len(opportunities)} opportunities to {opportunities_file}")
    else:
        print("  No opportunities found")

    print()
    print(f"✅ Conversion complete!")
    print(f"   Output: {output_base_dir}/")
    print(f"   Symbol data (CSV): {len(list(symbol_data_dir.glob('*.csv')))} files")
    print(f"   Price history (Parquet): {len(list(price_history_dir.glob('*.parquet')))} files")
    print(f"   Opportunities: {len(opportunities)} rows")


if __name__ == '__main__':
    main()
