"""
Prepare RL Training Dataset with Configurable Interval Data from Klines

This script creates clean training data for RL with configurable time intervals:
1. Extracts opportunities (entry points for RL episodes)
2. For each symbol in opportunities:
   - Loads minute-level prices from klines data
   - Resamples to specified interval (5min, 15min, 1h, 4h, etc.)
   - Combines prices with funding rates from opportunities
   - Adds funding payment flags (when fees are actually charged)
3. Saves symbol data as CSV files with columns:
   timestamp, binance_price, bybit_price, binance_funding_rate,
   bybit_funding_rate, binance_funding_paid, bybit_funding_paid

This provides exactly what the RL environment needs: price data at configurable intervals
and funding payment times. Uses klines (minute-level price data) for complete coverage.

Note: When using non-hourly intervals (e.g., 5min), make sure to train the RL model
with matching step_hours parameter (e.g., --step-minutes 5 in train_ppo.py).
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
import argparse


def parse_date_args(args):
    """Parse and validate date range arguments."""
    start_date = None
    end_date = None

    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')

    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')

    if args.days:
        if end_date is None:
            end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)

    return start_date, end_date


def load_opportunities(
    opportunities_dir: str,
    start_date: datetime = None,
    end_date: datetime = None
) -> List[dict]:
    """
    Load opportunities from JSON date folders.

    Returns:
        List of opportunity dictionaries
    """
    opportunities_path = Path(opportunities_dir)

    if not opportunities_path.exists():
        raise ValueError(f"Opportunities directory not found: {opportunities_dir}")

    json_files = sorted(list(opportunities_path.glob("*/opportunities.json")))

    if len(json_files) == 0:
        raise ValueError(f"No opportunities.json files found in {opportunities_dir}")

    # Filter by date range
    if start_date or end_date:
        filtered_files = []
        for json_file in json_files:
            folder_name = json_file.parent.name
            try:
                folder_date = datetime.strptime(folder_name, '%Y-%m-%d')
                if start_date and folder_date < start_date:
                    continue
                if end_date and folder_date > end_date:
                    continue
                filtered_files.append(json_file)
            except ValueError:
                continue
        json_files = filtered_files

    print(f"\n{'='*80}")
    print(f"LOADING OPPORTUNITIES")
    print(f"{'='*80}")
    print(f"Found {len(json_files)} date folders to process")
    if start_date:
        print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    if end_date:
        print(f"End date: {end_date.strftime('%Y-%m-%d')}")
    print()

    all_opportunities = []

    for idx, json_file in enumerate(json_files):
        folder_date = json_file.parent.name

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            opps_count = 0

            if isinstance(data, list):
                for entry in data:
                    timestamp = entry.get('timestamp')

                    if 'opportunities' in entry:
                        for opp in entry['opportunities']:
                            opp['hourly_timestamp'] = timestamp
                            all_opportunities.append(opp)
                            opps_count += 1

            if (idx + 1) % 5 == 0 or idx == len(json_files) - 1:
                print(f"Processed {idx + 1}/{len(json_files)}: {folder_date} ({opps_count} opportunities)")

        except Exception as e:
            print(f"Warning: Could not load {json_file.name}: {e}")
            continue

    print(f"\n{'─'*60}")
    print(f"Total opportunities loaded: {len(all_opportunities):,}")
    print(f"{'─'*60}\n")

    return all_opportunities


def extract_symbols_from_opportunities(opportunities: List[dict]) -> Set[str]:
    """Extract unique symbols that appear in opportunities."""
    symbols = set()
    for opp in opportunities:
        if 'symbol' in opp:
            symbols.add(opp['symbol'])
    return symbols


def load_price_and_funding_data(
    opportunities_dir: str,
    raw_data_dir: str,
    symbols: Set[str],
    start_date: datetime = None,
    end_date: datetime = None
) -> Tuple[Dict[str, List], Dict[Tuple[str, str], List]]:
    """
    Load price history from klines and funding rates from opportunities.

    Args:
        opportunities_dir: Path to opportunities directory (for funding rates)
        raw_data_dir: Path to raw data directory (for klines)
        symbols: Set of symbols to load
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        (price_data, funding_data)
        price_data: {symbol: [(timestamp, exchange, price)]}
        funding_data: {(exchange, symbol): [(timestamp, rate, funding_time, next_funding_time)]}
    """
    opportunities_path = Path(opportunities_dir)
    raw_path = Path(raw_data_dir)

    # Get list of date folders
    json_files = sorted(list(opportunities_path.glob("*/opportunities.json")))

    # Filter by date range
    if start_date or end_date:
        filtered_files = []
        for json_file in json_files:
            folder_name = json_file.parent.name
            try:
                folder_date = datetime.strptime(folder_name, '%Y-%m-%d')
                if start_date and folder_date < start_date:
                    continue
                if end_date and folder_date > end_date:
                    continue
                filtered_files.append(json_file)
            except ValueError:
                continue
        json_files = filtered_files

    print(f"\n{'='*80}")
    print(f"LOADING KLINES PRICES & FUNDING DATA FOR {len(symbols)} SYMBOLS")
    print(f"{'='*80}\n")

    price_data = {symbol: [] for symbol in symbols}
    funding_data = {}

    for idx, json_file in enumerate(json_files):
        folder_date = json_file.parent.name

        try:
            prices_count = 0
            funding_count = 0

            # Load klines data from raw directory
            klines_dir = raw_path / folder_date / "klines"
            if klines_dir.exists():
                # Load Binance klines
                binance_klines_file = klines_dir / "binance.json"
                if binance_klines_file.exists():
                    with open(binance_klines_file, 'r') as f:
                        binance_klines = json.load(f)

                    # Extract prices for our symbols
                    for symbol in symbols:
                        if symbol in binance_klines:
                            for kline in binance_klines[symbol]:
                                price_data[symbol].append({
                                    'timestamp': kline['timestamp'],
                                    'exchange': 'binance',
                                    'price': kline['price']
                                })
                                prices_count += 1

                # Load Bybit klines
                bybit_klines_file = klines_dir / "bybit.json"
                if bybit_klines_file.exists():
                    with open(bybit_klines_file, 'r') as f:
                        bybit_klines = json.load(f)

                    # Extract prices for our symbols
                    for symbol in symbols:
                        if symbol in bybit_klines:
                            for kline in bybit_klines[symbol]:
                                price_data[symbol].append({
                                    'timestamp': kline['timestamp'],
                                    'exchange': 'bybit',
                                    'price': kline['price']
                                })
                                prices_count += 1

            # Load funding rates from raw/*/funding_rates/*.json
            funding_rates_dir = raw_path / folder_date / "funding_rates"
            if funding_rates_dir.exists():
                # Load Binance funding rates
                binance_funding_file = funding_rates_dir / "binance.json"
                if binance_funding_file.exists():
                    with open(binance_funding_file, 'r') as f:
                        binance_funding_list = json.load(f)

                    # Extract funding rates for our symbols
                    for rate_entry in binance_funding_list:
                        symbol = rate_entry.get('symbol')
                        if symbol in symbols:
                            key = ('Binance', symbol)
                            if key not in funding_data:
                                funding_data[key] = []

                            funding_data[key].append({
                                'funding_time': rate_entry.get('fundingTime'),
                                'rate': rate_entry.get('rate')
                            })
                            funding_count += 1

                # Load Bybit funding rates
                bybit_funding_file = funding_rates_dir / "bybit.json"
                if bybit_funding_file.exists():
                    with open(bybit_funding_file, 'r') as f:
                        bybit_funding_list = json.load(f)

                    # Extract funding rates for our symbols
                    for rate_entry in bybit_funding_list:
                        symbol = rate_entry.get('symbol')
                        if symbol in symbols:
                            key = ('Bybit', symbol)
                            if key not in funding_data:
                                funding_data[key] = []

                            funding_data[key].append({
                                'funding_time': rate_entry.get('fundingTime'),
                                'rate': rate_entry.get('rate')
                            })
                            funding_count += 1

            if (idx + 1) % 5 == 0 or idx == len(json_files) - 1:
                print(f"Processed {idx + 1}/{len(json_files)}: {folder_date} ({prices_count} prices from klines, {funding_count} funding records)")

        except Exception as e:
            print(f"Warning: Could not load data for {folder_date}: {e}")
            continue

    print(f"\n{'─'*60}")
    print(f"Data extraction complete")
    print(f"{'─'*60}\n")

    return price_data, funding_data


def create_symbol_data(
    symbol: str,
    price_data: List[dict],
    funding_data_binance: List[dict],
    funding_data_bybit: List[dict],
    resample_interval: str = '1h'
) -> pd.DataFrame:
    """
    Create resampled interval data for a symbol combining prices and funding rates.

    Args:
        symbol: Symbol name
        price_data: List of price dictionaries
        funding_data_binance: Binance funding rate records
        funding_data_bybit: Bybit funding rate records
        resample_interval: Resample interval (e.g., '5min', '15min', '1h'). Default: '1h'

    Returns:
        DataFrame with columns: timestamp, binance_price, bybit_price,
        binance_funding_rate, bybit_funding_rate
    """
    if not price_data:
        return pd.DataFrame()

    # Convert prices to DataFrame
    price_df = pd.DataFrame(price_data)
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], utc=True)

    # Pivot to have columns per exchange
    price_pivot = price_df.pivot_table(
        index='timestamp',
        columns='exchange',
        values='price',
        aggfunc='first'
    )

    # Resample to specified interval (take first value in each interval)
    resampled_prices = price_pivot.resample(resample_interval).first()

    # Forward-fill missing prices (limit to 2 intervals)
    resampled_prices = resampled_prices.ffill(limit=2)

    # Rename columns
    resampled_prices.columns = [f'{col}_price' for col in resampled_prices.columns]

    # Process funding rates for Binance
    # Only set rate when funding payment occurs (based on fundingTime)
    binance_rates = pd.Series(0.0, index=resampled_prices.index)
    if funding_data_binance:
        funding_df_binance = pd.DataFrame(funding_data_binance)
        funding_df_binance['funding_time'] = pd.to_datetime(funding_df_binance['funding_time'], utc=True, format='mixed')

        # For each funding event, match it to the interval when it occurs
        for _, row in funding_df_binance.iterrows():
            funding_time = row['funding_time']
            rate = row['rate']

            # Round funding time to the resampled interval
            interval_timestamp = funding_time.floor(resample_interval)

            # Set rate at the interval when funding is paid
            if interval_timestamp in binance_rates.index:
                binance_rates[interval_timestamp] = rate

    # Process funding rates for Bybit
    bybit_rates = pd.Series(0.0, index=resampled_prices.index)
    if funding_data_bybit:
        funding_df_bybit = pd.DataFrame(funding_data_bybit)
        funding_df_bybit['funding_time'] = pd.to_datetime(funding_df_bybit['funding_time'], utc=True, format='mixed')

        # For each funding event, match it to the interval when it occurs
        for _, row in funding_df_bybit.iterrows():
            funding_time = row['funding_time']
            rate = row['rate']

            # Round funding time to the resampled interval
            interval_timestamp = funding_time.floor(resample_interval)

            # Set rate at the interval when funding is paid
            if interval_timestamp in bybit_rates.index:
                bybit_rates[interval_timestamp] = rate

    # Combine everything (no *_funding_paid columns)
    result = pd.DataFrame({
        'timestamp': resampled_prices.index,
        'binance_price': resampled_prices.get('binance_price', np.nan),
        'bybit_price': resampled_prices.get('bybit_price', np.nan),
        'binance_funding_rate': binance_rates,
        'bybit_funding_rate': bybit_rates
    })

    result = result.reset_index(drop=True)

    return result


def create_opportunities_dataframe(opportunities: List[dict]) -> pd.DataFrame:
    """Convert opportunities list to structured DataFrame."""
    df = pd.DataFrame(opportunities)

    # Use hourly_timestamp (historical time) NOT detectedAt (system processing time)
    if 'hourly_timestamp' in df.columns:
        df['entry_time'] = pd.to_datetime(df['hourly_timestamp'], format='mixed', utc=True)
    else:
        df['entry_time'] = pd.to_datetime(df['detectedAt'], format='mixed', utc=True)

    if 'longNextFundingTime' in df.columns:
        df['longNextFundingTime'] = pd.to_datetime(df['longNextFundingTime'], format='mixed', utc=True)
    if 'shortNextFundingTime' in df.columns:
        df['shortNextFundingTime'] = pd.to_datetime(df['shortNextFundingTime'], format='mixed', utc=True)

    # Rename columns to match environment expectations
    column_mapping = {
        'longExchange': 'long_exchange',
        'shortExchange': 'short_exchange',
        'longExchangePrice': 'entry_long_price',
        'shortExchangePrice': 'entry_short_price',
        'longFundingRate': 'long_funding_rate',
        'shortFundingRate': 'short_funding_rate',
        'longFundingIntervalHours': 'long_funding_interval_hours',
        'shortFundingIntervalHours': 'short_funding_interval_hours',
        'longNextFundingTime': 'long_next_funding_time',
        'shortNextFundingTime': 'short_next_funding_time',
        'fundProfit8h': 'fund_profit_8h',
        'fundApr': 'fund_apr',
        'volume24h': 'volume_24h',
        'spreadVolatilityStdDev': 'spread_volatility_stddev'
    }

    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_columns)

    df = df.sort_values('entry_time').reset_index(drop=True)

    return df


def prepare_rl_dataset(
    opportunities_dir: str,
    raw_data_dir: str,
    output_dir: str,
    start_date: datetime = None,
    end_date: datetime = None,
    resample_interval: str = '1h'
):
    """
    Main function to prepare RL dataset with configurable interval data.

    Args:
        opportunities_dir: Path to opportunities directory (for funding rates and opportunities)
        raw_data_dir: Path to raw data directory (for klines price data)
        output_dir: Output directory for processed data
        start_date: Optional start date filter
        end_date: Optional end date filter
        resample_interval: Resample interval (e.g., '5min', '15min', '1h', '4h'). Default: '1h'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"PREPARING RL DATASET (INTERVAL: {resample_interval})")
    print(f"{'='*80}")
    print(f"Opportunities: {opportunities_dir}")
    print(f"Klines (raw): {raw_data_dir}")
    print(f"Output: {output_dir}")
    print(f"Resample interval: {resample_interval}")
    print(f"{'='*80}\n")

    # Step 1: Load opportunities
    print(f"[1/5] Loading opportunities...")
    opportunities = load_opportunities(opportunities_dir, start_date, end_date)

    if len(opportunities) == 0:
        raise ValueError("No opportunities loaded")

    # Step 2: Extract unique symbols
    print(f"[2/5] Extracting unique symbols from opportunities...")
    symbols = extract_symbols_from_opportunities(opportunities)
    print(f"  Found {len(symbols)} unique symbols")
    print()

    # Step 3: Load price and funding data for these symbols
    print(f"[3/5] Loading price and funding data...")
    price_data, funding_data = load_price_and_funding_data(
        opportunities_dir,
        raw_data_dir,
        symbols,
        start_date,
        end_date
    )

    # Step 4: Create resampled interval CSVs for each symbol
    print(f"[4/5] Creating {resample_interval} interval data for each symbol...")
    symbol_data_dir = output_path / 'symbol_data'
    symbol_data_dir.mkdir(exist_ok=True)

    symbols_saved = 0
    symbols_with_issues = []

    for idx, symbol in enumerate(sorted(symbols)):
        try:
            # Get funding data for both exchanges
            funding_binance = funding_data.get(('Binance', symbol), [])
            funding_bybit = funding_data.get(('Bybit', symbol), [])

            # Create resampled data
            resampled_df = create_symbol_data(
                symbol,
                price_data.get(symbol, []),
                funding_binance,
                funding_bybit,
                resample_interval
            )

            if resampled_df.empty:
                symbols_with_issues.append(symbol)
                continue

            # Save as CSV
            output_file = symbol_data_dir / f"{symbol}.csv"
            resampled_df.to_csv(output_file, index=False)

            symbols_saved += 1

            if (idx + 1) % 20 == 0 or idx == len(symbols) - 1:
                print(f"Saved {idx + 1}/{len(symbols)}: {symbol} ({len(resampled_df)} intervals)")

        except Exception as e:
            print(f"Warning: Could not create data for {symbol}: {e}")
            symbols_with_issues.append(symbol)
            continue

    print(f"\n{'─'*60}")
    print(f"Symbol data saved: {symbols_saved}/{len(symbols)} symbols")
    if symbols_with_issues:
        print(f"Symbols with issues: {len(symbols_with_issues)}")
    print(f"Location: {symbol_data_dir}")
    print(f"{'─'*60}\n")

    # Step 4.5: Convert CSVs to parquet for efficient loading
    print(f"[4.5/6] Converting symbol CSVs to parquet format...")
    price_history_dir = output_path / 'price_history'
    price_history_dir.mkdir(exist_ok=True)

    parquet_count = 0
    for csv_file in symbol_data_dir.glob("*.csv"):
        try:
            symbol = csv_file.stem
            df = pd.read_csv(csv_file)
            parquet_file = price_history_dir / f"{symbol}.parquet"
            df.to_parquet(parquet_file, index=False)
            parquet_count += 1
        except Exception as e:
            print(f"  Warning: Could not convert {symbol} to parquet: {e}")

    print(f"  Converted {parquet_count} symbols to parquet")
    print(f"  Location: {price_history_dir}")
    print()

    # Step 5: Create and save opportunities DataFrame
    print(f"[5/5] Creating opportunities dataset...")
    opportunities_df = create_opportunities_dataframe(opportunities)

    print(f"  Processed {len(opportunities_df):,} opportunities")
    print(f"  Unique symbols: {opportunities_df['symbol'].nunique()}")
    print(f"  Date range: {opportunities_df['entry_time'].min()} to {opportunities_df['entry_time'].max()}")
    print()

    opportunities_file = output_path / 'rl_opportunities.csv'
    opportunities_df.to_csv(opportunities_file, index=False)
    print(f"  Saved: {opportunities_file}")
    print(f"  Size: {opportunities_file.stat().st_size / 1024 / 1024:.1f} MB")
    print()

    # Step 5: Split into train/test sets (80/20 by time)
    print(f"[5/6] Splitting data into train/test sets (80/20)...")
    split_idx = int(len(opportunities_df) * 0.8)
    train_df = opportunities_df.iloc[:split_idx].copy()
    test_df = opportunities_df.iloc[split_idx:].copy()

    # Save splits
    train_file = output_path / 'rl_train.csv'
    test_file = output_path / 'rl_test.csv'
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"  Train set: {len(train_df):,} opportunities")
    print(f"    Date range: {train_df['entry_time'].min()} to {train_df['entry_time'].max()}")
    print(f"    File: {train_file}")
    print(f"  Test set: {len(test_df):,} opportunities")
    print(f"    Date range: {test_df['entry_time'].min()} to {test_df['entry_time'].max()}")
    print(f"    File: {test_file}")
    print()

    # Summary
    print(f"{'='*80}")
    print(f"PREPARATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output files:")
    print(f"  Full dataset: {opportunities_file} ({len(opportunities_df):,} opportunities)")
    print(f"  Train set: {train_file} ({len(train_df):,} opportunities)")
    print(f"  Test set: {test_file} ({len(test_df):,} opportunities)")
    print(f"  Symbol data (CSV, {resample_interval} intervals): {symbol_data_dir}/ ({symbols_saved} symbols)")
    print(f"  Price history (Parquet): {price_history_dir}/ ({parquet_count} symbols)")
    print(f"\nData format:")
    print(f"  Each symbol CSV/parquet contains:")
    print(f"    - timestamp ({resample_interval} intervals)")
    print(f"    - binance_price, bybit_price")
    print(f"    - binance_funding_rate, bybit_funding_rate")
    print(f"\nUsage:")
    print(f"  - RL training uses: {train_file} + {price_history_dir}/")
    print(f"  - RL evaluation uses: {test_file} + {price_history_dir}/")
    print(f"  - Parquet files are loaded automatically by the environment")
    print(f"{'='*80}\n")

    return opportunities_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepare RL training dataset with configurable interval data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all available data (default: 1-hour intervals)
  python prepare_rl_data.py

  # Process last 30 days with 5-minute intervals
  python prepare_rl_data.py --days 30 --resample-interval 5min

  # Process last 30 days with 15-minute intervals
  python prepare_rl_data.py --days 30 --resample-interval 15min

  # Process specific date range with 1-hour intervals (default)
  python prepare_rl_data.py --start-date 2025-09-01 --end-date 2025-09-30

  # Custom output directory with 5-minute intervals
  python prepare_rl_data.py --output-dir data/rl_sept --days 30 --resample-interval 5min

Note: When training with non-hourly intervals, use matching --step-minutes in train_ppo.py
  Example: python train_ppo.py --step-minutes 5 (for 5-minute data)
        """
    )

    parser.add_argument(
        '--opportunities-dir',
        type=str,
        default='/Users/sansel/Projects/CryptoArbitrage/src/CryptoArbitrage.HistoricalCollector/data/opportunities',
        help='Path to opportunities folder'
    )

    parser.add_argument(
        '--raw-data-dir',
        type=str,
        default='/Users/sansel/Projects/CryptoArbitrage/src/CryptoArbitrage.HistoricalCollector/data/raw',
        help='Path to raw data folder (for klines)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for prepared datasets'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD, inclusive)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD, inclusive)'
    )

    parser.add_argument(
        '--days',
        type=int,
        help='Process last N days (alternative to start/end dates)'
    )

    parser.add_argument(
        '--resample-interval',
        type=str,
        default='1h',
        help='Resample interval for price data (e.g., "5min", "15min", "1h", "4h"). Default: "1h"'
    )

    args = parser.parse_args()

    # Parse date arguments
    start_date, end_date = parse_date_args(args)

    # Run preparation
    prepare_rl_dataset(
        opportunities_dir=args.opportunities_dir,
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        start_date=start_date,
        end_date=end_date,
        resample_interval=args.resample_interval
    )
