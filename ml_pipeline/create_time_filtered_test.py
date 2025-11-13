"""
Create filtered test data for specific time range.

Usage:
    python create_time_filtered_test.py \
        --start "2025-11-13 09:20:00" \
        --end "2025-11-13 09:30:00" \
        --input data/rl_test.csv \
        --output data/rl_test_filtered.csv
"""

import pandas as pd
import argparse
from pathlib import Path


def filter_test_data(input_path: str, output_path: str, start_time: str, end_time: str):
    """Filter test data to specific time range."""
    print(f"\n{'='*80}")
    print(f"FILTERING TEST DATA TO TIME RANGE")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Start: {start_time}")
    print(f"End: {end_time}")
    print()

    # Load test data
    print("Loading test data...")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    print(f"Original data: {len(df)} rows")
    print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Parse time range
    start_dt = pd.to_datetime(start_time, utc=True)
    end_dt = pd.to_datetime(end_time, utc=True)

    # Filter
    print(f"\nFiltering to: {start_dt} to {end_dt}")
    filtered_df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()

    print(f"Filtered data: {len(filtered_df)} rows")
    print(f"Time range: {filtered_df['timestamp'].min()} to {filtered_df['timestamp'].max()}")

    # Save
    print(f"\nSaving to {output_path}...")
    filtered_df.to_csv(output_path, index=False)
    print("✅ Done!")

    # Show sample
    print(f"\nFirst few rows:")
    print(filtered_df.head())

    return filtered_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter test data to specific time range')
    parser.add_argument('--start', type=str, required=True,
                        help='Start time (e.g., "2025-11-13 09:20:00")')
    parser.add_argument('--end', type=str, required=True,
                        help='End time (e.g., "2025-11-13 09:30:00")')
    parser.add_argument('--input', type=str, default='data/rl_test.csv',
                        help='Input test data CSV (default: data/rl_test.csv)')
    parser.add_argument('--output', type=str, default='data/rl_test_filtered.csv',
                        help='Output filtered CSV (default: data/rl_test_filtered.csv)')

    args = parser.parse_args()

    filter_test_data(args.input, args.output, args.start, args.end)
