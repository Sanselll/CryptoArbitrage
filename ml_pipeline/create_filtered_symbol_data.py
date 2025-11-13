#!/usr/bin/env python3
"""
Filter symbol_data CSVs to specific time ranges for testing.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import shutil

def filter_symbol_data(source_dir: str, target_dir: str, start_time: str, end_time: str):
    """
    Filter all CSV files in source_dir to the specified time range and save to target_dir.

    Args:
        source_dir: Source directory with symbol CSV files
        target_dir: Target directory to save filtered CSVs
        start_time: Start datetime string (ISO format)
        end_time: End datetime string (ISO format)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)

    # Parse time range
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)

    print(f"Filtering symbol data from {start_dt} to {end_dt}")
    print(f"Source: {source_path}")
    print(f"Target: {target_path}")
    print()

    # Get all CSV files
    csv_files = list(source_path.glob("*.csv"))
    print(f"Found {len(csv_files)} symbol CSV files")

    processed = 0
    skipped = 0

    for csv_file in csv_files:
        try:
            # Read CSV
            df = pd.read_csv(csv_file)

            # Check if dataframe has timestamp column
            if 'timestamp' not in df.columns:
                print(f"⚠️  Skipping {csv_file.name}: no 'timestamp' column")
                skipped += 1
                continue

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter to time range
            filtered_df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)].copy()

            if len(filtered_df) == 0:
                print(f"⚠️  Skipping {csv_file.name}: no data in time range")
                skipped += 1
                continue

            # Save filtered data
            output_file = target_path / csv_file.name
            filtered_df.to_csv(output_file, index=False)

            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed} files...")

        except Exception as e:
            print(f"❌ Error processing {csv_file.name}: {e}")
            skipped += 1

    print()
    print(f"✅ Complete!")
    print(f"   Processed: {processed} files")
    print(f"   Skipped: {skipped} files")
    print(f"   Output: {target_path}")


if __name__ == "__main__":
    # Define time ranges
    ranges = [
        {
            "name": "2-day (Nov 12-13)",
            "source": "data/symbol_data",
            "target": "data/symbol_data_1112_2d",
            "start": "2025-11-12 00:00:00+00:00",
            "end": "2025-11-13 09:45:00+00:00"
        },
        {
            "name": "3-hour (Nov 12)",
            "source": "data/symbol_data",
            "target": "data/symbol_data_1112_3h",
            "start": "2025-11-12 00:00:00+00:00",
            "end": "2025-11-12 02:55:00+00:00"
        }
    ]

    # Process each range
    for range_config in ranges:
        print("=" * 80)
        print(f"Creating {range_config['name']}")
        print("=" * 80)

        filter_symbol_data(
            source_dir=range_config['source'],
            target_dir=range_config['target'],
            start_time=range_config['start'],
            end_time=range_config['end']
        )
        print()
