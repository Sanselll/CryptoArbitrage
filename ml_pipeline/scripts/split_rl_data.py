"""
Split RL Opportunities Data into Train/Test Sets

Splits opportunities by time to avoid data leakage.
"""

import pandas as pd
from datetime import datetime, timedelta


def split_rl_data(
    input_file: str = 'data/rl_opportunities.csv',
    train_file: str = 'data/rl_train.csv',
    test_file: str = 'data/rl_test.csv',
    test_days: int = 1
):
    """
    Split RL data by time.

    Args:
        input_file: Input CSV file
        train_file: Output training CSV
        test_file: Output test CSV
        test_days: Number of days to reserve for testing
    """
    print("="*80)
    print("SPLITTING RL DATA INTO TRAIN/TEST SETS")
    print("="*80)

    # Load data
    print(f"\nLoading data from: {input_file}")
    df = pd.read_csv(input_file)

    # Parse entry_time
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)

    # Sort by time
    df = df.sort_values('entry_time').reset_index(drop=True)

    # Get date range
    min_date = df['entry_time'].min()
    max_date = df['entry_time'].max()
    total_days = (max_date - min_date).total_seconds() / 86400

    print(f"\nData range:")
    print(f"  Start: {min_date}")
    print(f"  End: {max_date}")
    print(f"  Total days: {total_days:.1f}")
    print(f"  Total opportunities: {len(df):,}")

    # Calculate split point
    # Reserve last N days for testing
    test_start = max_date - timedelta(days=test_days)

    # Split data
    train_df = df[df['entry_time'] < test_start].copy()
    test_df = df[df['entry_time'] >= test_start].copy()

    print(f"\n{'─'*60}")
    print(f"Train Set:")
    print(f"  Date range: {train_df['entry_time'].min()} to {train_df['entry_time'].max()}")
    print(f"  Opportunities: {len(train_df):,}")
    print(f"  Symbols: {train_df['symbol'].nunique()}")

    print(f"\nTest Set:")
    print(f"  Date range: {test_df['entry_time'].min()} to {test_df['entry_time'].max()}")
    print(f"  Opportunities: {len(test_df):,}")
    print(f"  Symbols: {test_df['symbol'].nunique()}")
    print(f"{'─'*60}")

    # Save splits
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\n✅ Split complete!")
    print(f"  Train set saved to: {train_file}")
    print(f"  Test set saved to: {test_file}")
    print("="*80)

    return train_df, test_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Split RL data into train/test sets')
    parser.add_argument('--input', type=str, default='data/rl_opportunities.csv',
                        help='Input CSV file')
    parser.add_argument('--train', type=str, default='data/rl_train.csv',
                        help='Output training CSV')
    parser.add_argument('--test', type=str, default='data/rl_test.csv',
                        help='Output test CSV')
    parser.add_argument('--test-days', type=int, default=1,
                        help='Number of days to reserve for testing')

    args = parser.parse_args()

    split_rl_data(
        input_file=args.input,
        train_file=args.train,
        test_file=args.test,
        test_days=args.test_days
    )
