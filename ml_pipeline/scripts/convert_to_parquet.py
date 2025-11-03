"""
Convert hourly symbol CSV files to parquet format for efficient loading
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.hourly_market_data_loader import HourlyMarketDataLoader

def convert_all_symbols():
    """Convert all symbol CSV files to parquet format."""

    print("=" * 60)
    print("Converting Symbol CSVs to Parquet")
    print("=" * 60)

    # Initialize loader
    data_dir = "data/symbol_data"
    output_dir = "data/price_history"

    loader = HourlyMarketDataLoader(data_dir)

    # Get all available symbols
    symbols = loader.get_available_symbols()
    print(f"\nFound {len(symbols)} symbols to convert")

    # Convert all to parquet
    loader.export_all_to_parquet(output_dir)

    print(f"\n✅ Conversion complete!")
    print(f"Parquet files saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    convert_all_symbols()
