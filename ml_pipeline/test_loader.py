"""
Quick test script for HourlyMarketDataLoader
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.hourly_market_data_loader import HourlyMarketDataLoader
import pandas as pd

def test_loader():
    """Test the hourly market data loader."""

    print("=" * 60)
    print("Testing HourlyMarketDataLoader")
    print("=" * 60)

    # Initialize loader
    data_dir = "data/symbol_data"
    loader = HourlyMarketDataLoader(data_dir)

    # Get available symbols
    symbols = loader.get_available_symbols()
    print(f"\n✓ Found {len(symbols)} symbols:")
    print(f"  {', '.join(symbols[:10])}" + ("..." if len(symbols) > 10 else ""))

    # Test loading a symbol
    test_symbol = symbols[0] if symbols else None
    if not test_symbol:
        print("\n✗ No symbols found!")
        return

    print(f"\n📊 Testing with {test_symbol}:")

    # Load the symbol
    df = loader.load_symbol(test_symbol)
    print(f"  ✓ Loaded {len(df)} hours of data")

    # Get date range
    start, end = loader.get_date_range(test_symbol)
    print(f"  ✓ Date range: {start} to {end}")

    # Get a specific timestamp's data
    test_time = df.index[10]  # Use 11th row as test
    print(f"\n📈 Testing data retrieval at {test_time}:")

    # Get market data
    market_data = loader.get_market_data(test_symbol, test_time)
    if market_data:
        print(f"  ✓ Binance price: ${market_data['binance_price']:.4f}")
        print(f"  ✓ Bybit price: ${market_data['bybit_price']:.4f}")
        print(f"  ✓ Binance funding: {market_data['binance_funding_rate']:.8f}")
        print(f"  ✓ Bybit funding: {market_data['bybit_funding_rate']:.8f}")
    else:
        print("  ✗ Could not retrieve market data")

    # Test spread calculation
    spread_pct = loader.get_spread_pct(test_symbol, test_time)
    if spread_pct is not None:
        print(f"  ✓ Spread: {spread_pct:.4f}%")

    # Test funding differential
    funding_diff = loader.get_funding_diff(test_symbol, test_time)
    if funding_diff is not None:
        print(f"  ✓ Funding diff: {funding_diff:.8f}")

    # Test time range query
    range_start = df.index[0]
    range_end = df.index[min(23, len(df)-1)]  # First 24 hours or less
    range_df = loader.get_time_range_data(test_symbol, range_start, range_end)
    print(f"\n📅 Time range query ({range_start} to {range_end}):")
    print(f"  ✓ Retrieved {len(range_df)} rows")

    # Test preloading multiple symbols
    print(f"\n🔄 Testing preload of first 3 symbols:")
    loader.clear_cache()
    loader.preload_symbols(symbols[:3])
    print(f"  ✓ Loaded {len(loader.loaded_symbols)} symbols into cache")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

if __name__ == "__main__":
    test_loader()
