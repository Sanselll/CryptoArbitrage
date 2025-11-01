"""
Hourly Market Data Loader for RL Training

Loads and provides efficient access to hourly price and funding rate data
prepared from historical klines and funding rate snapshots.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class HourlyMarketDataLoader:
    """
    Loads and provides efficient access to hourly market data.

    Data format per symbol CSV:
    - timestamp: Hourly timestamp (UTC)
    - binance_price: Price on Binance at that hour
    - bybit_price: Price on Bybit at that hour
    - binance_funding_rate: Funding rate on Binance (0.0 if no payment that hour)
    - bybit_funding_rate: Funding rate on Bybit (0.0 if no payment that hour)

    This loader provides:
    - Efficient symbol-by-symbol loading
    - Fast timestamp-based lookups
    - Price and funding rate queries
    - Multi-symbol data access
    """

    def __init__(self, data_dir: str):
        """
        Initialize hourly market data loader.

        Args:
            data_dir: Directory containing symbol CSV files (e.g., data/symbol_data/)
        """
        self.data_dir = Path(data_dir)
        self.symbol_data: Dict[str, pd.DataFrame] = {}
        self.loaded_symbols = set()

        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Load hourly data for a single symbol.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)

        Returns:
            DataFrame with columns: timestamp, binance_price, bybit_price,
                                   binance_funding_rate, bybit_funding_rate
        """
        if symbol in self.loaded_symbols:
            return self.symbol_data[symbol]

        csv_file = self.data_dir / f"{symbol}.csv"

        if not csv_file.exists():
            raise ValueError(f"Data not found for symbol: {symbol}")

        # Load CSV file
        df = pd.read_csv(csv_file)

        # Ensure timestamp column is datetime with UTC timezone
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Set timestamp as index for faster lookups
        df = df.set_index('timestamp').sort_index()

        # Verify required columns
        required_cols = ['binance_price', 'bybit_price',
                        'binance_funding_rate', 'bybit_funding_rate']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns for {symbol}: {missing_cols}")

        self.symbol_data[symbol] = df
        self.loaded_symbols.add(symbol)

        return df

    def get_price(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        exchange: str = 'binance',
        fallback_nearest: bool = True
    ) -> Optional[float]:
        """
        Get price for a symbol at a specific timestamp.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp (will be floored to nearest hour)
            exchange: Exchange name ('binance' or 'bybit')
            fallback_nearest: If True, use nearest hour if exact match not found

        Returns:
            Price at timestamp, or None if not available
        """
        # Ensure symbol is loaded
        if symbol not in self.loaded_symbols:
            try:
                self.load_symbol(symbol)
            except ValueError:
                return None

        df = self.symbol_data[symbol]
        price_col = f"{exchange.lower()}_price"

        if price_col not in df.columns:
            return None

        # Ensure timestamp is timezone-aware and floored to hour
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        timestamp = timestamp.floor('1h')

        # Try exact match first
        if timestamp in df.index:
            price = df.loc[timestamp, price_col]
            if pd.notna(price):
                return float(price)

        if not fallback_nearest:
            return None

        # Find nearest timestamp
        try:
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            if idx >= 0:
                price = df.iloc[idx][price_col]
                if pd.notna(price):
                    return float(price)
        except Exception:
            pass

        return None

    def get_funding_rate(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        exchange: str = 'binance'
    ) -> Optional[float]:
        """
        Get funding rate for a symbol at a specific timestamp.

        Returns 0.0 if no funding payment occurred at that hour.
        Returns None if data not available.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp (will be floored to nearest hour)
            exchange: Exchange name ('binance' or 'bybit')

        Returns:
            Funding rate at timestamp (0.0 if no payment), or None if not available
        """
        # Ensure symbol is loaded
        if symbol not in self.loaded_symbols:
            try:
                self.load_symbol(symbol)
            except ValueError:
                return None

        df = self.symbol_data[symbol]
        funding_col = f"{exchange.lower()}_funding_rate"

        if funding_col not in df.columns:
            return None

        # Ensure timestamp is timezone-aware and floored to hour
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        timestamp = timestamp.floor('1h')

        # Try exact match
        if timestamp in df.index:
            rate = df.loc[timestamp, funding_col]
            if pd.notna(rate):
                return float(rate)

        return None

    def get_market_data(
        self,
        symbol: str,
        timestamp: pd.Timestamp
    ) -> Optional[Dict[str, float]]:
        """
        Get all market data for a symbol at a specific timestamp.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp (will be floored to nearest hour)

        Returns:
            Dict with keys: binance_price, bybit_price,
                           binance_funding_rate, bybit_funding_rate
            Or None if not available
        """
        # Ensure symbol is loaded
        if symbol not in self.loaded_symbols:
            try:
                self.load_symbol(symbol)
            except ValueError:
                return None

        df = self.symbol_data[symbol]

        # Ensure timestamp is timezone-aware and floored to hour
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')
        timestamp = timestamp.floor('1h')

        # Try exact match
        if timestamp in df.index:
            row = df.loc[timestamp]
            return {
                'binance_price': float(row['binance_price']) if pd.notna(row['binance_price']) else None,
                'bybit_price': float(row['bybit_price']) if pd.notna(row['bybit_price']) else None,
                'binance_funding_rate': float(row['binance_funding_rate']) if pd.notna(row['binance_funding_rate']) else 0.0,
                'bybit_funding_rate': float(row['bybit_funding_rate']) if pd.notna(row['bybit_funding_rate']) else 0.0,
            }

        return None

    def get_time_range_data(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Get all market data for a symbol over a time range.

        Args:
            symbol: Trading pair symbol
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            DataFrame with all market data in the time range
        """
        # Ensure symbol is loaded
        if symbol not in self.loaded_symbols:
            self.load_symbol(symbol)

        df = self.symbol_data[symbol]

        # Ensure timestamps are timezone-aware
        if start_time.tz is None:
            start_time = start_time.tz_localize('UTC')
        if end_time.tz is None:
            end_time = end_time.tz_localize('UTC')

        # Filter by time range
        mask = (df.index >= start_time) & (df.index <= end_time)
        return df.loc[mask].copy()

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all symbols with data available.

        Returns:
            List of symbol strings
        """
        csv_files = list(self.data_dir.glob("*.csv"))
        return sorted([f.stem for f in csv_files])

    def get_date_range(self, symbol: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the date range covered by a symbol's data.

        Args:
            symbol: Trading pair symbol

        Returns:
            Tuple of (start_timestamp, end_timestamp)
        """
        if symbol not in self.loaded_symbols:
            self.load_symbol(symbol)

        df = self.symbol_data[symbol]
        return df.index.min(), df.index.max()

    def preload_symbols(self, symbols: List[str]):
        """
        Pre-load multiple symbols for faster access.

        Args:
            symbols: List of symbol strings
        """
        for symbol in symbols:
            if symbol not in self.loaded_symbols:
                try:
                    self.load_symbol(symbol)
                    print(f"  ✓ Loaded {symbol}")
                except ValueError as e:
                    print(f"  ✗ Could not load {symbol}: {e}")

    def get_spread(
        self,
        symbol: str,
        timestamp: pd.Timestamp
    ) -> Optional[float]:
        """
        Calculate the price spread between exchanges at a given timestamp.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp

        Returns:
            Absolute price spread (|binance_price - bybit_price|), or None if not available
        """
        data = self.get_market_data(symbol, timestamp)
        if data and data['binance_price'] is not None and data['bybit_price'] is not None:
            return abs(data['binance_price'] - data['bybit_price'])
        return None

    def get_spread_pct(
        self,
        symbol: str,
        timestamp: pd.Timestamp
    ) -> Optional[float]:
        """
        Calculate the percentage price spread between exchanges.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp

        Returns:
            Percentage spread relative to average price, or None if not available
        """
        data = self.get_market_data(symbol, timestamp)
        if data and data['binance_price'] is not None and data['bybit_price'] is not None:
            avg_price = (data['binance_price'] + data['bybit_price']) / 2
            if avg_price > 0:
                return abs(data['binance_price'] - data['bybit_price']) / avg_price * 100
        return None

    def get_funding_diff(
        self,
        symbol: str,
        timestamp: pd.Timestamp
    ) -> Optional[float]:
        """
        Calculate the funding rate differential between exchanges.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp

        Returns:
            Funding rate difference (binance_rate - bybit_rate), or None if not available
        """
        data = self.get_market_data(symbol, timestamp)
        if data:
            binance_rate = data['binance_funding_rate'] or 0.0
            bybit_rate = data['bybit_funding_rate'] or 0.0
            return binance_rate - bybit_rate
        return None

    def clear_cache(self):
        """Clear cached data to free memory."""
        self.symbol_data.clear()
        self.loaded_symbols.clear()

    def export_to_parquet(self, symbol: str, output_dir: str):
        """
        Export a symbol's data to parquet format for faster loading.

        Args:
            symbol: Trading pair symbol
            output_dir: Directory to save parquet file
        """
        if symbol not in self.loaded_symbols:
            self.load_symbol(symbol)

        df = self.symbol_data[symbol].reset_index()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        parquet_file = output_path / f"{symbol}.parquet"
        df.to_parquet(parquet_file, index=False)

        print(f"  Exported {symbol} to {parquet_file}")

    def export_all_to_parquet(self, output_dir: str):
        """
        Export all available symbols to parquet format.

        Args:
            output_dir: Directory to save parquet files
        """
        symbols = self.get_available_symbols()
        print(f"Exporting {len(symbols)} symbols to parquet...")

        for symbol in symbols:
            try:
                self.export_to_parquet(symbol, output_dir)
            except Exception as e:
                print(f"  ✗ Failed to export {symbol}: {e}")
