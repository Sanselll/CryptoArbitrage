"""
Price History Loader for RL Environment

Efficient loading and querying of historical price data for accurate P&L simulation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import pyarrow.parquet as pq


class PriceHistoryLoader:
    """
    Loads and provides efficient access to historical price data.

    Supports:
    - Loading from parquet files (one per symbol)
    - Efficient timestamp-based lookups
    - Interpolation for missing data
    - Multiple exchanges
    """

    def __init__(self, price_history_dir: str):
        """
        Initialize price history loader.

        Args:
            price_history_dir: Directory containing price history parquet files
        """
        self.price_history_dir = Path(price_history_dir)
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.loaded_symbols = set()

        if not self.price_history_dir.exists():
            raise ValueError(f"Price history directory not found: {price_history_dir}")

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Load price history for a single symbol.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)

        Returns:
            DataFrame with columns: timestamp, binance_price, bybit_price, ...
        """
        if symbol in self.loaded_symbols:
            return self.price_data[symbol]

        price_file = self.price_history_dir / f"{symbol}.parquet"

        if not price_file.exists():
            raise ValueError(f"Price history not found for symbol: {symbol}")

        # Load parquet file
        df = pd.read_parquet(price_file)

        # Ensure timestamp column is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        # Set timestamp as index for faster lookups
        df = df.set_index('timestamp').sort_index()

        self.price_data[symbol] = df
        self.loaded_symbols.add(symbol)

        return df

    def get_price(
        self,
        symbol: str,
        timestamp: pd.Timestamp,
        exchange: str = 'binance',
        fallback: bool = True
    ) -> Optional[float]:
        """
        Get price for a symbol at a specific timestamp.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            exchange: Exchange name (binance or bybit)
            fallback: If True, use nearest price if exact match not found

        Returns:
            Price at timestamp, or None if not available
        """
        # Ensure symbol is loaded
        if symbol not in self.loaded_symbols:
            try:
                self.load_symbol(symbol)
            except ValueError:
                return None

        df = self.price_data[symbol]
        price_col = f"{exchange.lower()}_price"

        if price_col not in df.columns:
            return None

        # Ensure timestamp is timezone-aware
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')

        # Try exact match first
        if timestamp in df.index:
            price = df.loc[timestamp, price_col]
            # Handle case where duplicate timestamps return a Series
            if isinstance(price, pd.Series):
                price = price.iloc[0]
            if pd.notna(price):
                return float(price)

        if not fallback:
            return None

        # Find nearest timestamp (forward fill, then backward fill)
        try:
            # Get the closest timestamp
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
        exchange: str = 'binance',
        fallback: bool = True
    ) -> Optional[float]:
        """
        Get funding rate for a symbol at a specific timestamp.

        Args:
            symbol: Trading pair symbol
            timestamp: Target timestamp
            exchange: Exchange name (binance or bybit)
            fallback: If True, use nearest rate if exact match not found

        Returns:
            Funding rate at timestamp, or None if not available
        """
        # Ensure symbol is loaded
        if symbol not in self.loaded_symbols:
            try:
                self.load_symbol(symbol)
            except ValueError:
                return None

        df = self.price_data[symbol]
        rate_col = f"{exchange.lower()}_funding_rate"

        if rate_col not in df.columns:
            return None

        # Ensure timestamp is timezone-aware
        if timestamp.tz is None:
            timestamp = timestamp.tz_localize('UTC')

        # Try exact match first
        if timestamp in df.index:
            rate = df.loc[timestamp, rate_col]
            # Handle case where duplicate timestamps return a Series
            if isinstance(rate, pd.Series):
                rate = rate.iloc[0]
            if pd.notna(rate):
                return float(rate)

        if not fallback:
            return None

        # Find nearest timestamp (forward fill, then backward fill)
        try:
            # Get the closest timestamp
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            if idx >= 0:
                rate = df.iloc[idx][rate_col]
                if pd.notna(rate):
                    return float(rate)
        except Exception:
            pass

        return None

    def get_price_range(
        self,
        symbol: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        exchange: str = 'binance'
    ) -> pd.Series:
        """
        Get price series for a time range.

        Args:
            symbol: Trading pair symbol
            start_time: Start timestamp
            end_time: End timestamp
            exchange: Exchange name

        Returns:
            Series of prices indexed by timestamp
        """
        # Ensure symbol is loaded
        if symbol not in self.loaded_symbols:
            self.load_symbol(symbol)

        df = self.price_data[symbol]
        price_col = f"{exchange.lower()}_price"

        if price_col not in df.columns:
            return pd.Series(dtype=float)

        # Ensure timestamps are timezone-aware
        if start_time.tz is None:
            start_time = start_time.tz_localize('UTC')
        if end_time.tz is None:
            end_time = end_time.tz_localize('UTC')

        # Filter by time range
        mask = (df.index >= start_time) & (df.index <= end_time)
        return df.loc[mask, price_col]

    def get_available_symbols(self) -> list:
        """
        Get list of all symbols with price history available.

        Returns:
            List of symbol strings
        """
        parquet_files = list(self.price_history_dir.glob("*.parquet"))
        return [f.stem for f in parquet_files]

    def get_date_range(self, symbol: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        Get the date range covered by a symbol's price history.

        Args:
            symbol: Trading pair symbol

        Returns:
            Tuple of (start_timestamp, end_timestamp)
        """
        if symbol not in self.loaded_symbols:
            self.load_symbol(symbol)

        df = self.price_data[symbol]
        return df.index.min(), df.index.max()

    def preload_symbols(self, symbols: list):
        """
        Pre-load multiple symbols for faster access.

        Args:
            symbols: List of symbol strings
        """
        for symbol in symbols:
            if symbol not in self.loaded_symbols:
                try:
                    self.load_symbol(symbol)
                except ValueError as e:
                    print(f"Warning: Could not load {symbol}: {e}")

    def clear_cache(self):
        """Clear cached price data to free memory."""
        self.price_data.clear()
        self.loaded_symbols.clear()
