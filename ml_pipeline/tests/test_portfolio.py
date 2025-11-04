"""
Unit tests for Portfolio enhancements

Tests margin calculation, liquidation tracking, and execution state features.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from models.rl.core.portfolio import Portfolio, Position


class TestPositionWithLeverage:
    """Test Position class with leverage support."""

    def test_position_creation_with_leverage(self):
        """Test creating a position with leverage."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="Binance",
            short_exchange="Bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50010.0,
            position_size_usd=1000.0,
            leverage=5.0,
            long_funding_rate=0.0001,
            short_funding_rate=-0.0001,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        assert position.leverage == 5.0
        assert position.position_size_usd == 1000.0
        # Margin = (long_size + short_size) / leverage = 2000 / 5 = 400
        assert position.margin_used_usd == 400.0

    def test_liquidation_price_calculation_long(self):
        """Test long liquidation price calculation."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="Binance",
            short_exchange="Bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50000.0,
            position_size_usd=1000.0,
            leverage=10.0,
            long_funding_rate=0.0,
            short_funding_rate=0.0,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        # Long liquidation: price * (1 - 0.9/leverage)
        # 50000 * (1 - 0.9/10) = 50000 * 0.91 = 45500
        expected_long_liq = 50000.0 * (1 - 0.9 / 10.0)
        assert abs(position.long_liquidation_price - expected_long_liq) < 1.0

    def test_liquidation_price_calculation_short(self):
        """Test short liquidation price calculation."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="Binance",
            short_exchange="Bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50000.0,
            position_size_usd=1000.0,
            leverage=10.0,
            long_funding_rate=0.0,
            short_funding_rate=0.0,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        # Short liquidation: price * (1 + 0.9/leverage)
        # 50000 * (1 + 0.9/10) = 50000 * 1.09 = 54500
        expected_short_liq = 50000.0 * (1 + 0.9 / 10.0)
        assert abs(position.short_liquidation_price - expected_short_liq) < 1.0

    def test_liquidation_distance(self):
        """Test liquidation distance calculation."""
        position = Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="Binance",
            short_exchange="Bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50000.0,
            position_size_usd=1000.0,
            leverage=5.0,
            long_funding_rate=0.0,
            short_funding_rate=0.0,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

        # Current prices same as entry
        distance = position.get_liquidation_distance(50000.0, 50000.0)

        # Should be around 0.18 (18%) for 5x leverage
        assert 0.15 < distance < 0.20


class TestPortfolioEnhancements:
    """Test Portfolio class enhancements."""

    def create_test_portfolio(self):
        """Helper to create a test portfolio."""
        return Portfolio(
            initial_capital=10000.0,
            max_positions=5,
            max_position_size_pct=30.0
        )

    def create_test_position(self, leverage=1.0):
        """Helper to create a test position."""
        return Position(
            opportunity_id="test_1",
            symbol="BTCUSDT",
            long_exchange="Binance",
            short_exchange="Bybit",
            entry_time=pd.Timestamp.now(tz='UTC'),
            entry_long_price=50000.0,
            entry_short_price=50000.0,
            position_size_usd=1000.0,
            leverage=leverage,
            long_funding_rate=0.0001,
            short_funding_rate=-0.0001,
            long_funding_interval_hours=8,
            short_funding_interval_hours=8,
            long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
            short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
        )

    def test_get_total_margin_used_no_positions(self):
        """Test total margin with no positions."""
        portfolio = self.create_test_portfolio()
        assert portfolio.get_total_margin_used() == 0.0

    def test_get_total_margin_used_one_position(self):
        """Test total margin with one position."""
        portfolio = self.create_test_portfolio()
        position = self.create_test_position(leverage=5.0)

        portfolio.open_position(position)

        # Margin = 2000 / 5 = 400
        assert portfolio.get_total_margin_used() == 400.0

    def test_get_total_margin_used_multiple_positions(self):
        """Test total margin with multiple positions."""
        portfolio = self.create_test_portfolio()

        # Add 3 positions with different leverages
        for i, leverage in enumerate([2.0, 5.0, 10.0]):
            position = Position(
                opportunity_id=f"test_{i}",
                symbol="BTCUSDT",
                long_exchange="Binance",
                short_exchange="Bybit",
                entry_time=pd.Timestamp.now(tz='UTC'),
                entry_long_price=50000.0,
                entry_short_price=50000.0,
                position_size_usd=1000.0,
                leverage=leverage,
                long_funding_rate=0.0,
                short_funding_rate=0.0,
                long_funding_interval_hours=8,
                short_funding_interval_hours=8,
                long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
                short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
            )
            portfolio.open_position(position)

        # Total margin = 2000/2 + 2000/5 + 2000/10 = 1000 + 400 + 200 = 1600
        expected_margin = 1000.0 + 400.0 + 200.0
        assert abs(portfolio.get_total_margin_used() - expected_margin) < 0.01

    def test_available_margin(self):
        """Test available margin calculation."""
        portfolio = self.create_test_portfolio()
        position = self.create_test_position(leverage=5.0)

        portfolio.open_position(position)

        # Available margin = 10000 - 400 = 9600
        assert abs(portfolio.available_margin - 9600.0) < 0.01

    def test_margin_utilization(self):
        """Test margin utilization percentage."""
        portfolio = self.create_test_portfolio()
        position = self.create_test_position(leverage=5.0)

        portfolio.open_position(position)

        # Margin utilization = 400 / 10000 * 100 = 4%
        assert abs(portfolio.margin_utilization - 4.0) < 0.01

    def test_can_open_position_with_leverage(self):
        """Test position validation with leverage."""
        portfolio = self.create_test_portfolio()

        # With 5x leverage, need 400 margin for 1000 position
        # Should be able to open
        can_open = portfolio.can_open_position(1000.0, leverage=5.0)
        assert can_open is True

    def test_cannot_open_position_insufficient_margin(self):
        """Test position rejection with insufficient margin."""
        portfolio = Portfolio(
            initial_capital=500.0,  # Small capital
            max_positions=5,
            max_position_size_pct=50.0
        )

        # Need 2000 / 2 = 1000 margin, but only have 500
        can_open = portfolio.can_open_position(1000.0, leverage=2.0)
        assert can_open is False

    def test_get_min_liquidation_distance_no_positions(self):
        """Test min liquidation distance with no positions."""
        portfolio = self.create_test_portfolio()
        price_data = {}

        distance = portfolio.get_min_liquidation_distance(price_data)
        assert distance == 1.0  # No positions = no risk

    def test_get_min_liquidation_distance_one_position(self):
        """Test min liquidation distance with one position."""
        portfolio = self.create_test_portfolio()
        position = self.create_test_position(leverage=5.0)
        portfolio.open_position(position)

        price_data = {
            "BTCUSDT": {
                "long_price": 50000.0,
                "short_price": 50000.0
            }
        }

        distance = portfolio.get_min_liquidation_distance(price_data)
        assert 0.15 < distance < 0.20  # Should be ~18% for 5x leverage

    def test_get_execution_state_no_position(self):
        """Test execution state for empty slot."""
        portfolio = self.create_test_portfolio()
        price_data = {}

        state = portfolio.get_execution_state(0, price_data)

        assert state.shape == (12,)
        assert np.all(state == 0.0)  # All zeros for empty slot

    def test_get_execution_state_with_position(self):
        """Test execution state for active position."""
        portfolio = self.create_test_portfolio()
        position = self.create_test_position(leverage=5.0)
        portfolio.open_position(position)

        price_data = {
            "BTCUSDT": {
                "long_price": 50000.0,
                "short_price": 50000.0
            }
        }

        state = portfolio.get_execution_state(0, price_data)

        assert state.shape == (12,)
        assert state[0] == 1.0  # is_active
        # Other features should be non-zero or calculated correctly

    def test_get_all_execution_states(self):
        """Test getting all execution states."""
        portfolio = self.create_test_portfolio()

        # Add 2 positions
        for i in range(2):
            position = Position(
                opportunity_id=f"test_{i}",
                symbol="BTCUSDT",
                long_exchange="Binance",
                short_exchange="Bybit",
                entry_time=pd.Timestamp.now(tz='UTC'),
                entry_long_price=50000.0,
                entry_short_price=50000.0,
                position_size_usd=1000.0,
                leverage=5.0,
                long_funding_rate=0.0,
                short_funding_rate=0.0,
                long_funding_interval_hours=8,
                short_funding_interval_hours=8,
                long_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8),
                short_next_funding_time=pd.Timestamp.now(tz='UTC') + timedelta(hours=8)
            )
            portfolio.open_position(position)

        price_data = {
            "BTCUSDT": {
                "long_price": 50000.0,
                "short_price": 50000.0
            }
        }

        all_states = portfolio.get_all_execution_states(price_data, max_positions=5)

        # Should return 60 dimensions (5 slots × 12 features)
        assert all_states.shape == (60,)

        # First 2 slots should be active (feature 0 = 1.0)
        assert all_states[0] == 1.0  # Slot 0 active
        assert all_states[12] == 1.0  # Slot 1 active

        # Last 3 slots should be inactive (all zeros)
        assert np.all(all_states[24:36] == 0.0)  # Slot 2 inactive
        assert np.all(all_states[36:48] == 0.0)  # Slot 3 inactive
        assert np.all(all_states[48:60] == 0.0)  # Slot 4 inactive
