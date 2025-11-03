"""
Portfolio Management for RL Trading Environment

Tracks arbitrage executions (2 positions: long + short) with accurate P&L calculation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta


@dataclass
class Position:
    """
    Represents a single funding arbitrage EXECUTION (long + short positions).

    Each execution has:
    - Long position on one exchange
    - Short position on another exchange
    - Separate funding payments on each exchange
    - Entry and exit fees for both sides
    """

    # Identification
    opportunity_id: str
    symbol: str
    long_exchange: str
    short_exchange: str
    entry_time: pd.Timestamp

    # Entry prices and sizing
    entry_long_price: float
    entry_short_price: float
    position_size_usd: float  # Size per side (total capital used = 2x this)

    # Funding rates (per payment interval)
    long_funding_rate: float  # e.g., 0.0001 = 0.01%
    short_funding_rate: float
    long_funding_interval_hours: int  # 1 or 8
    short_funding_interval_hours: int
    long_next_funding_time: pd.Timestamp
    short_next_funding_time: pd.Timestamp

    # Fee structure (realistic Binance/Bybit levels)
    maker_fee: float = 0.0001  # 0.01% maker fee (reduced from 0.02%)
    taker_fee: float = 0.0002  # 0.02% taker fee (reduced from 0.05%)
    entry_fee_pct: float = field(init=False)  # Calculated from maker/taker
    exit_fee_pct: float = field(init=False)

    # State tracking
    hours_held: float = 0.0
    long_pnl_pct: float = 0.0  # Price P&L on long side
    short_pnl_pct: float = 0.0  # Price P&L on short side
    long_net_funding_usd: float = 0.0  # Cumulative net funding on long (positive = received, negative = paid)
    short_net_funding_usd: float = 0.0  # Cumulative net funding on short (positive = received, negative = paid)
    entry_fees_paid_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Funding tracking (last payment times)
    last_long_funding_time: Optional[pd.Timestamp] = None
    last_short_funding_time: Optional[pd.Timestamp] = None

    # Exit information
    exit_time: Optional[pd.Timestamp] = None
    exit_long_price: Optional[float] = None
    exit_short_price: Optional[float] = None
    exit_fees_paid_usd: float = 0.0
    realized_pnl_usd: Optional[float] = None
    realized_pnl_pct: Optional[float] = None

    def __post_init__(self):
        """Calculate entry fees after initialization."""
        # Entry fees (assume maker orders for limit entry)
        self.entry_fee_pct = self.maker_fee * 2  # Long + short
        self.entry_fees_paid_usd = self.position_size_usd * 2 * self.maker_fee

        # Exit fees (assume taker for market exit, calculated on close)
        self.exit_fee_pct = self.taker_fee * 2

        # Initial funding tracking
        self.last_long_funding_time = self.long_next_funding_time - timedelta(hours=self.long_funding_interval_hours)
        self.last_short_funding_time = self.short_next_funding_time - timedelta(hours=self.short_funding_interval_hours)

    def update_funding_rates(self, long_funding_rate: float, short_funding_rate: float):
        """
        Update the funding rates for this position.

        CRITICAL FIX: Funding rates change hourly in the market.
        We must update them from current market data, not freeze at entry!

        Args:
            long_funding_rate: Current funding rate for long side
            short_funding_rate: Current funding rate for short side
        """
        self.long_funding_rate = long_funding_rate
        self.short_funding_rate = short_funding_rate

    def update_hourly(self, current_time: pd.Timestamp,
                     current_long_price: float,
                     current_short_price: float) -> float:
        """
        Update position P&L for the current hour.

        Returns:
            Change in unrealized P&L (in USD) since last update
        """
        previous_pnl_usd = self.unrealized_pnl_usd

        # Update time held
        self.hours_held = (current_time - self.entry_time).total_seconds() / 3600

        # === LONG POSITION P&L ===
        # Price change on long (profit if price goes up)
        long_price_change_pct = ((current_long_price - self.entry_long_price) /
                                 self.entry_long_price)
        long_price_pnl_usd = self.position_size_usd * long_price_change_pct

        # === SHORT POSITION P&L ===
        # Price change on short (profit if price goes down)
        short_price_change_pct = ((self.entry_short_price - current_short_price) /
                                  self.entry_short_price)
        short_price_pnl_usd = self.position_size_usd * short_price_change_pct

        # === FUNDING PAYMENTS ===
        # Check if funding time(s) passed and process payments
        long_funding_this_step = self._process_long_funding(current_time)
        short_funding_this_step = self._process_short_funding(current_time)

        # === TOTAL P&L ===
        # Net P&L = Long price P&L + Short price P&L + Funding - Entry fees
        # New format: funding values are already net (positive = received, negative = paid)
        net_funding_usd = self.long_net_funding_usd + self.short_net_funding_usd

        self.unrealized_pnl_usd = (
            long_price_pnl_usd +
            short_price_pnl_usd +
            net_funding_usd -
            self.entry_fees_paid_usd
        )

        # Calculate percentage (relative to total capital used = 2x position size)
        total_capital_used = self.position_size_usd * 2
        if total_capital_used > 0:
            self.unrealized_pnl_pct = (self.unrealized_pnl_usd / total_capital_used) * 100
            self.long_pnl_pct = (long_price_pnl_usd / self.position_size_usd) * 100
            self.short_pnl_pct = (short_price_pnl_usd / self.position_size_usd) * 100
        else:
            # Position size is zero (shouldn't happen, but safety check)
            self.unrealized_pnl_pct = 0.0
            self.long_pnl_pct = 0.0
            self.short_pnl_pct = 0.0

        # Return change in P&L for reward calculation
        pnl_change_usd = self.unrealized_pnl_usd - previous_pnl_usd

        return pnl_change_usd

    def _process_long_funding(self, current_time: pd.Timestamp) -> float:
        """
        Process funding payments on long position.

        Returns:
            Net funding for this step on long side (positive = received, negative = paid)
        """
        funding_this_step = 0.0

        # Check if funding time has passed
        while current_time >= self.long_next_funding_time:
            # Calculate net funding (negative rate means we receive, positive means we pay)
            # Since we're LONG: positive rate = we pay (negative value), negative rate = we receive (positive value)
            # So we need to negate to match our convention (positive = received)
            funding_payment_pct = self.long_funding_rate
            funding_payment_usd = self.position_size_usd * funding_payment_pct

            # Store as net funding (negate so positive = received, negative = paid)
            net_funding_usd = -funding_payment_usd

            funding_this_step += net_funding_usd
            self.long_net_funding_usd += net_funding_usd

            # Update next funding time
            self.last_long_funding_time = self.long_next_funding_time
            self.long_next_funding_time += timedelta(hours=self.long_funding_interval_hours)

        return funding_this_step

    def _process_short_funding(self, current_time: pd.Timestamp) -> float:
        """
        Process funding payments on short position.

        Returns:
            Net funding for this step on short side (positive = received, negative = paid)
        """
        funding_this_step = 0.0

        # Check if funding time has passed
        while current_time >= self.short_next_funding_time:
            # For SHORT positions in perpetual futures:
            # - Positive funding rate → longs pay shorts → we RECEIVE (profit)
            # - Negative funding rate → shorts pay longs → we PAY (cost)
            # Therefore: short_net_funding = position_size * rate (direct, no negation needed)
            funding_payment_pct = self.short_funding_rate
            net_funding_usd = self.position_size_usd * funding_payment_pct

            funding_this_step += net_funding_usd
            self.short_net_funding_usd += net_funding_usd

            # Update next funding time
            self.last_short_funding_time = self.short_next_funding_time
            self.short_next_funding_time += timedelta(hours=self.short_funding_interval_hours)

        return funding_this_step

    def close(self, exit_time: pd.Timestamp,
             exit_long_price: float,
             exit_short_price: float) -> float:
        """
        Close the position and calculate realized P&L.

        Returns:
            Realized P&L in USD
        """
        self.exit_time = exit_time
        self.exit_long_price = exit_long_price
        self.exit_short_price = exit_short_price

        # Final update to get latest P&L and funding
        self.update_hourly(exit_time, exit_long_price, exit_short_price)

        # Calculate exit fees (market orders = taker fees)
        self.exit_fees_paid_usd = self.position_size_usd * 2 * self.taker_fee

        # Realized P&L = Unrealized P&L - Exit fees
        self.realized_pnl_usd = self.unrealized_pnl_usd - self.exit_fees_paid_usd

        # Percentage relative to total capital used
        total_capital_used = self.position_size_usd * 2
        if total_capital_used > 0:
            self.realized_pnl_pct = (self.realized_pnl_usd / total_capital_used) * 100
        else:
            self.realized_pnl_pct = 0.0

        return self.realized_pnl_usd

    def get_breakdown(self) -> Dict:
        """Get detailed P&L breakdown for analysis."""
        total_capital = self.position_size_usd * 2

        # Safe division helper
        if total_capital > 0:
            net_funding_pct = ((self.long_net_funding_usd + self.short_net_funding_usd) / total_capital) * 100
            total_fees_pct = ((self.entry_fees_paid_usd + self.exit_fees_paid_usd) / total_capital) * 100
        else:
            net_funding_pct = 0.0
            total_fees_pct = 0.0

        return {
            'symbol': self.symbol,
            'hours_held': self.hours_held,

            # Component P&Ls
            'long_price_pnl_pct': self.long_pnl_pct,
            'short_price_pnl_pct': self.short_pnl_pct,
            'long_net_funding_usd': self.long_net_funding_usd,
            'short_net_funding_usd': self.short_net_funding_usd,
            'net_funding_usd': self.long_net_funding_usd + self.short_net_funding_usd,
            'net_funding_pct': net_funding_pct,

            # Fees
            'entry_fees_usd': self.entry_fees_paid_usd,
            'exit_fees_usd': self.exit_fees_paid_usd,
            'total_fees_usd': self.entry_fees_paid_usd + self.exit_fees_paid_usd,
            'total_fees_pct': total_fees_pct,

            # Total P&L
            'unrealized_pnl_usd': self.unrealized_pnl_usd,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl_usd': self.realized_pnl_usd,
            'realized_pnl_pct': self.realized_pnl_pct,
        }

    def to_dict(self) -> Dict:
        """Convert position to dictionary for logging."""
        return {
            'opportunity_id': self.opportunity_id,
            'symbol': self.symbol,
            'long_exchange': self.long_exchange,
            'short_exchange': self.short_exchange,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'hours_held': self.hours_held,
            'position_size_usd': self.position_size_usd,
            'unrealized_pnl_usd': self.unrealized_pnl_usd,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'realized_pnl_usd': self.realized_pnl_usd,
            'realized_pnl_pct': self.realized_pnl_pct,
        }


class Portfolio:
    """
    Manages multiple arbitrage executions and tracks overall portfolio P&L.

    Enforces position limits and capital constraints.
    """

    def __init__(self,
                 initial_capital: float = 10000.0,
                 max_positions: int = 3,
                 max_position_size_pct: float = 33.3):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital in USD
            max_positions: Maximum number of concurrent executions
            max_position_size_pct: Max % of capital per side (total used = 2x)
        """
        self.initial_capital = initial_capital
        self.total_capital = initial_capital
        self.max_positions = max_positions
        self.max_position_size_pct = max_position_size_pct

        # Position tracking
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []

        # Performance tracking
        self.total_pnl_usd = 0.0
        self.total_pnl_pct = 0.0
        self.peak_capital = initial_capital
        self.max_drawdown_pct = 0.0

        # Fee and funding tracking
        self.total_fees_paid_usd = 0.0
        self.total_funding_net_usd = 0.0

    @property
    def available_capital(self) -> float:
        """Capital not allocated to open positions."""
        # Each position uses 2x position_size_usd (long + short)
        allocated = sum(pos.position_size_usd * 2 for pos in self.positions)
        return self.total_capital - allocated

    @property
    def capital_utilization(self) -> float:
        """Percentage of capital currently in use."""
        if self.total_capital == 0:
            return 0.0
        allocated = sum(pos.position_size_usd * 2 for pos in self.positions)
        return (allocated / self.total_capital) * 100

    @property
    def unrealized_pnl_usd(self) -> float:
        """Total unrealized P&L across all open positions."""
        return sum(pos.unrealized_pnl_usd for pos in self.positions)

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value (capital + unrealized P&L)."""
        return self.total_capital + self.unrealized_pnl_usd

    def get_execution_avg_pnl_pct(self) -> float:
        """
        Get weighted average P&L across all closed executions.
        P&L is based on capital used per execution (position_size_usd * 2).

        This is the capital-independent metric that measures execution quality,
        not portfolio impact.
        """
        if len(self.closed_positions) == 0:
            return 0.0

        total_capital_used = sum(pos.position_size_usd * 2 for pos in self.closed_positions)
        if total_capital_used == 0:
            return 0.0

        # Weighted average by capital used in each execution
        weighted_pnl = sum(pos.realized_pnl_pct * (pos.position_size_usd * 2)
                           for pos in self.closed_positions)

        return weighted_pnl / total_capital_used

    def can_open_position(self, size_per_side_usd: float) -> bool:
        """
        Check if we can open a new arbitrage execution.

        Args:
            size_per_side_usd: Position size per side (total = 2x this)
        """
        # Position limit
        if len(self.positions) >= self.max_positions:
            return False

        # Capital availability (need 2x for long + short)
        total_size_needed = size_per_side_usd * 2
        if total_size_needed > self.available_capital:
            return False

        # Position size limit per side
        max_size_per_side = self.total_capital * (self.max_position_size_pct / 100)
        if size_per_side_usd > max_size_per_side:
            return False

        return True

    def open_position(self, position: Position) -> bool:
        """
        Open a new arbitrage execution.

        Returns:
            True if position opened successfully, False otherwise
        """
        if not self.can_open_position(position.position_size_usd):
            return False

        self.positions.append(position)
        return True

    def close_position(self, position_idx: int,
                      exit_time: pd.Timestamp,
                      exit_long_price: float,
                      exit_short_price: float) -> float:
        """
        Close an arbitrage execution and realize P&L.

        Returns:
            Realized P&L in USD
        """
        if position_idx >= len(self.positions):
            raise ValueError(f"Invalid position index: {position_idx}")

        position = self.positions.pop(position_idx)

        # Close position and get realized P&L
        realized_pnl_usd = position.close(exit_time, exit_long_price, exit_short_price)

        # Update portfolio capital
        # Return capital (2x position size) + P&L
        capital_returned = (position.position_size_usd * 2) + realized_pnl_usd
        self.total_capital += realized_pnl_usd  # Only add P&L, capital already tracked

        self.total_pnl_usd += realized_pnl_usd
        self.total_pnl_pct = ((self.total_capital - self.initial_capital) /
                             self.initial_capital) * 100

        # Track fees and funding
        self.total_fees_paid_usd += position.entry_fees_paid_usd + position.exit_fees_paid_usd
        self.total_funding_net_usd += (position.long_net_funding_usd +
                                       position.short_net_funding_usd)

        # Track drawdown
        self.peak_capital = max(self.peak_capital, self.total_capital)
        current_drawdown_pct = ((self.peak_capital - self.total_capital) /
                               self.peak_capital) * 100
        self.max_drawdown_pct = max(self.max_drawdown_pct, current_drawdown_pct)

        # Store closed position
        self.closed_positions.append(position)

        return realized_pnl_usd

    def update_positions_hourly(self, current_time: pd.Timestamp,
                               price_data: Dict[str, Dict[str, float]],
                               funding_rates: Optional[Dict[str, Dict[str, float]]] = None) -> float:
        """
        Update all positions with current prices and funding rates.

        Args:
            current_time: Current timestamp
            price_data: Dict mapping symbol to {'long_price': float, 'short_price': float}
            funding_rates: Dict mapping symbol to {'long_rate': float, 'short_rate': float}
                          CRITICAL FIX: Must be provided to update funding rates hourly!

        Returns:
            Total P&L change across all positions (for reward calculation)
        """
        total_pnl_change_usd = 0.0

        for position in self.positions:
            if position.symbol in price_data:
                prices = price_data[position.symbol]

                # CRITICAL FIX: Update funding rates from current market data
                if funding_rates and position.symbol in funding_rates:
                    rates = funding_rates[position.symbol]
                    position.update_funding_rates(
                        rates['long_rate'],
                        rates['short_rate']
                    )

                pnl_change_usd = position.update_hourly(
                    current_time,
                    prices['long_price'],
                    prices['short_price']
                )
                total_pnl_change_usd += pnl_change_usd

        return total_pnl_change_usd

    def get_state_features(self) -> np.ndarray:
        """
        Get portfolio state as feature vector for RL agent.

        Returns:
            Array of portfolio and position features
        """
        features = [
            self.total_capital / self.initial_capital,  # Capital ratio
            self.capital_utilization / 100,  # Utilization ratio
            len(self.positions) / self.max_positions,  # Position count ratio
            self.total_pnl_pct / 100,  # Total P&L ratio
            self.max_drawdown_pct / 100,  # Drawdown ratio
        ]

        # Add individual position P&L (pad to max_positions)
        for i in range(self.max_positions):
            if i < len(self.positions):
                pos = self.positions[i]
                features.append(pos.unrealized_pnl_pct / 100)
                features.append(pos.hours_held / 72.0)  # Normalize to 72h max
                # Net funding ratio (safeguard against zero division)
                total_capital = pos.position_size_usd * 2
                if total_capital > 0:
                    net_funding_ratio = (pos.long_net_funding_usd + pos.short_net_funding_usd) / total_capital
                else:
                    net_funding_ratio = 0.0
                features.append(net_funding_ratio)
            else:
                features.append(0.0)  # No position
                features.append(0.0)
                features.append(0.0)

        return np.array(features, dtype=np.float32)

    def get_summary(self) -> Dict:
        """Get portfolio summary statistics."""
        return {
            'total_capital': self.total_capital,
            'initial_capital': self.initial_capital,
            'total_pnl_usd': self.total_pnl_usd,
            'total_pnl_pct': self.total_pnl_pct,
            'portfolio_value': self.portfolio_value,
            'available_capital': self.available_capital,
            'capital_utilization': self.capital_utilization,
            'num_open_positions': len(self.positions),
            'num_closed_positions': len(self.closed_positions),
            'unrealized_pnl_usd': self.unrealized_pnl_usd,
            'max_drawdown_pct': self.max_drawdown_pct,
            'peak_capital': self.peak_capital,
            'total_fees_paid_usd': self.total_fees_paid_usd,
            'total_funding_net_usd': self.total_funding_net_usd,
        }
