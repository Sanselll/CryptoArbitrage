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
    - Leverage and margin tracking
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

    # Leverage and fees (with defaults)
    leverage: float = 1.0      # Leverage multiplier (1-10x)
    maker_fee: float = 0.0002  # 0.02% maker fee (realistic rate)
    taker_fee: float = 0.00055  # 0.055% taker fee (Bybit rate)
    slippage_pct: float = 0.0015  # 0.15% slippage per side
    entry_fee_pct: float = field(init=False)  # Calculated from maker/taker
    exit_fee_pct: float = field(init=False)

    # Margin and liquidation tracking
    margin_used_usd: float = field(init=False)  # Actual margin locked
    long_liquidation_price: float = field(init=False)
    short_liquidation_price: float = field(init=False)

    # State tracking
    hours_held: float = 0.0
    long_pnl_pct: float = 0.0  # Price P&L on long side
    short_pnl_pct: float = 0.0  # Price P&L on short side
    long_net_funding_usd: float = 0.0  # Cumulative net funding on long (positive = received, negative = paid)
    short_net_funding_usd: float = 0.0  # Cumulative net funding on short (positive = received, negative = paid)
    long_funding_payment_count: int = 0  # Number of funding payments on long side
    short_funding_payment_count: int = 0  # Number of funding payments on short side
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

    # NEW: Peak tracking for exit timing (Phase 1 - Observation Features)
    peak_pnl_pct: float = 0.0  # Highest P&L % achieved
    peak_pnl_time: Optional[pd.Timestamp] = None  # When peak occurred
    pnl_history: List[float] = field(default_factory=list)  # Last 6 hours of P&L % for velocity
    entry_apr: float = 0.0  # Annualized profit rate from opportunity (fund_apr)

    # V3: Velocity tracking for new features (RL-v3 refactoring)
    prev_estimated_pnl_pct: float = 0.0  # Previous estimated P&L (price only, no fees)
    prev_estimated_funding_8h_pct: float = 0.0  # Previous 8h funding estimate
    prev_spread_pct: float = 0.0  # Previous spread percentage

    def to_trade_record(self) -> dict:
        """Convert position to trade record for CSV logging."""
        total_funding_usd = self.long_net_funding_usd + self.short_net_funding_usd
        total_fees_usd = self.entry_fees_paid_usd + self.exit_fees_paid_usd

        return {
            'entry_datetime': self.entry_time,
            'exit_datetime': self.exit_time if self.exit_time else None,
            'symbol': self.symbol,
            'long_exchange': self.long_exchange,
            'short_exchange': self.short_exchange,
            'position_size_usd': self.position_size_usd,
            'leverage': self.leverage,
            'entry_long_price': self.entry_long_price,
            'entry_short_price': self.entry_short_price,
            'exit_long_price': self.exit_long_price,
            'exit_short_price': self.exit_short_price,
            'long_funding_rate': self.long_funding_rate,
            'short_funding_rate': self.short_funding_rate,
            'funding_earned_usd': total_funding_usd,
            'long_funding_earned_usd': self.long_net_funding_usd,
            'short_funding_earned_usd': self.short_net_funding_usd,
            'entry_fees_usd': self.entry_fees_paid_usd,
            'exit_fees_usd': self.exit_fees_paid_usd,
            'total_fees_usd': total_fees_usd,
            'realized_pnl_usd': self.realized_pnl_usd,
            'realized_pnl_pct': self.realized_pnl_pct,
            'unrealized_pnl_usd': self.unrealized_pnl_usd,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'hours_held': self.hours_held,
            'margin_used_usd': self.margin_used_usd,
            'status': 'closed' if self.exit_time else 'open',
        }

    def __post_init__(self):
        """Calculate entry fees, margin, and liquidation prices after initialization."""
        # Entry fees (assume maker orders for limit entry)
        self.entry_fee_pct = self.maker_fee * 2  # Long + short
        self.entry_fees_paid_usd = self.position_size_usd * 2 * self.maker_fee

        # Exit fees (assume taker for market exit, calculated on close)
        self.exit_fee_pct = self.taker_fee * 2

        # Calculate margin required (total for both long + short)
        # Margin = (position_size_long + position_size_short) / leverage
        # For cross-exchange: both sides use position_size_usd
        self.margin_used_usd = (self.position_size_usd * 2) / self.leverage

        # Calculate liquidation prices
        # Using simplified formula (maintenance margin ~10% of position value)
        # Liquidation occurs when unrealized loss reaches ~90% of margin

        # Long liquidation: price drops by leverage factor
        # Approximation: liq_price = entry_price * (1 - 0.9/leverage)
        self.long_liquidation_price = self.entry_long_price * (1 - 0.9 / self.leverage)

        # Short liquidation: price rises by leverage factor
        # Approximation: liq_price = entry_price * (1 + 0.9/leverage)
        self.short_liquidation_price = self.entry_short_price * (1 + 0.9 / self.leverage)

        # Initial funding tracking
        self.last_long_funding_time = self.long_next_funding_time - timedelta(hours=self.long_funding_interval_hours)
        self.last_short_funding_time = self.short_next_funding_time - timedelta(hours=self.short_funding_interval_hours)

        # CRITICAL FIX: Skip funding times at or before entry time
        # You can only earn funding for periods you actually held the position
        # Example: Enter at 20:00, next funding at 20:00 → skip to 04:00 (8h interval)
        #          The 20:00 funding is for positions held from 12:00-20:00, which we didn't hold
        while self.long_next_funding_time <= self.entry_time:
            self.last_long_funding_time = self.long_next_funding_time
            self.long_next_funding_time += timedelta(hours=self.long_funding_interval_hours)

        while self.short_next_funding_time <= self.entry_time:
            self.last_short_funding_time = self.short_next_funding_time
            self.short_next_funding_time += timedelta(hours=self.short_funding_interval_hours)

        # NEW: entry_apr is now passed directly from opportunity (fund_apr field)
        # No calculation needed - APR comes from opportunity detection

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

        # === LONG POSITION P&L (unrealized - NO slippage) ===
        # Slippage is only applied at actual exit in close_position()
        # Price change on long (profit if price goes up)
        long_price_change_pct = ((current_long_price - self.entry_long_price) /
                                 self.entry_long_price)
        long_price_pnl_usd = self.position_size_usd * long_price_change_pct

        # === SHORT POSITION P&L (unrealized - NO slippage) ===
        # Price change on short (profit if price goes down)
        short_price_change_pct = ((self.entry_short_price - current_short_price) /
                                  self.entry_short_price)
        short_price_pnl_usd = self.position_size_usd * short_price_change_pct

        # === FUNDING PAYMENTS ===
        # Check if funding time(s) passed and process payments
        long_funding_this_step = self._process_long_funding(current_time)
        short_funding_this_step = self._process_short_funding(current_time)

        # === TOTAL P&L ===
        # Net P&L = Long price P&L + Short price P&L + Funding - Entry fees - Exit fees
        # This shows the "true" P&L if position were closed now
        # NOTE: Slippage is NOT included here to match production behavior
        #       Slippage is only applied at actual close in close() method
        # New format: funding values are already net (positive = received, negative = paid)
        net_funding_usd = self.long_net_funding_usd + self.short_net_funding_usd

        # Exit costs (fees only, no slippage) - estimated if position were closed now
        estimated_exit_fees_usd = self.position_size_usd * 2 * self.taker_fee

        self.unrealized_pnl_usd = (
            long_price_pnl_usd +
            short_price_pnl_usd +
            net_funding_usd -
            self.entry_fees_paid_usd -
            estimated_exit_fees_usd
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

        # NEW: Track P&L history for velocity calculation (Phase 1 - Observation Features)
        # Keep last 6 hours of P&L percentages
        self.pnl_history.append(self.unrealized_pnl_pct)
        if len(self.pnl_history) > 6:
            self.pnl_history.pop(0)  # Remove oldest

        # NEW: Track peak P&L (Phase 1 - Observation Features)
        if self.unrealized_pnl_pct > self.peak_pnl_pct:
            self.peak_pnl_pct = self.unrealized_pnl_pct
            self.peak_pnl_time = current_time

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
            self.long_funding_payment_count += 1

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
            self.short_funding_payment_count += 1

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

        # Record exit fees for logging (already included in unrealized P&L)
        self.exit_fees_paid_usd = self.position_size_usd * 2 * self.taker_fee

        # Calculate slippage cost at actual close
        slippage_usd = self.position_size_usd * 2 * self.slippage_pct

        # Realized P&L = Unrealized P&L - Slippage (slippage only applied at close)
        self.realized_pnl_usd = self.unrealized_pnl_usd - slippage_usd

        # Percentage relative to total capital used
        total_capital_used = self.position_size_usd * 2
        if total_capital_used > 0:
            self.realized_pnl_pct = (self.realized_pnl_usd / total_capital_used) * 100
        else:
            self.realized_pnl_pct = 0.0

        return self.realized_pnl_usd

    def get_liquidation_distance(self, current_long_price: float, current_short_price: float) -> float:
        """
        Calculate distance to liquidation as percentage.

        Returns the minimum distance across both long and short sides.
        Lower values indicate higher risk.

        Args:
            current_long_price: Current price on long exchange
            current_short_price: Current price on short exchange

        Returns:
            Minimum liquidation distance percentage (0-1, where 0.15 = 15%)
        """
        # Long side: distance to liquidation price
        # If current < liq, we're underwater (distance is negative or zero)
        long_distance = abs(current_long_price - self.long_liquidation_price) / current_long_price

        # Short side: distance to liquidation price
        # If current > liq, we're underwater
        short_distance = abs(self.short_liquidation_price - current_short_price) / current_short_price

        # Return minimum (most dangerous side)
        return min(long_distance, short_distance)

    # NEW: Phase 1 - Observation Features for Exit Timing
    def get_pnl_velocity(self) -> float:
        """Calculate hourly P&L change rate from history.

        Returns:
            P&L velocity in percentage points per hour (e.g., -0.5 means declining at 0.5% per hour)
        """
        if len(self.pnl_history) < 2:
            return 0.0
        # Linear change: (latest - earliest) / hours_elapsed
        hours_elapsed = len(self.pnl_history) - 1
        return (self.pnl_history[-1] - self.pnl_history[0]) / hours_elapsed

    def get_peak_drawdown(self) -> float:
        """Calculate percentage decline from peak P&L.

        Returns:
            Drawdown as a ratio (0-1), where 0.3 means current P&L is 30% below peak
        """
        if self.peak_pnl_pct <= 0:
            return 0.0  # No peak yet (never been profitable)
        drawdown = (self.peak_pnl_pct - self.unrealized_pnl_pct) / self.peak_pnl_pct
        return max(0.0, drawdown)  # Clamp to 0 if above peak

    def get_apr_ratio(self, current_apr: float) -> float:
        """Calculate current APR / entry APR to detect funding rate deterioration.

        Args:
            current_apr: Current APR for this symbol (looked up from opportunities)

        Returns:
            Ratio (e.g., 0.5 means current APR is half of entry APR)
        """
        if self.entry_apr == 0:
            return 1.0  # Neutral if no entry APR

        # Use the looked-up current APR from opportunities
        return current_apr / self.entry_apr

    def get_return_efficiency(self) -> float:
        """Calculate P&L percentage per hour held (age-adjusted performance).

        Returns:
            P&L efficiency in percentage points per hour (e.g., 0.1 means earning 0.1% per hour)
        """
        if self.hours_held == 0:
            return 0.0
        return self.unrealized_pnl_pct / self.hours_held

    def is_old_loser(self, age_threshold_hours: float = 48.0) -> bool:
        """Check if position is old AND losing (for conditional age penalty).

        Args:
            age_threshold_hours: Age threshold in hours (default 48h)

        Returns:
            True if position is losing AND held longer than threshold
        """
        return self.unrealized_pnl_pct < 0 and self.hours_held > age_threshold_hours

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

    def calculate_current_apr(self) -> float:
        """
        Calculate current APR from the position's current funding rates.

        Since funding rates are updated dynamically at each timestep, this returns
        the APR the position is currently earning based on the latest market rates.

        Returns:
            float: Annualized APR in percentage (e.g., 15.5 for 15.5% APR)
        """
        # Calculate net funding rate (short pays, long receives, so net = short - long)
        net_funding_rate_per_interval = self.short_funding_rate - self.long_funding_rate

        # Calculate weighted average interval if different
        if self.long_funding_interval_hours == self.short_funding_interval_hours:
            avg_interval_hours = self.long_funding_interval_hours
        else:
            # Weight by absolute funding rate magnitude
            long_weight = abs(self.long_funding_rate)
            short_weight = abs(self.short_funding_rate)
            total_weight = long_weight + short_weight + 1e-9  # Avoid division by zero

            avg_interval_hours = (
                (long_weight * self.long_funding_interval_hours +
                 short_weight * self.short_funding_interval_hours) /
                total_weight
            )

        # Calculate annualized rate
        payments_per_year = (365 * 24) / avg_interval_hours
        annual_rate = net_funding_rate_per_interval * payments_per_year

        return annual_rate * 100  # Convert to percentage

    def calculate_estimated_funding_8h_pct(self) -> float:
        """
        Calculate expected net funding profit in next 8 hours (V3 feature).

        Matches Position._process_long_funding and _process_short_funding logic.
        See RL_FEATURE_REFACTORING_PLAN.txt Appendix A for detailed explanation.

        Returns:
            float: Percentage profit expected in 8h (e.g., 0.5 means 0.5% profit in 8h)
        """
        # Calculate number of funding payments in 8 hours
        long_payments_8h = 8.0 / self.long_funding_interval_hours
        short_payments_8h = 8.0 / self.short_funding_interval_hours

        # Long: negative rate = receive, positive rate = pay (portfolio.py line 269)
        # From _process_long_funding: net_funding_usd = -funding_payment_usd
        long_funding_8h = -self.long_funding_rate * long_payments_8h

        # Short: positive rate = receive, negative rate = pay (portfolio.py line 296)
        # From _process_short_funding: net_funding_usd = position_size * rate (direct, no negation)
        short_funding_8h = self.short_funding_rate * short_payments_8h

        # Net funding as decimal, convert to percentage
        estimated_funding_8h_pct = (long_funding_8h + short_funding_8h) * 100

        return estimated_funding_8h_pct

    def get_funding_summary(self) -> dict:
        """
        Get detailed funding summary for this position.

        Returns:
            dict with funding details for logging/analysis
        """
        total_capital = self.position_size_usd * 2

        return {
            'symbol': self.symbol,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'hours_held': self.hours_held,
            'entry_apr': self.entry_apr,

            # Long side funding
            'long_funding_total_usd': self.long_net_funding_usd,
            'long_funding_payment_count': self.long_funding_payment_count,
            'long_funding_interval_hours': self.long_funding_interval_hours,
            'long_avg_payment_usd': self.long_net_funding_usd / self.long_funding_payment_count if self.long_funding_payment_count > 0 else 0.0,

            # Short side funding
            'short_funding_total_usd': self.short_net_funding_usd,
            'short_funding_payment_count': self.short_funding_payment_count,
            'short_funding_interval_hours': self.short_funding_interval_hours,
            'short_avg_payment_usd': self.short_net_funding_usd / self.short_funding_payment_count if self.short_funding_payment_count > 0 else 0.0,

            # Net funding
            'net_funding_usd': self.long_net_funding_usd + self.short_net_funding_usd,
            'net_funding_pct': ((self.long_net_funding_usd + self.short_net_funding_usd) / total_capital * 100) if total_capital > 0 else 0.0,

            # Price P&L
            'long_price_pnl_usd': self.position_size_usd * (self.long_pnl_pct / 100),
            'short_price_pnl_usd': self.position_size_usd * (self.short_pnl_pct / 100),
            'total_price_pnl_usd': self.position_size_usd * (self.long_pnl_pct + self.short_pnl_pct) / 100,

            # Fees
            'entry_fees_usd': self.entry_fees_paid_usd,
            'exit_fees_usd': self.exit_fees_paid_usd,
            'total_fees_usd': self.entry_fees_paid_usd + self.exit_fees_paid_usd,

            # Total P&L
            'realized_pnl_usd': self.realized_pnl_usd if self.realized_pnl_usd is not None else self.unrealized_pnl_usd,
            'realized_pnl_pct': self.realized_pnl_pct if self.realized_pnl_pct is not None else self.unrealized_pnl_pct,
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
        """Capital available for new positions.

        Since margin is deducted from total_capital when opening positions,
        this now returns total_capital directly.
        """
        return self.total_capital

    @property
    def available_margin(self) -> float:
        """
        Margin available for new positions.

        Since margin is deducted from total_capital when opening positions,
        this now returns total_capital directly.
        """
        return self.total_capital

    @property
    def capital_utilization(self) -> float:
        """Percentage of initial capital currently in use (as margin)."""
        if self.initial_capital == 0:
            return 0.0
        margin_used = self.get_total_margin_used()
        return (margin_used / self.initial_capital) * 100

    @property
    def margin_utilization(self) -> float:
        """Percentage of initial capital locked as margin."""
        if self.initial_capital == 0:
            return 0.0
        margin_used = self.get_total_margin_used()
        return (margin_used / self.initial_capital) * 100

    @property
    def unrealized_pnl_usd(self) -> float:
        """Total unrealized P&L across all open positions."""
        return sum(pos.unrealized_pnl_usd for pos in self.positions)

    @property
    def portfolio_value(self) -> float:
        """Current portfolio value (capital + unrealized P&L)."""
        return self.total_capital + self.unrealized_pnl_usd

    def get_total_margin_used(self) -> float:
        """
        Get total margin locked across all open positions.

        Returns:
            Total margin in USD
        """
        return sum(pos.margin_used_usd for pos in self.positions)

    def get_min_liquidation_distance(self, price_data: Dict[str, Dict[str, float]]) -> float:
        """
        Get minimum liquidation distance across all open positions.

        Args:
            price_data: Dict mapping symbol to {'long_price': float, 'short_price': float}

        Returns:
            Minimum liquidation distance (0-1), or 1.0 if no positions
        """
        if len(self.positions) == 0:
            return 1.0  # No positions, no liquidation risk

        min_distance = 1.0
        for pos in self.positions:
            if pos.symbol in price_data:
                prices = price_data[pos.symbol]
                distance = pos.get_liquidation_distance(
                    prices['long_price'],
                    prices['short_price']
                )
                min_distance = min(min_distance, distance)

        return min_distance

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

    def get_total_funding_usd(self) -> float:
        """
        Get total funding earned across all open positions (USD).

        This is used to separate funding P&L from price P&L for reward calculation.
        Funding fees are the real profit source in arbitrage trading (paid every 1-8h).

        Returns:
            Total net funding earned in USD (sum of long + short funding for all positions)
        """
        return sum(
            pos.long_net_funding_usd + pos.short_net_funding_usd
            for pos in self.positions
        )

    def get_execution_state(self, exec_idx: int, price_data: Dict[str, Dict[str, float]],
                           best_available_apr: float = 0.0,
                           current_opportunities: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Get state features for a single execution slot (17 dimensions - V5.4).

        V5.4 Changes (203→213 total dims):
        - Execution features unchanged (17 per slot)
        - spread_change_from_entry now active (was disabled, always 0.0)
        - Added opportunity feature: spread_mean_reversion_potential (12 per slot)

        V3 Changes (301→203 total dims):
        - Removed: net_funding_ratio, net_funding_rate, funding_efficiency, entry_spread_pct,
                   long/short_pnl_pct, old pnl_velocity, peak_drawdown, is_old_loser
        - Added: estimated_pnl_pct, estimated_pnl_velocity, estimated_funding_8h_pct,
                 funding_velocity, spread_change_from_entry, pnl_imbalance
        - Updated: hours_held (log norm), APR (clip ±5000%)

        Args:
            exec_idx: Execution index (0-4)
            price_data: Current price data for all symbols
            best_available_apr: Maximum APR among current opportunities (for comparison)
            current_opportunities: List of current market opportunities to look up APR by symbol

        Returns:
            17-dimensional array of execution features
        """
        if exec_idx >= len(self.positions):
            # No position in this slot - return zeros
            return np.zeros(17, dtype=np.float32)

        pos = self.positions[exec_idx]

        # Get current prices for this position
        if pos.symbol in price_data:
            prices = price_data[pos.symbol]
            current_long_price = prices['long_price']
            current_short_price = prices['short_price']
        else:
            # Fallback to entry prices
            current_long_price = pos.entry_long_price
            current_short_price = pos.entry_short_price

        # Calculate base metrics
        total_capital = pos.position_size_usd * 2

        # ==================================================================
        # V3 FEATURES (17 dimensions)
        # ==================================================================

        # 1. is_active
        is_active = 1.0

        # 2. net_pnl_pct = (price P&L + funding - fees) / capital
        net_pnl_pct = pos.unrealized_pnl_pct / 100

        # 3. hours_held_norm = log(hours + 1) / log(73)
        # V3: Changed from linear /72 to log normalization
        hours_held_norm = np.log(pos.hours_held + 1) / np.log(73)

        # 4. estimated_pnl_pct = price P&L only (no fees, no funding, no slippage)
        # V3: NEW FEATURE - isolates price risk from funding profit
        # Slippage is only applied at actual exit, not in estimates
        long_price_pnl = pos.position_size_usd * ((current_long_price - pos.entry_long_price) / pos.entry_long_price)
        short_price_pnl = pos.position_size_usd * ((pos.entry_short_price - current_short_price) / pos.entry_short_price)

        estimated_pnl_pct = ((long_price_pnl + short_price_pnl) / total_capital) / 100 if total_capital > 0 else 0.0

        # 5. estimated_pnl_velocity = change in estimated P&L
        # V3: DISABLED - removed to eliminate state tracking complexity
        estimated_pnl_velocity = 0.0  # (estimated_pnl_pct - pos.prev_estimated_pnl_pct) / 100

        # 6. estimated_funding_8h_pct = expected funding profit in next 8h
        # V3: NEW FEATURE - replaces confusing net_funding_rate
        estimated_funding_8h_pct = pos.calculate_estimated_funding_8h_pct() / 100

        # 7. funding_velocity = change in 8h funding estimate
        # V3: DISABLED - removed to eliminate state tracking complexity
        funding_velocity = 0.0  # (estimated_funding_8h_pct - pos.prev_estimated_funding_8h_pct) / 100

        # 8. spread_pct = current price spread
        # V3: Renamed from current_spread_pct for clarity
        avg_price = (current_long_price + current_short_price) / 2
        spread_pct = abs(current_long_price - current_short_price) / avg_price if avg_price > 0 else 0.0

        # 9. spread_change_from_entry = entry_spread - current_spread
        # Positive = spread narrowed since entry = PROFIT from spread convergence
        # Negative = spread widened since entry = LOSS from spread divergence
        entry_avg = (pos.entry_long_price + pos.entry_short_price) / 2
        entry_spread = abs(pos.entry_short_price - pos.entry_long_price) / entry_avg if entry_avg > 0 else 0.0
        spread_change_from_entry = entry_spread - spread_pct
        spread_change_from_entry = np.clip(spread_change_from_entry, -0.05, 0.05)  # Clip to ±5%

        # 10. liquidation_distance_pct
        liquidation_distance_pct = pos.get_liquidation_distance(current_long_price, current_short_price)

        # 11. apr_ratio = current_apr / entry_apr (funding rate deterioration)
        # Look up current APR for this symbol
        current_position_apr_value = 0.0
        if current_opportunities:
            for opp in current_opportunities:
                if opp['symbol'] == pos.symbol:
                    current_position_apr_value = opp.get('fund_apr', 0.0)
                    break
        apr_ratio_raw = pos.get_apr_ratio(current_position_apr_value)
        apr_ratio = np.clip(apr_ratio_raw, 0, 3) / 3  # Clip [0, 3] → [0, 1]

        # 12. current_position_apr
        # V3: Changed from /100 to /5000 (APR can reach ±5000%)
        current_position_apr = np.clip(current_position_apr_value, -5000, 5000) / 5000

        # 13. best_available_apr
        # V3: Changed from /100 to /5000
        best_available_apr_norm = np.clip(best_available_apr, -5000, 5000) / 5000

        # 14. apr_advantage = current - best (negative = better opportunities exist)
        apr_advantage = current_position_apr - best_available_apr_norm

        # 15. return_efficiency = P&L per hour held (age-adjusted performance)
        # V3: Added clipping to prevent outliers from very short holds
        return_efficiency_raw = pos.get_return_efficiency()
        return_efficiency = np.clip(return_efficiency_raw, -50, 50) / 50

        # 16. value_to_capital_ratio = capital allocated to this position (relative to initial)
        value_to_capital_ratio = total_capital / self.initial_capital if self.initial_capital > 0 else 0.0

        # 17. pnl_imbalance = (long_pnl - short_pnl) / 200
        # V3: NEW FEATURE - detects directional exposure (arbitrage breaking down)
        pnl_imbalance = (pos.long_pnl_pct - pos.short_pnl_pct) / 200

        return np.array([
            is_active,                  # 1
            net_pnl_pct,                # 2
            hours_held_norm,            # 3
            estimated_pnl_pct,          # 4 (NEW)
            estimated_pnl_velocity,     # 5 (NEW)
            estimated_funding_8h_pct,   # 6 (NEW)
            funding_velocity,           # 7 (NEW)
            spread_pct,                 # 8
            spread_change_from_entry,   # 9
            liquidation_distance_pct,   # 10
            apr_ratio,                  # 11
            current_position_apr,       # 12
            best_available_apr_norm,    # 13
            apr_advantage,              # 14
            return_efficiency,          # 15
            value_to_capital_ratio,     # 16
            pnl_imbalance,              # 17 (NEW)
        ], dtype=np.float32)

    def get_all_execution_states(self, price_data: Dict[str, Dict[str, float]],
                                max_positions: int = 5,
                                best_available_apr: float = 0.0,
                                current_opportunities: Optional[List[Dict]] = None) -> np.ndarray:
        """
        Get execution state features for all 5 slots (85 dimensions total - V3 refactoring).

        V3: Changed from 100 dims (5×20) to 85 dims (5×17 features)

        Args:
            price_data: Current price data for all symbols
            max_positions: Maximum number of position slots (default 5)
            best_available_apr: Maximum APR among current opportunities (for comparison)
            current_opportunities: List of current market opportunities to look up APR by symbol

        Returns:
            85-dimensional array (5 slots × 17 features)
        """
        all_features = []
        for i in range(max_positions):
            exec_features = self.get_execution_state(i, price_data, best_available_apr, current_opportunities)
            all_features.extend(exec_features)

        return np.array(all_features, dtype=np.float32)

    def update_velocity_tracking(self, price_data: Dict[str, Dict[str, float]]):
        """
        Update velocity tracking for all open positions (V3 feature).

        Call this at the END of each timestep to store current values
        as "previous" for velocity calculations in the next timestep.

        Args:
            price_data: Current price data for all symbols
        """
        for pos in self.positions:
            # Get current prices
            if pos.symbol in price_data:
                prices = price_data[pos.symbol]
                current_long_price = prices['long_price']
                current_short_price = prices['short_price']
            else:
                current_long_price = pos.entry_long_price
                current_short_price = pos.entry_short_price

            # Calculate and store current values as "previous" for next step
            total_capital = pos.position_size_usd * 2

            # estimated_pnl_pct (price P&L only, no fees/funding, no slippage)
            # Slippage is only applied at actual exit, not in estimates
            long_price_pnl = pos.position_size_usd * ((current_long_price - pos.entry_long_price) / pos.entry_long_price)
            short_price_pnl = pos.position_size_usd * ((pos.entry_short_price - current_short_price) / pos.entry_short_price)

            pos.prev_estimated_pnl_pct = ((long_price_pnl + short_price_pnl) / total_capital) / 100 if total_capital > 0 else 0.0

            # estimated_funding_8h_pct
            pos.prev_estimated_funding_8h_pct = pos.calculate_estimated_funding_8h_pct() / 100

            # spread_pct
            avg_price = (current_long_price + current_short_price) / 2
            pos.prev_spread_pct = abs(current_long_price - current_short_price) / avg_price if avg_price > 0 else 0.0

    def can_open_position(self, size_per_side_usd: float, leverage: float = 1.0) -> bool:
        """
        Check if we can open a new arbitrage execution.

        Args:
            size_per_side_usd: Position size per side (total = 2x this)
            leverage: Leverage multiplier (1-10x)

        Returns:
            True if position can be opened, False otherwise
        """
        # Position limit
        if len(self.positions) >= self.max_positions:
            return False

        # Margin availability (with leverage, we need less capital)
        margin_needed = (size_per_side_usd * 2) / leverage
        if margin_needed > self.available_margin:
            return False

        # REMOVED: Capital availability check (replaced by margin check above)
        # With leverage, we don't need full capital, just margin

        # Position size limit per side (based on initial capital, not current)
        max_size_per_side = self.initial_capital * (self.max_position_size_pct / 100)
        if size_per_side_usd > max_size_per_side:
            return False

        return True

    def open_position(self, position: Position) -> bool:
        """
        Open a new arbitrage execution.

        Returns:
            True if position opened successfully, False otherwise
        """
        if not self.can_open_position(position.position_size_usd, position.leverage):
            return False

        self.positions.append(position)

        # Deduct margin for SINGLE EXCHANGE (one leg only)
        # Production queries individual exchange balances, each exchange only has one leg
        # margin_used_usd is for BOTH legs, so we deduct half to match single-exchange reporting
        # Formula: (position_size_usd * 2 / leverage) / 2 = position_size_usd / leverage
        single_exchange_margin = position.position_size_usd / position.leverage
        self.total_capital -= single_exchange_margin

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
        # Return single-exchange margin (was deducted in open_position) + realized P&L
        single_exchange_margin = position.position_size_usd / position.leverage
        self.total_capital += single_exchange_margin + realized_pnl_usd

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
