"""
Pydantic schemas for raw data validation.

Defines the expected structure of raw data sent from the backend to the ML API.
These schemas ensure type safety and data validation at runtime.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator


class TradingConfigRaw(BaseModel):
    """Trading configuration raw data."""
    max_leverage: float = Field(default=1.0, ge=1.0, le=10.0)
    target_utilization: float = Field(default=0.5, ge=0.0, le=1.0)
    max_positions: int = Field(default=1, ge=1, le=3)  # Allow 1-3 for backwards compatibility, model uses 1
    stop_loss_threshold: float = Field(default=-0.10, ge=-1.0, le=0.0)
    liquidation_buffer: float = Field(default=0.15, ge=0.0, le=1.0)


class PositionRawData(BaseModel):
    """Raw position data from backend."""
    # Identity
    is_active: bool = False
    symbol: Optional[str] = None

    # Position basics
    position_size_usd: float = Field(default=0.0, ge=0.0)
    position_age_hours: Optional[float] = Field(default=None, ge=0.0)
    leverage: float = Field(default=1.0, ge=1.0, le=10.0)

    # Prices
    entry_long_price: float = Field(default=0.0, ge=0.0)
    entry_short_price: float = Field(default=0.0, ge=0.0)
    current_long_price: float = Field(default=0.0, ge=0.0)
    current_short_price: float = Field(default=0.0, ge=0.0)
    slippage_pct: float = Field(default=0.0, ge=0.0)  # Slippage percentage for exit calculations

    # Raw funding and fees (Python calculates P&L)
    long_funding_earned_usd: float = Field(default=0.0)
    short_funding_earned_usd: float = Field(default=0.0)
    long_fees_usd: float = Field(default=0.0)
    short_fees_usd: float = Field(default=0.0)

    # Funding rates
    long_funding_rate: float = Field(default=0.0)
    short_funding_rate: float = Field(default=0.0)
    long_funding_interval_hours: float = Field(default=8.0, gt=0.0)
    short_funding_interval_hours: float = Field(default=8.0, gt=0.0)

    # APR
    entry_apr: float = Field(default=0.0)
    current_position_apr: float = Field(default=0.0)

    # Risk
    liquidation_distance: Optional[float] = Field(default=None, ge=0.0)

    class Config:
        extra = "allow"  # Allow additional fields from backend


class OpportunityRawData(BaseModel):
    """Raw opportunity data from backend."""
    # Identity
    symbol: str
    long_exchange: str
    short_exchange: str

    # Funding profit projections (6 features) - Optional to allow NaN/null passthrough
    fund_profit_8h: Optional[float] = Field(default=0.0)
    fund_profit_8h_24h_proj: Optional[float] = Field(default=0.0)
    fund_profit_8h_3d_proj: Optional[float] = Field(default=0.0)
    fund_apr: Optional[float] = Field(default=0.0)
    fund_apr_24h_proj: Optional[float] = Field(default=0.0)
    fund_apr_3d_proj: Optional[float] = Field(default=0.0)

    # Spread metrics (4 features - can be negative when short exchange price < long exchange price)
    # Optional to allow NaN/null passthrough for parity with direct mode
    spread_30_sample_avg: Optional[float] = Field(default=0.0)
    price_spread_24h_avg: Optional[float] = Field(default=0.0)
    price_spread_3d_avg: Optional[float] = Field(default=0.0)
    spread_volatility_stddev: Optional[float] = Field(default=0.0)

    # Liquidity (0.0=Good, 1.0=Medium, 2.0=Low)
    liquidityStatus: Optional[float] = Field(default=0.0)

    # Position tracking
    has_existing_position: bool = Field(default=False)

    class Config:
        extra = "allow"  # Allow additional fields from backend


class PortfolioRawData(BaseModel):
    """Raw portfolio data from backend."""
    # Agent session ID for per-session feature builder tracking
    # Ensures velocity state is isolated between different agent sessions
    session_id: Optional[str] = None

    positions: List[PositionRawData] = Field(default_factory=list, max_items=1)  # V9: single position only
    total_capital: float = Field(default=10000.0, gt=0.0)
    capital_utilization: float = Field(default=0.0, ge=0.0, le=200.0)  # Can exceed 100% during drawdowns

    @validator('positions')
    def validate_positions(cls, v):
        """Ensure max 1 position (V9)."""
        if len(v) > 1:
            raise ValueError(f"Maximum 1 position allowed, got {len(v)}")
        return v

    class Config:
        extra = "allow"


class RLRawDataRequest(BaseModel):
    """Complete raw data request from backend to ML API."""
    trading_config: Optional[TradingConfigRaw] = None
    portfolio: PortfolioRawData
    opportunities: List[OpportunityRawData] = Field(default_factory=list, max_items=5)  # V8: reduced from 10

    @validator('opportunities')
    def validate_opportunities(cls, v):
        """Ensure max 5 opportunities (V8)."""
        if len(v) > 5:
            raise ValueError(f"Maximum 5 opportunities allowed, got {len(v)}")
        return v

    class Config:
        extra = "allow"


class RLPredictionResponse(BaseModel):
    """Response from ML API with prediction results (V9: 17 actions)."""
    action: int = Field(ge=0, le=16)  # V9: reduced from 17
    action_name: str
    action_type: str  # "HOLD", "ENTER", "EXIT"
    confidence: float = Field(ge=0.0, le=1.0)
    action_probabilities: List[float] = Field(min_items=17, max_items=17)  # V9: reduced from 18

    # Action details
    opportunity_index: Optional[int] = Field(default=None, ge=0, le=4)  # V8: reduced from 9
    position_index: Optional[int] = Field(default=None, ge=0, le=0)  # V9: single position only
    size: Optional[str] = None  # "SMALL", "MEDIUM", "LARGE"
    size_multiplier: Optional[float] = None

    # Selected opportunity/position info
    selected_symbol: Optional[str] = None
    selected_long_exchange: Optional[str] = None
    selected_short_exchange: Optional[str] = None
    selected_fund_apr: Optional[float] = None

    # Mask info
    valid_actions: int = Field(ge=1, le=17)  # V9: reduced from 18
    masked_actions: int = Field(ge=0, le=16)  # V9: reduced from 17

    class Config:
        extra = "allow"
