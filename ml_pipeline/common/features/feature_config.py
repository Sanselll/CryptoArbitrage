"""
Feature configuration constants for RL model.

Defines all feature dimensions and configuration for the modular RL architecture.
Version: V7 (229 dimensions total)

V6 Changes:
- Added portfolio feature: time_to_next_funding_norm (+1)
- Added execution feature: pnl_vs_peak_pct (+1 per slot = +5)

V7 Changes:
- Added execution feature: apr_sign_match (+1 per slot = +5) - APR direction flip indicator
- Added execution feature: apr_velocity (+1 per slot = +5) - APR deterioration rate
- Total execution features: 18 -> 20
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class FeatureDimensions:
    """Feature dimensions for V7 modular architecture."""

    # Component dimensions
    CONFIG: Final[int] = 5
    PORTFOLIO: Final[int] = 4  # V6: +1 (time_to_next_funding_norm)
    EXECUTIONS_PER_SLOT: Final[int] = 20  # V7: +2 (apr_sign_match, apr_velocity)
    EXECUTIONS_SLOTS: Final[int] = 5
    EXECUTIONS_TOTAL: Final[int] = EXECUTIONS_PER_SLOT * EXECUTIONS_SLOTS  # 100
    OPPORTUNITIES_PER_SLOT: Final[int] = 12
    OPPORTUNITIES_SLOTS: Final[int] = 10
    OPPORTUNITIES_TOTAL: Final[int] = OPPORTUNITIES_PER_SLOT * OPPORTUNITIES_SLOTS  # 120

    # Total observation dimension
    TOTAL: Final[int] = CONFIG + PORTFOLIO + EXECUTIONS_TOTAL + OPPORTUNITIES_TOTAL  # 229

    # Action space
    ACTION_HOLD: Final[int] = 0
    ACTION_ENTER_START: Final[int] = 1
    ACTION_ENTER_END: Final[int] = 30
    ACTION_EXIT_START: Final[int] = 31
    ACTION_EXIT_END: Final[int] = 35
    TOTAL_ACTIONS: Final[int] = 36


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for feature engineering."""

    # Hours normalization (log scale)
    HOURS_HELD_LOG_BASE: Final[float] = 10.0

    # Clipping ranges
    APR_CLIP_MIN: Final[float] = -5000.0
    APR_CLIP_MAX: Final[float] = 5000.0
    RETURN_EFFICIENCY_CLIP_MIN: Final[float] = -50.0
    RETURN_EFFICIENCY_CLIP_MAX: Final[float] = 50.0

    # Feature scaler path (relative to ml_pipeline/)
    # V5.4: StandardScaler (12 features)
    FEATURE_SCALER_PATH: Final[str] = "trained_models/rl/feature_scaler_v3.pkl"

    # Feature names for execution slots (20 features - V7)
    EXECUTION_FEATURE_NAMES: Final[tuple] = (
        "is_active",
        "net_pnl_pct",
        "hours_held_norm",
        "estimated_pnl_pct",
        "estimated_pnl_velocity",
        "estimated_funding_8h_pct",
        "funding_velocity",
        "spread_pct",
        "spread_change_from_entry",
        "liquidation_distance_pct",
        "apr_ratio",
        "current_position_apr",
        "best_available_apr_norm",
        "apr_advantage",
        "return_efficiency",
        "value_to_capital_ratio",
        "pnl_imbalance",
        "pnl_vs_peak_pct",  # V6: Current P&L / Peak P&L (signals when to take profits)
        "apr_sign_match",   # V7: APR direction flip indicator (1.0 = same sign, -1.0 = flipped)
        "apr_velocity",     # V7: APR deterioration rate (negative = getting worse)
    )

    # Feature names for opportunity slots (12 features - V5.4)
    OPPORTUNITY_FEATURE_NAMES: Final[tuple] = (
        "fund_profit_8h",
        "fund_profit_8h_24h_proj",
        "fund_profit_8h_3d_proj",
        "fund_apr",
        "fund_apr_24h_proj",
        "fund_apr_3d_proj",
        "spread_30_sample_avg",
        "price_spread_24h_avg",
        "price_spread_3d_avg",
        "spread_volatility_stddev",
        "apr_velocity",
        "spread_mean_reversion_potential",  # V5.4: Sign-agnostic spread profitability
    )


# Singleton instances
DIMS = FeatureDimensions()
CONFIG = FeatureConfig()
