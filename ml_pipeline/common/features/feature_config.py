"""
Feature configuration constants for RL model.

Defines all feature dimensions and configuration for the modular RL architecture.
Version: V10 (91 dimensions total)

V6 Changes:
- Added portfolio feature: time_to_next_funding_norm (+1)
- Added execution feature: pnl_vs_peak_pct (+1 per slot = +5)

V7 Changes:
- Added execution feature: apr_sign_match (+1 per slot = +5) - APR direction flip indicator
- Added execution feature: apr_velocity (+1 per slot = +5) - APR deterioration rate
- Total execution features: 18 -> 20

V8 Changes (Optimization):
- Reduced opportunity slots: 10 -> 5 (position opps first, then best APR)
- Reduced position slots: 5 -> 2
- Observation space: 229 -> 109 dimensions
- Action space: 36 -> 18 actions

V9 Changes (Simplification):
- Removed capital features: num_positions_ratio, capital_utilization from portfolio
- Removed value_to_capital_ratio from execution features
- Reduced position slots: 2 -> 1 (single position only)
- Portfolio features: 4 -> 2
- Execution features per slot: 20 -> 19
- Observation space: 109 -> 86 dimensions
- Action space: 18 -> 17 actions

V10 Changes (Funding Timing):
- Added opportunity feature: time_to_profitable_funding (+1 per slot = +5)
- Uses actual next funding times from exchange data (not hardcoded 8h schedule)
- Opportunity features per slot: 12 -> 13
- Observation space: 86 -> 91 dimensions
"""

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class FeatureDimensions:
    """Feature dimensions for V10 modular architecture (single position)."""

    # Component dimensions
    CONFIG: Final[int] = 5
    PORTFOLIO: Final[int] = 2  # min_liq_distance, time_to_next_funding (V9: removed capital features)
    EXECUTIONS_PER_SLOT: Final[int] = 19  # V9: 19 features per slot (removed value_to_capital_ratio)
    EXECUTIONS_SLOTS: Final[int] = 1  # V9: single position only
    EXECUTIONS_TOTAL: Final[int] = EXECUTIONS_PER_SLOT * EXECUTIONS_SLOTS  # 19
    OPPORTUNITIES_PER_SLOT: Final[int] = 13  # V10: added time_to_profitable_funding
    OPPORTUNITIES_SLOTS: Final[int] = 5  # V8: reduced from 10
    OPPORTUNITIES_TOTAL: Final[int] = OPPORTUNITIES_PER_SLOT * OPPORTUNITIES_SLOTS  # 65

    # Total observation dimension
    TOTAL: Final[int] = CONFIG + PORTFOLIO + EXECUTIONS_TOTAL + OPPORTUNITIES_TOTAL  # 91 (V10)

    # Action space (V9: 17 actions)
    # 0: HOLD
    # 1-5: ENTER_OPP_0-4_SMALL
    # 6-10: ENTER_OPP_0-4_MEDIUM
    # 11-15: ENTER_OPP_0-4_LARGE
    # 16: EXIT_POS_0
    ACTION_HOLD: Final[int] = 0
    ACTION_ENTER_START: Final[int] = 1
    ACTION_ENTER_END: Final[int] = 15  # V8: reduced from 30
    ACTION_EXIT_START: Final[int] = 16  # V8: reduced from 31
    ACTION_EXIT_END: Final[int] = 16  # V9: only EXIT_POS_0 (was 17)
    TOTAL_ACTIONS: Final[int] = 17  # V9: reduced from 18


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

    # Feature names for execution slots (19 features - V9)
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
        "pnl_imbalance",
        "pnl_vs_peak_pct",
        "apr_sign_match",
        "apr_velocity",
    )

    # Feature names for opportunity slots (13 features - V10)
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
        "time_to_profitable_funding",  # V10: Minutes to next profitable funding / 480
    )


# Singleton instances
DIMS = FeatureDimensions()
CONFIG = FeatureConfig()
