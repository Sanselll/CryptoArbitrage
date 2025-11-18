"""
Unified Feature Builder for RL Inference (V3)

This module provides the SINGLE SOURCE OF TRUTH for all feature engineering
used in the modular RL arbitrage system. All components (backend inference,
ML API server, training, and testing) must use this class to ensure consistency.

Architecture V3: 203-dimensional observation space
- Config: 5 dims
- Portfolio: 3 dims
- Executions: 85 dims (5 slots × 17 features)
- Opportunities: 110 dims (10 slots × 11 features)

Key Principles:
1. All feature calculation logic lives HERE and only here
2. Backend sends raw data, this module transforms it to features
3. Training environment uses same code paths
4. Feature scaler is applied here for opportunity features
5. No feature logic duplication across Python/C# codebases
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .feature_config import DIMS, CONFIG


class UnifiedFeatureBuilder:
    """
    Unified feature preparation for RL model inference.

    This class contains all feature engineering logic used across:
    - Backend API inference (via ML API server)
    - Training environment
    - Test inference

    Usage:
        builder = UnifiedFeatureBuilder(feature_scaler_path='trained_models/rl/feature_scaler_v2.pkl')
        observation = builder.build_observation_from_raw_data(raw_data)
    """

    def __init__(self, feature_scaler_path: Optional[str] = None):
        """
        Initialize the unified feature builder.

        Args:
            feature_scaler_path: Path to fitted StandardScaler pickle file (for opportunity features)
                               If None, uses default path from CONFIG
        """
        self.feature_scaler = None

        if feature_scaler_path is None:
            feature_scaler_path = CONFIG.FEATURE_SCALER_PATH

        if feature_scaler_path:
            scaler_path = Path(feature_scaler_path)
            if not scaler_path.is_absolute():
                # Make relative to ml_pipeline directory
                ml_pipeline_dir = Path(__file__).parent.parent.parent
                scaler_path = ml_pipeline_dir / feature_scaler_path

            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"✅ Feature scaler loaded from: {scaler_path}")
            else:
                print(f"⚠️  WARNING: Feature scaler not found at: {scaler_path}")
                print(f"   Using raw features (may cause poor predictions!)")

    def build_observation_from_raw_data(self, raw_data: Dict[str, Any], log_file: Optional[str] = None) -> np.ndarray:
        """
        Build complete 203-dim observation vector from raw backend data.

        This is the main entry point for converting raw data to model features.

        Args:
            raw_data: Dictionary with trading_config, portfolio, opportunities
            log_file: Optional path to log features for debugging

        Args:
            raw_data: Dict containing:
                - 'trading_config': Dict with max_leverage, target_utilization, etc.
                - 'portfolio': Dict with positions, total_capital, capital_utilization
                - 'opportunities': List of opportunity dicts

        Returns:
            203-dimensional observation array (5 + 3 + 85 + 110)
        """
        trading_config = raw_data.get('trading_config', {})
        portfolio = raw_data.get('portfolio', {})
        opportunities = raw_data.get('opportunities', [])

        # Build each component
        config_features = self.build_config_features(trading_config)
        portfolio_features = self.build_portfolio_features(portfolio, trading_config)

        # Calculate best available APR for execution features
        best_available_apr = 0.0
        if opportunities:
            best_available_apr = max(opp.get('fund_apr', 0.0) for opp in opportunities)

        execution_features = self.build_execution_features(portfolio, best_available_apr)
        opportunity_features = self.build_opportunity_features(opportunities)

        # Concatenate all components
        observation = np.concatenate([
            config_features,
            portfolio_features,
            execution_features,
            opportunity_features
        ]).astype(np.float32)

        assert observation.shape == (DIMS.TOTAL,), \
            f"Observation shape mismatch: expected {DIMS.TOTAL}, got {observation.shape[0]}"

        # Log features if requested
        if log_file:
            import json
            log_entry = {
                'config_features': config_features.tolist(),
                'portfolio_features': portfolio_features.tolist(),
                'execution_features': execution_features.tolist(),
                'opportunity_features': opportunity_features.tolist(),
                'full_observation': observation.tolist(),
                'num_opportunities': len(raw_data.get('opportunities', [])),
                'num_positions': len([p for p in raw_data.get('portfolio', {}).get('positions', []) if p.get('is_active')])
            }
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

        return observation

    def build_config_features(self, trading_config: Dict) -> np.ndarray:
        """
        Build config features from trading configuration.

        Features (5 dimensions):
        1. max_leverage
        2. target_utilization
        3. max_positions
        4. stop_loss_threshold
        5. liquidation_buffer

        Args:
            trading_config: Dict with max_leverage, target_utilization, max_positions, etc.

        Returns:
            5-dimensional array
        """
        return np.array([
            float(trading_config.get('max_leverage', 1.0)),
            float(trading_config.get('target_utilization', 0.5)),
            float(trading_config.get('max_positions', 3)),
            float(trading_config.get('stop_loss_threshold', -0.02)),
            float(trading_config.get('liquidation_buffer', 0.15)),
        ], dtype=np.float32)

    def build_portfolio_features(self, portfolio: Dict, trading_config: Dict) -> np.ndarray:
        """
        Build portfolio features (V3: 3 dimensions).

        Features:
        1. num_positions_ratio: Active positions / max_positions
        2. min_liq_distance: Minimum liquidation distance across all positions
        3. capital_utilization: Used capital / total capital

        V3 Changes: Removed historical metrics (avg_position_pnl_pct, total_pnl_pct, max_drawdown_pct)

        Args:
            portfolio: Portfolio state dict
            trading_config: Trading configuration dict

        Returns:
            3-dimensional array
        """
        # Count only ACTIVE positions, not empty padded slots
        positions = portfolio.get('positions', [])
        num_positions = sum(
            1 for p in positions
            if p.get('is_active', False) or p.get('position_is_active', 0.0) > 0.5
        )
        max_positions = trading_config.get('max_positions', 2)
        num_positions_ratio = num_positions / max_positions if max_positions > 0 else 0.0

        # Calculate minimum liquidation distance from active positions
        min_liq_distance = 1.0  # Default for no positions
        if num_positions > 0:
            active_liq_distances = [
                p.get('liquidation_distance', 1.0)
                for p in positions
                if p.get('is_active', False) or p.get('position_is_active', 0.0) > 0.5
            ]
            if active_liq_distances:
                min_liq_distance = min(active_liq_distances)

        capital_utilization = portfolio.get('capital_utilization', 0.0) / 100.0

        return np.array([
            num_positions_ratio,
            min_liq_distance,
            capital_utilization,
        ], dtype=np.float32)

    def build_execution_features(self, portfolio: Dict, best_available_apr: float = 0.0) -> np.ndarray:
        """
        Build execution features for up to 5 position slots (85 dimensions total).

        Each position has 17 features:
        1. is_active
        2. net_pnl_pct (price P&L + funding - fees) / capital
        3. hours_held_norm: log(hours + 1) / log(73)
        4. estimated_pnl_pct: price P&L only (no fees, no funding)
        5. estimated_pnl_velocity: change in estimated P&L (currently disabled)
        6. estimated_funding_8h_pct: expected funding profit in next 8h
        7. funding_velocity: change in 8h funding estimate (currently disabled)
        8. spread_pct: current price spread
        9. spread_velocity: change in spread (currently disabled)
        10. liquidation_distance_pct
        11. apr_ratio: current_apr / entry_apr (clipped to [0,3])
        12. current_position_apr (normalized to ±5000%)
        13. best_available_apr_norm (normalized to ±5000%)
        14. apr_advantage: current - best
        15. return_efficiency: P&L per hour held (clipped to ±50)
        16. value_to_capital_ratio: capital allocated to this position
        17. pnl_imbalance: (long_pnl - short_pnl) / 200

        Args:
            portfolio: Portfolio state dict with positions list
            best_available_apr: Maximum APR among current opportunities

        Returns:
            85-dimensional array (5 slots × 17 features)
        """
        positions = portfolio.get('positions', [])
        capital = portfolio.get('total_capital', portfolio.get('capital', 10000.0))

        all_features = []

        for slot_idx in range(DIMS.EXECUTIONS_SLOTS):
            if slot_idx < len(positions):
                pos = positions[slot_idx]

                # Check if position is ACTUALLY active
                is_active = pos.get('is_active', False) or pos.get('position_is_active', 0.0) > 0.5

                if not is_active:
                    # Empty/inactive slot - all zeros
                    all_features.extend([0.0] * DIMS.EXECUTIONS_PER_SLOT)
                    continue

                # Extract position data
                total_capital_used = pos.get('position_size_usd', 0.0) * 2
                raw_hours_held = pos.get('position_age_hours', pos.get('hours_held', 0.0))

                # Prices
                current_long_price = pos.get('current_long_price', pos.get('current_price', 0.0))
                current_short_price = pos.get('current_short_price', pos.get('current_price', 0.0))
                entry_long_price = pos.get('entry_long_price', pos.get('entry_price', 0.0))
                entry_short_price = pos.get('entry_short_price', pos.get('entry_price', 0.0))

                # ===== Calculate 17 features =====

                # 1. is_active
                is_active_feat = 1.0

                # 2. net_pnl_pct (for feature output - divide by 100 to get decimal)
                unrealized_pnl_pct_raw = pos.get('unrealized_pnl_pct', 0.0)  # As percentage (e.g., 2.5)
                net_pnl_pct = unrealized_pnl_pct_raw / 100  # Convert to decimal (e.g., 0.025)

                # 3. hours_held_norm (log scale)
                hours_held_norm = np.log(raw_hours_held + 1) / np.log(CONFIG.HOURS_HELD_LOG_BASE + 63)

                # 4. estimated_pnl_pct (price P&L only - WITH slippage applied on exit)
                long_price_pnl = 0.0
                short_price_pnl = 0.0
                if entry_long_price > 0 and entry_short_price > 0:
                    position_size = pos.get('position_size_usd', 0.0)
                    slippage_pct = pos.get('slippage_pct', 0.0)

                    # Apply slippage ONLY on exit (not entry)
                    effective_current_long = current_long_price * (1 + slippage_pct)  # Receive less at exit
                    effective_current_short = current_short_price * (1 - slippage_pct)  # Pay more at exit

                    long_price_pnl = position_size * ((effective_current_long - entry_long_price) / entry_long_price)
                    short_price_pnl = position_size * ((entry_short_price - effective_current_short) / entry_short_price)
                estimated_pnl_pct = ((long_price_pnl + short_price_pnl) / total_capital_used) / 100 if total_capital_used > 0 else 0.0

                # 5. estimated_pnl_velocity (disabled - requires state tracking)
                estimated_pnl_velocity = 0.0

                # 6. estimated_funding_8h_pct
                long_interval = pos.get('long_funding_interval_hours', 8)
                short_interval = pos.get('short_funding_interval_hours', 8)
                long_rate = pos.get('long_funding_rate', 0.0)
                short_rate = pos.get('short_funding_rate', 0.0)

                long_payments_8h = 8.0 / long_interval if long_interval > 0 else 1.0
                short_payments_8h = 8.0 / short_interval if short_interval > 0 else 1.0
                long_funding_8h = -long_rate * long_payments_8h
                short_funding_8h = short_rate * short_payments_8h
                # Convert to percentage then back to decimal to match original implementation
                estimated_funding_8h_pct = ((long_funding_8h + short_funding_8h) * 100) / 100

                # 7. funding_velocity (disabled - requires state tracking)
                funding_velocity = 0.0

                # 8. spread_pct
                avg_price = (current_long_price + current_short_price) / 2
                spread_pct = abs(current_long_price - current_short_price) / avg_price if avg_price > 0 else 0.0

                # 9. spread_velocity (disabled - requires state tracking)
                spread_velocity = 0.0

                # 10. liquidation_distance_pct
                liquidation_distance_pct = pos.get('liquidation_distance', 1.0)
                if liquidation_distance_pct is None or liquidation_distance_pct == 0.0:
                    leverage = pos.get('leverage', 1.0)
                    if leverage > 0 and entry_long_price > 0 and entry_short_price > 0:
                        long_liq = entry_long_price * (1 - 0.9 / leverage)
                        short_liq = entry_short_price * (1 + 0.9 / leverage)
                        long_dist = abs(current_long_price - long_liq) / current_long_price if current_long_price > 0 else 1.0
                        short_dist = abs(short_liq - current_short_price) / current_short_price if current_short_price > 0 else 1.0
                        liquidation_distance_pct = min(long_dist, short_dist)
                    else:
                        liquidation_distance_pct = 1.0

                # 11. apr_ratio
                current_position_apr_value = pos.get('current_position_apr', 0.0)
                entry_apr = pos.get('entry_apr', 0.0)
                apr_ratio_raw = current_position_apr_value / entry_apr if entry_apr > 0 else 1.0
                apr_ratio = np.clip(apr_ratio_raw, 0, 3) / 3

                # 12. current_position_apr
                current_position_apr = np.clip(
                    current_position_apr_value,
                    CONFIG.APR_CLIP_MIN,
                    CONFIG.APR_CLIP_MAX
                ) / CONFIG.APR_CLIP_MAX

                # 13. best_available_apr_norm
                best_available_apr_norm = np.clip(
                    best_available_apr,
                    CONFIG.APR_CLIP_MIN,
                    CONFIG.APR_CLIP_MAX
                ) / CONFIG.APR_CLIP_MAX

                # 14. apr_advantage
                apr_advantage = current_position_apr - best_available_apr_norm

                # 15. return_efficiency (P&L per hour - use raw percentage, not divided by 100)
                if raw_hours_held > 0:
                    return_efficiency_raw = unrealized_pnl_pct_raw / raw_hours_held  # Use raw % value
                else:
                    return_efficiency_raw = 0.0
                return_efficiency = np.clip(
                    return_efficiency_raw,
                    CONFIG.RETURN_EFFICIENCY_CLIP_MIN,
                    CONFIG.RETURN_EFFICIENCY_CLIP_MAX
                ) / CONFIG.RETURN_EFFICIENCY_CLIP_MAX

                # 16. value_to_capital_ratio
                value_to_capital_ratio = total_capital_used / capital if capital > 0 else 0.0

                # 17. pnl_imbalance
                long_pnl_pct = pos.get('long_pnl_pct', 0.0)
                short_pnl_pct = pos.get('short_pnl_pct', 0.0)
                pnl_imbalance = (long_pnl_pct - short_pnl_pct) / 200

                # CRITICAL: Feature order must match exactly across all components
                slot_features = [
                    is_active_feat,
                    net_pnl_pct,
                    hours_held_norm,
                    estimated_pnl_pct,
                    estimated_pnl_velocity,
                    estimated_funding_8h_pct,
                    funding_velocity,
                    spread_pct,
                    spread_velocity,
                    liquidation_distance_pct,
                    apr_ratio,
                    current_position_apr,
                    best_available_apr_norm,
                    apr_advantage,
                    return_efficiency,
                    value_to_capital_ratio,
                    pnl_imbalance,
                ]
            else:
                # Empty slot - all zeros
                slot_features = [0.0] * DIMS.EXECUTIONS_PER_SLOT

            all_features.extend(slot_features)

        return np.array(all_features, dtype=np.float32)

    def build_opportunity_features(self, opportunities: List[Dict]) -> np.ndarray:
        """
        Build opportunity features for up to 10 slots (110 dimensions total).

        Each opportunity has 11 features:
        1-6: fund_profit_8h, fundProfit8h24hProj, fundProfit8h3dProj,
             fund_apr, fundApr24hProj, fundApr3dProj
        7-10: spread30SampleAvg, priceSpread24hAvg, priceSpread3dAvg, spread_volatility_stddev
        11: apr_velocity (fund_profit_8h - fundProfit8h24hProj)

        CRITICAL: Feature scaler is applied to ALL slots (including empty ones)
        to match training behavior. Empty slots are padded with zeros BEFORE scaling.

        Args:
            opportunities: List of up to 10 opportunity dicts

        Returns:
            110-dimensional array (10 slots × 11 features)
        """
        all_features = []

        for slot_idx in range(DIMS.OPPORTUNITIES_SLOTS):
            if slot_idx < len(opportunities):
                opp = opportunities[slot_idx]

                fund_profit_8h = opp.get('fund_profit_8h', 0)
                fundProfit8h24hProj = opp.get('fundProfit8h24hProj', 0)

                features = [
                    # Profit projections (6 features)
                    fund_profit_8h,
                    fundProfit8h24hProj,
                    opp.get('fundProfit8h3dProj', 0),
                    opp.get('fund_apr', 0),
                    opp.get('fundApr24hProj', 0),
                    opp.get('fundApr3dProj', 0),
                    # Spread metrics (4 features)
                    opp.get('spread30SampleAvg', 0),
                    opp.get('priceSpread24hAvg', 0),
                    opp.get('priceSpread3dAvg', 0),
                    opp.get('spread_volatility_stddev', 0),
                    # Velocity (1 feature)
                    fund_profit_8h - fundProfit8h24hProj,  # apr_velocity
                ]

                # Convert to float32 and ensure no NaN/inf
                features = [
                    float(np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0))
                    for x in features
                ]
            else:
                # Empty slot - all zeros BEFORE scaling
                features = [0.0] * DIMS.OPPORTUNITIES_PER_SLOT

            # Apply feature scaler to ALL slots (including empty ones)
            # This matches training environment behavior
            if self.feature_scaler is not None:
                features_array = np.array(features, dtype=np.float32)
                features_scaled = self.feature_scaler.transform(
                    features_array.reshape(1, DIMS.OPPORTUNITIES_PER_SLOT)
                )
                features = features_scaled.flatten().tolist()

            all_features.extend(features)

        return np.array(all_features, dtype=np.float32)

    def get_action_mask(
        self,
        opportunities: List[Dict],
        num_positions: int,
        max_positions: int
    ) -> np.ndarray:
        """
        Generate action mask (36 dimensions).

        Args:
            opportunities: List of opportunity dicts
            num_positions: Current number of active positions
            max_positions: Maximum allowed positions

        Returns:
            Boolean array (36,) where True = valid action
        """
        mask = np.zeros(DIMS.TOTAL_ACTIONS, dtype=bool)

        # HOLD is always valid
        mask[DIMS.ACTION_HOLD] = True

        # ENTER actions: valid if opportunity exists AND we have capacity
        has_capacity = num_positions < max_positions

        if has_capacity:
            for i in range(DIMS.OPPORTUNITIES_SLOTS):
                if i < len(opportunities):
                    # Prevent duplicate positions for same symbol
                    if not opportunities[i].get('has_existing_position', False):
                        mask[1 + i] = True      # SMALL
                        mask[11 + i] = True     # MEDIUM
                        mask[21 + i] = True     # LARGE

        # EXIT actions: valid if position exists
        for i in range(DIMS.EXECUTIONS_SLOTS):
            if i < num_positions:
                mask[DIMS.ACTION_EXIT_START + i] = True

        return mask
