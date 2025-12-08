"""
Unified Feature Builder for RL Inference (V6)

This module provides the SINGLE SOURCE OF TRUTH for all feature engineering
used in the modular RL arbitrage system. All components (backend inference,
ML API server, training, and testing) must use this class to ensure consistency.

Architecture V6: 219-dimensional observation space
- Config: 5 dims
- Portfolio: 4 dims (V6: +1 time_to_next_funding_norm)
- Executions: 90 dims (5 slots × 18 features, V6: +1 pnl_vs_peak_pct per slot)
- Opportunities: 120 dims (10 slots × 12 features)

V6 Changes:
- Added portfolio feature: time_to_next_funding_norm (minutes until next funding / 480)
- Added execution feature #18: pnl_vs_peak_pct (current P&L / peak P&L, signals profit-taking)

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
from datetime import datetime
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

    def __init__(self, feature_scaler_path: Optional[str] = None, enable_velocity_tracking: bool = True):
        """
        Initialize the unified feature builder.

        Args:
            feature_scaler_path: Path to fitted StandardScaler pickle file (for opportunity features)
                               If None, uses default path from CONFIG
            enable_velocity_tracking: If True, track state for velocity feature calculation
        """
        self.feature_scaler = None
        self.enable_velocity_tracking = enable_velocity_tracking

        # Velocity tracking state (per position slot)
        self._prev_estimated_pnl = {}  # slot_idx -> previous value
        self._prev_funding_8h = {}     # slot_idx -> previous value
        self._prev_spread = {}         # slot_idx -> previous value

        # V6: Peak P&L tracking (per position slot) for pnl_vs_peak_pct feature
        self._peak_pnl = {}  # slot_idx -> peak P&L value
        self._slot_symbols = {}  # slot_idx -> symbol (for detecting position changes)

        # V7: APR velocity tracking (per position slot) for apr_velocity feature
        self._prev_position_apr = {}  # slot_idx -> previous APR value

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

    def reset_velocity_tracking(self):
        """
        Reset velocity tracking state.

        Call this at the start of each episode to prevent stale velocity values
        from bleeding across episodes.
        """
        self._prev_estimated_pnl.clear()
        self._prev_funding_8h.clear()
        self._prev_spread.clear()
        self._peak_pnl.clear()  # V6: Reset peak P&L tracking
        self._slot_symbols.clear()  # V6: Reset symbol tracking
        self._prev_position_apr.clear()  # V7: Reset APR velocity tracking

    def build_observation_from_raw_data(self, raw_data: Dict[str, Any], log_file: Optional[str] = None) -> np.ndarray:
        """
        Build complete 219-dim observation vector from raw backend data (V6).

        This is the main entry point for converting raw data to model features.

        Args:
            raw_data: Dict containing:
                - 'trading_config': Dict with max_leverage, target_utilization, etc.
                - 'portfolio': Dict with positions, total_capital, capital_utilization
                - 'opportunities': List of opportunity dicts
                - 'current_time': Optional datetime for time_to_next_funding calculation
            log_file: Optional path to log features for debugging

        Returns:
            219-dimensional observation array (5 + 4 + 90 + 120)
        """
        trading_config = raw_data.get('trading_config', {})
        portfolio = raw_data.get('portfolio', {})
        opportunities = raw_data.get('opportunities', [])
        current_time = raw_data.get('current_time', None)  # V6: For time_to_next_funding

        # Build each component
        config_features = self.build_config_features(trading_config)
        portfolio_features = self.build_portfolio_features(portfolio, trading_config, current_time)

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

    def build_portfolio_features(self, portfolio: Dict, trading_config: Dict, current_time: Optional[datetime] = None) -> np.ndarray:
        """
        Build portfolio features (V6: 4 dimensions).

        Features:
        1. num_positions_ratio: Active positions / max_positions
        2. min_liq_distance: Minimum liquidation distance across all positions
        3. capital_utilization: Used capital / total capital
        4. time_to_next_funding_norm: V6 - Minutes until next funding payment / 480 (0-1)

        Args:
            portfolio: Portfolio state dict
            trading_config: Trading configuration dict
            current_time: Optional datetime for funding time calculation

        Returns:
            4-dimensional array
        """
        # Count only ACTIVE positions, not empty padded slots
        positions = portfolio.get('positions', [])
        num_positions = sum(
            1 for p in positions
            if p.get('is_active', False)
        )
        max_positions = trading_config.get('max_positions', 2)
        num_positions_ratio = num_positions / max_positions if max_positions > 0 else 0.0

        # Calculate minimum liquidation distance from active positions
        min_liq_distance = 1.0  # Default for no positions
        if num_positions > 0:
            active_liq_distances = [
                p.get('liquidation_distance', 1.0)
                for p in positions
                if p.get('is_active', False)
            ]
            if active_liq_distances:
                min_liq_distance = min(active_liq_distances)

        capital_utilization = portfolio.get('capital_utilization', 0.0) / 100.0

        # V6: Calculate time to next funding payment
        time_to_next_funding_norm = self._calc_time_to_next_funding(current_time)

        return np.array([
            num_positions_ratio,
            min_liq_distance,
            capital_utilization,
            time_to_next_funding_norm,  # V6: New feature
        ], dtype=np.float32)

    def _calc_time_to_next_funding(self, current_time: Optional[datetime]) -> float:
        """
        Calculate normalized time until next funding payment (V6).

        Funding payments occur at 00:00, 08:00, 16:00 UTC.
        Returns minutes until next payment / 480 (8 hours max).

        Args:
            current_time: Current datetime (UTC)

        Returns:
            Normalized value 0-1 (0 = funding imminent, 1 = just passed)
        """
        if current_time is None:
            return 0.5  # Default to middle if no time available

        # Funding hours: 00, 08, 16 UTC
        funding_hours = [0, 8, 16, 24]  # 24 = midnight next day
        current_hour = current_time.hour
        current_minute = current_time.minute

        # Find next funding hour
        next_funding_hour = 24  # Default to midnight
        for fh in funding_hours:
            if fh > current_hour or (fh == current_hour and current_minute == 0):
                next_funding_hour = fh
                break

        # Calculate minutes until next funding
        minutes_until = (next_funding_hour - current_hour) * 60 - current_minute
        if minutes_until < 0:
            minutes_until += 24 * 60  # Wrap to next day

        # Normalize by 480 minutes (8 hours)
        return min(minutes_until / 480.0, 1.0)

    def build_execution_features(self, portfolio: Dict, best_available_apr: float = 0.0) -> np.ndarray:
        """
        Build execution features for up to 5 position slots (90 dimensions total, V6).

        Each position has 18 features:
        1. is_active
        2. net_pnl_pct (price P&L + funding - fees) / capital
        3. hours_held_norm: log(hours + 1) / log(73)
        4. estimated_pnl_pct: price P&L only (no fees, no funding)
        5. estimated_pnl_velocity: change in estimated P&L (currently disabled)
        6. estimated_funding_8h_pct: expected funding profit in next 8h
        7. funding_velocity: change in 8h funding estimate (currently disabled)
        8. spread_pct: current price spread
        9. spread_change_from_entry: entry_spread - current_spread (+ = profit from spread narrowing)
        10. liquidation_distance_pct
        11. apr_ratio: current_apr / entry_apr (clipped to [0,3])
        12. current_position_apr (normalized to ±5000%)
        13. best_available_apr_norm (normalized to ±5000%)
        14. apr_advantage: current - best
        15. return_efficiency: P&L per hour held (clipped to ±50)
        16. value_to_capital_ratio: capital allocated to this position
        17. pnl_imbalance: (long_pnl - short_pnl) / 200
        18. pnl_vs_peak_pct: V6 - current P&L / peak P&L (signals profit-taking opportunity)

        Args:
            portfolio: Portfolio state dict with positions list
            best_available_apr: Maximum APR among current opportunities

        Returns:
            90-dimensional array (5 slots × 18 features)
        """
        positions = portfolio.get('positions', [])
        capital = portfolio.get('total_capital', 10000.0)

        all_features = []

        for slot_idx in range(DIMS.EXECUTIONS_SLOTS):
            if slot_idx < len(positions):
                pos = positions[slot_idx]

                # Check if position is ACTUALLY active
                is_active = pos.get('is_active', False)

                # V6/V7: Detect position change and reset tracking for this slot
                pos_symbol = pos.get('symbol', '')
                if slot_idx in self._slot_symbols and self._slot_symbols[slot_idx] != pos_symbol:
                    # Different symbol in this slot - reset tracking
                    if slot_idx in self._peak_pnl:
                        del self._peak_pnl[slot_idx]
                    if slot_idx in self._prev_estimated_pnl:
                        del self._prev_estimated_pnl[slot_idx]
                    if slot_idx in self._prev_funding_8h:
                        del self._prev_funding_8h[slot_idx]
                    if slot_idx in self._prev_position_apr:  # V7: Reset APR tracking
                        del self._prev_position_apr[slot_idx]
                self._slot_symbols[slot_idx] = pos_symbol

                if not is_active:
                    # Empty/inactive slot - all zeros
                    all_features.extend([0.0] * DIMS.EXECUTIONS_PER_SLOT)
                    continue

                # Extract position data
                total_capital_used = pos.get('position_size_usd', 0.0) * 2
                raw_hours_held = pos.get('position_age_hours', 0.0)

                # Prices
                current_long_price = pos.get('current_long_price', 0.0)
                current_short_price = pos.get('current_short_price', 0.0)
                entry_long_price = pos.get('entry_long_price', 0.0)
                entry_short_price = pos.get('entry_short_price', 0.0)

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
                estimated_pnl_pct = (long_price_pnl + short_price_pnl) / total_capital_used if total_capital_used > 0 else 0.0

                # 5. estimated_pnl_velocity (change per step)
                estimated_pnl_velocity = 0.0
                if self.enable_velocity_tracking:
                    prev_pnl = self._prev_estimated_pnl.get(slot_idx, estimated_pnl_pct)
                    estimated_pnl_velocity = np.clip(estimated_pnl_pct - prev_pnl, -0.1, 0.1)  # Clip to ±10%
                    self._prev_estimated_pnl[slot_idx] = estimated_pnl_pct

                # 6. estimated_funding_8h_pct
                long_interval = pos.get('long_funding_interval_hours', 8)
                short_interval = pos.get('short_funding_interval_hours', 8)
                # Use estimated rates (with fallback chain) instead of actual rates
                # Estimated rates fallback: parquet → opportunity CSV → last known
                long_rate = pos.get('estimated_long_funding_rate', pos.get('long_funding_rate', 0.0))
                short_rate = pos.get('estimated_short_funding_rate', pos.get('short_funding_rate', 0.0))

                long_payments_8h = 8.0 / long_interval if long_interval > 0 else 1.0
                short_payments_8h = 8.0 / short_interval if short_interval > 0 else 1.0
                long_funding_8h = -long_rate * long_payments_8h
                short_funding_8h = short_rate * short_payments_8h
                estimated_funding_8h_pct = long_funding_8h + short_funding_8h

                # 7. funding_velocity (change per step)
                funding_velocity = 0.0
                if self.enable_velocity_tracking:
                    prev_funding = self._prev_funding_8h.get(slot_idx, estimated_funding_8h_pct)
                    funding_velocity = np.clip(estimated_funding_8h_pct - prev_funding, -0.05, 0.05)  # Clip to ±5%
                    self._prev_funding_8h[slot_idx] = estimated_funding_8h_pct

                # 8. spread_pct
                avg_price = (current_long_price + current_short_price) / 2
                spread_pct = abs(current_long_price - current_short_price) / avg_price if avg_price > 0 else 0.0

                # 9. spread_change_from_entry = entry_spread - current_spread
                # Positive = spread narrowed since entry = PROFIT from spread convergence
                # Negative = spread widened since entry = LOSS from spread divergence
                entry_avg = (entry_long_price + entry_short_price) / 2
                entry_spread = abs(entry_short_price - entry_long_price) / entry_avg if entry_avg > 0 else 0.0
                spread_change_from_entry = entry_spread - spread_pct
                spread_change_from_entry = np.clip(spread_change_from_entry, -0.05, 0.05)  # Clip to ±5%

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

                # 11. apr_ratio (V7: Allow negative values to signal direction flip)
                current_position_apr_value = pos.get('current_position_apr', 0.0)
                entry_apr = pos.get('entry_apr', 0.0)
                apr_ratio_raw = current_position_apr_value / entry_apr if entry_apr > 0 else 1.0
                apr_ratio = np.clip(apr_ratio_raw, -3, 3) / 3  # V7: Changed from [0,3] to [-3,3]

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

                # 14. apr_advantage (V6: Log scale for large gaps to amplify signal)
                # Problem: /5000 normalization makes 1239% gap only -0.25 (too weak!)
                # Solution: Use log scale for large gaps to create stronger exit signal
                raw_current_apr = current_position_apr_value  # Raw APR (not normalized)
                raw_best_apr = best_available_apr  # Raw APR (not normalized)
                apr_gap = raw_best_apr - raw_current_apr  # Positive = better opportunity exists

                if apr_gap > 100:  # Significant gap (>100% APR)
                    # Log scale: 100% gap → -0.3, 1000% gap → -1.0, 10000% gap → -2.0
                    apr_advantage = -np.log10(apr_gap / 100 + 1)
                else:
                    # Linear for small gaps (preserves sensitivity for normal differences)
                    apr_advantage = -apr_gap / 500

                apr_advantage = np.clip(apr_advantage, -2.0, 0.5)  # Allow slight positive when current > best

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

                # 18. pnl_vs_peak_pct (V6: signals profit-taking opportunity)
                # Track peak P&L for this slot and calculate ratio
                current_pnl = unrealized_pnl_pct_raw  # Use raw percentage
                if slot_idx not in self._peak_pnl:
                    self._peak_pnl[slot_idx] = current_pnl
                else:
                    # Update peak if current P&L is higher
                    if current_pnl > self._peak_pnl[slot_idx]:
                        self._peak_pnl[slot_idx] = current_pnl

                peak_pnl = self._peak_pnl[slot_idx]
                if peak_pnl > 0.001:  # Avoid division by zero, only meaningful when peak is positive
                    pnl_vs_peak_pct = current_pnl / peak_pnl
                    pnl_vs_peak_pct = np.clip(pnl_vs_peak_pct, 0.0, 1.0)  # Clip to 0-1
                else:
                    pnl_vs_peak_pct = 1.0  # Default: at peak or no meaningful peak yet

                # 19. apr_sign_match (V7: Explicit APR direction flip indicator)
                # 1.0 = same sign as entry (good), -1.0 = sign flipped (bad - exit signal!)
                if entry_apr > 0 and current_position_apr_value < 0:
                    apr_sign_match = -1.0  # CRITICAL: Flipped from positive to negative!
                elif entry_apr < 0 and current_position_apr_value > 0:
                    apr_sign_match = -1.0  # Flipped from negative to positive
                else:
                    apr_sign_match = 1.0   # Same sign or both zero

                # 20. apr_velocity (V7: Rate of APR change - signals deterioration)
                # Negative = APR getting worse, positive = APR improving
                apr_velocity = 0.0
                if self.enable_velocity_tracking:
                    prev_apr = self._prev_position_apr.get(slot_idx, current_position_apr_value)
                    apr_velocity = (current_position_apr_value - prev_apr) / 100  # Normalize by 100% APR
                    apr_velocity = np.clip(apr_velocity, -1.0, 1.0)  # Clip to ±1
                    self._prev_position_apr[slot_idx] = current_position_apr_value

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
                    spread_change_from_entry,
                    liquidation_distance_pct,
                    apr_ratio,
                    current_position_apr,
                    best_available_apr_norm,
                    apr_advantage,
                    return_efficiency,
                    value_to_capital_ratio,
                    pnl_imbalance,
                    pnl_vs_peak_pct,  # V6: New feature
                    apr_sign_match,   # V7: APR direction flip indicator
                    apr_velocity,     # V7: APR deterioration rate
                ]
            else:
                # Empty slot - all zeros
                slot_features = [0.0] * DIMS.EXECUTIONS_PER_SLOT

            all_features.extend(slot_features)

        return np.array(all_features, dtype=np.float32)

    def build_opportunity_features(self, opportunities: List[Dict]) -> np.ndarray:
        """
        Build opportunity features for up to 10 slots (120 dimensions total).

        Each opportunity has 12 features (V5.4):
        1-6: fund_profit_8h, fund_profit_8h_24h_proj, fund_profit_8h_3d_proj,
             fund_apr, fund_apr_24h_proj, fund_apr_3d_proj
        7-10: spread_30_sample_avg, price_spread_24h_avg, price_spread_3d_avg, spread_volatility_stddev
        11: apr_velocity (fund_profit_8h - fund_profit_8h_24h_proj)
        12: spread_mean_reversion_potential = |spread_30| - |spread_3d| (sign-agnostic)

        CRITICAL: Feature scaler is applied to ALL slots (including empty ones)
        to match training behavior. Empty slots are padded with zeros BEFORE scaling.

        Args:
            opportunities: List of up to 10 opportunity dicts

        Returns:
            120-dimensional array (10 slots × 12 features)
        """
        all_features = []

        for slot_idx in range(DIMS.OPPORTUNITIES_SLOTS):
            if slot_idx < len(opportunities):
                opp = opportunities[slot_idx]

                fund_profit_8h = opp.get('fund_profit_8h', 0)
                fund_profit_8h_24h_proj = opp.get('fund_profit_8h_24h_proj', 0)
                spread_30_sample_avg = opp.get('spread_30_sample_avg', 0)
                price_spread_3d_avg = opp.get('price_spread_3d_avg', 0)

                # V5.4: Sign-agnostic spread mean-reversion potential
                # Positive = spread wider than 3d avg = mean-reversion opportunity
                # Works regardless of spread sign (+/-)
                spread_mean_reversion_potential = abs(spread_30_sample_avg) - abs(price_spread_3d_avg)
                spread_mean_reversion_potential = np.clip(spread_mean_reversion_potential, -0.05, 0.05)

                features = [
                    # Profit projections (6 features)
                    fund_profit_8h,
                    fund_profit_8h_24h_proj,
                    opp.get('fund_profit_8h_3d_proj', 0),
                    opp.get('fund_apr', 0),
                    opp.get('fund_apr_24h_proj', 0),
                    opp.get('fund_apr_3d_proj', 0),
                    # Spread metrics (4 features)
                    spread_30_sample_avg,
                    opp.get('price_spread_24h_avg', 0),
                    price_spread_3d_avg,
                    opp.get('spread_volatility_stddev', 0),
                    # Velocity (1 feature)
                    fund_profit_8h - fund_profit_8h_24h_proj,  # apr_velocity
                    # V5.4: Spread mean-reversion potential (1 feature)
                    spread_mean_reversion_potential,
                ]

                # Convert to float32 and ensure no NaN/inf
                features = [
                    float(np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0))
                    for x in features
                ]
            else:
                # Empty slot - all zeros (DO NOT SCALE - keep as zeros)
                # Scaling zeros produces non-zero values which misleads the model
                all_features.extend([0.0] * DIMS.OPPORTUNITIES_PER_SLOT)
                continue

            # Apply feature scaler ONLY to non-empty slots
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
