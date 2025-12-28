"""
Unified Feature Builder for RL Inference (V10)

This module provides the SINGLE SOURCE OF TRUTH for all feature engineering
used in the modular RL arbitrage system. All components (backend inference,
ML API server, training, and testing) must use this class to ensure consistency.

Architecture V10: 91-dimensional observation space
- Config: 5 dims
- Portfolio: 2 dims (min_liq_distance, time_to_next_funding_norm)
- Executions: 19 dims (1 slot × 19 features)
- Opportunities: 65 dims (5 slots × 13 features)

V10 Changes (Funding Timing):
- Added opportunity feature: time_to_profitable_funding (+1 per slot = +5)
- Uses actual next funding times from exchange data (not hardcoded 8h schedule)
- Calculates time to next funding on profitable side (long if rate<0, short if rate>0)
- Opportunity features per slot: 12 → 13
- Observation space: 86 → 91 dimensions

Key Principles:
1. All feature calculation logic lives HERE and only here
2. Backend sends raw data, this module transforms it to features
3. Training environment uses same code paths
4. Feature scaler is applied here for opportunity features (first 12 only)
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
        Build complete 91-dim observation vector from raw backend data (V10).

        This is the main entry point for converting raw data to model features.

        Args:
            raw_data: Dict containing:
                - 'trading_config': Dict with max_leverage, target_utilization, etc.
                - 'portfolio': Dict with positions, total_capital, capital_utilization
                - 'opportunities': List of opportunity dicts (with funding times and rates)
                - 'current_time': Optional datetime for time_to_profitable_funding calculation
            log_file: Optional path to log features for debugging

        Returns:
            91-dimensional observation array (5 + 2 + 19 + 65)
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
        opportunity_features = self.build_opportunity_features(opportunities, current_time)

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
        Build portfolio features (V9: 2 dimensions).

        Features:
        1. min_liq_distance: Minimum liquidation distance across all positions
        2. time_to_next_funding_norm: Minutes until next funding payment / 480 (0-1)

        V9: Removed num_positions_ratio and capital_utilization

        Args:
            portfolio: Portfolio state dict
            trading_config: Trading configuration dict
            current_time: Optional datetime for funding time calculation

        Returns:
            2-dimensional array
        """
        # Count only ACTIVE positions, not empty padded slots
        positions = portfolio.get('positions', [])
        num_positions = sum(
            1 for p in positions
            if p.get('is_active', False)
        )

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

        # V6: Calculate time to next funding payment
        time_to_next_funding_norm = self._calc_time_to_next_funding(current_time)

        return np.array([
            min_liq_distance,
            time_to_next_funding_norm,
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
        Build execution features for 1 position slot (19 dimensions total, V9).

        Each position has 19 features:
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
        11. apr_ratio: current_apr / entry_apr (clipped to [-3,3])
        12. current_position_apr (normalized to ±5000%)
        13. best_available_apr_norm (normalized to ±5000%)
        14. apr_advantage: current - best
        15. return_efficiency: P&L per hour held (clipped to ±50)
        16. pnl_imbalance: (long_pnl - short_pnl) / 200
        17. pnl_vs_peak_pct: current P&L / peak P&L (signals profit-taking opportunity)
        18. apr_sign_match: APR direction flip indicator
        19. apr_velocity: APR deterioration rate

        V9: Removed value_to_capital_ratio, reduced to 1 slot

        Args:
            portfolio: Portfolio state dict with positions list
            best_available_apr: Maximum APR among current opportunities

        Returns:
            19-dimensional array (1 slot × 19 features)
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

                # ===== Calculate features =====

                # 1. is_active
                is_active_feat = 1.0

                # Get position size and leverage for P&L calculation
                position_size = pos.get('position_size_usd', 0.0)  # This is margin per leg
                leverage = pos.get('leverage', 1.0)
                notional_per_leg = position_size * leverage  # Actual position value per leg

                # Calculate PRICE P&L from entry/current prices (using NOTIONAL, not margin)
                long_price_pnl_usd = 0.0
                short_price_pnl_usd = 0.0
                long_price_pnl_pct = 0.0
                short_price_pnl_pct = 0.0
                if entry_long_price > 0 and entry_short_price > 0:
                    slippage_pct = pos.get('slippage_pct', 0.0)

                    # Apply slippage ONLY on exit (not entry)
                    effective_current_long = current_long_price * (1 - slippage_pct)  # Receive less at exit
                    effective_current_short = current_short_price * (1 + slippage_pct)  # Pay more at exit

                    # Price change percentages
                    long_price_pnl_pct = (effective_current_long - entry_long_price) / entry_long_price
                    short_price_pnl_pct = (entry_short_price - effective_current_short) / entry_short_price

                    # P&L in USD (using NOTIONAL value, not margin)
                    long_price_pnl_usd = notional_per_leg * long_price_pnl_pct
                    short_price_pnl_usd = notional_per_leg * short_price_pnl_pct

                # Get raw funding earned (from backend - actual exchange data)
                long_funding_earned_usd = pos.get('long_funding_earned_usd', 0.0)
                short_funding_earned_usd = pos.get('short_funding_earned_usd', 0.0)
                net_funding_usd = long_funding_earned_usd + short_funding_earned_usd

                # Get raw fees (from backend - actual exchange data)
                long_fees_usd = pos.get('long_fees_usd', 0.0)
                short_fees_usd = pos.get('short_fees_usd', 0.0)
                total_fees_usd = long_fees_usd + short_fees_usd

                # Calculate TOTAL unrealized P&L (price + funding - fees)
                unrealized_pnl_usd = long_price_pnl_usd + short_price_pnl_usd + net_funding_usd - total_fees_usd
                unrealized_pnl_pct_raw = (unrealized_pnl_usd / total_capital_used * 100) if total_capital_used > 0 else 0.0

                # 2. net_pnl_pct (for feature output - divide by 100 to get decimal)
                net_pnl_pct = unrealized_pnl_pct_raw / 100  # Convert to decimal (e.g., 0.025)

                # 3. hours_held_norm (log scale)
                hours_held_norm = np.log(raw_hours_held + 1) / np.log(CONFIG.HOURS_HELD_LOG_BASE + 63)

                # 4. estimated_pnl_pct (price P&L only - no funding, no fees)
                estimated_pnl_pct = (long_price_pnl_usd + short_price_pnl_usd) / total_capital_used if total_capital_used > 0 else 0.0

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

                # V9: Removed value_to_capital_ratio

                # 16. pnl_imbalance (calculated from price P&L percentages)
                # long_price_pnl_pct and short_price_pnl_pct calculated above from entry/current prices
                pnl_imbalance = (long_price_pnl_pct * 100 - short_price_pnl_pct * 100) / 200

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
                # V9: 19 features (removed value_to_capital_ratio)
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
                    # V9: Removed value_to_capital_ratio
                    pnl_imbalance,
                    pnl_vs_peak_pct,
                    apr_sign_match,
                    apr_velocity,
                ]
            else:
                # Empty slot - all zeros
                slot_features = [0.0] * DIMS.EXECUTIONS_PER_SLOT

            all_features.extend(slot_features)

        return np.array(all_features, dtype=np.float32)

    def build_opportunity_features(self, opportunities: List[Dict], current_time: Optional[datetime] = None) -> np.ndarray:
        """
        Build opportunity features for up to 5 slots (65 dimensions total, V10).

        Each opportunity has 13 features:
        1-6: fund_profit_8h, fund_profit_8h_24h_proj, fund_profit_8h_3d_proj,
             fund_apr, fund_apr_24h_proj, fund_apr_3d_proj
        7-10: spread_30_sample_avg, price_spread_24h_avg, price_spread_3d_avg, spread_volatility_stddev
        11: apr_velocity (fund_profit_8h - fund_profit_8h_24h_proj)
        12: spread_mean_reversion_potential = |spread_30| - |spread_3d| (sign-agnostic)
        13: time_to_profitable_funding = minutes to next funding on profitable side / 480 (V10)

        CRITICAL: Feature scaler is applied to first 12 features only.
        time_to_profitable_funding is already normalized 0-1 and added after scaling.

        Args:
            opportunities: List of up to 5 opportunity dicts
            current_time: Current datetime for funding time calculation

        Returns:
            65-dimensional array (5 slots × 13 features)
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

                # First 12 features (to be scaled)
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

                # Apply feature scaler to first 12 features ONLY
                if self.feature_scaler is not None:
                    features_array = np.array(features, dtype=np.float32)
                    features_scaled = self.feature_scaler.transform(
                        features_array.reshape(1, 12)  # Scaler expects 12 features
                    )
                    features = features_scaled.flatten().tolist()

                # V10: Calculate time_to_profitable_funding (not scaled - already normalized)
                time_to_profitable_funding = self._calc_time_to_profitable_funding(opp, current_time)
                features.append(time_to_profitable_funding)

                all_features.extend(features)
            else:
                # Empty slot - all zeros
                all_features.extend([0.0] * DIMS.OPPORTUNITIES_PER_SLOT)

        return np.array(all_features, dtype=np.float32)

    def _calc_time_to_profitable_funding(self, opp: Dict, current_time: Optional[datetime]) -> float:
        """
        Calculate normalized time until next funding payment on the profitable side (V10).

        For funding arbitrage:
        - Long side profitable when long_funding_rate < 0 (we receive funding)
        - Short side profitable when short_funding_rate > 0 (we receive funding)

        Returns the time to the SOONEST profitable funding payment.

        Time source priority:
        1. Opportunity's entry_time (for training/backtesting with historical data)
        2. Passed current_time parameter (fallback)
        3. datetime.utcnow() (for live inference)

        Args:
            opp: Opportunity dict with funding rates, next funding times, and optionally entry_time
            current_time: Fallback datetime (UTC) if entry_time not in opportunity

        Returns:
            Normalized value 0-1 (0 = funding imminent, 1 = ~8h away)
            Returns 0.5 if no time available
        """
        import pandas as pd
        from datetime import datetime as dt, timezone

        # Priority: opportunity's entry_time > passed current_time > now (UTC)
        ref_time = opp.get('entry_time')
        if ref_time is None:
            ref_time = current_time
        if ref_time is None:
            ref_time = dt.now(timezone.utc)  # Use timezone-aware UTC

        # Get funding rates to determine which side(s) are profitable
        long_rate = opp.get('long_funding_rate', 0.0)
        short_rate = opp.get('short_funding_rate', 0.0)

        # Get next funding times
        long_next_funding = opp.get('long_next_funding_time')
        short_next_funding = opp.get('short_next_funding_time')

        # Convert to pandas Timestamp if needed
        if long_next_funding is not None and not isinstance(long_next_funding, pd.Timestamp):
            try:
                long_next_funding = pd.to_datetime(long_next_funding)
            except:
                long_next_funding = None

        if short_next_funding is not None and not isinstance(short_next_funding, pd.Timestamp):
            try:
                short_next_funding = pd.to_datetime(short_next_funding)
            except:
                short_next_funding = None

        # Convert ref_time to pandas Timestamp if needed
        if not isinstance(ref_time, pd.Timestamp):
            ref_time = pd.to_datetime(ref_time)

        # Calculate minutes to each profitable side
        profitable_times = []

        # Long is profitable if rate < 0 (we receive)
        if long_rate < 0 and long_next_funding is not None:
            try:
                minutes_to_long = (long_next_funding - ref_time).total_seconds() / 60
                if minutes_to_long > 0:
                    profitable_times.append(minutes_to_long)
            except:
                pass

        # Short is profitable if rate > 0 (we receive)
        if short_rate > 0 and short_next_funding is not None:
            try:
                minutes_to_short = (short_next_funding - ref_time).total_seconds() / 60
                if minutes_to_short > 0:
                    profitable_times.append(minutes_to_short)
            except:
                pass

        if not profitable_times:
            # No profitable funding coming - return 1.0 (max wait)
            # This could happen if both rates are unfavorable
            return 1.0

        # Return minimum (soonest profitable funding)
        min_minutes = min(profitable_times)

        # Normalize by 480 minutes (8 hours max)
        return min(min_minutes / 480.0, 1.0)

    def get_action_mask(
        self,
        opportunities: List[Dict],
        num_positions: int,
        max_positions: int,
        current_time: Optional[datetime] = None,
        max_minutes_to_funding: float = 30.0,
        min_apr: float = 2500.0
    ) -> np.ndarray:
        """
        Generate action mask (17 dimensions, V10).

        Action space:
        - 0: HOLD
        - 1-5: ENTER_OPP_0-4_SMALL
        - 6-10: ENTER_OPP_0-4_MEDIUM
        - 11-15: ENTER_OPP_0-4_LARGE
        - 16: EXIT_POS_0

        ENTER masking criteria (V10):
        - Opportunity must have fund_apr >= min_apr
        - Time to profitable funding must be <= max_minutes_to_funding
        - Must have position capacity
        - Must not have existing position for same symbol

        Args:
            opportunities: List of opportunity dicts
            num_positions: Current number of active positions
            max_positions: Maximum allowed positions
            current_time: Fallback datetime for funding time calculation
            max_minutes_to_funding: Max minutes until next profitable funding (default: 30)
            min_apr: Minimum APR required to enter (default: 2500)

        Returns:
            Boolean array (17,) where True = valid action
        """
        mask = np.zeros(DIMS.TOTAL_ACTIONS, dtype=bool)

        # HOLD is always valid
        mask[DIMS.ACTION_HOLD] = True

        # APR + TIME MASKING (min 2500%, max 30min to funding)
        effective_max_positions = min(max_positions, DIMS.EXECUTIONS_SLOTS)
        has_capacity = num_positions < effective_max_positions

        if has_capacity:
            for i in range(DIMS.OPPORTUNITIES_SLOTS):
                if i < len(opportunities):
                    opp = opportunities[i]

                    if opp.get('has_existing_position', False):
                        continue

                    fund_apr = opp.get('fund_apr', 0.0)
                    if fund_apr < min_apr:
                        continue

                    time_to_funding_norm = self._calc_time_to_profitable_funding(opp, current_time)
                    minutes_to_funding = time_to_funding_norm * 480.0
                    if minutes_to_funding > max_minutes_to_funding:
                        continue

                    mask[1 + i] = True
                    mask[6 + i] = True
                    mask[11 + i] = True

        # EXIT actions
        for i in range(DIMS.EXECUTIONS_SLOTS):
            if i < num_positions:
                mask[DIMS.ACTION_EXIT_START + i] = True

        return mask
