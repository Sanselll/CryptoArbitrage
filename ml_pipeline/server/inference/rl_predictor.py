"""
Modular RL Predictor - Uses trained ModularPPONetwork for inference (V3)

Loads and runs the trained modular PPO model to provide action predictions
for funding arbitrage opportunities.

Architecture (V3 Refactoring: 301→203 dimensions):
- 203-dim observation space:
  * Config: 5 dims (max_leverage, target_utilization, max_positions, stop_loss, liq_buffer)
  * Portfolio: 3 dims (num_positions_ratio, liq_distance, utilization) [V3: removed historical metrics]
  * Executions: 85 dims (5 positions × 17 features each) [V3: 20→17 features, added velocities]
  * Opportunities: 110 dims (10 opportunities × 11 features each) [V3: removed market quality]

- 36 actions:
  * 0: HOLD
  * 1-10: ENTER_OPP_0-9_SMALL (10% of max allowed size)
  * 11-20: ENTER_OPP_0-9_MEDIUM (20% of max allowed size)
  * 21-30: ENTER_OPP_0-9_LARGE (30% of max allowed size)
  * 31-35: EXIT_POS_0-4

Usage:
    predictor = ModularRLPredictor('checkpoints/best_model.pt')
    predictions = predictor.predict_opportunities(opportunities, portfolio, trading_config)
"""

import numpy as np
import torch
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import ML model components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer
from models.rl.core.config import TradingConfig


# Action space mapping
ACTION_NAMES = {
    0: 'HOLD',
    **{i: f'ENTER_OPP_{i-1}_SMALL' for i in range(1, 11)},
    **{i: f'ENTER_OPP_{i-11}_MEDIUM' for i in range(11, 21)},
    **{i: f'ENTER_OPP_{i-21}_LARGE' for i in range(21, 31)},
    **{i: f'EXIT_POS_{i-31}' for i in range(31, 36)},
}

# Position sizes (as % of max allowed size per side)
SIZE_MULTIPLIERS = {
    'SMALL': 0.10,   # 10%
    'MEDIUM': 0.20,  # 20%
    'LARGE': 0.30,   # 30%
}


class ModularRLPredictor:
    """
    RL Model predictor using ModularPPONetwork for crypto arbitrage opportunities.

    Processes up to 10 opportunities simultaneously and returns action predictions
    with probabilities, confidence scores, and recommended position sizes.
    """

    def __init__(self, model_path: str = 'checkpoints/v3_production/best_model.pt',
                 feature_scaler_path: str = 'trained_models/rl/feature_scaler_v2.pkl',
                 device: str = 'cpu'):
        """
        Initialize the modular RL predictor (V3).

        Args:
            model_path: Path to trained PPOTrainer checkpoint (.pt file)
            feature_scaler_path: Path to fitted StandardScaler pickle (V3: 11 features, was 19)
            device: Device to use ('cpu' or 'cuda')
        """
        print(f"Loading Modular RL model from: {model_path}")

        self.device = device

        # Load feature scaler (critical for proper predictions!)
        self.feature_scaler = None
        if feature_scaler_path is not None:
            scaler_path = Path(feature_scaler_path)
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print(f"✅ Feature scaler loaded from: {feature_scaler_path}")
                print(f"   Features will be standardized (mean=0, std=1)")
            else:
                print(f"⚠️  WARNING: Feature scaler not found at: {feature_scaler_path}")
                print(f"   Using raw features (may cause poor predictions!)")

        # Create network
        self.network = ModularPPONetwork()

        # Create trainer and load checkpoint
        self.trainer = PPOTrainer(
            network=self.network,
            learning_rate=3e-4,
            device=device
        )

        # Load trained weights
        checkpoint_path = Path(model_path)
        if checkpoint_path.exists():
            self.trainer.load(str(checkpoint_path))
            print(f"✅ Model loaded successfully")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        # Set to evaluation mode
        self.network.eval()

        # Compile network for faster inference (PyTorch 2.0+)
        # Using 'default' mode for balanced compilation
        try:
            self.network = torch.compile(self.network, mode='default')
            print("✅ Model compiled with torch.compile for faster inference")
        except Exception as e:
            print(f"⚠️  torch.compile not available (requires PyTorch 2.0+): {e}")

        # Get model info
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"   Network parameters: {total_params:,}")
        print(f"   Observation space: 203 dimensions (V3: 301→203, streamlined features)")
        print(f"   Action space: 36 actions")

    def _build_config_features(self, trading_config: Dict) -> np.ndarray:
        """
        Build config features from trading configuration.

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

    def _build_portfolio_features(self, portfolio: Dict, trading_config: Dict) -> np.ndarray:
        """
        Build portfolio features (V3: 3 dimensions, was 6).

        V3: Removed avg_position_pnl_pct, total_pnl_pct, max_drawdown_pct (historical metrics)

        Args:
            portfolio: Portfolio state dict
            trading_config: Trading configuration dict

        Returns:
            3-dimensional array (V3: was 6)
        """
        # Count only ACTIVE positions, not empty padded slots
        positions = portfolio.get('positions', [])
        num_positions = sum(1 for p in positions if p.get('is_active', False) or p.get('position_is_active', 0.0) > 0.5)
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

    def _build_execution_features(self, portfolio: Dict, best_available_apr: float = 0.0) -> np.ndarray:
        """
        Build execution features for up to 5 position slots (85 dimensions total - V3 refactoring).

        V3 Changes (20→17 features per slot):
        - Removed: net_funding_ratio, net_funding_rate, funding_efficiency, entry_spread_pct,
                   long/short_pnl_pct, old pnl_velocity, peak_drawdown, is_old_loser
        - Added: estimated_pnl_pct, estimated_pnl_velocity, estimated_funding_8h_pct,
                 funding_velocity, spread_velocity, pnl_imbalance
        - Updated: hours_held (log norm), APR (clip ±5000%)

        Each position has 17 features:
        1. is_active
        2. net_pnl_pct = (price P&L + funding - fees) / capital
        3. hours_held_norm = log(hours + 1) / log(73)
        4. estimated_pnl_pct = price P&L only (no fees, no funding) [NEW]
        5. estimated_pnl_velocity = change in estimated P&L [NEW]
        6. estimated_funding_8h_pct = expected funding profit in next 8h [NEW]
        7. funding_velocity = change in 8h funding estimate [NEW]
        8. spread_pct = current price spread
        9. spread_velocity = change in spread [NEW]
        10. liquidation_distance_pct
        11. apr_ratio = current_apr / entry_apr (clipped to [0,3])
        12. current_position_apr (normalized to ±5000%)
        13. best_available_apr_norm (normalized to ±5000%)
        14. apr_advantage = current - best
        15. return_efficiency = P&L per hour held (clipped to ±50)
        16. value_to_capital_ratio = capital allocated to this position
        17. pnl_imbalance = (long_pnl - short_pnl) / 200 [NEW]

        Args:
            portfolio: Portfolio state dict
            best_available_apr: Maximum APR among current opportunities (for APR comparison)

        Returns:
            85-dimensional array (5 slots × 17 features)
        """
        positions = portfolio.get('positions', [])
        capital = portfolio.get('total_capital', portfolio.get('capital', 10000.0))

        all_features = []

        for slot_idx in range(5):
            if slot_idx < len(positions):
                pos = positions[slot_idx]

                # Check if position is ACTUALLY active (not just a padded empty slot)
                is_active = pos.get('is_active', False) or pos.get('position_is_active', 0.0) > 0.5
                if not is_active:
                    # Empty/inactive slot - all zeros (matches environment's behavior)
                    slot_features = [0.0] * 17  # V3: 17 features per slot
                    all_features.extend(slot_features)
                    continue

                # Get position data
                total_capital_used = pos.get('position_size_usd', 0.0) * 2
                raw_hours_held = pos.get('position_age_hours', pos.get('hours_held', 0.0))

                # Prices
                current_long_price = pos.get('current_long_price', pos.get('current_price', 0.0))
                current_short_price = pos.get('current_short_price', pos.get('current_price', 0.0))
                entry_long_price = pos.get('entry_long_price', pos.get('entry_price', 0.0))
                entry_short_price = pos.get('entry_short_price', pos.get('entry_price', 0.0))

                # ==================================================================
                # V3 FEATURES (17 dimensions)
                # ==================================================================

                # 1. is_active
                is_active_feat = 1.0

                # 2. net_pnl_pct = (price P&L + funding - fees) / capital
                net_pnl_pct = pos.get('unrealized_pnl_pct', 0.0) / 100

                # 3. hours_held_norm = log(hours + 1) / log(73)
                # V3: Changed from linear /72 to log normalization
                hours_held_norm = np.log(raw_hours_held + 1) / np.log(73)

                # 4. estimated_pnl_pct = price P&L only (no fees, no funding)
                # V3: NEW FEATURE - isolates price risk from funding profit
                long_price_pnl = 0.0
                short_price_pnl = 0.0
                if entry_long_price > 0 and entry_short_price > 0:
                    position_size = pos.get('position_size_usd', 0.0)
                    long_price_pnl = position_size * ((current_long_price - entry_long_price) / entry_long_price)
                    short_price_pnl = position_size * ((entry_short_price - current_short_price) / entry_short_price)
                estimated_pnl_pct = ((long_price_pnl + short_price_pnl) / total_capital_used) / 100 if total_capital_used > 0 else 0.0

                # 5. estimated_pnl_velocity = change in estimated P&L
                # V3: DISABLED - removed to eliminate state tracking complexity
                estimated_pnl_velocity = 0.0  # (estimated_pnl_pct - prev_estimated_pnl_pct) / 100

                # 6. estimated_funding_8h_pct = expected funding profit in next 8h
                # V3: NEW FEATURE - replaces confusing net_funding_rate
                # Calculate: (long_payments * -long_rate + short_payments * short_rate) * 100
                long_interval = pos.get('long_funding_interval_hours', 8)
                short_interval = pos.get('short_funding_interval_hours', 8)
                long_rate = pos.get('long_funding_rate', 0.0)
                short_rate = pos.get('short_funding_rate', 0.0)

                long_payments_8h = 8.0 / long_interval if long_interval > 0 else 1.0
                short_payments_8h = 8.0 / short_interval if short_interval > 0 else 1.0
                long_funding_8h = -long_rate * long_payments_8h
                short_funding_8h = short_rate * short_payments_8h
                estimated_funding_8h_pct = (long_funding_8h + short_funding_8h) * 100

                # 7. funding_velocity = change in 8h funding estimate
                # V3: DISABLED - removed to eliminate state tracking complexity
                funding_velocity = 0.0  # (estimated_funding_8h_pct - prev_funding_8h_pct) / 100

                # 8. spread_pct = current price spread
                # V3: Renamed from current_spread_pct for clarity
                avg_price = (current_long_price + current_short_price) / 2
                spread_pct = abs(current_long_price - current_short_price) / avg_price if avg_price > 0 else 0.0

                # 9. spread_velocity = change in spread
                # V3: DISABLED - removed to eliminate state tracking complexity
                spread_velocity = 0.0  # spread_pct - prev_spread_pct

                # 10. liquidation_distance_pct
                liquidation_distance_pct = pos.get('liquidation_distance', 1.0)
                if liquidation_distance_pct is None or liquidation_distance_pct == 0.0:
                    # Calculate from leverage and prices
                    leverage = pos.get('leverage', 1.0)
                    if leverage > 0 and entry_long_price > 0 and entry_short_price > 0:
                        # Calculate liquidation prices
                        long_liq = entry_long_price * (1 - 0.9 / leverage)
                        short_liq = entry_short_price * (1 + 0.9 / leverage)

                        # Calculate distances
                        long_dist = abs(current_long_price - long_liq) / current_long_price if current_long_price > 0 else 1.0
                        short_dist = abs(short_liq - current_short_price) / current_short_price if current_short_price > 0 else 1.0

                        liquidation_distance_pct = min(long_dist, short_dist)
                    else:
                        liquidation_distance_pct = 1.0

                # 11. apr_ratio = current_apr / entry_apr (funding rate deterioration)
                # Look up current APR for this symbol from current_position_apr
                current_position_apr_value = pos.get('current_position_apr', 0.0)
                entry_apr = pos.get('entry_apr', 0.0)
                if entry_apr > 0:
                    apr_ratio_raw = current_position_apr_value / entry_apr
                else:
                    apr_ratio_raw = 1.0
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
                if raw_hours_held > 0:
                    return_efficiency_raw = net_pnl_pct / raw_hours_held
                else:
                    return_efficiency_raw = 0.0
                return_efficiency = np.clip(return_efficiency_raw, -50, 50) / 50

                # 16. value_to_capital_ratio = capital allocated to this position
                value_to_capital_ratio = total_capital_used / capital if capital > 0 else 0.0

                # 17. pnl_imbalance = (long_pnl - short_pnl) / 200
                # V3: NEW FEATURE - detects directional exposure (arbitrage breaking down)
                long_pnl_pct = pos.get('long_pnl_pct', 0.0)
                short_pnl_pct = pos.get('short_pnl_pct', 0.0)
                pnl_imbalance = (long_pnl_pct - short_pnl_pct) / 200

                # CRITICAL: Feature order MUST match environment.py get_execution_state() exactly!
                slot_features = [
                    is_active_feat,             # 1
                    net_pnl_pct,                # 2
                    hours_held_norm,            # 3
                    estimated_pnl_pct,          # 4 (NEW)
                    estimated_pnl_velocity,     # 5 (NEW)
                    estimated_funding_8h_pct,   # 6 (NEW)
                    funding_velocity,           # 7 (NEW)
                    spread_pct,                 # 8
                    spread_velocity,            # 9 (NEW)
                    liquidation_distance_pct,   # 10
                    apr_ratio,                  # 11
                    current_position_apr,       # 12
                    best_available_apr_norm,    # 13
                    apr_advantage,              # 14
                    return_efficiency,          # 15
                    value_to_capital_ratio,     # 16
                    pnl_imbalance,              # 17 (NEW)
                ]
            else:
                # Empty slot - all zeros
                slot_features = [0.0] * 17  # V3: 17 features per slot

            all_features.extend(slot_features)

        return np.array(all_features, dtype=np.float32)

    def _build_opportunity_features(self, opportunities: List[Dict]) -> np.ndarray:
        """
        Build opportunity features for up to 10 slots (110 dimensions total - V3 refactoring).

        V3 Changes (19→11 features per slot):
        - Removed: long_funding_rate, short_funding_rate, long_interval, short_interval,
                   volume_24h, bidAskSpreadPercent, orderbookDepthUsd,
                   estimatedProfitPercentage, positionCostPercent (9 features)
        - Added: apr_velocity (fund_profit_8h - fundProfit8h24hProj)
        - Kept: 6 funding profit/APR projections, 4 spread metrics

        Each opportunity has 11 features:
        1-6: fund_profit_8h, fundProfit8h24hProj, fundProfit8h3dProj,
             fund_apr, fundApr24hProj, fundApr3dProj
        7-10: spread30SampleAvg, priceSpread24hAvg, priceSpread3dAvg, spread_volatility_stddev
        11: apr_velocity (fund_profit_8h - fundProfit8h24hProj)

        Features are standardized using the feature scaler if available.

        CRITICAL: Feature scaler must be applied to ALL slots (including empty ones)
        to match training behavior. Empty slots are padded with zeros BEFORE scaling.

        Args:
            opportunities: List of up to 10 opportunity dicts

        Returns:
            110-dimensional array (10 slots × 11 features)
        """
        all_features = []

        for slot_idx in range(10):
            if slot_idx < len(opportunities):
                opp = opportunities[slot_idx]

                # 11 features (V3: removed 9 market quality, added 1 velocity)
                # Assumes market quality pre-filtering upstream
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
                    # Velocity (1 feature - NEW in V3)
                    fund_profit_8h - fundProfit8h24hProj,  # apr_velocity
                ]

                # Convert to float32 and ensure no NaN/inf
                features = [float(np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)) for x in features]
            else:
                # Empty slot - all zeros BEFORE scaling
                features = [0.0] * 11  # V3: 11 features per slot

            # CRITICAL FIX: Apply feature scaler to ALL slots (including empty ones)
            # This matches the training environment behavior where scaler transforms
            # all 10 slots regardless of whether they contain real opportunities
            # V3: Scaler expects 11 features (was 19)
            if self.feature_scaler is not None:
                features_array = np.array(features, dtype=np.float32)
                features_scaled = self.feature_scaler.transform(features_array.reshape(1, 11))
                features = features_scaled.flatten().tolist()

            all_features.extend(features)

        return np.array(all_features, dtype=np.float32)

    def _build_observation(
        self,
        trading_config: Dict,
        portfolio: Dict,
        opportunities: List[Dict]
    ) -> np.ndarray:
        """
        Build complete 203-dim observation vector (V3 refactoring: was 301).

        V3 Changes:
        - Portfolio: 6→3 dims (removed historical metrics)
        - Execution: 100→85 dims (5 slots × 20→17 features)
        - Opportunity: 190→110 dims (10 slots × 19→11 features)

        Args:
            trading_config: Trading configuration dict
            portfolio: Portfolio state dict
            opportunities: List of up to 10 opportunity dicts

        Returns:
            203-dimensional observation array (5 + 3 + 85 + 110 = 203)
        """
        # Config features (5)
        config_features = self._build_config_features(trading_config)

        # Portfolio features (3 - V3: was 6)
        portfolio_features = self._build_portfolio_features(portfolio, trading_config)

        # Calculate best available APR from opportunities (for APR comparison features)
        best_available_apr = 0.0
        if len(opportunities) > 0:
            best_available_apr = max(opp.get('fund_apr', 0.0) for opp in opportunities)

        # Execution features (85 = 5 slots × 17 features - V3: was 100)
        execution_features = self._build_execution_features(portfolio, best_available_apr)

        # Opportunity features (110 = 10 slots × 11 features - V3: was 190)
        opportunity_features = self._build_opportunity_features(opportunities)

        # Concatenate all (5 + 3 + 85 + 110 = 203)
        observation = np.concatenate([
            config_features,
            portfolio_features,
            execution_features,
            opportunity_features
        ]).astype(np.float32)

        return observation

    def _get_action_mask(
        self,
        opportunities: List[Dict],
        num_positions: int,
        max_positions: int
    ) -> np.ndarray:
        """
        Generate action mask (36 dimensions).

        IMPORTANT: Must match environment.py _get_action_mask() logic EXACTLY.

        Args:
            opportunities: List of opportunity dicts
            num_positions: Current number of positions (len(portfolio.positions))
            max_positions: Maximum allowed positions

        Returns:
            Boolean array (36,) where True = valid action
        """
        mask = np.zeros(36, dtype=bool)

        # HOLD is always valid
        mask[0] = True

        # ENTER actions: valid if opportunity exists AND we can open position
        has_capacity = num_positions < max_positions

        if has_capacity:
            for i in range(10):
                if i < len(opportunities):
                    # Check if this opportunity already has an existing position
                    # to prevent duplicate positions for the same symbol
                    if not opportunities[i].get('has_existing_position', False):
                        mask[1 + i] = True      # SMALL
                        mask[11 + i] = True     # MEDIUM
                        mask[21 + i] = True     # LARGE

        # EXIT actions: valid if position exists
        for i in range(5):
            if i < num_positions:
                mask[31 + i] = True

        return mask

    def _decode_action(self, action: int) -> Dict[str, Any]:
        """
        Decode action ID to human-readable format.

        Args:
            action: Action ID (0-35)

        Returns:
            Dict with 'type', 'opportunity_index', 'position_index', 'size'
        """
        if action == 0:
            return {
                'type': 'HOLD',
                'opportunity_index': None,
                'position_index': None,
                'size': None,
            }
        elif 1 <= action <= 10:
            return {
                'type': 'ENTER',
                'opportunity_index': action - 1,
                'position_index': None,
                'size': 'SMALL',
            }
        elif 11 <= action <= 20:
            return {
                'type': 'ENTER',
                'opportunity_index': action - 11,
                'position_index': None,
                'size': 'MEDIUM',
            }
        elif 21 <= action <= 30:
            return {
                'type': 'ENTER',
                'opportunity_index': action - 21,
                'position_index': None,
                'size': 'LARGE',
            }
        elif 31 <= action <= 35:
            return {
                'type': 'EXIT',
                'opportunity_index': None,
                'position_index': action - 31,
                'size': None,
            }
        else:
            # Invalid action
            return {
                'type': 'UNKNOWN',
                'opportunity_index': None,
                'position_index': None,
                'size': None,
            }

    def _log_position_features_readable(self, portfolio: Dict, obs: np.ndarray, opportunities: List[Dict]):
        """
        Log position features in human-readable format for debugging (V3).

        Logs raw feature values, normalized values sent to model, and APR calculation details.
        Helps identify feature distribution mismatches between training and production.

        Args:
            portfolio: Portfolio state dict with positions
            obs: Full observation vector (203 dims - V3: was 301)
            opportunities: List of opportunity dicts for best_available_apr
        """
        try:
            import datetime
            from pathlib import Path

            # Log to project directory
            log_path = Path(__file__).parent.parent.parent / 'production_features.log'

            # Get active positions
            positions = portfolio.get('positions', [])
            active_positions = [p for p in positions if p.get('is_active', False) or p.get('position_is_active', 0.0) > 0.5]

            if not active_positions:
                return  # Nothing to log

            # Find best available APR from opportunities
            best_available_apr = 0.0
            if opportunities:
                best_available_apr = max(opp.get('fund_apr', 0.0) for opp in opportunities)

            # Feature names for execution features (17 features per position - V3)
            feature_names = [
                "is_active",
                "net_pnl_pct",
                "hours_held_norm",
                "estimated_pnl_pct",
                "estimated_pnl_velocity",
                "estimated_funding_8h_pct",
                "funding_velocity",
                "spread_pct",
                "spread_velocity",
                "liquidation_distance_pct",
                "apr_ratio",
                "current_position_apr",
                "best_available_apr_norm",
                "apr_advantage",
                "return_efficiency",
                "value_to_capital_ratio",
                "pnl_imbalance"
            ]

            with open(log_path, 'a') as f:
                f.write("=" * 80 + "\n")
                f.write(f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC - Position Features (V3)\n")
                f.write("=" * 80 + "\n")

                for slot_idx, pos in enumerate(active_positions):
                    symbol = pos.get('symbol', 'UNKNOWN')
                    raw_hours_held = pos.get('position_age_hours', pos.get('hours_held', 0.0))
                    entry_apr = pos.get('entry_apr', 0.0)

                    f.write(f"\nSymbol: {symbol}\n")
                    f.write(f"Hours Held: {raw_hours_held:.2f}h ({raw_hours_held*60:.0f} minutes)\n")
                    f.write(f"Entry APR: {entry_apr:.0f}%\n")
                    f.write(f"\n")

                    # Extract normalized features from observation vector
                    # V3 Observation structure: config(5) + portfolio(3) + execution(85) + opportunities(110)
                    # Execution features start at index 8, each position has 17 features
                    feat_start_idx = 8 + (slot_idx * 17)
                    normalized_features = obs[feat_start_idx:feat_start_idx+17]

                    # Get raw values from position dict
                    net_pnl_pct_raw = pos.get('unrealized_pnl_pct', 0.0)
                    long_funding_rate = pos.get('long_funding_rate', 0.0)
                    short_funding_rate = pos.get('short_funding_rate', 0.0)
                    current_position_apr = pos.get('current_position_apr', 0.0)
                    long_pnl_pct = pos.get('long_pnl_pct', 0.0)
                    short_pnl_pct = pos.get('short_pnl_pct', 0.0)

                    # Write raw features (V3: 17 features)
                    f.write("Raw Features (before normalization) - V3:\n")
                    f.write(f"  1. is_active                  : {normalized_features[0]:.4f}\n")
                    f.write(f"  2. net_pnl_pct                : {net_pnl_pct_raw:.4f}%\n")
                    f.write(f"  3. hours_held                 : {raw_hours_held:.4f}h (log-normalized)\n")
                    f.write(f"  4. estimated_pnl_pct          : (price P&L only)\n")
                    f.write(f"  5. estimated_pnl_velocity     : (NEW - trend signal)\n")
                    f.write(f"  6. estimated_funding_8h_pct   : (NEW - 8h funding estimate)\n")
                    f.write(f"  7. funding_velocity           : (NEW - funding trend)\n")
                    f.write(f"  8. spread_pct                 : {normalized_features[7]:.6f}\n")
                    f.write(f"  9. spread_velocity            : (NEW - spread trend)\n")
                    f.write(f" 10. liquidation_distance_pct   : {normalized_features[9]:.4f}\n")
                    f.write(f" 11. apr_ratio                  : {normalized_features[10]:.4f}\n")
                    f.write(f" 12. current_position_apr       : {current_position_apr:.2f}% (±5000% range)\n")
                    f.write(f" 13. best_available_apr_norm    : {best_available_apr:.2f}% (±5000% range)\n")
                    f.write(f" 14. apr_advantage              : {current_position_apr - best_available_apr:.2f}%\n")
                    f.write(f" 15. return_efficiency          : (P&L per hour, clipped ±50)\n")
                    f.write(f" 16. value_to_capital_ratio     : {normalized_features[15]:.4f}\n")
                    f.write(f" 17. pnl_imbalance              : (NEW - long vs short P&L)\n")

                    f.write(f"\nNormalized Features (sent to model) - V3:\n")
                    for i, (name, val) in enumerate(zip(feature_names, normalized_features)):
                        f.write(f" {i+1:2d}. {name:30s}: {val:12.6f}\n")

                    f.write(f"\nV3 Feature Details:\n")
                    f.write(f"  long_funding_rate             : {long_funding_rate:.6f}\n")
                    f.write(f"  short_funding_rate            : {short_funding_rate:.6f}\n")
                    f.write(f"  current_position_apr (backend): {current_position_apr:.2f}%\n")
                    f.write(f"  long_pnl_pct                  : {long_pnl_pct:.4f}%\n")
                    f.write(f"  short_pnl_pct                 : {short_pnl_pct:.4f}%\n")
                    f.write(f"  pnl_imbalance                 : {(long_pnl_pct - short_pnl_pct) / 200:.6f}\n")

                    f.write("\n")

                f.write("\n\n")
        except Exception as e:
            # Don't crash if logging fails
            print(f"Warning: Failed to write readable feature log: {e}")
            pass

    def predict_opportunities(
        self,
        opportunities: List[Dict],
        portfolio: Dict,
        trading_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Predict best action for given opportunities and portfolio state.

        Args:
            opportunities: List of up to 10 opportunity dicts
            portfolio: Current portfolio state dict
            trading_config: Trading configuration dict (optional, uses defaults if not provided)

        Returns:
            Dict with:
                - action: Recommended action ('HOLD', 'ENTER', 'EXIT')
                - action_id: Action ID (0-35)
                - confidence: Probability of selected action
                - state_value: Estimated state value
                - opportunity_symbol: Symbol if ENTER action
                - opportunity_index: Index if ENTER action
                - position_index: Index if EXIT action
                - position_size: Recommended size ('SMALL', 'MEDIUM', 'LARGE') if ENTER
                - action_probabilities: Full distribution over all 36 actions
        """
        # Use default config if not provided
        if trading_config is None:
            trading_config = {
                'max_leverage': 1.0,
                'target_utilization': 0.5,
                'max_positions': 3,
                'stop_loss_threshold': -0.02,
                'liquidation_buffer': 0.15,
            }

        # Build observation
        obs = self._build_observation(trading_config, portfolio, opportunities)

        # Log readable features for debugging (production)
        self._log_position_features_readable(portfolio, obs, opportunities)

        # FEATURE ANALYSIS: Log observation vector to file for debugging
        # This helps identify feature distribution mismatches between test and production
        import json
        import datetime
        try:
            feature_log_path = '/tmp/ml_observation_log.jsonl'
            num_positions = sum(1 for p in portfolio.get('positions', []) if p.get('symbol', '') != '')

            log_entry = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'num_positions': num_positions,
                'num_opportunities': len(opportunities),
                'observation_vector': obs.tolist(),
                'config_features': obs[:5].tolist(),  # 5 dims
                'portfolio_features': obs[5:8].tolist(),  # 3 dims (V3: was 6)
                'execution_features_pos0': obs[8:25].tolist() if num_positions > 0 else [],  # 17 dims (V3: was 20)
                'opportunity_features_opp0': obs[93:104].tolist() if len(opportunities) > 0 else [],  # 11 dims (V3: was 19)
            }

            # Append to log file
            with open(feature_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            # Don't crash if logging fails
            pass

        # DEBUG: Log input data structure for analysis (DISABLED - set to True to enable)
        VERBOSE_DEBUG = False
        if VERBOSE_DEBUG:
            print(f"\n{'='*80}")
            print(f"ML PREDICTOR INPUT DATA")
            print(f"{'='*80}")
            print(f"\n📊 PORTFOLIO STATE:")
            print(f"  Positions count: {len(portfolio.get('positions', []))}")
            print(f"  Total capital: ${portfolio.get('total_capital', 0):.2f}")
            print(f"  Capital Utilization: {portfolio.get('capital_utilization', 0):.2f}%")
            print(f"  Total PnL %: {portfolio.get('total_pnl_pct', 0):.2f}%")

            # Log each position in detail
            for i, pos in enumerate(portfolio.get('positions', [])):
                print(f"\n  Position {i+1}:")
                print(f"    Symbol: {pos.get('symbol', 'N/A')}")
                print(f"    Hours held (raw): {pos.get('hours_held', 0):.2f}h")
                print(f"    Unrealized PnL %: {pos.get('unrealized_pnl_pct', 0):.2f}%")
                print(f"    Long net funding USD: ${pos.get('long_net_funding_usd', 0):.2f}")
                print(f"    Short net funding USD: ${pos.get('short_net_funding_usd', 0):.2f}")
                print(f"    Position size USD: ${pos.get('position_size_usd', 0):.2f}")
                print(f"    Current long price: ${pos.get('current_long_price', 0):.2f}")
                print(f"    Current short price: ${pos.get('current_short_price', 0):.2f}")
                print(f"    Entry long price: ${pos.get('entry_long_price', 0):.2f}")
                print(f"    Entry short price: ${pos.get('entry_short_price', 0):.2f}")
                print(f"    Current spread %: {pos.get('current_spread_pct', 0):.4f}")
                print(f"    Entry spread %: {pos.get('entry_spread_pct', 0):.4f}")
                print(f"    Long PnL %: {pos.get('long_pnl_pct', 0):.2f}%")
                print(f"    Short PnL %: {pos.get('short_pnl_pct', 0):.2f}%")
                print(f"    Liquidation distance: {pos.get('liquidation_distance', 1.0):.4f}")
                print(f"    Position is active: {pos.get('position_is_active', 0)}")

            print(f"\n⚙️  TRADING CONFIG:")
            print(f"  Max leverage: {trading_config.get('max_leverage', 1.0)}")
            print(f"  Target utilization: {trading_config.get('target_utilization', 0.5)}")
            print(f"  Max positions: {trading_config.get('max_positions', 3)}")

            print(f"\n🎯 OPPORTUNITIES: {len(opportunities)}")
            for i, opp in enumerate(opportunities[:3]):  # Show first 3
                print(f"  Opp {i+1}: {opp.get('symbol', 'N/A')} - Fund APR: {opp.get('fund_apr', 0):.2f}%")

            print(f"\n📈 OBSERVATION VECTOR (V3):")
            print(f"  Shape: {obs.shape}")
            print(f"  Config features (5): {obs[:5]}")
            print(f"  Portfolio features (3): {obs[5:8]}")
            print(f"  Execution features (85): First slot (17): {obs[8:25]}")
            print(f"  Opportunity features (110): First slot (11): {obs[93:104]}")
            print(f"{'='*80}\n")

        # Build action mask
        # CRITICAL: Must match environment logic exactly!
        # Environment uses len(self.portfolio.positions) which only counts real Position objects
        # We receive a padded list (5 slots), so count only non-empty positions (symbol != '')
        positions = portfolio.get('positions', [])
        num_positions = sum(1 for p in positions if p.get('symbol', '') != '')
        max_positions = trading_config.get('max_positions', 3)
        action_mask = self._get_action_mask(opportunities, num_positions, max_positions)

        # Select action (deterministic = greedy)
        action, value, log_prob = self.trainer.select_action(
            obs,
            action_mask,
            deterministic=True
        )

        # Get full probability distribution
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_tensor = torch.BoolTensor(action_mask).unsqueeze(0).to(self.device)

            action_logits, _ = self.network(obs_tensor, mask_tensor)
            probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]

        # Decode action
        action_info = self._decode_action(action)

        # Build result
        result = {
            'action': action_info['type'],
            'action_id': int(action),
            'confidence': float(probs[action]),
            'state_value': float(value),
            'action_probabilities': probs.tolist(),
        }

        # Add specific fields based on action type
        if action_info['type'] == 'ENTER':
            opp_idx = action_info['opportunity_index']
            result['opportunity_index'] = opp_idx
            result['opportunity_symbol'] = opportunities[opp_idx]['symbol'] if opp_idx < len(opportunities) else 'UNKNOWN'
            result['position_size'] = action_info['size']
            result['size_multiplier'] = SIZE_MULTIPLIERS[action_info['size']]
        elif action_info['type'] == 'EXIT':
            result['position_index'] = action_info['position_index']

        return result

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model (V3).

        Returns:
            Dict with model metadata
        """
        return {
            'model_type': 'ModularPPO',
            'architecture': 'modular_network_with_attention',
            'action_space': 36,
            'observation_space': 203,  # V3: was 301 (streamlined features)
            'max_opportunities': 10,
            'max_positions': 5,
            'position_features': 17,  # V3: was 20 (added velocities, removed historical)
            'opportunity_features': 11,  # V3: was 19 (removed market quality)
            'network_parameters': sum(p.numel() for p in self.network.parameters()),
        }


if __name__ == "__main__":
    # Test the predictor
    print("Testing ModularRLPredictor...")

    # Create predictor
    predictor = ModularRLPredictor('checkpoints/best_model.pt')

    # Test data
    test_opportunities = [
        {
            'symbol': 'BTCUSDT',
            'long_exchange': 'binance',
            'short_exchange': 'bybit',
            'long_funding_rate': 0.0001,
            'short_funding_rate': -0.0001,
            'fund_apr': 10.0,
            'volume_24h': 1000000000,
        }
    ]

    test_portfolio = {
        'total_capital': 10000,
        'initial_capital': 10000,
        'available_margin': 8000,
        'margin_utilization': 20,
        'utilization': 20,
        'total_pnl_pct': 2.5,
        'positions': [],
    }

    test_config = {
        'max_leverage': 1.0,
        'target_utilization': 0.5,
        'max_positions': 3,
    }

    # Predict
    result = predictor.predict_opportunities(test_opportunities, test_portfolio, test_config)

    print("\n" + "="*80)
    print("PREDICTION RESULT:")
    print("="*80)
    print(f"Action: {result['action']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"State Value: {result['state_value']:.4f}")

    if result['action'] == 'ENTER':
        print(f"Symbol: {result['opportunity_symbol']}")
        print(f"Size: {result['position_size']}")

    print(f"\nModel Info:")
    for k, v in predictor.get_model_info().items():
        print(f"  {k}: {v}")

    print("\n✅ Predictor test passed!")
