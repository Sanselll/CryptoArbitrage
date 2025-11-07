"""
Modular RL Predictor - Uses trained ModularPPONetwork for inference

Loads and runs the trained modular PPO model to provide action predictions
for funding arbitrage opportunities.

Architecture:
- 275-dim observation space:
  * Config: 5 dims (max_leverage, target_utilization, max_positions, stop_loss, liq_buffer)
  * Portfolio: 10 dims (capital_ratio, available_ratio, margin_util, etc.)
  * Executions: 60 dims (5 positions × 12 features each)
  * Opportunities: 200 dims (10 opportunities × 20 features each)

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

    def __init__(self, model_path: str = 'checkpoints/best_model.pt',
                 feature_scaler_path: str = 'trained_models/rl/feature_scaler.pkl',
                 device: str = 'cpu'):
        """
        Initialize the modular RL predictor.

        Args:
            model_path: Path to trained PPOTrainer checkpoint (.pt file)
            feature_scaler_path: Path to fitted StandardScaler pickle (default: trained_models/rl/feature_scaler.pkl)
            device: Device to use ('cpu' or 'cuda')
        """
        print(f"Loading Modular RL model from: {model_path}")

        self.device = device

        # Load feature scaler (critical for proper predictions!)
        self.feature_scaler = None
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

        # Get model info
        total_params = sum(p.numel() for p in self.network.parameters())
        print(f"   Network parameters: {total_params:,}")
        print(f"   Observation space: 275 dimensions")
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

    def _build_portfolio_features(self, portfolio: Dict) -> np.ndarray:
        """
        Build portfolio features (10 dimensions).

        Args:
            portfolio: Portfolio state dict

        Returns:
            10-dimensional array
        """
        capital = portfolio.get('total_capital', portfolio.get('capital', 10000.0))
        initial_capital = portfolio.get('initial_capital', 10000.0)
        capital_ratio = capital / initial_capital if initial_capital > 0 else 1.0

        available_margin = portfolio.get('available_margin', capital)
        available_ratio = available_margin / capital if capital > 0 else 1.0

        margin_util = portfolio.get('margin_utilization', 0.0) / 100.0  # Convert % to decimal

        num_positions = len(portfolio.get('positions', []))
        max_positions = portfolio.get('max_positions', 3)
        num_positions_ratio = num_positions / max_positions if max_positions > 0 else 0.0

        total_pnl_pct = portfolio.get('total_pnl_pct', 0.0) / 100.0
        max_drawdown_pct = portfolio.get('max_drawdown', 0.0) / 100.0

        # Episode progress (placeholder - backend should provide this)
        episode_progress = 0.5

        # Min liquidation distance (placeholder)
        min_liq_distance = 1.0

        capital_utilization = portfolio.get('utilization', 0.0) / 100.0

        return np.array([
            capital_ratio,
            available_ratio,
            margin_util,
            num_positions_ratio,
            0.0,  # avg_position_pnl_pct (placeholder)
            total_pnl_pct,
            max_drawdown_pct,
            episode_progress,
            min_liq_distance,
            capital_utilization,
        ], dtype=np.float32)

    def _build_execution_features(self, portfolio: Dict) -> np.ndarray:
        """
        Build execution features for up to 5 position slots (60 dimensions total).

        Each position has 12 features:
        1. unrealized_pnl_pct
        2. hours_held
        3. net_funding_ratio
        4. net_funding_rate
        5. current_spread_pct
        6. entry_spread_pct
        7. value_to_capital_ratio
        8. funding_efficiency
        9. long_pnl_pct
        10. short_pnl_pct
        11. liquidation_distance
        12. position_is_active (1.0 if slot filled, 0.0 if empty)

        Args:
            portfolio: Portfolio state dict

        Returns:
            60-dimensional array (5 slots × 12 features)
        """
        positions = portfolio.get('positions', [])
        capital = portfolio.get('total_capital', portfolio.get('capital', 10000.0))

        all_features = []

        for slot_idx in range(5):
            if slot_idx < len(positions):
                pos = positions[slot_idx]

                # 1. Net P&L %
                net_pnl_pct = pos.get('unrealized_pnl_pct', 0.0) / 100.0

                # 2. Hours held (normalize by 72h = 3 days)
                hours_held = pos.get('position_age_hours', pos.get('hours_held', 0.0)) / 72.0

                # 3. Net funding ratio
                total_capital_used = pos.get('position_size_usd', 0.0) * 2
                long_net_funding = pos.get('long_net_funding_usd', 0.0)
                short_net_funding = pos.get('short_net_funding_usd', 0.0)
                net_funding_usd = long_net_funding + short_net_funding
                net_funding_ratio = net_funding_usd / total_capital_used if total_capital_used > 0 else 0.0

                # 4. Net funding rate
                net_funding_rate = (pos.get('short_funding_rate', 0.0) -
                                   pos.get('long_funding_rate', 0.0))

                # 5. Current spread %
                current_long_price = pos.get('current_long_price', pos.get('current_price', 0.0))
                current_short_price = pos.get('current_short_price', pos.get('current_price', 0.0))
                if current_long_price > 0 and current_short_price > 0:
                    avg_current_price = (current_long_price + current_short_price) / 2
                    current_spread_pct = abs(current_long_price - current_short_price) / avg_current_price
                else:
                    current_spread_pct = 0.0

                # 6. Entry spread %
                entry_long_price = pos.get('entry_long_price', pos.get('entry_price', 0.0))
                entry_short_price = pos.get('entry_short_price', pos.get('entry_price', 0.0))
                if entry_long_price > 0 and entry_short_price > 0:
                    avg_entry_price = (entry_long_price + entry_short_price) / 2
                    entry_spread_pct = abs(entry_long_price - entry_short_price) / avg_entry_price
                else:
                    entry_spread_pct = 0.0

                # 7. Value-to-capital ratio
                value_ratio = total_capital_used / capital if capital > 0 else 0.0

                # 8. Funding efficiency
                total_fees = pos.get('entry_fees_paid_usd', 0.0) + (total_capital_used * 0.0006)
                funding_efficiency = net_funding_usd / total_fees if total_fees > 0 else 0.0

                # 9. Long side P&L %
                long_pnl_pct = pos.get('long_pnl_pct', 0.0) / 100.0

                # 10. Short side P&L %
                short_pnl_pct = pos.get('short_pnl_pct', 0.0) / 100.0

                # 11. Liquidation distance
                # Calculate from leverage and prices if not provided
                liquidation_distance = pos.get('liquidation_distance', None)
                if liquidation_distance is None:
                    leverage = pos.get('leverage', 1.0)
                    entry_long = pos.get('entry_long_price', 0.0)
                    entry_short = pos.get('entry_short_price', 0.0)
                    current_long = pos.get('current_long_price', entry_long)
                    current_short = pos.get('current_short_price', entry_short)

                    if leverage > 0 and entry_long > 0 and entry_short > 0:
                        # Calculate liquidation prices (matches training environment formula)
                        long_liq = entry_long * (1 - 0.9 / leverage)
                        short_liq = entry_short * (1 + 0.9 / leverage)

                        # Calculate distances (% from current to liquidation)
                        long_dist = abs(current_long - long_liq) / current_long if current_long > 0 else 1.0
                        short_dist = abs(short_liq - current_short) / current_short if current_short > 0 else 1.0

                        # Return minimum (closest to liquidation = highest risk)
                        liquidation_distance = min(long_dist, short_dist)
                    else:
                        liquidation_distance = 1.0  # Safe default (no liquidation risk)

                # 12. Position is active
                position_is_active = 1.0

                slot_features = [
                    net_pnl_pct, hours_held, net_funding_ratio, net_funding_rate,
                    current_spread_pct, entry_spread_pct, value_ratio, funding_efficiency,
                    long_pnl_pct, short_pnl_pct, liquidation_distance, position_is_active
                ]
            else:
                # Empty slot - all zeros
                slot_features = [0.0] * 12

            all_features.extend(slot_features)

        return np.array(all_features, dtype=np.float32)

    def _build_opportunity_features(self, opportunities: List[Dict]) -> np.ndarray:
        """
        Build opportunity features for up to 10 slots (200 dimensions total).

        Each opportunity has 20 features (no momentum features, matches full mode).
        Features are standardized using the feature scaler if available.

        Args:
            opportunities: List of up to 10 opportunity dicts

        Returns:
            200-dimensional array (10 slots × 20 features)
        """
        all_features = []

        for slot_idx in range(10):
            if slot_idx < len(opportunities):
                opp = opportunities[slot_idx]

                features = [
                    opp.get('long_funding_rate', 0.0),
                    opp.get('short_funding_rate', 0.0),
                    opp.get('long_funding_interval_hours', 8.0) / 8.0,
                    opp.get('short_funding_interval_hours', 8.0) / 8.0,
                    opp.get('fund_profit_8h', 0.0),
                    opp.get('fundProfit8h24hProj', 0.0),
                    opp.get('fundProfit8h3dProj', 0.0),
                    opp.get('fund_apr', 0.0),
                    opp.get('fundApr24hProj', 0.0),
                    opp.get('fundApr3dProj', 0.0),
                    opp.get('spread30SampleAvg', 0.0),
                    opp.get('priceSpread24hAvg', 0.0),
                    opp.get('priceSpread3dAvg', 0.0),
                    opp.get('spread_volatility_stddev', 0.0),
                    np.log10(max(float(opp.get('volume_24h', 1e6) or 1e6), 1e5)),
                    float(opp.get('bidAskSpreadPercent', 0) or 0),
                    np.log10(max(float(opp.get('orderbookDepthUsd', 1e4) or 1e4), 1e3)),
                    float(opp.get('estimatedProfitPercentage', 0) or 0),
                    float(opp.get('positionCostPercent', 0.2) or 0.2),
                    opp.get('short_funding_rate', 0.0) - opp.get('long_funding_rate', 0.0),  # Net funding rate
                ]

                # Convert to float32 and ensure no NaN/inf
                features = [float(np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)) for x in features]

                # Apply feature scaler if available
                if self.feature_scaler is not None:
                    # Scaler now expects 20 features (matches full mode)
                    features_array = np.array(features)
                    # Scale
                    features_scaled = self.feature_scaler.transform(features_array.reshape(1, 20))
                    features = features_scaled.flatten().tolist()
            else:
                # Empty slot - all zeros
                features = [0.0] * 20

            all_features.extend(features)

        return np.array(all_features, dtype=np.float32)

    def _build_observation(
        self,
        trading_config: Dict,
        portfolio: Dict,
        opportunities: List[Dict]
    ) -> np.ndarray:
        """
        Build complete 275-dim observation vector.

        Args:
            trading_config: Trading configuration dict
            portfolio: Portfolio state dict
            opportunities: List of up to 10 opportunity dicts

        Returns:
            275-dimensional observation array
        """
        # Config features (5)
        config_features = self._build_config_features(trading_config)

        # Portfolio features (10)
        portfolio_features = self._build_portfolio_features(portfolio)

        # Execution features (60)
        execution_features = self._build_execution_features(portfolio)

        # Opportunity features (200)
        opportunity_features = self._build_opportunity_features(opportunities)

        # Concatenate all (5 + 10 + 60 + 200 = 275)
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

        Args:
            opportunities: List of opportunity dicts (includes has_existing_position flag)
            num_positions: Current number of open positions
            max_positions: Maximum allowed positions

        Returns:
            Boolean array (36,) where True = valid action
        """
        mask = np.zeros(36, dtype=bool)

        # HOLD is always valid
        mask[0] = True

        # ENTER actions: valid if opportunity exists AND we can open position AND no existing position
        has_capacity = num_positions < max_positions

        if has_capacity:
            for i in range(10):
                if i < len(opportunities):
                    opp = opportunities[i]
                    # CRITICAL: Check if opportunity already has an open position
                    # This prevents duplicate ENTER actions on the same opportunity
                    has_existing_position = opp.get('has_existing_position', False)

                    if not has_existing_position:
                        # Enable all three size variants for this opportunity
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

        # DEBUG: Log input data structure for analysis
        print(f"\n{'='*80}")
        print(f"ML PREDICTOR INPUT DATA")
        print(f"{'='*80}")
        print(f"\n📊 PORTFOLIO STATE:")
        print(f"  Positions count: {len(portfolio.get('positions', []))}")
        print(f"  Total capital: ${portfolio.get('total_capital', 0):.2f}")
        print(f"  Utilization: {portfolio.get('utilization', 0):.2f}%")
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

        print(f"\n📈 OBSERVATION VECTOR:")
        print(f"  Shape: {obs.shape}")
        print(f"  Config features (5): {obs[:5]}")
        print(f"  Portfolio features (10): {obs[5:15]}")
        print(f"  Execution features (60): First 12: {obs[15:27]}")
        print(f"  Opportunity features (200): First 20: {obs[75:95]}")
        print(f"{'='*80}\n")

        # Build action mask (count only ACTIVE positions, not empty slots)
        positions = portfolio.get('positions', [])
        num_positions = sum(1 for p in positions if p.get('is_active', False) or p.get('position_is_active', 0.0) > 0.5)
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
        Get information about the loaded model.

        Returns:
            Dict with model metadata
        """
        return {
            'model_type': 'ModularPPO',
            'architecture': 'modular_network_with_attention',
            'action_space': 36,
            'observation_space': 275,
            'max_opportunities': 10,
            'max_positions': 5,
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
