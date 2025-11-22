"""
Modular RL Predictor V2 - Simplified with UnifiedFeatureBuilder

This version uses the UnifiedFeatureBuilder for all feature preparation,
eliminating code duplication and ensuring consistency across all components.

Architecture V3: 203-dimensional observation space
- Config: 5 dims
- Portfolio: 3 dims
- Executions: 85 dims (5 slots × 17 features)
- Opportunities: 110 dims (10 slots × 11 features)

Action space: 36 actions
- 0: HOLD
- 1-10: ENTER_OPP_0-9_SMALL (10%)
- 11-20: ENTER_OPP_0-9_MEDIUM (20%)
- 21-30: ENTER_OPP_0-9_LARGE (30%)
- 31-35: EXIT_POS_0-4
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.rl.networks.modular_ppo import ModularPPONetwork
from models.rl.algorithms.ppo_trainer import PPOTrainer
from common.features import UnifiedFeatureBuilder, DIMS


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
    Simplified RL Model predictor using UnifiedFeatureBuilder.

    All feature preparation is delegated to UnifiedFeatureBuilder,
    ensuring consistency with training environment and eliminating duplication.
    """

    def __init__(
        self,
        model_path: str = 'checkpoints/v3_production/best_model.pt',
        feature_scaler_path: str = 'trained_models/rl/feature_scaler_v2.pkl',
        device: str = 'cpu'
    ):
        """
        Initialize the modular RL predictor.

        Args:
            model_path: Path to trained PPOTrainer checkpoint (.pt file)
            feature_scaler_path: Path to fitted feature scaler pickle (V3: StandardScaler)
            device: Device to use ('cpu' or 'cuda')
        """
        print(f"Loading Modular RL model from: {model_path}")

        self.device = device

        # Initialize unified feature builder (SINGLE SOURCE OF TRUTH)
        self.feature_builder = UnifiedFeatureBuilder(feature_scaler_path=feature_scaler_path)

        # Create network and trainer (SAME AS test_inference.py)
        network = ModularPPONetwork()
        self.trainer = PPOTrainer(
            network=network,
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

        # Set to evaluation mode (network is already compiled by PPOTrainer)
        self.trainer.network.eval()
        print("✅ Model set to evaluation mode")

        # Get model info
        total_params = sum(p.numel() for p in self.trainer.network.parameters())
        print(f"   Network parameters: {total_params:,}")
        print(f"   Observation space: {DIMS.TOTAL} dimensions (V3)")
        print(f"   Action space: {DIMS.TOTAL_ACTIONS} actions")

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
            trading_config: Trading configuration dict (optional)

        Returns:
            Dict with:
                - action: Recommended action ('HOLD', 'ENTER', 'EXIT')
                - action_id: Action ID (0-35)
                - confidence: Probability of selected action
                - state_value: Estimated state value
                - opportunity_symbol: Symbol if ENTER action
                - opportunity_index: Index if ENTER action
                - position_index: Index if EXIT action
                - position_size: Recommended size if ENTER
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

        # Build observation using unified feature builder
        raw_data = {
            'trading_config': trading_config,
            'portfolio': portfolio,
            'opportunities': opportunities
        }
        obs = self.feature_builder.build_observation_from_raw_data(raw_data)

        # Log observation for debugging
        self._log_observation(obs, portfolio, opportunities)

        # Build action mask using unified feature builder
        positions = portfolio.get('positions', [])
        num_positions = sum(1 for p in positions if p.get('is_active', False) or p.get('symbol', '') != '')
        max_positions = trading_config.get('max_positions', 3)
        action_mask = self.feature_builder.get_action_mask(opportunities, num_positions, max_positions)

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

            action_logits, _ = self.trainer.network(obs_tensor, mask_tensor)
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
            if opp_idx < len(opportunities):
                result['opportunity_index'] = opp_idx
                result['opportunity_symbol'] = opportunities[opp_idx].get('symbol', 'UNKNOWN')
                result['opportunity_long_exchange'] = opportunities[opp_idx].get('long_exchange', 'UNKNOWN')
                result['opportunity_short_exchange'] = opportunities[opp_idx].get('short_exchange', 'UNKNOWN')
                result['opportunity_fund_apr'] = opportunities[opp_idx].get('fund_apr', 0.0)
            result['position_size'] = action_info['size']
            result['size_multiplier'] = SIZE_MULTIPLIERS[action_info['size']]
        elif action_info['type'] == 'EXIT':
            result['position_index'] = action_info['position_index']

        # Add mask info
        result['valid_actions'] = int(action_mask.sum())
        result['masked_actions'] = int((~action_mask).sum())

        return result

    def _decode_action(self, action: int) -> Dict[str, Any]:
        """
        Decode action ID to human-readable format.

        Args:
            action: Action ID (0-35)

        Returns:
            Dict with 'type', 'opportunity_index', 'position_index', 'size'
        """
        if action == DIMS.ACTION_HOLD:
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
        elif DIMS.ACTION_EXIT_START <= action <= DIMS.ACTION_EXIT_END:
            return {
                'type': 'EXIT',
                'opportunity_index': None,
                'position_index': action - DIMS.ACTION_EXIT_START,
                'size': None,
            }
        else:
            raise ValueError(f"Invalid action: {action}")

    def _log_observation(self, obs: np.ndarray, portfolio: Dict, opportunities: List[Dict]):
        """Log observation vector for debugging."""
        import json
        import datetime

        try:
            feature_log_path = '/tmp/ml_observation_log.jsonl'
            num_positions = sum(
                1 for p in portfolio.get('positions', [])
                if p.get('is_active', False) or p.get('symbol', '') != ''
            )

            log_entry = {
                'timestamp': datetime.datetime.utcnow().isoformat(),
                'num_positions': num_positions,
                'num_opportunities': len(opportunities),
                'observation_vector': obs.tolist(),
                'config_features': obs[:DIMS.CONFIG].tolist(),
                'portfolio_features': obs[DIMS.CONFIG:DIMS.CONFIG+DIMS.PORTFOLIO].tolist(),
                'execution_features_pos0': obs[DIMS.CONFIG+DIMS.PORTFOLIO:DIMS.CONFIG+DIMS.PORTFOLIO+DIMS.EXECUTIONS_PER_SLOT].tolist() if num_positions > 0 else [],
                'opportunity_features_opp0': obs[DIMS.CONFIG+DIMS.PORTFOLIO+DIMS.EXECUTIONS_TOTAL:DIMS.CONFIG+DIMS.PORTFOLIO+DIMS.EXECUTIONS_TOTAL+DIMS.OPPORTUNITIES_PER_SLOT].tolist() if len(opportunities) > 0 else [],
            }

            with open(feature_log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception:
            pass  # Don't crash if logging fails

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dict with model metadata
        """
        return {
            'model_type': 'ModularPPO',
            'architecture': 'modular_network_with_attention',
            'action_space': DIMS.TOTAL_ACTIONS,
            'observation_space': DIMS.TOTAL,
            'max_opportunities': DIMS.OPPORTUNITIES_SLOTS,
            'max_positions': DIMS.EXECUTIONS_SLOTS,
            'position_features': DIMS.EXECUTIONS_PER_SLOT,
            'opportunity_features': DIMS.OPPORTUNITIES_PER_SLOT,
            'network_parameters': sum(p.numel() for p in self.trainer.network.parameters()),
            'feature_builder': 'UnifiedFeatureBuilder (single source of truth)',
        }
