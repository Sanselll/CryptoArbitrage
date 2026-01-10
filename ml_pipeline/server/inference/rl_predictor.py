"""
Modular RL Predictor V2 - Simplified with UnifiedFeatureBuilder (V9)

This version uses the UnifiedFeatureBuilder for all feature preparation,
eliminating code duplication and ensuring consistency across all components.

Architecture V9: 86-dimensional observation space (simplified)
- Config: 5 dims
- Portfolio: 2 dims (removed capital features)
- Executions: 19 dims (1 slot × 19 features)
- Opportunities: 60 dims (5 slots × 12 features)

Action space: 17 actions (V9: single position)
- 0: HOLD
- 1-5: ENTER_OPP_0-4_SMALL (10%)
- 6-10: ENTER_OPP_0-4_MEDIUM (20%)
- 11-15: ENTER_OPP_0-4_LARGE (30%)
- 16: EXIT_POS_0
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


# Action space mapping (V9: 17 actions)
ACTION_NAMES = {
    0: 'HOLD',
    **{i: f'ENTER_OPP_{i-1}_SMALL' for i in range(1, 6)},     # 1-5
    **{i: f'ENTER_OPP_{i-6}_MEDIUM' for i in range(6, 11)},   # 6-10
    **{i: f'ENTER_OPP_{i-11}_LARGE' for i in range(11, 16)},  # 11-15
    16: 'EXIT_POS_0',                                          # V9: single position
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

    Per-Session Feature Builders:
    - Each agent session has its own UnifiedFeatureBuilder instance
    - This isolates velocity tracking state between different users/sessions
    - Without isolation, velocity features would be contaminated across sessions
    - Sessions without ID use a default builder with velocity tracking disabled
    """

    def __init__(
        self,
        model_path: str = 'trained_models/rl/modular_ppo_v10.pt',
        feature_scaler_path: str = 'trained_models/rl/feature_scaler_v3.pkl',
        device: str = 'cpu'
    ):
        """
        Initialize the modular RL predictor.

        Args:
            model_path: Path to trained PPOTrainer checkpoint (.pt file)
            feature_scaler_path: Path to fitted feature scaler pickle (V5.4: StandardScaler with 12 features)
            device: Device to use ('cpu' or 'cuda')
        """
        print(f"Loading Modular RL model from: {model_path}")

        self.device = device

        # Store scaler path for creating per-session builders
        self.feature_scaler_path = feature_scaler_path

        # Per-session feature builders for velocity tracking isolation
        # Key: session_id (string), Value: UnifiedFeatureBuilder instance
        self.session_builders: Dict[str, UnifiedFeatureBuilder] = {}

        # Default builder for requests without session_id (velocity tracking disabled)
        self.default_builder = UnifiedFeatureBuilder(
            feature_scaler_path=feature_scaler_path,
            enable_velocity_tracking=False
        )

        # Legacy: Keep single feature_builder for backward compatibility
        self.feature_builder = self.default_builder

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

    def _get_feature_builder(self, session_id: Optional[str]) -> UnifiedFeatureBuilder:
        """
        Get or create a feature builder for the given session.

        Per-session builders ensure velocity tracking state isolation:
        - Each session has its own velocity history (_prev_estimated_pnl, etc.)
        - Without isolation, sessions would contaminate each other's velocity features
        - Sessions without ID use the default builder (velocity tracking disabled)

        Args:
            session_id: Agent session ID (Guid string) or None

        Returns:
            UnifiedFeatureBuilder instance for this session
        """
        if not session_id:
            return self.default_builder

        if session_id not in self.session_builders:
            print(f"Creating new feature builder for session: {session_id}")
            self.session_builders[session_id] = UnifiedFeatureBuilder(
                feature_scaler_path=self.feature_scaler_path,
                enable_velocity_tracking=True
            )

        return self.session_builders[session_id]

    def cleanup_session(self, session_id: str) -> bool:
        """
        Remove a session's feature builder when the session ends.
        Call this when an agent session is stopped to free memory.

        Args:
            session_id: Agent session ID to cleanup

        Returns:
            True if session was found and removed, False otherwise
        """
        if session_id in self.session_builders:
            del self.session_builders[session_id]
            print(f"Cleaned up feature builder for session: {session_id}")
            return True
        return False

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs with feature builders."""
        return list(self.session_builders.keys())

    def predict_opportunities(
        self,
        opportunities: List[Dict],
        portfolio: Dict,
        trading_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Predict best action for given opportunities and portfolio state (V8).

        Args:
            opportunities: List of up to 5 opportunity dicts
            portfolio: Current portfolio state dict
            trading_config: Trading configuration dict (optional)

        Returns:
            Dict with:
                - action: Recommended action ('HOLD', 'ENTER', 'EXIT')
                - action_id: Action ID (0-16)
                - confidence: Probability of selected action
                - state_value: Estimated state value
                - opportunity_symbol: Symbol if ENTER action
                - opportunity_index: Index if ENTER action (0-4)
                - position_index: Index if EXIT action (0 only, V9)
                - position_size: Recommended size if ENTER
                - action_probabilities: Full distribution over all 17 actions
        """
        # Use default config if not provided
        if trading_config is None:
            trading_config = {
                'max_leverage': 1.0,
                'target_utilization': 0.5,
                'max_positions': 1,  # V9: single position only
                'stop_loss_threshold': -0.10,
                'liquidation_buffer': 0.15,
            }

        # Get session-specific feature builder for velocity tracking isolation
        session_id = portfolio.get('session_id')
        feature_builder = self._get_feature_builder(session_id)

        # Build observation using session-specific feature builder
        from datetime import datetime
        import pandas as pd

        # Use current_time from portfolio if provided (for backtesting), else use real time
        current_time = portfolio.get('current_time')
        if current_time is None:
            current_time = datetime.utcnow()
        elif isinstance(current_time, str):
            # Parse timestamp string (handles various formats including nanoseconds)
            current_time = pd.Timestamp(current_time).to_pydatetime()

        raw_data = {
            'trading_config': trading_config,
            'portfolio': portfolio,
            'opportunities': opportunities,
            'current_time': current_time,  # V6: For time_to_next_funding feature
        }
        obs = feature_builder.build_observation_from_raw_data(raw_data)

        # Log observation for debugging
        self._log_observation(obs, portfolio, opportunities, session_id)

        # Build action mask using session-specific feature builder (V10)
        positions = portfolio.get('positions', [])
        num_positions = sum(1 for p in positions if p.get('is_active', False) or p.get('symbol', '') != '')
        max_positions = trading_config.get('max_positions', 1)

        # V10 masking criteria from trading config (with defaults)
        min_apr = trading_config.get('min_apr', 2000.0)
        max_apr = trading_config.get('max_apr', 15000.0)  # Block high APR traps
        max_minutes_to_funding = trading_config.get('max_minutes_to_funding', 30.0)
        allowed_liquidity = [0.0]  # Good (0) only, block Medium (1) and Low (2)

        action_mask = feature_builder.get_action_mask(
            opportunities, num_positions, max_positions,
            current_time=current_time,
            max_minutes_to_funding=max_minutes_to_funding,
            min_apr=min_apr,
            max_apr=max_apr,
            allowed_liquidity=allowed_liquidity
        )

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
            pos_idx = action_info['position_index']
            result['position_index'] = pos_idx
            # Add symbol for EXIT actions (fixes index mismatch between ML API and backend)
            positions = portfolio.get('positions', [])
            if pos_idx < len(positions):
                result['exit_symbol'] = positions[pos_idx].get('symbol', 'UNKNOWN')

        # Add mask info
        result['valid_actions'] = int(action_mask.sum())
        result['masked_actions'] = int((~action_mask).sum())

        # Add timing info for debugging (V6 feature verification)
        result['current_time_utc'] = current_time.isoformat() + 'Z'
        result['time_to_next_funding_norm'] = float(obs[6])  # Index: 5 config + 1 (second portfolio feature)

        return result

    def _decode_action(self, action: int) -> Dict[str, Any]:
        """
        Decode action ID to human-readable format (V9: 17 actions).

        Args:
            action: Action ID (0-16)

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
        elif 1 <= action <= 5:  # V8: 1-5 (was 1-10)
            return {
                'type': 'ENTER',
                'opportunity_index': action - 1,
                'position_index': None,
                'size': 'SMALL',
            }
        elif 6 <= action <= 10:  # V8: 6-10 (was 11-20)
            return {
                'type': 'ENTER',
                'opportunity_index': action - 6,
                'position_index': None,
                'size': 'MEDIUM',
            }
        elif 11 <= action <= 15:  # V8: 11-15 (was 21-30)
            return {
                'type': 'ENTER',
                'opportunity_index': action - 11,
                'position_index': None,
                'size': 'LARGE',
            }
        elif DIMS.ACTION_EXIT_START <= action <= DIMS.ACTION_EXIT_END:  # V8: 16-17
            return {
                'type': 'EXIT',
                'opportunity_index': None,
                'position_index': action - DIMS.ACTION_EXIT_START,
                'size': None,
            }
        else:
            raise ValueError(f"Invalid action: {action}")

    def _log_observation(self, obs: np.ndarray, portfolio: Dict, opportunities: List[Dict], session_id: Optional[str] = None):
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
                'session_id': session_id,
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
