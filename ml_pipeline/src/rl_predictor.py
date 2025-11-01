"""
RL Predictor Module

Loads and runs the trained RL model (PPO) to provide action predictions
for opportunities and positions.

Usage:
    predictor = RLPredictor('models/pbt_20251101_083701/agent_2_model.zip')
    predictions = predictor.evaluate_opportunities(opportunities, portfolio)
"""

import numpy as np
import pickle
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.policies import obs_as_tensor


class RLPredictor:
    """
    RL Model predictor for crypto arbitrage opportunities and positions.

    Uses the trained PPO model to evaluate:
    - ENTER probability for each opportunity
    - EXIT probability for each open position
    """

    def __init__(self, model_path: str, feature_scaler_path: str = 'models/rl/feature_scaler.pkl'):
        """
        Initialize the RL predictor.

        Args:
            model_path: Path to trained PPO model (.zip file)
            feature_scaler_path: Path to fitted StandardScaler
        """
        print(f"Loading RL model from: {model_path}")
        self.model = PPO.load(model_path)

        print(f"Loading feature scaler from: {feature_scaler_path}")
        with open(feature_scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        self.model_version = Path(model_path).parent.name
        print(f"RL Predictor initialized (model: {self.model_version})")

    def predict_probabilities(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Get action probabilities from the policy network.

        Args:
            observation: State observation (124 dimensions)

        Returns:
            action_probs: Probability for each action (9 actions)
            state_value: Value estimate for the state
        """
        # Add batch dimension if needed (1, 124)
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        # Convert observation to tensor
        obs_tensor = obs_as_tensor(observation, self.model.policy.device)

        # Get action distribution
        distribution = self.model.policy.get_distribution(obs_tensor)

        # Get probability distribution
        action_probs = distribution.distribution.probs
        action_probs_np = action_probs.detach().cpu().numpy()[0]

        # Get value estimate
        values = self.model.policy.predict_values(obs_tensor)
        value = values.detach().cpu().numpy()[0][0]

        return action_probs_np, value

    def _build_observation(self, opportunities: List[Dict], portfolio: Dict) -> np.ndarray:
        """
        Build observation vector from opportunities and portfolio state.

        Observation structure (124 dims):
        - Portfolio state: 14 dimensions
        - Opportunities: 5 × 22 features = 110 dimensions

        Args:
            opportunities: List of opportunity dicts (max 5)
            portfolio: Portfolio state dict

        Returns:
            observation: numpy array (124,)
        """
        # Portfolio features (14 dims)
        capital = portfolio.get('capital', 10000.0)
        initial_capital = portfolio.get('initial_capital', 10000.0)
        capital_ratio = capital / initial_capital if initial_capital > 0 else 1.0

        num_positions = portfolio.get('num_positions', 0)
        utilization = portfolio.get('utilization', 0.0)
        total_pnl = portfolio.get('total_pnl_pct', 0.0) / 100.0  # Convert % to decimal
        drawdown = portfolio.get('drawdown', 0.0) / 100.0

        # Position features (3 positions × 3 features = 9 dims)
        position_features = []
        positions = portfolio.get('positions', [])
        for i in range(3):  # max_positions = 3
            if i < len(positions):
                pos = positions[i]
                position_features.extend([
                    pos.get('pnl_pct', 0.0) / 100.0,
                    pos.get('hours_held', 0.0) / 72.0,  # Normalize by max episode length
                    pos.get('funding_rate', 0.0) / 100.0
                ])
            else:
                position_features.extend([0.0, 0.0, 0.0])

        portfolio_obs = [capital_ratio, utilization, num_positions / 3.0, total_pnl, drawdown] + position_features

        # Opportunity features (5 opps × 22 features = 110 dims)
        opportunity_features = []
        for i in range(5):  # max_opportunities_per_hour = 5
            if i < len(opportunities):
                opp = opportunities[i]
                features = self._extract_opportunity_features(opp)
            else:
                features = [0.0] * 22  # Padding
            opportunity_features.extend(features)

        # Combine and normalize
        observation = np.array(portfolio_obs + opportunity_features, dtype=np.float32)

        # Apply StandardScaler (only to opportunity features, portfolio already normalized)
        # Feature scaler was fitted on all 22 features per opportunity
        opp_features_only = observation[14:].reshape(5, 22)  # Extract 5 opps × 22 features
        opp_features_scaled = self.scaler.transform(opp_features_only.reshape(-1, 22))
        observation[14:] = opp_features_scaled.flatten()

        return observation

    def _extract_opportunity_features(self, opp: Dict) -> List[float]:
        """
        Extract 22 features from opportunity dict.

        Features match the training data format:
        1-2: Funding rates
        3-4: Funding intervals
        5-7: Profit projections (8h, 24h, 3d)
        8-10: APR projections
        11-13: Spread metrics
        14: Volatility
        15: Volume (log)
        16: Bid-ask spread
        17: Liquidity (log)
        18-19: Estimated profit and cost
        20-22: Momentum features
        """
        features = [
            opp.get('long_funding_rate', 0.0),
            opp.get('short_funding_rate', 0.0),
            opp.get('long_funding_interval_hours', 8.0) / 8.0,  # Normalized
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
            np.log10(max(opp.get('volume_24h', 1.0), 1.0)),  # Log-scaled
            opp.get('bidAskSpreadPercent', 0.0),
            np.log10(max(opp.get('orderbookDepthUsd', 1.0), 1.0)),  # Log-scaled
            opp.get('estimatedProfitPercentage', 0.0),
            opp.get('positionCostPercent', 0.0),
            # Momentum features (calculated)
            opp.get('spread30SampleAvg', 0.0) - opp.get('priceSpread24hAvg', 0.0),
            opp.get('fund_apr', 0.0) - opp.get('fundApr24hProj', 0.0),
            opp.get('priceSpread24hAvg', 0.0) - opp.get('priceSpread3dAvg', 0.0),
        ]
        return features

    def _calculate_confidence(self, probability: float, entropy: float) -> str:
        """
        Calculate confidence level based on probability and entropy.

        Args:
            probability: Action probability (0-1)
            entropy: Distribution entropy (lower = more confident)

        Returns:
            confidence: "HIGH", "MEDIUM", or "LOW"
        """
        if probability > 0.7 and entropy < 1.0:
            return "HIGH"
        elif probability > 0.4 or entropy < 1.5:
            return "MEDIUM"
        else:
            return "LOW"

    def evaluate_opportunities(
        self,
        opportunities: List[Dict],
        portfolio: Dict
    ) -> List[Dict[str, Any]]:
        """
        Evaluate ENTER probabilities for ALL opportunities using batch processing.

        The model can only evaluate 5 opportunities at once (fixed observation space).
        This method processes opportunities in batches of 5, padding the last batch if needed.

        Args:
            opportunities: List of opportunity dicts (any number)
            portfolio: Current portfolio state

        Returns:
            predictions: List of dicts with {
                'opportunity_index': int,
                'symbol': str,
                'enter_probability': float,
                'confidence': str,
                'hold_probability': float,
                'state_value': float
            }
        """
        if not opportunities:
            return []

        all_predictions = []

        # Process opportunities in batches of 5
        for batch_start in range(0, len(opportunities), 5):
            batch_end = min(batch_start + 5, len(opportunities))
            batch = opportunities[batch_start:batch_end]

            # Evaluate this batch (will be auto-padded to 5 by _build_observation)
            batch_predictions = self._evaluate_opportunity_batch(batch, portfolio, batch_start)
            all_predictions.extend(batch_predictions)

        return all_predictions

    def _evaluate_opportunity_batch(
        self,
        opportunities: List[Dict],
        portfolio: Dict,
        batch_offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Evaluate a single batch of opportunities (max 5).

        Args:
            opportunities: List of opportunity dicts (max 5)
            portfolio: Current portfolio state
            batch_offset: Index offset for opportunity_index in results

        Returns:
            predictions: List of dicts with predictions
        """
        # Build observation (will pad to 5 if needed)
        observation = self._build_observation(opportunities, portfolio)

        # Get action probabilities
        action_probs, state_value = self.predict_probabilities(observation)

        # Calculate entropy for confidence
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        # Extract ENTER probabilities for each opportunity in this batch
        predictions = []
        for i, opp in enumerate(opportunities):
            enter_action = i + 1  # Actions 1-5 correspond to entering opportunities 0-4
            enter_prob = float(action_probs[enter_action])

            predictions.append({
                'opportunity_index': batch_offset + i,
                'symbol': opp.get('symbol', 'UNKNOWN'),
                'enter_probability': enter_prob,
                'confidence': self._calculate_confidence(enter_prob, entropy),
                'hold_probability': float(action_probs[0]),  # Action 0 = hold
                'state_value': float(state_value)
            })

        return predictions

    def evaluate_positions(
        self,
        positions: List[Dict],
        portfolio: Dict,
        opportunities: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate EXIT probabilities for open positions.

        Args:
            positions: List of open position dicts (max 3)
            portfolio: Current portfolio state
            opportunities: Current opportunities (for full observation)

        Returns:
            predictions: List of dicts with {
                'position_index': int,
                'symbol': str,
                'exit_probability': float,
                'confidence': str
            }
        """
        if not positions:
            return []

        # Limit to first 3 positions (model constraint)
        positions = positions[:3]

        # Use empty opportunities if not provided
        if opportunities is None:
            opportunities = []

        # Build observation
        observation = self._build_observation(opportunities, portfolio)

        # Get action probabilities
        action_probs, state_value = self.predict_probabilities(observation)

        # Calculate entropy
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        # Extract EXIT probabilities for each position
        predictions = []
        for i, pos in enumerate(positions):
            exit_action = 6 + i  # Actions 6-8 correspond to exiting positions 0-2
            exit_prob = float(action_probs[exit_action])

            predictions.append({
                'position_index': i,
                'symbol': pos.get('symbol', 'UNKNOWN'),
                'exit_probability': exit_prob,
                'confidence': self._calculate_confidence(exit_prob, entropy),
                'hold_probability': float(action_probs[0]),
                'state_value': float(state_value)
            })

        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            info: Dict with model metadata
        """
        return {
            'model_version': self.model_version,
            'model_type': 'PPO',
            'action_space': 9,
            'observation_space': 124,
            'max_opportunities': 5,
            'max_positions': 3
        }
