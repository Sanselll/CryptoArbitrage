"""
RL Predictor Module - Execution Mode

Loads and runs the trained RL model (PPO) to provide action predictions
for opportunities and executions using execution-based architecture:
- 1 opportunity + 1 execution per evaluation
- 36-dim observation space (14 portfolio+execution + 22 opportunity features)
- 3 actions (HOLD=0, ENTER=1, EXIT=2)

Usage:
    predictor = RLPredictor('models/simple_mode_pbt/pbt_20251103_104418/best_global_model.zip')
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
    RL Model predictor for crypto arbitrage opportunities and executions (Execution Mode).

    Architecture:
    - 1 opportunity + 1 execution per evaluation
    - Observation: 36 dims (14 portfolio+execution + 22 opportunity)
      * Portfolio base: 4 dims (capital_ratio, utilization, pnl, drawdown)
      * Execution features: 10 dims (net_pnl, hours, net_funding_ratio, net_funding_rate,
                                     current_spread, entry_spread, value_ratio, funding_efficiency,
                                     long_pnl, short_pnl)
      * Opportunity: 22 features
    - Actions: 3 (HOLD=0, ENTER=1, EXIT=2)

    Uses the trained PPO model to evaluate:
    - ENTER probability for a single opportunity (Action 1)
    - EXIT probability for a single open execution (Action 2)
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
            observation: State observation (36 dimensions)

        Returns:
            action_probs: Probability for each action (3 actions: HOLD, ENTER, EXIT)
            state_value: Value estimate for the state
        """
        # Add batch dimension if needed (1, 36)
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

    def _build_observation(self, opportunity: Dict, portfolio: Dict) -> np.ndarray:
        """
        Build observation vector from single opportunity and portfolio state (Execution Mode).

        Observation structure (36 dims):
        - Portfolio base: 4 dimensions
          * capital_ratio, utilization, total_pnl, drawdown
        - Execution features: 10 dimensions
          * net_pnl %, hours_held, net_funding_ratio, net_funding_rate
          * current_spread %, entry_spread %, value_ratio, funding_efficiency
          * long_pnl %, short_pnl %
        - Opportunity: 22 features

        Args:
            opportunity: Single opportunity dict
            portfolio: Portfolio state dict

        Returns:
            observation: numpy array (36,)
        """
        # Portfolio base features (4 dims)
        capital = portfolio.get('total_capital', portfolio.get('capital', 10000.0))
        initial_capital = portfolio.get('initial_capital', 10000.0)
        capital_ratio = capital / initial_capital if initial_capital > 0 else 1.0

        utilization = portfolio.get('utilization', 0.0)
        total_pnl = portfolio.get('total_pnl_pct', 0.0) / 100.0  # Convert % to decimal
        drawdown = portfolio.get('max_drawdown', portfolio.get('drawdown', 0.0)) / 100.0

        # Execution features (10 dims)
        positions = portfolio.get('positions', [])
        if len(positions) > 0:
            pos = positions[0]  # Only first execution

            # 1. Net P&L % (unrealized_pnl_pct)
            net_pnl_pct = pos.get('unrealized_pnl_pct', 0.0) / 100.0

            # 2. Hours held
            hours_held = pos.get('position_age_hours', pos.get('hours_held', 0.0)) / 72.0

            # 3. Net funding ratio (cumulative funding / capital)
            total_capital_used = pos.get('position_size_usd', 10000.0) * 2
            # Updated to use new feature names from C# (net funding per side)
            long_net_funding = pos.get('long_net_funding_usd', 0.0)
            short_net_funding = pos.get('short_net_funding_usd', 0.0)
            net_funding_usd = long_net_funding + short_net_funding
            net_funding_ratio = net_funding_usd / total_capital_used if total_capital_used > 0 else 0.0

            # 4. Net funding rate (short_rate - long_rate)
            net_funding_rate = (pos.get('short_funding_rate', 0.0) -
                               pos.get('long_funding_rate', 0.0))

            # 5. Current spread % (price difference between long/short)
            current_long_price = pos.get('current_long_price', pos.get('current_price', 0.0))
            current_short_price = pos.get('current_short_price', pos.get('current_price', 0.0))
            if current_long_price > 0 and current_short_price > 0:
                avg_current_price = (current_long_price + current_short_price) / 2
                current_spread_pct = abs(current_long_price - current_short_price) / avg_current_price
            else:
                current_spread_pct = 0.0

            # 6. Entry spread % (initial price difference)
            entry_long_price = pos.get('entry_long_price', pos.get('entry_price', 0.0))
            entry_short_price = pos.get('entry_short_price', pos.get('entry_price', 0.0))
            if entry_long_price > 0 and entry_short_price > 0:
                avg_entry_price = (entry_long_price + entry_short_price) / 2
                entry_spread_pct = abs(entry_long_price - entry_short_price) / avg_entry_price
            else:
                entry_spread_pct = 0.0

            # 7. Value-to-capital ratio
            value_ratio = total_capital_used / capital if capital > 0 else 0.0

            # 8. Funding efficiency (net_funding / total_fees)
            total_fees = pos.get('entry_fees_paid_usd', 0.0) + (total_capital_used * 0.0006)  # Approx taker fee
            funding_efficiency = net_funding_usd / total_fees if total_fees > 0 else 0.0

            # 9. Long side P&L %
            long_pnl_pct = pos.get('long_pnl_pct', 0.0) / 100.0

            # 10. Short side P&L %
            short_pnl_pct = pos.get('short_pnl_pct', 0.0) / 100.0

            execution_features = [
                net_pnl_pct, hours_held, net_funding_ratio, net_funding_rate,
                current_spread_pct, entry_spread_pct, value_ratio, funding_efficiency,
                long_pnl_pct, short_pnl_pct
            ]
        else:
            execution_features = [0.0] * 10

        portfolio_obs = [capital_ratio, utilization, total_pnl, drawdown] + execution_features

        # Single opportunity features (22 dims)
        opportunity_features = self._extract_opportunity_features(opportunity)

        # Combine
        observation = np.array(portfolio_obs + opportunity_features, dtype=np.float32)

        # Apply StandardScaler (only to opportunity features, portfolio+execution already normalized)
        opp_features_scaled = self.scaler.transform([observation[14:]])  # Skip first 14 (4 portfolio + 10 execution)
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
        Evaluate ENTER probabilities for opportunities (Simple Mode: 1 at a time).

        Each opportunity is evaluated independently by the model.

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

        # Process each opportunity individually
        for idx, opp in enumerate(opportunities):
            prediction = self._evaluate_single_opportunity(opp, portfolio, idx)
            all_predictions.append(prediction)

        return all_predictions

    def _evaluate_single_opportunity(
        self,
        opportunity: Dict,
        portfolio: Dict,
        opportunity_index: int = 0
    ) -> Dict[str, Any]:
        """
        Evaluate a single opportunity (Simple Mode).

        Args:
            opportunity: Single opportunity dict
            portfolio: Current portfolio state
            opportunity_index: Index for tracking in results

        Returns:
            prediction: Dict with prediction details
        """
        # Build observation for single opportunity
        observation = self._build_observation(opportunity, portfolio)

        # Get action probabilities
        action_probs, state_value = self.predict_probabilities(observation)

        # Calculate entropy for confidence
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        # Extract ENTER probability (Action 1) and HOLD probability (Action 0)
        # For opportunities, EXIT action doesn't make sense, so normalize ENTER + HOLD
        enter_prob_raw = float(action_probs[1])
        hold_prob_raw = float(action_probs[0])

        # Normalize to 100% (discard EXIT probability)
        total_relevant = enter_prob_raw + hold_prob_raw
        if total_relevant > 0:
            enter_prob = enter_prob_raw / total_relevant
            hold_prob = hold_prob_raw / total_relevant
        else:
            # Fallback if both are zero (shouldn't happen)
            enter_prob = 0.5
            hold_prob = 0.5

        return {
            'opportunity_index': opportunity_index,
            'symbol': opportunity.get('symbol', 'UNKNOWN'),
            'enter_probability': enter_prob,
            'confidence': self._calculate_confidence(enter_prob, entropy),
            'hold_probability': hold_prob,
            'state_value': float(state_value)
        }

    def evaluate_positions(
        self,
        positions: List[Dict],
        portfolio: Dict,
        opportunity: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate EXIT probability for open position (Simple Mode: 1 position only).

        Args:
            positions: List of open position dicts (only first is evaluated)
            portfolio: Current portfolio state
            opportunity: Current opportunity (required for full observation)

        Returns:
            predictions: List with single dict {
                'position_index': int,
                'symbol': str,
                'exit_probability': float,
                'confidence': str,
                'hold_probability': float,
                'state_value': float
            }
        """
        if not positions:
            return []

        # Only evaluate first position (simple mode constraint)
        pos = positions[0]

        # Use empty opportunity if not provided
        if opportunity is None:
            opportunity = {}

        # Build observation
        observation = self._build_observation(opportunity, portfolio)

        # Get action probabilities
        action_probs, state_value = self.predict_probabilities(observation)

        # Calculate entropy
        entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))

        # Extract EXIT probability (Action 2) and HOLD probability (Action 0)
        # For positions, ENTER action doesn't make sense, so normalize EXIT + HOLD
        exit_prob_raw = float(action_probs[2])
        hold_prob_raw = float(action_probs[0])

        # Normalize to 100% (discard ENTER probability)
        total_relevant = exit_prob_raw + hold_prob_raw
        if total_relevant > 0:
            exit_prob = exit_prob_raw / total_relevant
            hold_prob = hold_prob_raw / total_relevant
        else:
            # Fallback if both are zero (shouldn't happen)
            exit_prob = 0.5
            hold_prob = 0.5

        return [{
            'position_index': 0,
            'symbol': pos.get('symbol', 'UNKNOWN'),
            'exit_probability': exit_prob,
            'confidence': self._calculate_confidence(exit_prob, entropy),
            'hold_probability': hold_prob,
            'state_value': float(state_value)
        }]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model (Execution Mode).

        Returns:
            info: Dict with model metadata
        """
        return {
            'model_version': self.model_version,
            'model_type': 'PPO',
            'action_space': 3,
            'observation_space': 36,
            'max_opportunities': 1,
            'max_executions': 1,
            'architecture': 'execution_mode'
        }
