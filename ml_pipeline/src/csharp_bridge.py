"""
C# Bridge Module for ML Predictions

This module provides a simple interface for C# to call Python ML models.
It takes opportunity data as JSON and returns predictions.
"""

import json
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, List, Any

from .data.preprocessor import FeaturePreprocessor


class MLPredictor:
    """ML prediction service for C# integration."""

    def __init__(self, model_dir: str = 'models/xgboost'):
        """Initialize models and preprocessor."""
        self.model_dir = Path(model_dir)
        self._load_models()

    def _load_models(self):
        """Load trained models and preprocessor."""
        # Load models
        profit_dict = joblib.load(self.model_dir / 'profit_model.pkl')
        success_dict = joblib.load(self.model_dir / 'success_model.pkl')
        duration_dict = joblib.load(self.model_dir / 'duration_model.pkl')

        self.profit_model = profit_dict['model']
        self.success_model = success_dict['model']
        self.duration_model = duration_dict['model']

        # Load preprocessor
        self.preprocessor = FeaturePreprocessor()
        self.preprocessor.load(self.model_dir / 'preprocessor.pkl')

    def _calculate_composite_score(self, profit: float, success_proba: float, duration: float) -> float:
        """
        Calculate weighted composite score (0-100).

        Weights:
        - Success probability: 80% (most accurate metric)
        - Predicted profit: 15% (second most important)
        - Hold duration: 5% (least important)

        Args:
            profit: Predicted profit percentage
            success_proba: Success probability (0-1)
            duration: Predicted hold duration in hours

        Returns:
            Composite score in 0-100 range
        """
        # Normalize profit to 0-100 (expected range: -5% to 20%)
        profit_score = max(0, min(100, (profit - (-5.0)) / (20.0 - (-5.0)) * 100))

        # Success probability already 0-1, scale to 0-100
        success_score = success_proba * 100

        # Duration score - inverse (shorter = better, range: 1h to 168h)
        duration_score = max(0, min(100, (1 - (duration - 1.0) / (168.0 - 1.0)) * 100))

        # Weighted composite: Success 80%, Profit 15%, Duration 5%
        composite = (
            success_score * 0.80 +
            profit_score * 0.15 +
            duration_score * 0.05
        )

        # Clip to 0-100 range
        return max(0.0, min(100.0, composite))

    def predict_single(self, opportunity: Dict[str, Any]) -> Dict[str, float]:
        """
        Predict for a single opportunity.

        Args:
            opportunity: Dictionary of opportunity data

        Returns:
            Dictionary with predictions: {
                'predicted_profit_percent': float,
                'success_probability': float,
                'predicted_duration_hours': float,
                'composite_score': float
            }
        """
        df = self._convert_to_dataframe([opportunity])

        # Transform features
        X = self.preprocessor.transform(df)

        # Make predictions
        profit = float(self.profit_model.predict(X)[0])
        success_proba = float(self.success_model.predict_proba(X)[0, 1])
        duration = float(self.duration_model.predict(X)[0])
        composite = self._calculate_composite_score(profit, success_proba, duration)

        return {
            'predicted_profit_percent': profit,
            'success_probability': success_proba,
            'predicted_duration_hours': duration,
            'composite_score': composite
        }

    def predict_batch(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """
        Predict for multiple opportunities.

        Args:
            opportunities: List of opportunity dictionaries

        Returns:
            List of prediction dictionaries
        """
        df = self._convert_to_dataframe(opportunities)

        # Transform features
        X = self.preprocessor.transform(df)

        # Make predictions
        profits = self.profit_model.predict(X)
        success_probas = self.success_model.predict_proba(X)[:, 1]
        durations = self.duration_model.predict(X)

        # Build results
        results = []
        for i in range(len(opportunities)):
            composite = self._calculate_composite_score(
                float(profits[i]),
                float(success_probas[i]),
                float(durations[i])
            )
            results.append({
                'predicted_profit_percent': float(profits[i]),
                'success_probability': float(success_probas[i]),
                'predicted_duration_hours': float(durations[i]),
                'composite_score': float(composite)
            })

        return results

    def _convert_to_dataframe(self, opportunities: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert C# opportunity objects to pandas DataFrame with ML features."""
        rows = []

        for opp in opportunities:
            # Parse detected_at timestamp
            detected_at = pd.to_datetime(opp['detectedAt'])

            row = {
                # Identifiers
                'symbol': opp['symbol'],
                'long_exchange': opp['longExchange'],
                'short_exchange': opp['shortExchange'],
                'entry_time': detected_at,

                # Temporal features
                'hour_of_day': detected_at.hour,
                'day_of_week': detected_at.dayofweek,  # Monday=0, Sunday=6

                # Funding rate features
                'long_funding_rate': opp['longFundingRate'],
                'short_funding_rate': opp['shortFundingRate'],
                'long_funding_interval_hours': opp['longFundingIntervalHours'],
                'short_funding_interval_hours': opp['shortFundingIntervalHours'],

                # Calculate minutes until funding
                'long_next_funding_minutes': (pd.to_datetime(opp['longNextFundingTime']) - detected_at).total_seconds() / 60,
                'short_next_funding_minutes': (pd.to_datetime(opp['shortNextFundingTime']) - detected_at).total_seconds() / 60,

                # Funding projections
                'fund_profit_8h': opp['fundProfit8h'],
                'fund_apr': opp['fundApr'],
                'fund_profit_8h_24h_proj': opp.get('fundProfit8h24hProj', 0),
                'fund_apr_24h_proj': opp.get('fundApr24hProj', 0),
                'fund_profit_8h_3d_proj': opp.get('fundProfit8h3dProj', 0),
                'fund_apr_3d_proj': opp.get('fundApr3dProj', 0),

                # Price spread statistics
                'price_spread_24h_avg': opp.get('priceSpread24hAvg', 0),
                'price_spread_3d_avg': opp.get('priceSpread3dAvg', 0),
                'spread_30sample_avg': opp.get('spread30SampleAvg', 0),
                'spread_volatility_stddev': opp.get('spreadVolatilityStdDev', 0),
                'spread_volatility_cv': opp.get('spreadVolatilityCv', 0),

                # Volume
                'volume_24h': opp['volume24h'],

                # Target flags (set to 0 since these are unknown at detection time)
                'hit_profit_target': 0,
                'hit_stop_loss': 0,

                # Metadata (unknown at detection time, added for consistency with training data)
                'strategy_name': '',
                'exit_reason': '',
            }

            rows.append(row)

        df = pd.DataFrame(rows)
        return df


# Global predictor instance
_predictor = None


def initialize(model_dir: str = 'models/xgboost') -> None:
    """Initialize the predictor. Call this once at startup."""
    global _predictor
    _predictor = MLPredictor(model_dir)


def predict_opportunity(opportunity_json: str) -> str:
    """
    Predict for a single opportunity.

    Args:
        opportunity_json: JSON string of opportunity data

    Returns:
        JSON string of predictions
    """
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Call initialize() first.")

    result = _predictor.predict_single(opportunity_json)
    return json.dumps(result)


def predict_opportunities(opportunities_json: str) -> str:
    """
    Predict for multiple opportunities.

    Args:
        opportunities_json: JSON string of array of opportunity data

    Returns:
        JSON string of array of predictions
    """
    if _predictor is None:
        raise RuntimeError("Predictor not initialized. Call initialize() first.")

    results = _predictor.predict_batch(opportunities_json)
    return json.dumps(results)
