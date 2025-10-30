"""
C# Bridge Module for ML Predictions

This module provides a simple interface for C# to call Python ML models.
It takes opportunity data as JSON and returns predictions.
"""

import json
import numpy as np
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

    def predict_single(self, opportunity: Dict[str, Any], hold_duration_hours: float = 8.0) -> Dict[str, float]:
        """
        Predict for a single opportunity at a specific hold duration.

        Args:
            opportunity: Dictionary of opportunity data
            hold_duration_hours: Target hold duration in hours (default: 8.0)

        Returns:
            Dictionary with predictions: {
                'predicted_profit_percent': float,
                'success_probability': float,
                'predicted_duration_hours': float (returns same as input with fixed-duration model),
                'composite_score': float
            }
        """
        df = self._convert_to_dataframe([opportunity], hold_duration_hours)

        # Transform features
        X = self.preprocessor.transform(df)

        # Make predictions
        profit = float(self.profit_model.predict(X)[0])
        success_proba = float(self.success_model.predict_proba(X)[0, 1])

        # With fixed-duration model, duration prediction is less relevant
        # Return the target duration instead (what the user wants)
        duration = hold_duration_hours

        composite = self._calculate_composite_score(profit, success_proba, duration)

        return {
            'predicted_profit_percent': profit,
            'success_probability': success_proba,
            'predicted_duration_hours': duration,
            'composite_score': composite
        }

    def predict_batch(self, opportunities: List[Dict[str, Any]], hold_duration_hours: float = 8.0) -> List[Dict[str, float]]:
        """
        Predict for multiple opportunities, automatically finding optimal hold duration.

        For each opportunity, tests all 9 durations [0.5, 1, 2, 4, 8, 12, 24, 48, 72]
        and returns predictions at the optimal duration (highest profit with success > threshold).

        Args:
            opportunities: List of opportunity dictionaries
            hold_duration_hours: Ignored (kept for API compatibility)

        Returns:
            List of prediction dictionaries at optimal duration for each opportunity
        """
        # All durations to test (same as training data)
        durations_to_test = [0.5, 1, 2, 4, 8, 12, 24, 48, 72]

        # Minimum success probability threshold
        min_success_threshold = 0.05

        results = []

        for opp in opportunities:
            best_profit = -999.0
            best_success_prob = 0.0
            best_duration = -1  # Use -1 as sentinel (not a valid duration)
            best_composite = 0.0

            # Test all durations for this opportunity
            for test_duration in durations_to_test:
                # Convert to dataframe with this test duration
                df = self._convert_to_dataframe([opp], test_duration)

                # Transform features
                X = self.preprocessor.transform(df)

                # Make predictions
                profit = float(self.profit_model.predict(X)[0])
                success_prob = float(self.success_model.predict_proba(X)[0, 1])

                # Calculate composite score
                composite = self._calculate_composite_score(profit, success_prob, test_duration)

                # Select if this is better (higher profit AND meets success threshold)
                # Also prefer shorter duration if profit is similar (within 0.1%)
                if success_prob >= min_success_threshold:
                    if profit > best_profit + 0.1:
                        # Significantly better profit
                        best_profit = profit
                        best_success_prob = success_prob
                        best_duration = test_duration
                        best_composite = composite
                    elif abs(profit - best_profit) <= 0.1 and test_duration < best_duration:
                        # Similar profit but shorter duration (better capital efficiency)
                        best_profit = profit
                        best_success_prob = success_prob
                        best_duration = test_duration
                        best_composite = composite

            # If no duration met threshold, default to 8h prediction
            if best_duration == -1:
                df = self._convert_to_dataframe([opp], 8.0)
                X = self.preprocessor.transform(df)
                best_profit = float(self.profit_model.predict(X)[0])
                best_success_prob = float(self.success_model.predict_proba(X)[0, 1])
                best_duration = 8.0
                best_composite = self._calculate_composite_score(best_profit, best_success_prob, 8.0)

            print(f"[OPTIMIZER] {opp.get('symbol', 'UNKNOWN')}: best={best_duration}h, profit={best_profit:.2f}%, success={best_success_prob:.1%}")

            # Add result for this opportunity
            results.append({
                'predicted_profit_percent': best_profit,
                'success_probability': best_success_prob,
                'predicted_duration_hours': best_duration,
                'composite_score': best_composite
            })

        return results

    def _extract_tft_static_features(self, opp: Dict[str, Any]) -> np.ndarray:
        """
        Extract generalizable static features for TFT (prevent memorization).

        Returns array of [volatility_class, liquidity_tier, funding_interval_type]
        """
        # 1. Volatility class (0=low, 1=medium, 2=high)
        vol = opp.get('spreadVolatilityStdDev', 0)
        if vol < 0.01:
            volatility_class = 0  # Low volatility
        elif vol < 0.05:
            volatility_class = 1  # Medium volatility
        else:
            volatility_class = 2  # High volatility

        # 2. Liquidity tier (0=low, 1=medium, 2=high)
        volume = opp.get('volume24h', 0)
        if volume < 1e6:
            liquidity_tier = 0  # Low liquidity
        elif volume < 1e7:
            liquidity_tier = 1  # Medium liquidity
        else:
            liquidity_tier = 2  # High liquidity

        # 3. Funding interval type (0=hourly, 1=8h, 2=mixed)
        long_interval = opp.get('longFundingIntervalHours', 8)
        short_interval = opp.get('shortFundingIntervalHours', 8)
        if long_interval == 1 and short_interval == 1:
            funding_interval_type = 0  # Both hourly
        elif long_interval == 8 and short_interval == 8:
            funding_interval_type = 1  # Both 8-hourly
        else:
            funding_interval_type = 2  # Mixed

        return np.array([volatility_class, liquidity_tier, funding_interval_type], dtype=np.int64)

    def _extract_tft_time_varying_features(self, opp: Dict[str, Any]) -> np.ndarray:
        """
        Extract time-varying features for TFT (20 raw features).

        Returns array matching TFTDataLoader.TIME_VARYING_FEATURES order.
        """
        detected_at = pd.to_datetime(opp['detectedAt'])
        long_next_funding_minutes = (pd.to_datetime(opp['longNextFundingTime']) - detected_at).total_seconds() / 60
        short_next_funding_minutes = (pd.to_datetime(opp['shortNextFundingTime']) - detected_at).total_seconds() / 60

        features = np.array([
            # Current funding rates
            opp['fundProfit8h'],
            opp['fundApr'],
            opp['longFundingRate'],
            opp['shortFundingRate'],
            opp['longFundingIntervalHours'],
            opp['shortFundingIntervalHours'],

            # Funding projections (multi-timeframe)
            opp.get('fundProfit8h24hProj', 0),
            opp.get('fundApr24hProj', 0),
            opp.get('fundProfit8h3dProj', 0),
            opp.get('fundApr3dProj', 0),

            # Spread analysis (multi-timeframe) - INCLUDING BOTH current and 30sample
            opp.get('priceSpread24hAvg', 0),
            opp.get('priceSpread3dAvg', 0),
            opp.get('spread30SampleAvg', 0),

            # Volatility & liquidity
            opp.get('spreadVolatilityStdDev', 0),
            opp.get('spreadVolatilityCv', 0),
            opp['volume24h'],

            # Timing features
            long_next_funding_minutes,
            short_next_funding_minutes,
            detected_at.hour,
            detected_at.dayofweek,
        ], dtype=np.float32)

        return features

    def _convert_to_dataframe(self, opportunities: List[Dict[str, Any]], hold_duration_hours: float = 8.0) -> pd.DataFrame:
        """
        Convert C# opportunity objects to pandas DataFrame with ML features.

        Args:
            opportunities: List of opportunity dictionaries
            hold_duration_hours: Target hold duration in hours

        Returns:
            DataFrame with features including target_hold_hours
        """
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

                # Fixed-duration prediction feature (NEW)
                'target_hold_hours': hold_duration_hours,

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
