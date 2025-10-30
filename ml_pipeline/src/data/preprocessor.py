"""
Feature Preprocessor

Handles feature engineering, encoding, and scaling for ML training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, List, Optional, Tuple
import joblib
from pathlib import Path


class FeaturePreprocessor:
    """
    Feature engineering and preprocessing pipeline.

    Transforms raw opportunity data into ML-ready features:
    - Cyclical encoding for time features
    - Categorical encoding
    - Feature scaling
    - Feature creation (interactions, ratios)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.scaler = None
        self.feature_names = None
        self.categorical_mappings = {}
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        df = self._create_features(df)
        df = self._encode_categorical(df, fit=True)

        # Drop non-numeric columns that can't be used for training
        # entry_time is datetime, symbol is object, strategy/exchange are objects (not encoding them)
        # strategy_name and exit_reason are target variables (what we're trying to predict), not features
        cols_to_drop = ['entry_time', 'symbol', 'exit_time', 'strategy', 'long_exchange', 'short_exchange',
                        'strategy_name', 'exit_reason']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        df = self._scale_features(df, fit=True)

        self.is_fitted = True
        self.feature_names = df.columns.tolist()

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            df: Input DataFrame

        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted first. Call fit_transform().")

        df = self._create_features(df)
        df = self._encode_categorical(df, fit=False)

        # Drop non-numeric columns that can't be used for training
        cols_to_drop = ['entry_time', 'symbol', 'exit_time', 'strategy', 'long_exchange', 'short_exchange']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        df = self._scale_features(df, fit=False)

        # Ensure same columns as training
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with zeros

        df = df[self.feature_names]  # Reorder to match training

        return df

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features with multi-timeframe analysis.

        Strategy: Distinguish between long-term sustained profitability and short-term spikes,
        while avoiding mean reversion bias.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        df = df.copy()

        # === TEMPORAL FEATURES ===
        # REMOVED: Session and weekend flags caused overfitting to training period
        # These patterns (Asian session profitable, weekend effect) don't generalize
        #
        # # Market session (Asian/European/US hours) - REMOVED
        # if 'hour_of_day' in df.columns:
        #     df['is_asian_session'] = df['hour_of_day'].between(0, 8).astype(float)
        #     df['is_european_session'] = df['hour_of_day'].between(8, 16).astype(float)
        #     df['is_us_session'] = df['hour_of_day'].between(16, 24).astype(float)
        #
        # # Weekend effect - REMOVED
        # if 'day_of_week' in df.columns:
        #     df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(float)
        #     df['is_weekday'] = (~df['day_of_week'].isin([5, 6])).astype(float)

        # Cyclical encoding for hour (captures 24h cyclicality) - KEPT
        # These generalize better than raw hour_of_day
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Cyclical encoding for day of week (captures 7-day cyclicality) - KEPT
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # === FIXED-DURATION PREDICTION FEATURES ===
        # These features help the model understand profit expectations at different hold times
        if 'target_hold_hours' in df.columns:
            # Duration categories
            df['is_short_duration'] = (df['target_hold_hours'] <= 2).astype(float)  # 0.5h, 1h, 2h
            df['is_medium_duration'] = (df['target_hold_hours'].between(4, 12)).astype(float)  # 4h, 8h, 12h
            df['is_long_duration'] = (df['target_hold_hours'] >= 24).astype(float)  # 24h, 48h, 72h

            # Expected funding payments at target duration
            if 'long_funding_interval_hours' in df.columns:
                # Estimate how many funding payments will be received
                df['expected_funding_payments_long'] = df['target_hold_hours'] / df['long_funding_interval_hours'].fillna(8)
            if 'short_funding_interval_hours' in df.columns:
                df['expected_funding_payments_short'] = df['target_hold_hours'] / df['short_funding_interval_hours'].fillna(8)

            # Expected cumulative funding profit at target duration
            if 'fund_profit_8h' in df.columns:
                # Extrapolate 8h profit to target duration
                df['expected_funding_profit_at_target'] = (df['fund_profit_8h'] / 8.0) * df['target_hold_hours']

            # Duration-specific profit projections
            if 'fund_apr' in df.columns:
                # Convert APR to expected profit at target duration
                df['expected_profit_from_apr'] = (df['fund_apr'] / 365.0 / 24.0) * df['target_hold_hours']

            # Risk exposure based on duration
            if 'spread_volatility_cv' in df.columns:
                # Longer durations = more volatility exposure
                df['volatility_risk_scaled'] = df['spread_volatility_cv'] * np.sqrt(df['target_hold_hours'])

        # Funding cycle timing
        if 'long_next_funding_minutes' in df.columns:
            df['hours_until_long_funding'] = df['long_next_funding_minutes'] / 60
        if 'short_next_funding_minutes' in df.columns:
            df['hours_until_short_funding'] = df['short_next_funding_minutes'] / 60

        # Time until NEXT funding event (minimum of both exchanges)
        if 'hours_until_long_funding' in df.columns and 'hours_until_short_funding' in df.columns:
            df['hours_until_next_funding'] = df[['hours_until_long_funding', 'hours_until_short_funding']].min(axis=1)
            df['near_funding_event'] = (df['hours_until_next_funding'] < 1.0).astype(float)
            df['can_capture_funding'] = (df['hours_until_next_funding'] < 2.0).astype(float)

        # === LONG-TERM SUSTAINABILITY FEATURES (for 24-72h holds) ===

        # Primary profitability signal (use 3d projection directly - most stable)
        df['sustainable_profit'] = df['fund_apr_3d_proj']
        df['long_term_viable'] = (df['fund_apr_3d_proj'] > 20).astype(float)
        df['long_term_stable'] = (np.abs(df['fund_apr_24h_proj'] - df['fund_apr_3d_proj']) < 10).astype(float)
        df['long_term_improving'] = (df['fund_apr_24h_proj'] > df['fund_apr_3d_proj']).astype(float)
        df['long_term_deteriorating'] = (df['fund_apr_24h_proj'] < df['fund_apr_3d_proj'] - 20).astype(float)

        # === SHORT-TERM SPIKE FEATURES (for 1-12h holds) ===

        # Spike magnitude (how much current rate exceeds projections)
        df['rate_spike_magnitude'] = np.maximum(0, df['fund_profit_8h'] - df['fund_profit_8h_3d_proj'])
        df['rate_spike_magnitude_24h'] = np.maximum(0, df['fund_profit_8h'] - df['fund_profit_8h_24h_proj'])

        # Extreme rate flags
        df['extreme_rate'] = (np.abs(df['fund_profit_8h']) > 0.2).astype(float)
        df['very_high_rate'] = (np.abs(df['fund_profit_8h']) > 0.1).astype(float)

        # Spike opportunity (high rate + can capture it before reversal)
        if 'can_capture_funding' in df.columns:
            df['spike_opportunity_score'] = df['rate_spike_magnitude'] * (1 + df['can_capture_funding'])
        else:
            df['spike_opportunity_score'] = df['rate_spike_magnitude']

        # === CRITICAL: REVERSAL WARNING FEATURES (Prevents MEUSDT-like false positives) ===

        # Current high but 3d projection negative = DANGER (mean reversion)
        df['false_positive_reversal'] = (
            (df['fund_profit_8h'] > 0.05) &
            (df['fund_apr_3d_proj'] < -10)
        ).astype(float)

        # Rate deteriorating across timeframes (current > 24h > 3d but trending down)
        df['rate_deterioration_severe'] = (
            (df['fund_profit_8h'] > df['fund_profit_8h_24h_proj']) &
            (df['fund_profit_8h_24h_proj'] < df['fund_profit_8h_3d_proj'])
        ).astype(float)

        # Mean reversion risk (current far above 3d average)
        df['mean_reversion_risk'] = np.where(
            np.abs(df['fund_profit_8h_3d_proj']) > 0.001,
            np.abs(df['fund_profit_8h'] - df['fund_profit_8h_3d_proj']) / np.abs(df['fund_profit_8h_3d_proj']),
            0
        )

        # === MULTI-TIMEFRAME CONSISTENCY ===

        # All timeframes positive (strong signal)
        df['all_timeframes_positive'] = (
            (df['fund_profit_8h'] > 0.02) &
            (df['fund_profit_8h_24h_proj'] > 0.01) &
            (df['fund_apr_3d_proj'] > 10)
        ).astype(float)

        # Timeframe divergence (how much current differs from 3d projection)
        df['timeframe_divergence'] = np.abs(df['fund_apr'] - df['fund_apr_3d_proj'])

        # === SPREAD FEATURES (Multi-Timeframe Analysis) ===

        # Short-term momentum (30-sample vs 24h)
        if 'spread_30sample_avg' in df.columns and 'price_spread_24h_avg' in df.columns:
            df['spread_short_term_momentum'] = df['spread_30sample_avg'] - df['price_spread_24h_avg']
            df['spread_widening_short_term'] = (df['spread_30sample_avg'] > df['price_spread_24h_avg']).astype(float)
            df['spread_narrowing_short_term'] = (df['spread_30sample_avg'] < df['price_spread_24h_avg']).astype(float)

        # Medium-term trend (24h vs 3d)
        if 'price_spread_24h_avg' in df.columns and 'price_spread_3d_avg' in df.columns:
            df['spread_medium_term_trend'] = df['price_spread_24h_avg'] - df['price_spread_3d_avg']
            df['spread_widening_medium_term'] = (df['price_spread_24h_avg'] > df['price_spread_3d_avg']).astype(float)

        # Entry timing (is recent spread favorable vs historical?)
        if 'spread_30sample_avg' in df.columns and 'price_spread_3d_avg' in df.columns:
            df['spread_entry_favorable'] = (np.abs(df['spread_30sample_avg']) > np.abs(df['price_spread_3d_avg'])).astype(float)
            df['spread_stable_entry'] = (np.abs(df['spread_30sample_avg'] - df['price_spread_3d_avg']) < 0.2).astype(float)
            df['spread_deviation_from_mean'] = np.abs(df['spread_30sample_avg'] - df['price_spread_3d_avg'])

        # Multi-timeframe spread alignment
        if 'spread_30sample_avg' in df.columns and 'price_spread_24h_avg' in df.columns and 'price_spread_3d_avg' in df.columns:
            df['all_spreads_aligned'] = (
                (np.sign(df['spread_30sample_avg']) == np.sign(df['price_spread_24h_avg'])) &
                (np.sign(df['price_spread_24h_avg']) == np.sign(df['price_spread_3d_avg']))
            ).astype(float)

            # Spread mean reversion warning (current narrow, historically wide)
            df['spread_mean_reversion_risk'] = (
                (df['spread_short_term_momentum'] > 0.3) &
                (np.abs(df['spread_30sample_avg']) < np.abs(df['price_spread_3d_avg']) * 0.5) &
                (df['spread_deviation_from_mean'] > 0.2)
            ).astype(float)

        # === RISK-ADJUSTED RETURNS ===

        # Long-term risk-adjusted (use 3d projection)
        df['sustainable_return_per_vol'] = np.where(
            df['spread_volatility_cv'] > 0,
            df['fund_apr_3d_proj'] / (1 + df['spread_volatility_cv']),
            df['fund_apr_3d_proj']
        )

        # Short-term spike risk-adjusted
        df['spike_return_per_vol'] = np.where(
            df['spread_volatility_cv'] > 0,
            df['rate_spike_magnitude'] * 365 * 3 / (1 + df['spread_volatility_cv']),
            df['rate_spike_magnitude'] * 365 * 3
        )

        # === FUNDING RATE PATTERNS ===

        # Funding differential (the actual profit source)
        df['funding_differential'] = df['short_funding_rate'] - df['long_funding_rate']
        df['funding_favorable'] = (df['funding_differential'] > 0).astype(float)

        # Rate alignment patterns
        df['both_rates_positive'] = (
            (df['long_funding_rate'] > 0) & (df['short_funding_rate'] > 0)
        ).astype(float)
        df['both_rates_negative'] = (
            (df['long_funding_rate'] < 0) & (df['short_funding_rate'] < 0)
        ).astype(float)

        # Funding asymmetry (ratio of rates)
        df['funding_asymmetry'] = np.where(
            np.abs(df['short_funding_rate']) > 1e-10,
            df['long_funding_rate'] / df['short_funding_rate'],
            0
        )

        # === VOLATILITY CATEGORIES ===

        df['low_volatility'] = (df['spread_volatility_cv'] < 0.2).astype(float)
        df['medium_volatility'] = (
            (df['spread_volatility_cv'] >= 0.2) & (df['spread_volatility_cv'] < 0.5)
        ).astype(float)
        df['high_volatility'] = (df['spread_volatility_cv'] >= 0.5).astype(float)

        # === VOLUME & LIQUIDITY ===
        # REMOVED: Volume features have low variance (data is pre-filtered by volume)
        # Feature importance analysis showed 0-1.2% importance for volume features
        # Model was memorizing volume thresholds from training set
        #
        # # Volume magnitude features - REMOVED
        # df['volume_adequate'] = (df['volume_24h'] > 1_000_000).astype(float)
        # df['volume_high'] = (df['volume_24h'] > 10_000_000).astype(float)
        # df['volume_weight'] = np.minimum(1.0, df['volume_24h'] / 10_000_000)

        # Note: All volume/liquidity features (long_volume_24h, short_volume_24h,
        # bid_ask_spread, orderbook_depth, volume_24h) are NOT included because
        # opportunities are pre-filtered based on these thresholds.
        # Including them adds no variance/information and causes memorization.

        return df

    def _encode_categorical(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Encode categorical variables.

        Args:
            df: Input DataFrame
            fit: Whether to fit encoders

        Returns:
            DataFrame with encoded categoricals
        """
        df = df.copy()

        # Removed ALL categorical encoding:
        # - 'symbol' (197 unique values = too many features)
        # - 'market_regime' (not in dataset)
        # - 'strategy' (only 2 values, low variance)
        # - 'long_exchange', 'short_exchange' (learns names not patterns - won't generalize!)
        categorical_cols = []  # No categorical encoding - use engineered features instead

        if fit:
            # One-hot encoding
            df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)
        else:
            # Apply existing encodings
            for col in categorical_cols:
                if col in df.columns:
                    df = pd.get_dummies(df, columns=[col], drop_first=True, dtype=float)

        return df

    def _scale_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Scale numerical features.

        Args:
            df: Input DataFrame
            fit: Whether to fit scaler

        Returns:
            DataFrame with scaled features
        """
        # Columns to scale (numerical features only)
        scale_cols = [
            # Raw projections (stable signals)
            'fund_apr_24h_proj', 'fund_apr_3d_proj',
            'fund_profit_8h_24h_proj', 'fund_profit_8h_3d_proj',
            'fund_profit_8h', 'fund_apr',  # Keep for spike detection

            # Spread projections
            'price_spread_24h_avg', 'price_spread_3d_avg', 'spread_30sample_avg',

            # Volatility
            'spread_volatility_stddev', 'spread_volatility_cv',

            # REMOVED: Volume features (low variance, pre-filtered data)
            # 'volume_24h', 'volume_weight',

            # Long-term sustainability features
            'sustainable_profit',

            # Short-term spike features
            'rate_spike_magnitude', 'rate_spike_magnitude_24h', 'spike_opportunity_score',

            # Reversal warnings
            'mean_reversion_risk', 'timeframe_divergence',

            # Spread features
            'spread_short_term_momentum', 'spread_medium_term_trend',
            'spread_deviation_from_mean',

            # Risk-adjusted returns
            'sustainable_return_per_vol', 'spike_return_per_vol',

            # Funding patterns
            'funding_differential', 'funding_asymmetry',
            'long_funding_rate', 'short_funding_rate',

            # Timing
            'hours_until_next_funding', 'hours_until_long_funding', 'hours_until_short_funding'
        ]

        # Only scale columns that exist
        scale_cols = [col for col in scale_cols if col in df.columns]

        if not scale_cols:
            return df

        scaler_type = self.config.get('scaler_type', 'standard')

        if fit:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaler type: {scaler_type}")

            df[scale_cols] = self.scaler.fit_transform(df[scale_cols].fillna(0))
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit_transform() first.")

            df[scale_cols] = self.scaler.transform(df[scale_cols].fillna(0))

        return df

    def save(self, output_path: Path) -> None:
        """
        Save preprocessor to disk.

        Args:
            output_path: Path to save preprocessor
        """
        joblib.dump({
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'categorical_mappings': self.categorical_mappings,
            'config': self.config
        }, output_path)

        print(f"✅ Preprocessor saved to {output_path}")

    def load(self, model_path: Path) -> 'FeaturePreprocessor':
        """
        Load preprocessor from disk.

        Args:
            model_path: Path to saved preprocessor

        Returns:
            self
        """
        data = joblib.load(model_path)

        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.categorical_mappings = data['categorical_mappings']
        self.config = data['config']
        self.is_fitted = True

        print(f"✅ Preprocessor loaded from {model_path}")

        return self

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names after preprocessing.

        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted yet.")

        return self.feature_names
