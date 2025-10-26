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
        Create engineered features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with new features
        """
        df = df.copy()

        # Fill NaN values in break-even columns with a large value (indicates no break-even point)
        # This is more meaningful than 0 (which would indicate instant break-even)
        if 'fund_break_even_24h_proj' in df.columns:
            df['fund_break_even_24h_proj'] = df['fund_break_even_24h_proj'].fillna(999)
        if 'fund_break_even_3d_proj' in df.columns:
            df['fund_break_even_3d_proj'] = df['fund_break_even_3d_proj'].fillna(999)
        if 'break_even_hours' in df.columns:
            df['break_even_hours'] = df['break_even_hours'].fillna(999)

        # === TEMPORAL FEATURES ===

        # Market session (Asian/European/US hours)
        if 'hour_of_day' in df.columns:
            df['is_asian_session'] = df['hour_of_day'].between(0, 8).astype(float)
            df['is_european_session'] = df['hour_of_day'].between(8, 16).astype(float)
            df['is_us_session'] = df['hour_of_day'].between(16, 24).astype(float)

        # Weekend effect
        if 'day_of_week' in df.columns:
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(float)
            df['is_weekday'] = (~df['day_of_week'].isin([5, 6])).astype(float)

        # Cyclical encoding for hour (captures 24h cyclicality)
        if 'hour_of_day' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Cyclical encoding for day of week (captures 7-day cyclicality)
        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Funding cycle position (hours until next funding event)
        if 'long_next_funding_minutes' in df.columns:
            df['hours_until_long_funding'] = df['long_next_funding_minutes'] / 60
        if 'short_next_funding_minutes' in df.columns:
            df['hours_until_short_funding'] = df['short_next_funding_minutes'] / 60

        # Time until NEXT funding event (minimum of both exchanges)
        if 'hours_until_long_funding' in df.columns and 'hours_until_short_funding' in df.columns:
            df['hours_until_next_funding'] = df[['hours_until_long_funding', 'hours_until_short_funding']].min(axis=1)

        # === PROFITABILITY FEATURES ===

        # Rate momentum (how current rate compares to historical averages)
        df['rate_momentum_24h'] = np.where(
            df['fund_profit_8h_24h_proj'].notna() & (df['fund_profit_8h_24h_proj'] != 0),
            (df['fund_profit_8h'] - df['fund_profit_8h_24h_proj']) / df['fund_profit_8h_24h_proj'].abs(),
            0
        )

        df['rate_momentum_3d'] = np.where(
            df['fund_profit_8h_3d_proj'].notna() & (df['fund_profit_8h_3d_proj'] != 0),
            (df['fund_profit_8h'] - df['fund_profit_8h_3d_proj']) / df['fund_profit_8h_3d_proj'].abs(),
            0
        )

        # Rate stability (how much rates are fluctuating)
        df['rate_stability'] = np.where(
            df['fund_profit_8h_3d_proj'].notna() & (df['fund_profit_8h_3d_proj'] != 0),
            np.abs(df['fund_profit_8h'] - df['fund_profit_8h_3d_proj']) / df['fund_profit_8h_3d_proj'].abs(),
            0
        )

        # Rate trend (uptrend, downtrend, or stable)
        df['rate_trend_score'] = np.where(
            (df['fund_profit_8h'] > df['fund_profit_8h_24h_proj']) &
            (df['fund_profit_8h_24h_proj'] > df['fund_profit_8h_3d_proj']),
            1,  # Uptrend
            np.where(df['fund_profit_8h'] < df['fund_profit_8h_3d_proj'], -1, 0)  # Downtrend vs Stable
        )

        # Rate acceleration (change in momentum)
        df['rate_acceleration'] = df['rate_momentum_24h'] - df['rate_momentum_3d']

        # === EXCHANGE-AGNOSTIC FEATURES (instead of encoding exchange names) ===

        # Funding differential (absolute difference between exchanges)
        df['funding_differential'] = np.abs(df['long_funding_rate'] - df['short_funding_rate'])

        # Funding direction (which exchange has higher rate)
        df['funding_direction'] = (df['long_funding_rate'] > df['short_funding_rate']).astype(float)

        # Funding asymmetry (ratio of rates)
        df['funding_asymmetry'] = np.where(
            np.abs(df['short_funding_rate']) > 1e-10,
            df['long_funding_rate'] / df['short_funding_rate'],
            0
        )

        # === PROFIT SIGNAL FEATURES ===

        # Expected profit signal (profit adjusted for volatility)
        df['expected_profit_signal'] = np.where(
            df['spread_volatility_cv'] > 0.01,
            df['fund_apr'] / df['spread_volatility_cv'],
            df['fund_apr'] * 100  # Low volatility = high signal
        )

        # Sharpe-like proxy (profit per unit of spread volatility)
        df['sharpe_proxy'] = np.where(
            df['spread_volatility_stddev'] > 0.001,
            df['fund_profit_8h'] / df['spread_volatility_stddev'],
            df['fund_profit_8h'] * 1000
        )

        # Profit consistency score
        df['profit_consistency'] = 1.0 / (1.0 + np.abs(df['rate_momentum_24h']))

        # === RISK FEATURES ===

        # Volatility risk score
        df['volatility_risk_score'] = np.where(
            df['spread_volatility_cv'] < 0.2, 1.0,  # Low risk
            np.where(df['spread_volatility_cv'] < 0.5, 0.5, 0.0)  # Medium vs High risk
        )

        # === BREAK-EVEN FEATURES ===

        # Break-even feasibility (can we break even within 24 hours?)
        df['break_even_feasibility'] = np.where(
            (df['break_even_hours'].notna()) & (df['break_even_hours'] < 24),
            1.0,
            0.0
        )

        # === INTERACTION FEATURES ===

        # Risk-adjusted return (profit divided by volatility)
        df['risk_adjusted_return'] = np.where(
            (df['spread_volatility_cv'].notna()) & (df['spread_volatility_cv'] > 0),
            df['fund_apr'] / (1 + df['spread_volatility_cv']),
            df['fund_apr']
        )

        # Break-even efficiency (inverse of break-even time)
        df['breakeven_efficiency'] = np.where(
            (df['break_even_hours'].notna()) & (df['break_even_hours'] > 0),
            100 / df['break_even_hours'],
            0
        )

        # === SPREAD METRICS ===

        # Spread consistency (is current spread consistent with historical average?)
        df['spread_consistency'] = np.where(
            (df['spread_30sample_avg'].notna()) & (df['spread_30sample_avg'] != 0),
            np.abs(df['fund_profit_8h'] - df['spread_30sample_avg']) / df['spread_30sample_avg'].abs(),
            0
        )

        # === EXCHANGE FEATURES ===

        # Note: long_volume_24h and short_volume_24h not in dataset
        # Volume imbalance feature removed

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
        # Note: Only including columns that exist in the dataset
        scale_cols = [
            # Funding features
            'fund_profit_8h', 'fund_apr', 'fund_profit_8h_24h_proj', 'fund_profit_8h_3d_proj',
            'funding_differential', 'funding_asymmetry',
            'long_funding_rate', 'short_funding_rate',

            # Spread features
            'current_price_spread_pct', 'price_spread_24h_avg', 'price_spread_3d_avg',
            'spread_30sample_avg', 'spread_volatility_stddev', 'spread_volatility_cv',

            # Break-even features
            'break_even_hours', 'fund_break_even_24h_proj', 'fund_break_even_3d_proj',

            # Volume
            'volume_24h',

            # Engineered profit signals
            'expected_profit_signal', 'sharpe_proxy', 'profit_consistency',

            # Momentum and trend features
            'rate_momentum_24h', 'rate_momentum_3d', 'rate_acceleration', 'rate_stability',

            # Other engineered features
            'risk_adjusted_return', 'breakeven_efficiency', 'spread_consistency',

            # Temporal features (cyclical encodings don't need scaling)
            'hours_until_long_funding', 'hours_until_short_funding', 'hours_until_next_funding'
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
