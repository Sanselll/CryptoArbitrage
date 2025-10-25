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

        # === RISK FEATURES ===

        # Volatility risk score
        df['volatility_risk_score'] = np.where(
            df['spread_volatility_cv'] < 0.2, 1.0,  # Low risk
            np.where(df['spread_volatility_cv'] < 0.5, 0.5, 0.0)  # Medium vs High risk
        )

        # === LIQUIDITY FEATURES ===

        # Volume adequacy (normalized by position size)
        df['volume_adequacy'] = df['volume_24h'] / 1000  # Assuming $1000 position size

        # Liquidity score (categorical to numerical)
        liquidity_mapping = {'Good': 1.0, 'Medium': 0.6, 'Low': 0.2}
        df['liquidity_score'] = df['liquidity_status'].map(liquidity_mapping).fillna(0.5)

        # === TIMING FEATURES (Cyclical Encoding) ===

        # Hour of day (sin/cos encoding to capture cyclical nature)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Day of week (sin/cos encoding)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # === INTERACTION FEATURES ===

        # Profit weighted by liquidity (opportunities with good liquidity are more valuable)
        df['profit_x_liquidity'] = df['fund_profit_8h'] * df['liquidity_score']

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

        # Volume imbalance (if long/short volumes differ significantly)
        df['volume_imbalance'] = np.where(
            (df['long_volume_24h'].notna()) & (df['short_volume_24h'].notna()) &
            (df['long_volume_24h'] + df['short_volume_24h'] > 0),
            np.abs(df['long_volume_24h'] - df['short_volume_24h']) / (df['long_volume_24h'] + df['short_volume_24h']),
            0
        )

        # === BTC CONTEXT ===

        # BTC price (normalized to thousands)
        df['btc_price_k'] = df['btc_price_at_entry'] / 1000

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

        categorical_cols = ['symbol', 'strategy', 'long_exchange', 'short_exchange', 'market_regime']

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
            'fund_profit_8h', 'fund_apr', 'fund_profit_8h_24h_proj', 'fund_profit_8h_3d_proj',
            'break_even_hours', 'spread_volatility_cv', 'spread_volatility_stddev',
            'spread_30sample_avg', 'volume_24h', 'long_volume_24h', 'short_volume_24h',
            'bid_ask_spread_pct', 'orderbook_depth_usd', 'position_cost_pct',
            'position_size_usd', 'btc_price_at_entry',
            # Engineered features
            'rate_momentum_24h', 'rate_momentum_3d', 'rate_stability',
            'volume_adequacy', 'risk_adjusted_return', 'breakeven_efficiency',
            'spread_consistency', 'volume_imbalance', 'btc_price_k'
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
