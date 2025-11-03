"""
Position Feature Engineering

Additional feature engineering for position snapshot data.
Adds derived features, interaction terms, and temporal features.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PositionFeatureEngineer:
    """
    Feature engineering for position snapshot data.
    Adds derived features and prepares data for ML models.
    """

    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []

    def add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features to the dataframe.

        Args:
            df: DataFrame with snapshot data

        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()

        print("Adding derived features...")

        # ============================================
        # P&L ACCELERATION AND MOMENTUM
        # ============================================

        # P&L acceleration (change in velocity)
        df['pnl_acceleration'] = df.groupby('execution_id')['pnl_velocity_per_hour'].diff()

        # P&L momentum (velocity * time)
        df['pnl_momentum'] = df['pnl_velocity_per_hour'] * df['time_in_position_hours']

        # Distance from peak (absolute value)
        df['distance_from_peak'] = np.abs(df['drawdown_from_peak_percent'])

        # P&L trend (positive = improving, negative = deteriorating)
        df['pnl_trend'] = np.where(df['pnl_velocity_per_hour'] > 0, 1, -1)

        # ============================================
        # FUNDING RATE FEATURES
        # ============================================

        # Funding rate momentum (change over time)
        df['funding_diff_velocity'] = df.groupby('execution_id')['funding_rate_differential_change'].diff()

        # Funding reversal severity
        df['funding_reversal_severity'] = df['funding_reversal_magnitude'] * np.abs(df['entry_funding_rate_differential'])

        # Time until next funding (normalized)
        df['funding_proximity'] = 1 / (df['minutes_to_next_funding'] + 1)  # High when close to funding

        # ============================================
        # SPREAD DYNAMICS
        # ============================================

        # Spread expansion rate
        df['spread_expansion_rate'] = df['spread_change_since_entry_percent'] / (df['time_in_position_hours'] + 0.01)

        # Spread volatility ratio
        df['spread_volatility_ratio'] = np.where(
            df['entry_spread_volatility_stddev'].notna() & (df['entry_spread_volatility_stddev'] > 0),
            df['current_spread_volatility_stddev'] / df['entry_spread_volatility_stddev'],
            1.0
        )

        # ============================================
        # TIME-BASED FEATURES
        # ============================================

        # Position age (normalized by predicted duration)
        df['position_age_normalized'] = np.where(
            df['ml_predicted_duration_hours'].notna() & (df['ml_predicted_duration_hours'] > 0),
            df['time_in_position_hours'] / df['ml_predicted_duration_hours'],
            df['time_in_position_hours'] / 24.0  # Normalize by 24h if no prediction
        )

        # Time pressure (approaching max hold time)
        MAX_HOLD_HOURS = 72
        df['time_pressure'] = df['time_in_position_hours'] / MAX_HOLD_HOURS

        # Time squared (capture non-linear time effects)
        df['time_in_position_squared'] = df['time_in_position_hours'] ** 2

        # ============================================
        # EFFICIENCY METRICS
        # ============================================

        # P&L efficiency (current vs predicted)
        df['pnl_efficiency'] = np.where(
            df['ml_predicted_profit_percent'].notna() & (df['ml_predicted_profit_percent'] != 0),
            df['current_pnl_percent'] / df['ml_predicted_profit_percent'],
            0.0
        )

        # Time efficiency (actual vs predicted)
        df['time_efficiency'] = np.where(
            df['ml_predicted_duration_hours'].notna() & (df['ml_predicted_duration_hours'] > 0),
            df['ml_predicted_duration_hours'] / (df['time_in_position_hours'] + 0.01),
            1.0
        )

        # Combined efficiency score
        df['efficiency_score'] = (df['pnl_efficiency'] + df['time_efficiency']) / 2

        # ============================================
        # RISK INDICATORS
        # ============================================

        # Drawdown risk (current drawdown / max historical drawdown)
        df['drawdown_risk'] = np.where(
            df['max_drawdown_percent'] < 0,
            df['drawdown_from_peak_percent'] / df['max_drawdown_percent'],
            0.0
        )

        # Volatility shock (sudden change in volatility)
        df['volatility_shock'] = np.abs(df['volatility_change_ratio'] - 1.0)

        # Volume decline risk (volume dropping significantly)
        df['volume_decline_risk'] = np.where(
            df['volume_change_ratio'] < 1.0,
            1.0 - df['volume_change_ratio'],
            0.0
        )

        # ============================================
        # INTERACTION FEATURES
        # ============================================

        # P&L * Momentum
        df['pnl_momentum_interaction'] = df['current_pnl_percent'] * df['pnl_velocity_per_hour']

        # Funding reversal * Time
        df['funding_time_interaction'] = df['funding_reversal_magnitude'] * df['time_in_position_hours']

        # Spread change * Volume change
        df['spread_volume_interaction'] = df['spread_change_since_entry_percent'] * df['volume_change_ratio']

        # Efficiency * Maturity
        df['efficiency_maturity_interaction'] = df['hold_efficiency'] * df.get('position_maturity', 0)

        # ============================================
        # SIGNAL STRENGTH FEATURES
        # ============================================

        # Consecutive movement strength
        df['movement_strength'] = df['consecutive_positive_samples'] - df['consecutive_negative_samples']

        # Peak distance concern (far from peak = concerning)
        df['peak_distance_concern'] = np.where(
            df['peak_pnl_percent'] > 0,
            df['distance_from_peak'] / (df['peak_pnl_percent'] + 0.01),
            0.0
        )

        print(f"✅ Added {len(df.columns) - len(df.columns)} derived features")

        return df

    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.

        Args:
            df: DataFrame with snapshot data
            fit: If True, fit label encoders; otherwise use existing ones

        Returns:
            DataFrame with encoded categorical features
        """
        df = df.copy()

        categorical_cols = ['symbol', 'strategy', 'long_exchange', 'short_exchange', 'exit_reason']

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('UNKNOWN'))
                else:
                    if col in self.label_encoders:
                        # Handle unknown categories
                        known_labels = set(self.label_encoders[col].classes_)
                        df[col] = df[col].apply(lambda x: x if x in known_labels else 'UNKNOWN')
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].fillna('UNKNOWN'))

        return df

    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.

        Args:
            df: DataFrame with snapshot data
            fit: If True, fit scaler; otherwise use existing one

        Returns:
            DataFrame with normalized features
        """
        df = df.copy()

        # Get numerical columns (excluding IDs, labels, and timestamps)
        exclude_cols = [
            'execution_id', 'snapshot_index', 'snapshot_time', 'entry_time',
            'should_exit_now', 'exit_reason', 'hours_until_optimal_exit',
            'optimal_exit_pnl_percent', 'potential_pnl_loss',
            'symbol', 'strategy', 'long_exchange', 'short_exchange'
        ]

        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]

        if len(numerical_cols) == 0:
            return df

        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
            self.feature_names = numerical_cols
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])

        return df

    def prepare_for_training(
        self,
        df: pd.DataFrame,
        add_features: bool = True,
        normalize: bool = True,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Prepare snapshot data for model training.

        Args:
            df: Raw snapshot DataFrame
            add_features: Whether to add derived features
            normalize: Whether to normalize features
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            Tuple of (prepared_df, feature_columns, label_columns)
        """
        df = df.copy()

        # Add derived features
        if add_features:
            df = self.add_derived_features(df)

        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)

        # Normalize if requested
        if normalize:
            df = self.normalize_features(df, fit=fit)

        # Get feature and label columns
        label_cols = [
            'should_exit_now',
            'exit_reason',
            'hours_until_optimal_exit',
            'optimal_exit_pnl_percent',
            'potential_pnl_loss'
        ]

        # Feature columns are all numerical except IDs, timestamps, and labels
        exclude_cols = [
            'execution_id', 'snapshot_index', 'snapshot_time', 'entry_time',
            'symbol', 'strategy', 'long_exchange', 'short_exchange'
        ] + label_cols

        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col not in exclude_cols]

        print(f"\n✅ Prepared data for training:")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Labels: {len(label_cols)}")

        return df, feature_cols, label_cols

    def split_by_execution(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data by execution_id to prevent data leakage.
        All snapshots from the same execution stay together.

        Args:
            df: DataFrame with snapshot data
            train_ratio: Ratio of executions for training
            val_ratio: Ratio of executions for validation
            test_ratio: Ratio of executions for testing

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, "Ratios must sum to 1.0"

        # Get unique execution IDs
        execution_ids = df['execution_id'].unique()
        n_executions = len(execution_ids)

        # Shuffle execution IDs
        np.random.shuffle(execution_ids)

        # Calculate split points
        train_n = int(n_executions * train_ratio)
        val_n = int(n_executions * val_ratio)

        train_ids = execution_ids[:train_n]
        val_ids = execution_ids[train_n:train_n + val_n]
        test_ids = execution_ids[train_n + val_n:]

        # Split dataframes
        train_df = df[df['execution_id'].isin(train_ids)].copy()
        val_df = df[df['execution_id'].isin(val_ids)].copy()
        test_df = df[df['execution_id'].isin(test_ids)].copy()

        print(f"\n📊 Data split by execution:")
        print(f"   Train: {len(train_df):,} snapshots ({len(train_ids):,} executions)")
        print(f"   Val:   {len(val_df):,} snapshots ({len(val_ids):,} executions)")
        print(f"   Test:  {len(test_df):,} snapshots ({len(test_ids):,} executions)")

        # Label distribution
        print(f"\n🎯 Exit signal distribution:")
        print(f"   Train: {train_df['should_exit_now'].sum():,} / {len(train_df):,} ({train_df['should_exit_now'].mean()*100:.2f}%)")
        print(f"   Val:   {val_df['should_exit_now'].sum():,} / {len(val_df):,} ({val_df['should_exit_now'].mean()*100:.2f}%)")
        print(f"   Test:  {test_df['should_exit_now'].sum():,} / {len(test_df):,} ({test_df['should_exit_now'].mean()*100:.2f}%)")

        return train_df, val_df, test_df


# Example usage
if __name__ == "__main__":
    import sys
    from pathlib import Path
    from snapshot_loader import SnapshotLoader

    if len(sys.argv) < 2:
        print("Usage: python position_features.py <path_to_snapshots.csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])

    # Load data
    loader = SnapshotLoader()
    df = loader.load_csv(csv_path)

    # Feature engineering
    engineer = PositionFeatureEngineer()
    prepared_df, feature_cols, label_cols = engineer.prepare_for_training(df)

    print(f"\n📋 Feature columns ({len(feature_cols)}):")
    for col in feature_cols[:10]:
        print(f"   - {col}")
    if len(feature_cols) > 10:
        print(f"   ... and {len(feature_cols) - 10} more")

    # Split data
    train_df, val_df, test_df = engineer.split_by_execution(prepared_df)

    print(f"\n✅ Feature engineering complete!")
