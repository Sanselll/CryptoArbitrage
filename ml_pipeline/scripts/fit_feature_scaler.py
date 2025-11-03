"""
Fit and save a StandardScaler for opportunity features.

This script loads the training data, extracts the 22 opportunity features,
fits a StandardScaler, and saves it to disk for use during training and inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


def extract_opportunity_features(row):
    """Extract the 22 features from a single opportunity."""
    features = [
        # Funding rates (raw)
        row.get('long_funding_rate', 0),
        row.get('short_funding_rate', 0),
        # Funding intervals (normalized)
        row.get('long_funding_interval_hours', 8) / 8,
        row.get('short_funding_interval_hours', 8) / 8,
        # Funding profit projections (raw)
        row.get('fund_profit_8h', 0),
        row.get('fundProfit8h24hProj', 0),
        row.get('fundProfit8h3dProj', 0),
        # APR projections (raw)
        row.get('fund_apr', 0),
        row.get('fundApr24hProj', 0),
        row.get('fundApr3dProj', 0),
        # Spread statistics (raw)
        row.get('spread30SampleAvg', 0),
        row.get('priceSpread24hAvg', 0),
        row.get('priceSpread3dAvg', 0),
        row.get('spread_volatility_stddev', 0),

        # Critical additions for opportunity quality (5 features)
        np.log10(max(float(row.get('volume_24h', 1e6) or 1e6), 1e5)),
        float(row.get('bidAskSpreadPercent', 0) or 0),
        np.log10(max(float(row.get('orderbookDepthUsd', 1e4) or 1e4), 1e3)),
        float(row.get('estimatedProfitPercentage', 0) or 0),
        float(row.get('positionCostPercent', 0.2) or 0.2),

        # Momentum/Trend features (3 features)
        row.get('spread30SampleAvg', 0) - row.get('priceSpread24hAvg', 0),
        row.get('fund_apr', 0) - row.get('fundApr24hProj', 0),
        row.get('priceSpread24hAvg', 0) - row.get('priceSpread3dAvg', 0),
    ]

    # Convert to float and handle NaN/inf
    features = [float(np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)) for x in features]
    return features


def main():
    print("="*80)
    print("FITTING FEATURE SCALER FOR RL ENVIRONMENT")
    print("="*80)

    # Paths
    train_data_path = "data/rl_train.csv"
    scaler_output_path = "models/rl/feature_scaler.pkl"

    # Create output directory
    Path(scaler_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load training data
    print(f"\nLoading training data from: {train_data_path}")
    df = pd.read_csv(train_data_path, low_memory=False)
    print(f"   Loaded {len(df):,} opportunities")

    # Extract all features
    print("\nExtracting 22 features from each opportunity...")
    all_features = []
    for idx, row in df.iterrows():
        features = extract_opportunity_features(row)
        all_features.append(features)

        if (idx + 1) % 10000 == 0:
            print(f"   Processed {idx + 1:,} / {len(df):,} opportunities")

    # Convert to numpy array
    X = np.array(all_features, dtype=np.float32)
    print(f"\nFeature matrix shape: {X.shape}")

    # Display feature statistics before scaling
    print("\n" + "-"*80)
    print("FEATURE STATISTICS (BEFORE SCALING)")
    print("-"*80)
    feature_names = [
        'long_funding_rate', 'short_funding_rate',
        'long_funding_interval_norm', 'short_funding_interval_norm',
        'fund_profit_8h', 'fundProfit8h24hProj', 'fundProfit8h3dProj',
        'fund_apr', 'fundApr24hProj', 'fundApr3dProj',
        'spread30SampleAvg', 'priceSpread24hAvg', 'priceSpread3dAvg',
        'spread_volatility_stddev',
        'volume_24h_log', 'bidAskSpreadPercent', 'orderbookDepthUsd_log',
        'estimatedProfitPercentage', 'positionCostPercent',
        'spread_momentum', 'funding_momentum', 'trend_strength'
    ]

    for i, name in enumerate(feature_names):
        mean = X[:, i].mean()
        std = X[:, i].std()
        min_val = X[:, i].min()
        max_val = X[:, i].max()
        print(f"{i+1:2d}. {name:30s}: mean={mean:+10.4f}, std={std:8.4f}, range=[{min_val:+10.4f}, {max_val:+10.4f}]")

    # Fit StandardScaler
    print("\n" + "="*80)
    print("FITTING STANDARDSCALER")
    print("="*80)
    scaler = StandardScaler()
    scaler.fit(X)

    print("✅ Scaler fitted successfully")
    print(f"   Mean shape: {scaler.mean_.shape}")
    print(f"   Scale shape: {scaler.scale_.shape}")

    # Display feature statistics after scaling (for verification)
    X_scaled = scaler.transform(X)
    print("\n" + "-"*80)
    print("FEATURE STATISTICS (AFTER SCALING)")
    print("-"*80)
    for i, name in enumerate(feature_names):
        mean = X_scaled[:, i].mean()
        std = X_scaled[:, i].std()
        min_val = X_scaled[:, i].min()
        max_val = X_scaled[:, i].max()
        print(f"{i+1:2d}. {name:30s}: mean={mean:+10.4f}, std={std:8.4f}, range=[{min_val:+10.4f}, {max_val:+10.4f}]")

    # Save scaler
    print("\n" + "="*80)
    print(f"Saving scaler to: {scaler_output_path}")
    with open(scaler_output_path, 'wb') as f:
        pickle.dump(scaler, f)

    print("✅ Scaler saved successfully")
    print("="*80)
    print("\nUsage in environment:")
    print("   1. Load scaler: scaler = pickle.load(open('models/rl/feature_scaler.pkl', 'rb'))")
    print("   2. Transform features: X_scaled = scaler.transform(X)")
    print("="*80)


if __name__ == "__main__":
    main()
