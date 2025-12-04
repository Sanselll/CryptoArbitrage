"""
Fit and save a StandardScaler for opportunity features (V5.4).

V5.4 Changes: 11→12 features
- Added: spread_mean_reversion_potential = |spread_30| - |spread_3d| (sign-agnostic)

V3 Changes: 19→11 features
- Removed: raw funding rates/intervals, market quality features (9 features)
- Added: apr_velocity (1 feature)
- Kept: all profit projections (6) + all spread metrics (4)

Assumption: Market quality pre-filtering happens upstream
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


def extract_opportunity_features(row):
    """
    Extract the 12 features from a single opportunity (V5.4).

    V5.4: Added spread_mean_reversion_potential (sign-agnostic spread profitability)
    """
    spread_30 = row.get('spread_30_sample_avg', 0)
    spread_3d = row.get('price_spread_3d_avg', 0)

    # V5.4: Sign-agnostic spread mean-reversion potential
    spread_mean_reversion_potential = abs(spread_30) - abs(spread_3d)
    spread_mean_reversion_potential = np.clip(spread_mean_reversion_potential, -0.05, 0.05)

    features = [
        # Profit projections (6 features)
        row.get('fund_profit_8h', 0),
        row.get('fund_profit_8h_24h_proj', 0),
        row.get('fund_profit_8h_3d_proj', 0),
        row.get('fund_apr', 0),
        row.get('fund_apr_24h_proj', 0),
        row.get('fund_apr_3d_proj', 0),
        # Spread statistics (4 features)
        spread_30,
        row.get('price_spread_24h_avg', 0),
        spread_3d,
        row.get('spread_volatility_stddev', 0),
        # Velocity (1 feature)
        row.get('fund_profit_8h', 0) - row.get('fund_profit_8h_24h_proj', 0),  # apr_velocity
        # V5.4: Spread mean-reversion potential (1 feature)
        spread_mean_reversion_potential,
    ]

    # Convert to float and handle NaN/inf
    features = [float(np.nan_to_num(x, nan=0.0, posinf=100.0, neginf=-100.0)) for x in features]
    return features


def main():
    print("="*80)
    print("FITTING FEATURE SCALER FOR RL ENVIRONMENT (V5.4 - StandardScaler)")
    print("="*80)

    # Paths
    train_data_path = "data/rl_train.csv"
    scaler_output_path = "trained_models/rl/feature_scaler_v3.pkl"  # V5.4: 12 features

    # Create output directory
    Path(scaler_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load training data
    print(f"\nLoading training data from: {train_data_path}")
    df = pd.read_csv(train_data_path, low_memory=False)
    print(f"   Loaded {len(df):,} opportunities")

    # Extract all features
    print("\nExtracting 12 features from each opportunity (V5.4: 11→12)...")
    all_features = []
    for idx, row in df.iterrows():
        features = extract_opportunity_features(row)
        all_features.append(features)

        if (idx + 1) % 10000 == 0:
            print(f"   Processed {idx + 1:,} / {len(df):,} opportunities")

    # Convert to numpy array
    X = np.array(all_features, dtype=np.float32)
    print(f"\nFeature matrix shape: {X.shape} (expected: [N, 12])")

    # Display feature statistics before scaling
    print("\n" + "-"*80)
    print("FEATURE STATISTICS (BEFORE SCALING)")
    print("-"*80)
    feature_names = [
        'fund_profit_8h', 'fund_profit_8h_24h_proj', 'fund_profit_8h_3d_proj',
        'fund_apr', 'fund_apr_24h_proj', 'fund_apr_3d_proj',
        'spread_30_sample_avg', 'price_spread_24h_avg', 'price_spread_3d_avg',
        'spread_volatility_stddev',
        'apr_velocity',
        'spread_mean_reversion_potential',  # NEW in V5.4
    ]

    for i, name in enumerate(feature_names):
        mean = X[:, i].mean()
        std = X[:, i].std()
        min_val = X[:, i].min()
        max_val = X[:, i].max()
        print(f"{i+1:2d}. {name:30s}: mean={mean:+10.4f}, std={std:8.4f}, range=[{min_val:+10.4f}, {max_val:+10.4f}]")

    # Fit StandardScaler
    print("\n" + "="*80)
    print("FITTING STANDARDSCALER (mean/std-based)")
    print("="*80)
    scaler = StandardScaler()
    scaler.fit(X)

    print("✅ Scaler fitted successfully")
    print(f"   Mean shape: {scaler.mean_.shape}")
    print(f"   Scale (std) shape: {scaler.scale_.shape}")

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
    print("\nV5.4 SCALER USAGE:")
    print(f"   1. Load scaler: scaler = pickle.load(open('{scaler_output_path}', 'rb'))")
    print("   2. Transform features: X_scaled = scaler.transform(X)  # X must be shape [N, 12]")
    print("\nNOTE: This is V5.4 scaler (12 features, StandardScaler).")
    print("      Update FEATURE_SCALER_PATH in feature_config.py to 'trained_models/rl/feature_scaler_v3.pkl'")
    print("="*80)


if __name__ == "__main__":
    main()
