"""
Validate Backend ML Predictions

Loads opportunities from C# backend JSON dumps and validates predictions
by comparing them to Python ML pipeline predictions.
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from data.preprocessor import FeaturePreprocessor


def load_backend_opportunities(json_path):
    """Load opportunities from backend JSON dump."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract opportunities from the first snapshot
    if isinstance(data, list) and len(data) > 0:
        opportunities = data[0]['opportunities']
    else:
        raise ValueError("Unexpected JSON format")

    return opportunities


def convert_to_dataframe(opportunities):
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

            # Price spread
            'current_price_spread_pct': float(opp['currentPriceSpreadPercent']),

            # Funding projections
            'fund_profit_8h': opp['fundProfit8h'],
            'fund_apr': opp['fundApr'],
            'fund_profit_8h_24h_proj': opp.get('fundProfit8h24hProj'),
            'fund_apr_24h_proj': opp.get('fundApr24hProj'),
            'fund_break_even_24h_proj': opp.get('fundBreakEvenTime24hProj'),
            'fund_profit_8h_3d_proj': opp.get('fundProfit8h3dProj'),
            'fund_apr_3d_proj': opp.get('fundApr3dProj'),
            'fund_break_even_3d_proj': opp.get('priceSpread3dAvg'),  # Using as proxy

            # Break-even
            'break_even_hours': opp.get('breakEvenTimeHours', 0),

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

            # C# backend predictions (what we're validating)
            'csharp_predicted_profit': opp.get('mlPredictedProfitPercent'),
            'csharp_success_probability': opp.get('mlSuccessProbability'),
            'csharp_predicted_duration': opp.get('mlPredictedDurationHours'),
            'csharp_composite_score': opp.get('mlCompositeScore'),
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def load_models(model_dir='models/xgboost'):
    """Load trained models and preprocessor."""
    model_path = Path(model_dir)

    print(f"Loading models from {model_path}...")

    # Load models
    profit_dict = joblib.load(model_path / 'profit_model.pkl')
    success_dict = joblib.load(model_path / 'success_model.pkl')
    duration_dict = joblib.load(model_path / 'duration_model.pkl')

    profit_model = profit_dict['model']
    success_model = success_dict['model']
    duration_model = duration_dict['model']

    # Load preprocessor
    preprocessor = FeaturePreprocessor()
    preprocessor.load(model_path / 'preprocessor.pkl')

    print("✅ Models loaded successfully")
    return profit_model, success_model, duration_model, preprocessor


def validate_predictions(json_path, model_dir='models/xgboost'):
    """Validate backend predictions against Python ML pipeline."""

    print("=" * 100)
    print("BACKEND ML PREDICTION VALIDATION")
    print("=" * 100)

    # Load opportunities from backend JSON
    print(f"\n📂 Loading opportunities from: {json_path}")
    opportunities = load_backend_opportunities(json_path)
    print(f"   Found {len(opportunities)} opportunities")

    # Convert to DataFrame
    print("\n🔄 Converting to DataFrame format...")
    df = convert_to_dataframe(opportunities)

    # Save C# predictions before preprocessing
    csharp_predictions = df[['symbol', 'csharp_predicted_profit', 'csharp_success_probability',
                             'csharp_predicted_duration', 'csharp_composite_score']].copy()

    # Load models
    profit_model, success_model, duration_model, preprocessor = load_models(model_dir)

    # Prepare features for prediction
    print("\n🛠️  Extracting and preprocessing features...")
    # Drop target columns and C# predictions
    feature_cols_to_drop = ['csharp_predicted_profit', 'csharp_success_probability',
                            'csharp_predicted_duration', 'csharp_composite_score']
    df_features = df.drop(columns=feature_cols_to_drop, errors='ignore')

    # Transform using preprocessor
    X = preprocessor.transform(df_features)

    print(f"   Feature shape: {X.shape}")
    print(f"   Feature count: {len(preprocessor.feature_names)}")

    # Make predictions
    print("\n🤖 Making predictions with Python ML models...")
    python_profit = profit_model.predict(X)
    python_success_proba = success_model.predict_proba(X)[:, 1]
    python_duration = duration_model.predict(X)
    python_composite = python_success_proba * python_profit

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'symbol': df['symbol'],
        'long_exchange': df['long_exchange'],
        'short_exchange': df['short_exchange'],

        # Python predictions
        'py_profit_%': python_profit,
        'py_success_prob': python_success_proba,
        'py_duration_hrs': python_duration,
        'py_composite': python_composite,

        # C# predictions
        'cs_profit_%': csharp_predictions['csharp_predicted_profit'],
        'cs_success_prob': csharp_predictions['csharp_success_probability'],
        'cs_duration_hrs': csharp_predictions['csharp_predicted_duration'],
        'cs_composite': csharp_predictions['csharp_composite_score'],
    })

    # Calculate differences
    comparison['profit_diff'] = np.abs(comparison['py_profit_%'] - comparison['cs_profit_%'])
    comparison['success_diff'] = np.abs(comparison['py_success_prob'] - comparison['cs_success_prob'])
    comparison['duration_diff'] = np.abs(comparison['py_duration_hrs'] - comparison['cs_duration_hrs'])
    comparison['composite_diff'] = np.abs(comparison['py_composite'] - comparison['cs_composite'])

    # Print results
    print("\n" + "=" * 100)
    print("PREDICTION COMPARISON (Python ML Pipeline vs C# Backend)")
    print("=" * 100)

    for idx, row in comparison.iterrows():
        print(f"\n{'─' * 100}")
        print(f"📊 Opportunity #{idx + 1}: {row['symbol']} ({row['long_exchange']} vs {row['short_exchange']})")
        print(f"{'─' * 100}")

        print(f"\n   {'Metric':<20} {'Python':>15} {'C#':>15} {'Difference':>15} {'Match':>10}")
        print(f"   {'-' * 80}")

        # Profit
        profit_match = '✅' if row['profit_diff'] < 0.1 else '❌'
        print(f"   {'Profit %':<20} {row['py_profit_%']:>15.4f} {row['cs_profit_%']:>15.4f} {row['profit_diff']:>15.4f} {profit_match:>10}")

        # Success probability
        success_match = '✅' if row['success_diff'] < 0.05 else '❌'
        print(f"   {'Success Prob':<20} {row['py_success_prob']:>15.4f} {row['cs_success_prob']:>15.4f} {row['success_diff']:>15.4f} {success_match:>10}")

        # Duration
        duration_match = '✅' if row['duration_diff'] < 1.0 else '❌'
        print(f"   {'Duration (hrs)':<20} {row['py_duration_hrs']:>15.2f} {row['cs_duration_hrs']:>15.2f} {row['duration_diff']:>15.2f} {duration_match:>10}")

        # Composite
        composite_match = '✅' if row['composite_diff'] < 0.5 else '❌'
        print(f"   {'Composite Score':<20} {row['py_composite']:>15.4f} {row['cs_composite']:>15.4f} {row['composite_diff']:>15.4f} {composite_match:>10}")

    # Summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    print(f"\n📊 Prediction Accuracy:")
    print(f"   Profit MAE:           {comparison['profit_diff'].mean():.4f}%")
    print(f"   Success Prob MAE:     {comparison['success_diff'].mean():.4f}")
    print(f"   Duration MAE:         {comparison['duration_diff'].mean():.2f} hours")
    print(f"   Composite Score MAE:  {comparison['composite_diff'].mean():.4f}")

    # Check for systematic issues
    print(f"\n⚠️  Potential Issues:")

    # Check for negative probabilities (should be 0-1)
    invalid_cs_probs = (comparison['cs_success_prob'] < 0) | (comparison['cs_success_prob'] > 1)
    if invalid_cs_probs.any():
        print(f"   ❌ C# has {invalid_cs_probs.sum()} invalid success probabilities (outside 0-1 range)")
    else:
        print(f"   ✅ All C# success probabilities are in valid range (0-1)")

    # Check if all predictions are identical
    if comparison['cs_profit_%'].nunique() == 1:
        print(f"   ❌ All C# profit predictions are IDENTICAL: {comparison['cs_profit_%'].iloc[0]:.4f}")
    else:
        print(f"   ✅ C# profit predictions vary ({comparison['cs_profit_%'].nunique()} unique values)")

    if comparison['cs_success_prob'].nunique() == 1:
        print(f"   ❌ All C# success probabilities are IDENTICAL: {comparison['cs_success_prob'].iloc[0]:.4f}")
    else:
        print(f"   ✅ C# success probabilities vary ({comparison['cs_success_prob'].nunique()} unique values)")

    # Check for large discrepancies
    large_profit_diff = (comparison['profit_diff'] > 1.0).sum()
    large_success_diff = (comparison['success_diff'] > 0.1).sum()

    print(f"\n📉 Large Discrepancies:")
    print(f"   Profit differences > 1%:      {large_profit_diff}/{len(comparison)}")
    print(f"   Success prob differences > 0.1: {large_success_diff}/{len(comparison)}")

    return comparison


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Validate backend ML predictions')
    parser.add_argument('json_path', help='Path to opportunities JSON file from backend')
    parser.add_argument('--model-dir', default='models/xgboost', help='Directory containing trained models')

    args = parser.parse_args()

    try:
        comparison = validate_predictions(args.json_path, args.model_dir)

        # Optionally save results
        output_path = Path(args.json_path).parent / 'prediction_validation.csv'
        comparison.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
