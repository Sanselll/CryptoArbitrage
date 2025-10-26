"""
Test Trained Models

Load trained models and test predictions on sample opportunities.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from data.preprocessor import FeaturePreprocessor

def load_models(model_dir='models/xgboost'):
    """Load trained models and preprocessor."""
    model_path = Path(model_dir)

    print("Loading models...")
    # Models are saved as dicts with 'model' key
    profit_dict = joblib.load(model_path / 'profit_model.pkl')
    success_dict = joblib.load(model_path / 'success_model.pkl')
    duration_dict = joblib.load(model_path / 'duration_model.pkl')

    profit_model = profit_dict['model']
    success_model = success_dict['model']
    duration_model = duration_dict['model']

    # Load preprocessor properly
    preprocessor = FeaturePreprocessor()
    preprocessor.load(model_path / 'preprocessor.pkl')

    print("✅ Models loaded successfully")
    return profit_model, success_model, duration_model, preprocessor


def test_on_sample_data(data_path='../src/CryptoArbitrage.HistoricalCollector/data/training_data.csv', n_samples=20):
    """Test models on sample opportunities."""

    # Load models
    profit_model, success_model, duration_model, preprocessor = load_models()

    # Load test data
    print(f"\nLoading test data from {data_path}...")
    df = pd.read_csv(data_path)

    # Take random sample
    sample_df = df.sample(n=n_samples, random_state=42)

    print(f"Testing on {n_samples} random opportunities...\n")

    # Save actual values
    actual_profit = sample_df['actual_profit_pct'].values
    actual_success = sample_df['was_profitable'].values
    actual_duration = sample_df['actual_hold_hours'].values

    # Get entry info for display
    entry_times = sample_df['entry_time'].values
    symbols = sample_df['symbol'].values
    strategies = sample_df['strategy_name'].values if 'strategy_name' in sample_df.columns else ['N/A'] * n_samples

    # Preprocess features
    X = preprocessor.transform(sample_df.drop(columns=[
        'actual_hold_hours', 'actual_profit_pct', 'was_profitable',
        'exit_time', 'peak_profit_pct', 'max_drawdown_pct',
        'strategy_name', 'exit_reason', 'hit_profit_target', 'hit_stop_loss'
    ], errors='ignore'))

    # Make predictions
    pred_profit = profit_model.predict(X)
    pred_success_proba = success_model.predict_proba(X)[:, 1]  # Probability of profitable
    pred_success = success_model.predict(X)
    pred_duration = duration_model.predict(X)

    # Create results dataframe
    results = pd.DataFrame({
        'entry_time': entry_times,
        'symbol': symbols,
        'strategy': strategies,
        'success_prob': pred_success_proba,
        'pred_success': pred_success,
        'actual_success': actual_success,
        'pred_profit_pct': pred_profit,
        'actual_profit_pct': actual_profit,
        'pred_duration_hrs': pred_duration,
        'actual_duration_hrs': actual_duration,
    })

    # Add error columns
    results['profit_error'] = np.abs(results['pred_profit_pct'] - results['actual_profit_pct'])
    results['duration_error'] = np.abs(results['pred_duration_hrs'] - results['actual_duration_hrs'])
    results['success_correct'] = results['pred_success'] == results['actual_success']

    # Display results
    print("="*120)
    print("PREDICTION RESULTS")
    print("="*120)

    for idx, row in results.iterrows():
        print(f"\n📊 Opportunity {idx + 1}:")
        print(f"   Symbol: {row['symbol']:12s} | Strategy: {row['strategy']:20s} | Entry: {row['entry_time']}")
        print(f"   Success Probability: {row['success_prob']:.1%}")
        print(f"   Success Prediction:  {str(bool(row['pred_success'])):5s} | Actual: {str(bool(row['actual_success'])):5s} | {'✅ Correct' if row['success_correct'] else '❌ Wrong'}")
        print(f"   Profit Prediction:   {row['pred_profit_pct']:6.2f}% | Actual: {row['actual_profit_pct']:6.2f}% | Error: {row['profit_error']:.2f}%")
        print(f"   Duration Prediction: {row['pred_duration_hrs']:5.1f}h | Actual: {row['actual_duration_hrs']:5.1f}h | Error: {row['duration_error']:.1f}h")

    # Summary statistics
    print("\n" + "="*120)
    print("SUMMARY STATISTICS")
    print("="*120)

    print(f"\n📈 Success Classification:")
    print(f"   Accuracy: {results['success_correct'].mean():.1%}")
    print(f"   Average Success Probability: {results['success_prob'].mean():.1%}")

    print(f"\n💰 Profit Prediction:")
    print(f"   Mean Absolute Error: {results['profit_error'].mean():.2f}%")
    print(f"   RMSE: {np.sqrt((results['profit_error']**2).mean()):.2f}%")
    print(f"   Average Predicted Profit: {results['pred_profit_pct'].mean():.2f}%")
    print(f"   Average Actual Profit: {results['actual_profit_pct'].mean():.2f}%")

    print(f"\n⏱️  Duration Prediction:")
    print(f"   Mean Absolute Error: {results['duration_error'].mean():.1f} hours")
    print(f"   RMSE: {np.sqrt((results['duration_error']**2).mean()):.1f} hours")
    print(f"   Average Predicted Duration: {results['pred_duration_hrs'].mean():.1f} hours")
    print(f"   Average Actual Duration: {results['actual_duration_hrs'].mean():.1f} hours")

    # Show best opportunities by different criteria
    print("\n" + "="*120)
    print("TOP 5 OPPORTUNITIES BY DIFFERENT CRITERIA")
    print("="*120)

    print("\n🏆 Top 5 by Success Probability:")
    top_success = results.nlargest(5, 'success_prob')[['symbol', 'success_prob', 'pred_profit_pct', 'actual_profit_pct']]
    for i, row in enumerate(top_success.itertuples(), 1):
        print(f"   {i}. {row.symbol:12s} - Success: {row.success_prob:.1%}, Pred Profit: {row.pred_profit_pct:+6.2f}%, Actual: {row.actual_profit_pct:+6.2f}%")

    print("\n💎 Top 5 by Predicted Profit:")
    top_profit = results.nlargest(5, 'pred_profit_pct')[['symbol', 'success_prob', 'pred_profit_pct', 'actual_profit_pct']]
    for i, row in enumerate(top_profit.itertuples(), 1):
        print(f"   {i}. {row.symbol:12s} - Pred Profit: {row.pred_profit_pct:+6.2f}%, Success: {row.success_prob:.1%}, Actual: {row.actual_profit_pct:+6.2f}%")

    # Calculate composite score
    results['composite_score'] = results['success_prob'] * results['pred_profit_pct']

    print("\n🎯 Top 5 by Composite Score (success_prob × pred_profit):")
    top_composite = results.nlargest(5, 'composite_score')[['symbol', 'success_prob', 'pred_profit_pct', 'composite_score', 'actual_profit_pct']]
    for i, row in enumerate(top_composite.itertuples(), 1):
        print(f"   {i}. {row.symbol:12s} - Score: {row.composite_score:6.2f}, Success: {row.success_prob:.1%}, Pred: {row.pred_profit_pct:+6.2f}%, Actual: {row.actual_profit_pct:+6.2f}%")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test trained ML models')
    parser.add_argument('--data-path', type=str,
                        default='../src/CryptoArbitrage.HistoricalCollector/data/training_data.csv',
                        help='Path to test data CSV')
    parser.add_argument('--n-samples', type=int, default=20,
                        help='Number of samples to test')
    parser.add_argument('--model-dir', type=str, default='models/xgboost',
                        help='Directory containing trained models')

    args = parser.parse_args()

    results = test_on_sample_data(args.data_path, args.n_samples)

    print("\n✅ Testing complete!")
