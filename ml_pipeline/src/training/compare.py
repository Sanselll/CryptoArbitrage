"""
Model Comparison Script

Train all 4 model types and compare their backtest performance.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
from typing import Dict, List

from ..data.loader import DataLoader
from ..data.preprocessor import FeaturePreprocessor
from ..models.xgboost_model import XGBoostProfitPredictor, XGBoostSuccessClassifier, XGBoostDurationPredictor
from ..models.lightgbm_model import LightGBMProfitPredictor, LightGBMSuccessClassifier, LightGBMDurationPredictor
from ..models.catboost_model import CatBoostProfitPredictor, CatBoostSuccessClassifier, CatBoostDurationPredictor
from ..models.random_forest_model import RandomForestProfitPredictor, RandomForestSuccessClassifier, RandomForestDurationPredictor
from ..models.base_model import ModelEnsemble
from ..scoring.opportunity_scorer import OpportunityScorer
from ..backtesting.backtester import Backtester
from ..backtesting.metrics import MetricsComparison


def load_config(config_path: Path) -> Dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_single_model(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict,
    verbose: bool = False
) -> tuple:
    """
    Train a single model ensemble.

    Returns:
        (ensemble, preprocessor)
    """
    print(f"\n{'─'*80}")
    print(f"Training {model_type.upper()}...")
    print(f"{'─'*80}")

    # Create preprocessor
    preprocessor = FeaturePreprocessor(config.get('features', {}))

    # Preprocess
    X_train = preprocessor.fit_transform(train_df.drop(columns=[
        'target_hold_hours', 'target_profit_pct', 'target_profit_usd', 'target_was_profitable',
        'peak_profit_pct', 'max_drawdown_pct', 'funding_payments_count', 'total_funding_usd'
    ], errors='ignore'))

    X_val = preprocessor.transform(val_df.drop(columns=[
        'target_hold_hours', 'target_profit_pct', 'target_profit_usd', 'target_was_profitable',
        'peak_profit_pct', 'max_drawdown_pct', 'funding_payments_count', 'total_funding_usd'
    ], errors='ignore'))

    # Extract targets
    y_profit_train = train_df['target_profit_pct']
    y_success_train = train_df['target_was_profitable'].astype(int)
    y_duration_train = train_df['target_hold_hours']

    y_profit_val = val_df['target_profit_pct']
    y_success_val = val_df['target_was_profitable'].astype(int)
    y_duration_val = val_df['target_hold_hours']

    # Create models
    model_config = config.get(model_type, {})

    if model_type == 'xgboost':
        profit_model = XGBoostProfitPredictor(model_config.get('profit_predictor'))
        success_model = XGBoostSuccessClassifier(model_config.get('success_classifier'))
        duration_model = XGBoostDurationPredictor(model_config.get('duration_predictor'))
    elif model_type == 'lightgbm':
        profit_model = LightGBMProfitPredictor(model_config.get('profit_predictor'))
        success_model = LightGBMSuccessClassifier(model_config.get('success_classifier'))
        duration_model = LightGBMDurationPredictor(model_config.get('duration_predictor'))
    elif model_type == 'catboost':
        profit_model = CatBoostProfitPredictor(model_config.get('profit_predictor'))
        success_model = CatBoostSuccessClassifier(model_config.get('success_classifier'))
        duration_model = CatBoostDurationPredictor(model_config.get('duration_predictor'))
    elif model_type == 'random_forest':
        profit_model = RandomForestProfitPredictor(model_config.get('profit_predictor'))
        success_model = RandomForestSuccessClassifier(model_config.get('success_classifier'))
        duration_model = RandomForestDurationPredictor(model_config.get('duration_predictor'))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create and train ensemble
    ensemble = ModelEnsemble(profit_model, success_model, duration_model)
    ensemble.train_all(
        X_train, y_profit_train, y_success_train, y_duration_train,
        X_val, y_profit_val, y_success_val, y_duration_val,
        verbose=verbose
    )

    print(f"✅ {model_type.upper()} training complete")

    return ensemble, preprocessor


def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare all ML models via backtesting")
    parser.add_argument('--data-path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to training config')
    parser.add_argument('--scoring-config', type=str, default='config/scoring_config.yaml',
                        help='Path to scoring config')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--train-size', type=float, default=0.6,
                        help='Training set size (rest is test)')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Validation set size (of training data)')
    parser.add_argument('--initial-capital', type=float, default=10000,
                        help='Initial capital for backtesting')
    parser.add_argument('--selection-interval', type=int, default=24,
                        help='Selection interval in hours')

    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))
    scoring_config_path = Path(args.scoring_config) if Path(args.scoring_config).exists() else None

    # Load data
    print("\n" + "="*80)
    print("MODEL COMPARISON - BACKTESTING ALL MODELS")
    print("="*80 + "\n")

    print("Loading data...")
    loader = DataLoader()
    df = loader.load_csv(Path(args.data_path))

    # Split data: train/val/test
    train_val_df, test_df = loader.split_train_test(
        test_size=1 - args.train_size,
        stratify_column='target_was_profitable'
    )

    train_df, val_df = loader.split_train_test(
        test_size=args.val_size,
        stratify_column='target_was_profitable'
    )

    print(f"\nTrain: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    # Train all models
    model_types = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
    models = {}
    preprocessors = {}

    print(f"\n{'='*80}")
    print("TRAINING ALL MODELS")
    print(f"{'='*80}")

    for model_type in model_types:
        ensemble, preprocessor = train_single_model(
            model_type,
            train_df,
            val_df,
            config,
            verbose=False
        )
        models[model_type] = ensemble
        preprocessors[model_type] = preprocessor

    # Backtest all models
    print(f"\n{'='*80}")
    print("BACKTESTING ALL MODELS")
    print(f"{'='*80}")

    backtest_results = {}

    for model_type in model_types:
        ensemble = models[model_type]
        preprocessor = preprocessors[model_type]

        # Create scorer
        scorer = OpportunityScorer(ensemble, scoring_config_path)

        # Prepare test data with features
        test_features = preprocessor.transform(test_df.drop(columns=[
            'target_hold_hours', 'target_profit_pct', 'target_profit_usd', 'target_was_profitable',
            'peak_profit_pct', 'max_drawdown_pct', 'funding_payments_count', 'total_funding_usd'
        ], errors='ignore'))

        # Merge features back with original data for backtesting
        test_with_features = test_df.copy()
        for col in test_features.columns:
            test_with_features[col] = test_features[col].values

        # Run backtest
        backtester = Backtester(scorer, initial_capital=args.initial_capital)
        result = backtester.run_backtest(
            test_with_features,
            selection_interval_hours=args.selection_interval,
            model_name=model_type.upper()
        )

        backtest_results[model_type] = result.metrics

    # Compare results
    print(f"\n{'='*80}")
    print("MODEL COMPARISON RESULTS")
    print(f"{'='*80}\n")

    comparison_df = MetricsComparison.compare_models(backtest_results)
    MetricsComparison.print_comparison(comparison_df)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df.to_csv(output_dir / 'model_comparison.csv')
    print(f"\n✅ Results saved to {output_dir / 'model_comparison.csv'}")

    # Save detailed results for each model
    for model_type, metrics in backtest_results.items():
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(output_dir / f'{model_type}_metrics.csv', index=False)

    print(f"\n{'='*80}")
    print("✅ COMPARISON COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
