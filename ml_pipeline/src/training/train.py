"""
Main Training Script

Train ML models on historical arbitrage data.
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
from typing import Dict, Optional

from ..data.loader import DataLoader
from ..data.preprocessor import FeaturePreprocessor
from ..models.xgboost_model import XGBoostProfitPredictor, XGBoostSuccessClassifier, XGBoostDurationPredictor
from ..models.lightgbm_model import LightGBMProfitPredictor, LightGBMSuccessClassifier, LightGBMDurationPredictor
from ..models.catboost_model import CatBoostProfitPredictor, CatBoostSuccessClassifier, CatBoostDurationPredictor
from ..models.random_forest_model import RandomForestProfitPredictor, RandomForestSuccessClassifier, RandomForestDurationPredictor
from ..models.base_model import ModelEnsemble


def load_config(config_path: Path) -> Dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model_ensemble(
    model_type: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
    config: Dict,
    verbose: bool = True
) -> ModelEnsemble:
    """
    Train a complete model ensemble (profit, success, duration).

    Args:
        model_type: 'xgboost', 'lightgbm', 'catboost', or 'random_forest'
        train_df: Training data
        val_df: Validation data
        preprocessor: Feature preprocessor
        config: Training configuration
        verbose: Print progress

    Returns:
        Trained ModelEnsemble
    """
    print(f"\n{'='*80}")
    print(f"TRAINING {model_type.upper()} MODELS")
    print(f"{'='*80}\n")

    # Preprocess features
    print("Preprocessing features...")
    X_train = preprocessor.fit_transform(train_df.drop(columns=[
        'actual_hold_hours', 'actual_profit_pct', 'was_profitable',
        'exit_time', 'peak_profit_pct', 'max_drawdown_pct'
    ], errors='ignore'))

    X_val = preprocessor.transform(val_df.drop(columns=[
        'actual_hold_hours', 'actual_profit_pct', 'was_profitable',
        'exit_time', 'peak_profit_pct', 'max_drawdown_pct'
    ], errors='ignore'))

    # Extract targets
    # Clip profit targets to reduce noise from extreme outliers (-100% liquidations, +200% anomalies)
    y_profit_train = train_df['actual_profit_pct'].clip(-50, 50)
    y_success_train = train_df['was_profitable'].astype(int)
    y_duration_train = train_df['actual_hold_hours']

    y_profit_val = val_df['actual_profit_pct'].clip(-50, 50)
    y_success_val = val_df['was_profitable'].astype(int)
    y_duration_val = val_df['actual_hold_hours']

    # Create models based on type
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

    # Create ensemble
    ensemble = ModelEnsemble(profit_model, success_model, duration_model)

    # Train all models
    ensemble.train_all(
        X_train, y_profit_train, y_success_train, y_duration_train,
        X_val, y_profit_val, y_success_val, y_duration_val,
        verbose=verbose
    )

    print(f"\n✅ {model_type.upper()} training complete!")

    return ensemble, preprocessor


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ML models for opportunity scoring")
    parser.add_argument('--data-path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--model-type', type=str, default='xgboost',
                        choices=['xgboost', 'lightgbm', 'catboost', 'random_forest', 'all'],
                        help='Model type to train')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                        help='Path to training config')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--export-onnx', action='store_true',
                        help='Export models to ONNX format')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set size (0.0-1.0)')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='Validation set size (0.0-1.0)')

    args = parser.parse_args()

    # Load configuration
    config = load_config(Path(args.config))

    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80 + "\n")

    loader = DataLoader()
    df = loader.load_csv(Path(args.data_path))
    loader.print_summary()

    # Split data
    print("\nSplitting data...")
    train_val_df, test_df = loader.split_train_test(
        test_size=args.test_size,
        stratify_column='was_profitable'
    )

    train_df, val_df = loader.split_train_test(
        test_size=args.val_size,
        stratify_column='was_profitable'
    )

    # Determine which models to train
    if args.model_type == 'all':
        model_types = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
    else:
        model_types = [args.model_type]

    # Train models
    for model_type in model_types:
        # Create preprocessor for this model
        preprocessor = FeaturePreprocessor(config.get('features', {}))

        # Train ensemble
        ensemble, preprocessor = train_model_ensemble(
            model_type,
            train_df,
            val_df,
            preprocessor,
            config,
            verbose=True
        )

        # Evaluate on test set
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_type.upper()} ON TEST SET")
        print(f"{'='*80}\n")

        X_test = preprocessor.transform(test_df.drop(columns=[
            'actual_hold_hours', 'actual_profit_pct', 'was_profitable',
            'exit_time', 'peak_profit_pct', 'max_drawdown_pct'
        ], errors='ignore'))

        # Clip test targets too for consistent evaluation
        y_profit_test = test_df['actual_profit_pct'].clip(-50, 50)
        y_success_test = test_df['was_profitable'].astype(int)
        y_duration_test = test_df['actual_hold_hours']

        metrics = ensemble.evaluate_all(X_test, y_profit_test, y_success_test, y_duration_test)

        print("\nProfit Prediction Metrics:")
        for key, value in metrics['profit'].items():
            print(f"  {key}: {value:.4f}")

        print("\nSuccess Classification Metrics:")
        for key, value in metrics['success'].items():
            print(f"  {key}: {value:.4f}")

        print("\nDuration Prediction Metrics:")
        for key, value in metrics['duration'].items():
            print(f"  {key}: {value:.4f}")

        # Save models
        output_dir = Path(args.output_dir) / model_type
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nSaving models to {output_dir}...")
        ensemble.profit_model.save(output_dir / 'profit_model.pkl')
        ensemble.success_model.save(output_dir / 'success_model.pkl')
        ensemble.duration_model.save(output_dir / 'duration_model.pkl')

        # Save preprocessor
        preprocessor.save(output_dir / 'preprocessor.pkl')

        # Export to ONNX if requested
        if args.export_onnx:
            print(f"\nExporting to XGBoost JSON format...")
            try:
                ensemble.export_all_onnx(output_dir)
            except Exception as e:
                print(f"⚠️  Export to JSON skipped: {e}")
                print(f"   Models are saved as pickle (.pkl) files")

        # Save feature importance
        print(f"\nSaving feature importance...")
        profit_importance = ensemble.profit_model.get_feature_importance()
        profit_importance.to_csv(output_dir / 'profit_feature_importance.csv', index=False)

        success_importance = ensemble.success_model.get_feature_importance()
        success_importance.to_csv(output_dir / 'success_feature_importance.csv', index=False)

        duration_importance = ensemble.duration_model.get_feature_importance()
        duration_importance.to_csv(output_dir / 'duration_feature_importance.csv', index=False)

    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
