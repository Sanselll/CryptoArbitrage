"""
Hyperparameter Tuning with Optuna

Uses Optuna to find optimal hyperparameters for XGBoost models.
"""

import argparse
import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score
import xgboost as xgb

from ..data.loader import DataLoader
from ..data.preprocessor import FeaturePreprocessor


def objective_profit(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for profit predictor (minimize RMSE, maximize R²).

    Args:
        trial: Optuna trial
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Validation RMSE (lower is better)
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000, step=500),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    # Train model with early stopping
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return rmse


def objective_success(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for success classifier (maximize AUC).

    Args:
        trial: Optuna trial
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Negative validation AUC (for minimization)
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 3000, step=500),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 3.0, 10.0),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    # Train model with early stopping
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict and evaluate
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)

    # Return negative AUC (Optuna minimizes)
    return -auc


def objective_duration(trial, X_train, y_train, X_val, y_val):
    """
    Objective function for duration predictor (minimize RMSE).

    Args:
        trial: Optuna trial
        X_train, y_train: Training data
        X_val, y_val: Validation data

    Returns:
        Validation RMSE (lower is better)
    """
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 5000, step=500),
        'max_depth': trial.suggest_int('max_depth', 6, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    # Train model with early stopping
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Predict and evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))

    return rmse


def main():
    """Main tuning function."""
    parser = argparse.ArgumentParser(description="Tune hyperparameters with Optuna")
    parser.add_argument('--data-path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--model-type', type=str, default='success',
                        choices=['profit', 'success', 'duration'],
                        help='Which model to tune')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--output-config', type=str, default='config/tuned_config.yaml',
                        help='Where to save tuned config')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--val-size', type=float, default=0.2)

    args = parser.parse_args()

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

    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Preprocess
    print("\nPreprocessing features...")
    preprocessor = FeaturePreprocessor()

    X_train = preprocessor.fit_transform(train_df.drop(columns=[
        'actual_hold_hours', 'actual_profit_pct', 'was_profitable',
        'exit_time', 'peak_profit_pct', 'max_drawdown_pct'
    ], errors='ignore'))

    X_val = preprocessor.transform(val_df.drop(columns=[
        'actual_hold_hours', 'actual_profit_pct', 'was_profitable',
        'exit_time', 'peak_profit_pct', 'max_drawdown_pct'
    ], errors='ignore'))

    # Prepare targets
    if args.model_type == 'profit':
        y_train = train_df['actual_profit_pct'].clip(-50, 50)
        y_val = val_df['actual_profit_pct'].clip(-50, 50)
        objective_func = lambda trial: objective_profit(trial, X_train, y_train, X_val, y_val)
        direction = 'minimize'  # Minimize RMSE
        metric_name = 'RMSE'
    elif args.model_type == 'success':
        y_train = train_df['was_profitable'].astype(int)
        y_val = val_df['was_profitable'].astype(int)
        objective_func = lambda trial: objective_success(trial, X_train, y_train, X_val, y_val)
        direction = 'minimize'  # Minimize negative AUC
        metric_name = 'AUC'
    else:  # duration
        y_train = train_df['actual_hold_hours']
        y_val = val_df['actual_hold_hours']
        objective_func = lambda trial: objective_duration(trial, X_train, y_train, X_val, y_val)
        direction = 'minimize'  # Minimize RMSE
        metric_name = 'RMSE'

    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")

    # Run Optuna optimization
    print("\n" + "="*80)
    print(f"TUNING {args.model_type.upper()} MODEL - {args.n_trials} TRIALS")
    print("="*80 + "\n")

    # Create study
    study = optuna.create_study(
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=42)
    )

    # Optimize
    study.optimize(
        objective_func,
        n_trials=args.n_trials,
        show_progress_bar=True,
        n_jobs=1  # Run sequentially to avoid memory issues
    )

    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80 + "\n")

    print(f"Best {metric_name}: {abs(study.best_value):.4f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save to config file
    output_path = Path(args.output_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing config or create new
    if output_path.exists():
        with open(output_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # Update with tuned parameters
    if 'xgboost' not in config:
        config['xgboost'] = {}

    # Determine model key and parameters
    if args.model_type == 'profit':
        model_key = 'profit_predictor'
        objective = 'reg:squarederror'
        eval_metric = 'rmse'
    elif args.model_type == 'success':
        model_key = 'success_classifier'
        objective = 'binary:logistic'
        eval_metric = 'auc'
    else:  # duration
        model_key = 'duration_predictor'
        objective = 'reg:squarederror'
        eval_metric = 'rmse'

    config['xgboost'][model_key] = study.best_params
    config['xgboost'][model_key]['objective'] = objective
    config['xgboost'][model_key]['eval_metric'] = eval_metric
    config['xgboost'][model_key]['tree_method'] = 'hist'
    config['xgboost'][model_key]['early_stopping_rounds'] = 150

    # Save config
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✅ Tuned config saved to: {output_path}")
    print("\nTo use these parameters, run training with:")
    print(f"  --config {output_path}")

    # Show parameter importance
    print("\n" + "="*80)
    print("PARAMETER IMPORTANCE")
    print("="*80 + "\n")

    importance = optuna.importance.get_param_importances(study)
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {param:20s}: {imp:.4f}")


if __name__ == '__main__':
    main()
