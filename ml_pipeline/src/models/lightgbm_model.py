"""
LightGBM Model Implementation

Concrete implementation of BaseMLModel using LightGBM.
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from .base_model import BaseMLModel


class LightGBMModel(BaseMLModel):
    """
    LightGBM implementation of BaseMLModel.

    LightGBM is faster than XGBoost with similar accuracy.
    Good for large datasets.
    """

    def __init__(self, model_type: str, config: Dict[str, Any]):
        """
        Initialize LightGBM model.

        Args:
            model_type: 'profit', 'success', or 'duration'
            config: LightGBM hyperparameters
        """
        super().__init__(model_type, config)

        # Create model based on type
        if model_type == 'success':
            self.model = lgb.LGBMClassifier(**config)
        else:
            self.model = lgb.LGBMRegressor(**config)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'LightGBMModel':
        """
        Train the LightGBM model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Print training progress

        Returns:
            self
        """
        self.feature_names = X_train.columns.tolist()

        # Prepare eval set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=[lgb.log_evaluation(period=50)] if verbose else None
        )

        self.is_trained = True

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classifiers).

        Args:
            X: Features

        Returns:
            Probabilities [n_samples, n_classes]
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        if self.model_type != 'success':
            raise ValueError("predict_proba only available for classifiers")

        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = self.model.feature_importances_

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def export_onnx(self, output_path: Path) -> None:
        """
        Export to ONNX format.

        Args:
            output_path: Path to save ONNX model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            import onnx

            # Define input type
            initial_type = [('float_input', FloatTensorType([None, len(self.feature_names)]))]

            # Convert to ONNX
            onnx_model = convert_sklearn(
                self.model,
                initial_types=initial_type,
                target_opset=12
            )

            # Save
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            onnx.save_model(onnx_model, str(output_path))

            # Validate
            onnx_model_check = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model_check)

            print(f"✅ Model exported to ONNX: {output_path}")

        except ImportError as e:
            print(f"❌ ONNX export failed: {e}")
            print("Install with: pip install onnx skl2onnx")

    def save(self, output_path: Path) -> None:
        """
        Save model to disk.

        Args:
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'config': self.config
        }, output_path)

        print(f"✅ Model saved: {output_path}")

    def load(self, model_path: Path) -> 'LightGBMModel':
        """
        Load model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            self
        """
        data = joblib.load(model_path)

        self.model = data['model']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']
        self.config = data['config']
        self.is_trained = True

        print(f"✅ Model loaded: {model_path}")

        return self


# Convenience classes for specific use cases

class LightGBMProfitPredictor(LightGBMModel):
    """LightGBM model for profit prediction (regression)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'n_estimators': 500,
                'max_depth': 6,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': 42
            }

        super().__init__('profit', config)


class LightGBMSuccessClassifier(LightGBMModel):
    """LightGBM model for success classification (binary)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'n_estimators': 300,
                'max_depth': 5,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary',
                'metric': 'auc',
                'verbosity': -1,
                'random_state': 42
            }

        super().__init__('success', config)


class LightGBMDurationPredictor(LightGBMModel):
    """LightGBM model for hold duration prediction (regression)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'n_estimators': 400,
                'max_depth': 6,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': 42
            }

        super().__init__('duration', config)
