"""
CatBoost Model Implementation

Concrete implementation of BaseMLModel using CatBoost.
"""

import catboost as cb
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from .base_model import BaseMLModel


class CatBoostModel(BaseMLModel):
    """
    CatBoost implementation of BaseMLModel.

    CatBoost handles categorical features well and has good default parameters.
    """

    def __init__(self, model_type: str, config: Dict[str, Any]):
        """
        Initialize CatBoost model.

        Args:
            model_type: 'profit', 'success', or 'duration'
            config: CatBoost hyperparameters
        """
        super().__init__(model_type, config)

        # Create model based on type
        if model_type == 'success':
            self.model = cb.CatBoostClassifier(**config)
        else:
            self.model = cb.CatBoostRegressor(**config)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'CatBoostModel':
        """
        Train the CatBoost model.

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
            eval_set = cb.Pool(X_val, y_val)

        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=50 if verbose else False
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

        importance = self.model.get_feature_importance()

        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df

    def export_onnx(self, output_path: Path) -> None:
        """
        Export to ONNX format.

        Note: CatBoost has native ONNX export support.

        Args:
            output_path: Path to save ONNX model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # CatBoost native ONNX export
            self.model.save_model(
                str(output_path),
                format='onnx',
                export_parameters={
                    'onnx_domain': 'ai.catboost',
                    'onnx_model_version': 1,
                    'onnx_doc_string': f'CatBoost {self.model_type} model',
                    'onnx_graph_name': f'CatBoost_{self.model_type}'
                }
            )

            print(f"✅ Model exported to ONNX: {output_path}")

        except Exception as e:
            print(f"❌ ONNX export failed: {e}")
            print("Ensure CatBoost version supports ONNX export")

    def save(self, output_path: Path) -> None:
        """
        Save model to disk.

        Args:
            output_path: Path to save model
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CatBoost native save
        self.model.save_model(str(output_path))

        # Save metadata separately
        metadata_path = output_path.with_suffix('.meta')
        joblib.dump({
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'config': self.config
        }, metadata_path)

        print(f"✅ Model saved: {output_path}")

    def load(self, model_path: Path) -> 'CatBoostModel':
        """
        Load model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            self
        """
        # Load CatBoost model
        if self.model_type == 'success':
            self.model = cb.CatBoostClassifier()
        else:
            self.model = cb.CatBoostRegressor()

        self.model.load_model(str(model_path))

        # Load metadata
        metadata_path = Path(model_path).with_suffix('.meta')
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata['feature_names']
            self.model_type = metadata['model_type']
            self.config = metadata['config']

        self.is_trained = True

        print(f"✅ Model loaded: {model_path}")

        return self


# Convenience classes for specific use cases

class CatBoostProfitPredictor(CatBoostModel):
    """CatBoost model for profit prediction (regression)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'iterations': 500,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 1.0,
                'loss_function': 'RMSE',
                'verbose': False,
                'random_state': 42,
                'task_type': 'CPU'  # Use 'GPU' if CUDA available
            }

        super().__init__('profit', config)


class CatBoostSuccessClassifier(CatBoostModel):
    """CatBoost model for success classification (binary)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'iterations': 300,
                'depth': 5,
                'learning_rate': 0.05,
                'l2_leaf_reg': 1.0,
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'verbose': False,
                'random_state': 42,
                'task_type': 'CPU'
            }

        super().__init__('success', config)


class CatBoostDurationPredictor(CatBoostModel):
    """CatBoost model for hold duration prediction (regression)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'iterations': 400,
                'depth': 6,
                'learning_rate': 0.05,
                'l2_leaf_reg': 1.0,
                'loss_function': 'RMSE',
                'verbose': False,
                'random_state': 42,
                'task_type': 'CPU'
            }

        super().__init__('duration', config)
