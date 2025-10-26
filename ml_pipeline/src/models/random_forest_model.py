"""
Random Forest Model Implementation

Concrete implementation of BaseMLModel using Random Forest (sklearn).
"""

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import joblib

from .base_model import BaseMLModel


class RandomForestModel(BaseMLModel):
    """
    Random Forest implementation of BaseMLModel.

    Simple, interpretable baseline model.
    Good for understanding feature importance.
    """

    def __init__(self, model_type: str, config: Dict[str, Any]):
        """
        Initialize Random Forest model.

        Args:
            model_type: 'profit', 'success', or 'duration'
            config: Random Forest hyperparameters
        """
        super().__init__(model_type, config)

        # Create model based on type
        if model_type == 'success':
            self.model = RandomForestClassifier(**config)
        else:
            self.model = RandomForestRegressor(**config)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'RandomForestModel':
        """
        Train the Random Forest model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used for RF)
            y_val: Validation targets (not used for RF)
            verbose: Print training progress

        Returns:
            self
        """
        self.feature_names = X_train.columns.tolist()

        if verbose:
            print(f"Training Random Forest ({self.model_type})...")

        # Train
        self.model.fit(X_train, y_train)

        if verbose:
            print(f"✅ Training complete")

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

    def load(self, model_path: Path) -> 'RandomForestModel':
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

class RandomForestProfitPredictor(RandomForestModel):
    """Random Forest model for profit prediction (regression)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42,
                'verbose': 0
            }

        super().__init__('profit', config)


class RandomForestSuccessClassifier(RandomForestModel):
    """Random Forest model for success classification (binary)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'n_jobs': -1,
                'random_state': 42,
                'verbose': 0
            }

        super().__init__('success', config)


class RandomForestDurationPredictor(RandomForestModel):
    """Random Forest model for hold duration prediction (regression)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42,
                'verbose': 0
            }

        super().__init__('duration', config)
