"""
Base Model Abstract Class

Defines the interface that all ML models must implement.
This allows easy swapping between XGBoost, LightGBM, CatBoost, and Random Forest.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


class BaseMLModel(ABC):
    """
    Abstract base class for all ML models.

    Subclasses must implement:
    - train(): Training logic
    - predict(): Inference logic
    - predict_proba(): Probability predictions (for classifiers)
    - export_onnx(): Export to ONNX format
    - get_feature_importance(): Feature importance rankings
    """

    def __init__(self, model_type: str, config: Dict[str, Any]):
        """
        Initialize model.

        Args:
            model_type: Type of model ('profit', 'success', 'duration')
            config: Model-specific configuration (hyperparameters)
        """
        self.model_type = model_type
        self.config = config
        self.model = None
        self.feature_names = None
        self.is_trained = False

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'BaseMLModel':
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            verbose: Whether to print training progress

        Returns:
            self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Predictions array
        """
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classifiers only).

        Args:
            X: Features

        Returns:
            Probability array (shape: [n_samples, n_classes])
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.

        Returns:
            DataFrame with columns: ['feature', 'importance']
            Sorted by importance (descending)
        """
        pass

    @abstractmethod
    def export_onnx(self, output_path: Path) -> None:
        """
        Export model to ONNX format for C# inference.

        Args:
            output_path: Where to save the ONNX model
        """
        pass

    @abstractmethod
    def save(self, output_path: Path) -> None:
        """
        Save model to disk (native format).

        Args:
            output_path: Where to save the model
        """
        pass

    @abstractmethod
    def load(self, model_path: Path) -> 'BaseMLModel':
        """
        Load model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            self (for method chaining)
        """
        pass

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        predictions = self.predict(X_test)

        if self.model_type == 'success':
            # Classification metrics
            return self._evaluate_classifier(y_test, predictions, X_test)
        else:
            # Regression metrics
            return self._evaluate_regressor(y_test, predictions)

    def _evaluate_regressor(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate regression metrics.

        Returns:
            Dictionary with MAE, RMSE, R², MAPE
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }

    def _evaluate_classifier(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        X: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Returns:
            Dictionary with accuracy, precision, recall, F1, AUC
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score
        )

        y_pred_proba = self.predict_proba(X)[:, 1]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred_proba)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

    def __repr__(self) -> str:
        """String representation of model."""
        return f"{self.__class__.__name__}(type={self.model_type}, trained={self.is_trained})"


class ModelEnsemble:
    """
    Manages three related models: profit, success, duration.

    This makes it easy to work with all three models together:
    - Train all three at once
    - Make predictions from all three
    - Export all three to ONNX
    """

    def __init__(
        self,
        profit_model: BaseMLModel,
        success_model: BaseMLModel,
        duration_model: BaseMLModel
    ):
        """
        Initialize ensemble.

        Args:
            profit_model: Profit prediction model (regression)
            success_model: Success classification model
            duration_model: Duration prediction model (regression)
        """
        self.profit_model = profit_model
        self.success_model = success_model
        self.duration_model = duration_model

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_profit_train: pd.Series,
        y_success_train: pd.Series,
        y_duration_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_profit_val: Optional[pd.Series] = None,
        y_success_val: Optional[pd.Series] = None,
        y_duration_val: Optional[pd.Series] = None,
        verbose: bool = True
    ) -> 'ModelEnsemble':
        """
        Train all three models.

        Returns:
            self (for method chaining)
        """
        if verbose:
            print("Training Profit Predictor...")
        self.profit_model.train(X_train, y_profit_train, X_val, y_profit_val, verbose)

        if verbose:
            print("\nTraining Success Classifier...")
        self.success_model.train(X_train, y_success_train, X_val, y_success_val, verbose)

        if verbose:
            print("\nTraining Duration Predictor...")
        self.duration_model.train(X_train, y_duration_train, X_val, y_duration_val, verbose)

        return self

    def predict_all(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions from all three models.

        Returns:
            (profit_predictions, success_probabilities, duration_predictions)
        """
        profit_pred = self.profit_model.predict(X)
        success_prob = self.success_model.predict_proba(X)[:, 1]  # Probability of success
        duration_pred = self.duration_model.predict(X)

        return profit_pred, success_prob, duration_pred

    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_profit_test: pd.Series,
        y_success_test: pd.Series,
        y_duration_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all three models.

        Returns:
            Dictionary with metrics for each model
        """
        return {
            'profit': self.profit_model.evaluate(X_test, y_profit_test),
            'success': self.success_model.evaluate(X_test, y_success_test),
            'duration': self.duration_model.evaluate(X_test, y_duration_test)
        }

    def export_all_onnx(self, output_dir: Path) -> None:
        """
        Export all three models to ONNX.

        Args:
            output_dir: Directory to save ONNX models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.profit_model.export_onnx(output_dir / 'profit_model.onnx')
        self.success_model.export_onnx(output_dir / 'success_model.onnx')
        self.duration_model.export_onnx(output_dir / 'duration_model.onnx')

        print(f"✅ All models exported to {output_dir}")
