"""
Model training service for Customer Churn Prediction System.
Handles XGBoost model training, evaluation, and persistence with versioning.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import joblib
import json
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import config


class ModelTrainer:
    """
    Orchestrates the training process for XGBoost churn prediction models.
    Handles training, evaluation, and model persistence with versioning.
    """
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize ModelTrainer with hyperparameters.
        
        Args:
            params: XGBoost hyperparameters. If None, uses config.XGBOOST_PARAMS
        """
        self.params = params if params is not None else config.XGBOOST_PARAMS.copy()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ) -> XGBClassifier:
        """
        Train an XGBoost classifier on the provided data.
        
        Args:
            X_train: Training features array
            y_train: Training target array
            params: Optional hyperparameters to override instance params
            
        Returns:
            Trained XGBClassifier model
            
        Raises:
            ValueError: If training data is invalid
        """
        # Validate inputs
        if X_train.shape[0] == 0:
            raise ValueError("Training data is empty")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train have different lengths: "
                f"{X_train.shape[0]} vs {y_train.shape[0]}"
            )
        
        # Use provided params or instance params
        training_params = params if params is not None else self.params
        
        # Create and train model
        model = XGBClassifier(**training_params)
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate(
        self,
        model: XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Evaluate model performance and compute metrics.
        
        Args:
            model: Trained XGBClassifier model
            X_test: Test features array
            y_test: Test target array
            
        Returns:
            Dictionary containing precision, recall, f1_score, and confusion_matrix
            
        Raises:
            ValueError: If test data is invalid
        """
        # Validate inputs
        if X_test.shape[0] == 0:
            raise ValueError("Test data is empty")
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(
                f"X_test and y_test have different lengths: "
                f"{X_test.shape[0]} vs {y_test.shape[0]}"
            )
        
        # Generate predictions
        y_pred = model.predict(X_test)
        
        # Compute metrics
        precision = float(precision_score(y_test, y_pred, zero_division=0))
        recall = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix.tolist()  # Convert to list for JSON serialization
        }
        
        return metrics

    
    def save_model(
        self,
        model: XGBClassifier,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_dir: Optional[Path] = None
    ) -> str:
        """
        Save trained model with versioning (timestamp-based).
        
        Args:
            model: Trained XGBClassifier model to save
            version: Optional version string. If None, generates timestamp-based version
            metadata: Optional metadata dictionary to save with model
            model_dir: Optional directory to save model. If None, uses config.MODEL_REPOSITORY_DIR
            
        Returns:
            Version string of saved model
            
        Raises:
            ValueError: If model is not fitted
        """
        # Check if model is fitted
        if not hasattr(model, 'n_features_in_'):
            raise ValueError("Cannot save unfitted model")
        
        # Generate version if not provided (timestamp-based)
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"
        
        # Determine save directory
        save_dir = model_dir if model_dir is not None else config.MODEL_REPOSITORY_DIR
        version_dir = save_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Prepare and save metadata
        if metadata is None:
            metadata = {}
        
        metadata_with_version = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'n_features': int(model.n_features_in_),
            'params': model.get_params(),
            **metadata
        }
        
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_with_version, f, indent=2)
        
        return version
