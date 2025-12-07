"""
Explainability service for Customer Churn Prediction System.
Provides SHAP-based explanations for model predictions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from xgboost import XGBClassifier
import shap

import config


class SHAPExplainer:
    """
    Computes SHAP (Shapley Additive exPlanations) values to interpret model predictions.
    Provides feature importance rankings and validates explanation properties.
    """
    
    def __init__(self, model: XGBClassifier, background_data: np.ndarray):
        """
        Initialize SHAP explainer with trained model and background data.
        
        Args:
            model: Trained XGBClassifier model
            background_data: Background dataset for SHAP computation (preprocessed features)
                           Should be a representative sample of training data
        
        Raises:
            ValueError: If model is not fitted or background_data is invalid
        """
        # Validate model is fitted
        if not hasattr(model, 'n_features_in_'):
            raise ValueError("Model must be fitted before creating explainer")
        
        # Validate background data
        if not isinstance(background_data, np.ndarray):
            raise ValueError("background_data must be a numpy array")
        
        if background_data.ndim != 2:
            raise ValueError("background_data must be 2-dimensional")
        
        if background_data.shape[0] == 0:
            raise ValueError("background_data cannot be empty")
        
        if background_data.shape[1] != model.n_features_in_:
            raise ValueError(
                f"background_data has {background_data.shape[1]} features, "
                f"but model expects {model.n_features_in_}"
            )
        
        self.model = model
        self.background_data = background_data
        
        # Create SHAP explainer using TreeExplainer for XGBoost
        self.explainer = shap.TreeExplainer(model)
        
        # Compute base value (expected value over background data)
        self.base_value = self.explainer.expected_value
        
        # Handle case where base_value is an array (for multi-class)
        if isinstance(self.base_value, np.ndarray):
            # For binary classification, use the positive class base value
            self.base_value = float(self.base_value[1]) if len(self.base_value) > 1 else float(self.base_value[0])
        else:
            self.base_value = float(self.base_value)
    
    def explain(self, customer_data: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for customer data.
        
        Args:
            customer_data: Preprocessed customer features (n_samples, n_features)
        
        Returns:
            SHAP values array with shape (n_samples, n_features)
            Each value represents the contribution of that feature to the prediction
        
        Raises:
            ValueError: If customer_data has wrong shape or type
        """
        # Validate input
        if not isinstance(customer_data, np.ndarray):
            raise ValueError("customer_data must be a numpy array")
        
        if customer_data.ndim == 1:
            # Single sample, reshape to 2D
            customer_data = customer_data.reshape(1, -1)
        elif customer_data.ndim != 2:
            raise ValueError("customer_data must be 1D or 2D array")
        
        if customer_data.shape[1] != self.model.n_features_in_:
            raise ValueError(
                f"customer_data has {customer_data.shape[1]} features, "
                f"but model expects {self.model.n_features_in_}"
            )
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(customer_data)
        
        # Handle multi-output case (binary classification returns values for both classes)
        if isinstance(shap_values, list):
            # For binary classification, use positive class (index 1)
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        # Ensure 2D output
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        
        return shap_values
    
    def get_top_features(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        n: int = config.TOP_FEATURES_COUNT
    ) -> List[Tuple[str, float]]:
        """
        Identify and rank top contributing features by absolute SHAP value.
        
        Args:
            shap_values: SHAP values for a single prediction (1D array of length n_features)
            feature_names: List of feature names corresponding to SHAP values
            n: Number of top features to return (default from config)
        
        Returns:
            List of (feature_name, shap_value) tuples sorted by absolute value (descending)
            Length is min(n, number_of_features)
        
        Raises:
            ValueError: If shap_values and feature_names have different lengths
        """
        # Validate inputs
        if not isinstance(shap_values, np.ndarray):
            raise ValueError("shap_values must be a numpy array")
        
        # Handle 2D array (single sample)
        if shap_values.ndim == 2:
            if shap_values.shape[0] != 1:
                raise ValueError("shap_values must be for a single sample (1D or 2D with shape (1, n_features))")
            shap_values = shap_values.flatten()
        elif shap_values.ndim != 1:
            raise ValueError("shap_values must be 1D or 2D array")
        
        if len(shap_values) != len(feature_names):
            raise ValueError(
                f"shap_values length ({len(shap_values)}) must match "
                f"feature_names length ({len(feature_names)})"
            )
        
        if n <= 0:
            raise ValueError("n must be positive")
        
        # Create list of (feature_name, shap_value, abs_shap_value) tuples
        feature_contributions = [
            (name, float(value), abs(float(value)))
            for name, value in zip(feature_names, shap_values)
        ]
        
        # Sort by absolute value (descending)
        feature_contributions.sort(key=lambda x: x[2], reverse=True)
        
        # Return top n features (without the absolute value)
        top_n = min(n, len(feature_contributions))
        return [(name, value) for name, value, _ in feature_contributions[:top_n]]
    
    def validate_shap_sum(
        self,
        shap_values: np.ndarray,
        prediction: float,
        tolerance: float = 0.001
    ) -> bool:
        """
        Verify SHAP additivity property: sum of SHAP values equals prediction - base_value.
        
        Args:
            shap_values: SHAP values for a single prediction (1D array)
            prediction: Model prediction (probability or log-odds)
            tolerance: Numerical tolerance for equality check (default 0.001)
        
        Returns:
            True if additivity property holds within tolerance, False otherwise
        
        Raises:
            ValueError: If shap_values is not 1D array
        """
        # Validate input
        if not isinstance(shap_values, np.ndarray):
            raise ValueError("shap_values must be a numpy array")
        
        # Handle 2D array (single sample)
        if shap_values.ndim == 2:
            if shap_values.shape[0] != 1:
                raise ValueError("shap_values must be for a single sample")
            shap_values = shap_values.flatten()
        elif shap_values.ndim != 1:
            raise ValueError("shap_values must be 1D or 2D array")
        
        # Compute sum of SHAP values
        shap_sum = float(np.sum(shap_values))
        
        # Expected difference between prediction and base value
        expected_diff = float(prediction) - self.base_value
        
        # Check if they match within tolerance
        return abs(shap_sum - expected_diff) <= tolerance
