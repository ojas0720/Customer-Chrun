"""
Data validation and preprocessing module for Customer Churn Prediction System.
Handles data validation, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import config


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """
    Validates input data for schema compliance, type checking, and value ranges.
    Ensures data meets requirements before training or prediction.
    """
    
    def __init__(
        self,
        required_features: List[str],
        numerical_features: List[str],
        categorical_features: List[str],
        target_variable: Optional[str] = None,
        numerical_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize DataValidator with feature specifications.
        
        Args:
            required_features: List of feature names that must be present
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            target_variable: Name of target variable (required for training data)
            numerical_ranges: Optional dict mapping feature names to (min, max) tuples
        """
        self.required_features = required_features
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.target_variable = target_variable
        self.numerical_ranges = numerical_ranges or {}
    
    def validate(self, data: pd.DataFrame, require_target: bool = False) -> bool:
        """
        Validate dataframe against schema and value requirements.
        
        Args:
            data: DataFrame to validate
            require_target: Whether to require target variable presence
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails with detailed error message
        """
        # Check for empty dataframe
        if data.empty:
            raise ValidationError("Data is empty")
        
        # Schema validation - check for required features
        missing_features = set(self.required_features) - set(data.columns)
        if missing_features:
            raise ValidationError(
                f"Missing required features: {sorted(missing_features)}"
            )
        
        # Check for target variable if required
        if require_target:
            if self.target_variable is None:
                raise ValidationError("Target variable not specified in validator")
            if self.target_variable not in data.columns:
                raise ValidationError(
                    f"Missing target variable: {self.target_variable}"
                )
        
        # Type checking for numerical features
        for feature in self.numerical_features:
            if feature in data.columns:
                if not pd.api.types.is_numeric_dtype(data[feature]):
                    raise ValidationError(
                        f"Feature '{feature}' must be numeric, got {data[feature].dtype}"
                    )
        
        # Type checking for categorical features
        for feature in self.categorical_features:
            if feature in data.columns:
                if not (pd.api.types.is_string_dtype(data[feature]) or 
                       pd.api.types.is_categorical_dtype(data[feature]) or
                       pd.api.types.is_object_dtype(data[feature])):
                    raise ValidationError(
                        f"Feature '{feature}' must be categorical/string, got {data[feature].dtype}"
                    )
        
        # Value range validation for numerical features
        for feature, (min_val, max_val) in self.numerical_ranges.items():
            if feature in data.columns:
                # Check non-null values only
                valid_data = data[feature].dropna()
                if len(valid_data) > 0:
                    if valid_data.min() < min_val or valid_data.max() > max_val:
                        raise ValidationError(
                            f"Feature '{feature}' has values outside valid range "
                            f"[{min_val}, {max_val}]. Found range: "
                            f"[{valid_data.min()}, {valid_data.max()}]"
                        )
        
        return True
    
    def validate_for_training(self, data: pd.DataFrame) -> bool:
        """
        Validate data for training (requires target variable).
        
        Args:
            data: Training dataframe
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        return self.validate(data, require_target=True)
    
    def validate_for_prediction(self, data: pd.DataFrame) -> bool:
        """
        Validate data for prediction (no target variable required).
        
        Args:
            data: Prediction dataframe
            
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        return self.validate(data, require_target=False)



class DataTransformer:
    """
    Handles data transformation including missing value imputation,
    categorical encoding, and numerical scaling.
    Ensures consistent transformations between training and inference.
    """
    
    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str]
    ):
        """
        Initialize DataTransformer with feature specifications.
        
        Args:
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.is_fitted = False
        
        # Create preprocessing pipeline
        self.numerical_imputer = SimpleImputer(strategy='mean')
        self.numerical_scaler = StandardScaler()
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.categorical_encoder = OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False
        )
        
        # Store feature names after encoding
        self.feature_names_out_ = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataTransformer':
        """
        Fit the transformer on training data.
        
        Args:
            X: Training features dataframe
            y: Target variable (not used, for sklearn compatibility)
            
        Returns:
            self: Fitted transformer
        """
        # Fit numerical transformers
        if self.numerical_features:
            X_num = X[self.numerical_features]
            X_num_imputed = self.numerical_imputer.fit_transform(X_num)
            self.numerical_scaler.fit(X_num_imputed)
        
        # Fit categorical transformers
        if self.categorical_features:
            X_cat = X[self.categorical_features]
            X_cat_imputed = self.categorical_imputer.fit_transform(X_cat)
            self.categorical_encoder.fit(X_cat_imputed)
        
        # Build feature names
        self._build_feature_names()
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted transformers.
        
        Args:
            X: Features dataframe to transform
            
        Returns:
            Transformed feature array
            
        Raises:
            ValueError: If transformer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform")
        
        transformed_parts = []
        
        # Transform numerical features
        if self.numerical_features:
            X_num = X[self.numerical_features]
            X_num_imputed = self.numerical_imputer.transform(X_num)
            X_num_scaled = self.numerical_scaler.transform(X_num_imputed)
            transformed_parts.append(X_num_scaled)
        
        # Transform categorical features
        if self.categorical_features:
            X_cat = X[self.categorical_features]
            X_cat_imputed = self.categorical_imputer.transform(X_cat)
            X_cat_encoded = self.categorical_encoder.transform(X_cat_imputed)
            transformed_parts.append(X_cat_encoded)
        
        # Concatenate all transformed features
        if transformed_parts:
            return np.hstack(transformed_parts)
        else:
            return np.array([]).reshape(len(X), 0)
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """
        Fit transformer and transform data in one step.
        
        Args:
            X: Training features dataframe
            y: Target variable (not used, for sklearn compatibility)
            
        Returns:
            Transformed feature array
        """
        return self.fit(X, y).transform(X)
    
    def _build_feature_names(self) -> None:
        """Build list of feature names after transformation."""
        feature_names = []
        
        # Add numerical feature names
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        
        # Add encoded categorical feature names
        if self.categorical_features:
            cat_feature_names = self.categorical_encoder.get_feature_names_out(
                self.categorical_features
            )
            feature_names.extend(cat_feature_names)
        
        self.feature_names_out_ = feature_names
    
    def get_feature_names_out(self) -> List[str]:
        """
        Get feature names after transformation.
        
        Returns:
            List of feature names
            
        Raises:
            ValueError: If transformer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before getting feature names")
        return self.feature_names_out_
    
    def save(self, path: str) -> None:
        """
        Save fitted transformer to disk.
        
        Args:
            path: File path to save transformer
            
        Raises:
            ValueError: If transformer is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted transformer")
        
        transformer_data = {
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'numerical_imputer': self.numerical_imputer,
            'numerical_scaler': self.numerical_scaler,
            'categorical_imputer': self.categorical_imputer,
            'categorical_encoder': self.categorical_encoder,
            'feature_names_out_': self.feature_names_out_,
            'is_fitted': self.is_fitted
        }
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(transformer_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'DataTransformer':
        """
        Load fitted transformer from disk.
        
        Args:
            path: File path to load transformer from
            
        Returns:
            Loaded transformer instance
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Transformer file not found: {path}")
        
        transformer_data = joblib.load(path)
        
        # Create new instance
        transformer = cls(
            numerical_features=transformer_data['numerical_features'],
            categorical_features=transformer_data['categorical_features']
        )
        
        # Restore fitted state
        transformer.numerical_imputer = transformer_data['numerical_imputer']
        transformer.numerical_scaler = transformer_data['numerical_scaler']
        transformer.categorical_imputer = transformer_data['categorical_imputer']
        transformer.categorical_encoder = transformer_data['categorical_encoder']
        transformer.feature_names_out_ = transformer_data['feature_names_out_']
        transformer.is_fitted = transformer_data['is_fitted']
        
        return transformer



class FeatureEngineer:
    """
    Creates derived features from raw customer data.
    Ensures consistency between training and inference.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.is_fitted = False
        self.feature_specs = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineer':
        """
        Fit the feature engineer (learn any necessary statistics).
        
        Args:
            X: Training features dataframe
            y: Target variable (not used, for sklearn compatibility)
            
        Returns:
            self: Fitted feature engineer
        """
        # For now, feature engineering is stateless, but we keep fit for consistency
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by adding derived features.
        
        Args:
            X: Features dataframe
            
        Returns:
            DataFrame with original and derived features
        """
        X_transformed = X.copy()
        
        # Example derived features based on common churn prediction patterns
        
        # 1. Tenure-based features
        if 'tenure_months' in X.columns:
            # Tenure in years
            X_transformed['tenure_years'] = X['tenure_months'] / 12.0
            
            # Tenure categories
            X_transformed['is_new_customer'] = (X['tenure_months'] <= 12).astype(int)
            X_transformed['is_long_term_customer'] = (X['tenure_months'] >= 36).astype(int)
        
        # 2. Spending-based features
        if 'monthly_charges' in X.columns and 'total_charges' in X.columns:
            # Average monthly spend (handle division by zero)
            X_transformed['avg_monthly_spend'] = X['total_charges'] / (X['tenure_months'] + 1)
            
            # Ratio of current to average spend
            X_transformed['current_to_avg_ratio'] = (
                X['monthly_charges'] / (X_transformed['avg_monthly_spend'] + 0.01)
            )
        
        # 3. Total spend ratio (if both charges columns exist)
        if 'monthly_charges' in X.columns and 'total_charges' in X.columns:
            # Total spend ratio: what percentage of total charges is the monthly charge
            X_transformed['total_spend_ratio'] = (
                X['monthly_charges'] / (X['total_charges'] + 0.01)
            )
        
        # 4. Service complexity (if service-related columns exist)
        service_columns = [col for col in X.columns if 'service' in col.lower()]
        if service_columns:
            # Count number of services (assuming binary or categorical encoding)
            X_transformed['service_count'] = X[service_columns].notna().sum(axis=1)
        
        # 5. Contract value features
        if 'contract_type' in X.columns and 'monthly_charges' in X.columns:
            # High value customer indicator
            monthly_median = X['monthly_charges'].median()
            X_transformed['is_high_value'] = (X['monthly_charges'] > monthly_median).astype(int)
        
        return X_transformed
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit feature engineer and transform data in one step.
        
        Args:
            X: Training features dataframe
            y: Target variable (not used, for sklearn compatibility)
            
        Returns:
            DataFrame with original and derived features
        """
        return self.fit(X, y).transform(X)
    
    def get_derived_feature_names(self) -> List[str]:
        """
        Get list of derived feature names that will be created.
        
        Returns:
            List of derived feature names
        """
        # This is a static list based on the transform logic
        return [
            'tenure_years',
            'is_new_customer',
            'is_long_term_customer',
            'avg_monthly_spend',
            'current_to_avg_ratio',
            'total_spend_ratio',
            'service_count',
            'is_high_value'
        ]
