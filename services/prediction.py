"""
Prediction service for Customer Churn Prediction System.
Handles churn probability predictions with model loading and input validation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from xgboost import XGBClassifier

import config
from services.model_repository import ModelRepository, ModelNotFoundError
from services.preprocessing import DataValidator, DataTransformer, FeatureEngineer, ValidationError
from services.explainability import SHAPExplainer


@dataclass
class CustomerRecord:
    """
    Data structure containing customer attributes used for prediction.
    """
    customer_id: str
    features: Dict[str, Any]
    
    def validate(self) -> bool:
        """
        Validate customer record has required features and valid values.
        
        Returns:
            True if validation passes
            
        Raises:
            ValidationError: If validation fails
        """
        # Check customer_id
        if not self.customer_id or not isinstance(self.customer_id, str):
            raise ValidationError("customer_id must be a non-empty string")
        
        # Check features dict
        if not self.features or not isinstance(self.features, dict):
            raise ValidationError("features must be a non-empty dictionary")
        
        # Check for required features
        required_features = set(config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES)
        provided_features = set(self.features.keys())
        missing_features = required_features - provided_features
        
        if missing_features:
            raise ValidationError(
                f"Missing required features: {sorted(missing_features)}"
            )
        
        # Check for None/NaN in critical features (some missing values are OK, but not all)
        all_none = all(
            v is None or (isinstance(v, float) and np.isnan(v))
            for v in self.features.values()
        )
        if all_none:
            raise ValidationError("All feature values cannot be None/NaN")
        
        return True
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert customer record to pandas DataFrame.
        
        Returns:
            DataFrame with single row containing customer features
        """
        return pd.DataFrame([self.features])


@dataclass
class PredictionResult:
    """
    Result of a churn prediction containing probability, risk classification, and explanations.
    """
    customer_id: str
    churn_probability: float
    is_high_risk: bool
    prediction_timestamp: datetime
    model_version: str
    top_features: Optional[List[tuple]] = None  # List of (feature_name, shap_value) tuples
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert prediction result to dictionary.
        
        Returns:
            Dictionary representation of prediction result
        """
        return {
            'customer_id': self.customer_id,
            'churn_probability': self.churn_probability,
            'is_high_risk': self.is_high_risk,
            'prediction_timestamp': self.prediction_timestamp.isoformat(),
            'model_version': self.model_version,
            'top_features': self.top_features
        }


class ChurnPredictor:
    """
    Main prediction interface for generating churn probability predictions.
    Handles model loading, input validation, and batch predictions.
    """
    
    def __init__(
        self,
        model_version: Optional[str] = None,
        repository: Optional[ModelRepository] = None,
        high_risk_threshold: float = config.HIGH_RISK_THRESHOLD,
        enable_explanations: bool = False,
        background_data: Optional[np.ndarray] = None
    ):
        """
        Initialize ChurnPredictor.
        
        Args:
            model_version: Specific model version to use. If None, uses deployed/latest version
            repository: ModelRepository instance. If None, creates default repository
            high_risk_threshold: Probability threshold for high-risk classification
            enable_explanations: Whether to enable SHAP explanations (requires background_data)
            background_data: Background dataset for SHAP computation (preprocessed features)
        """
        self.repository = repository if repository is not None else ModelRepository()
        self.high_risk_threshold = high_risk_threshold
        
        # Load model and transformer
        if model_version is not None:
            self.model, self.transformer = self.repository.load(model_version)
            self.model_version = model_version
        else:
            # Load deployed model or fall back to latest
            self.model, self.transformer = self.repository.load_deployed()
            self.model_version = self.repository.get_deployed_version()
            if self.model_version is None:
                self.model_version = self.repository.get_latest_version()
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer()
        self.feature_engineer.fit(pd.DataFrame())  # Stateless, just set fitted flag
        
        # Initialize validator for predictions
        self.validator = DataValidator(
            required_features=config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES,
            numerical_features=config.NUMERICAL_FEATURES,
            categorical_features=config.CATEGORICAL_FEATURES,
            target_variable=None  # Not needed for predictions
        )
        
        # Initialize SHAP explainer if requested
        self.explainer = None
        self.feature_names = None
        if enable_explanations:
            if background_data is None:
                raise ValueError("background_data is required when enable_explanations=True")
            self.explainer = SHAPExplainer(self.model, background_data)
            # Get feature names from transformer
            self.feature_names = self.transformer.get_feature_names_out()
    
    def predict_proba(self, customer_data: Union[pd.DataFrame, CustomerRecord]) -> np.ndarray:
        """
        Generate churn probability predictions.
        
        Args:
            customer_data: DataFrame with customer features or single CustomerRecord
            
        Returns:
            Array of churn probabilities (values between 0 and 1)
            
        Raises:
            ValidationError: If input data is invalid
        """
        # Convert CustomerRecord to DataFrame if needed
        if isinstance(customer_data, CustomerRecord):
            customer_data.validate()
            df = customer_data.to_dataframe()
        else:
            df = customer_data.copy()
        
        # Validate input data
        self.validator.validate_for_prediction(df)
        
        # Apply feature engineering
        df_engineered = self.feature_engineer.transform(df)
        
        # Transform features
        X_transformed = self.transformer.transform(df_engineered)
        
        # Generate predictions (probability of churn - positive class)
        probabilities = self.model.predict_proba(X_transformed)[:, 1]
        
        return probabilities
    
    def predict_high_risk(
        self,
        customer_data: Union[pd.DataFrame, CustomerRecord],
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Classify customers as high-risk based on churn probability threshold.
        
        Args:
            customer_data: DataFrame with customer features or single CustomerRecord
            threshold: Optional custom threshold. If None, uses instance threshold
            
        Returns:
            Boolean array indicating high-risk classification
        """
        if threshold is None:
            threshold = self.high_risk_threshold
        
        probabilities = self.predict_proba(customer_data)
        return probabilities >= threshold
    
    def predict_batch(
        self,
        customers: List[Union[Dict[str, Any], CustomerRecord]],
        include_explanations: bool = False
    ) -> List[PredictionResult]:
        """
        Process batch predictions for multiple customers.
        
        Args:
            customers: List of customer dictionaries or CustomerRecord objects
            include_explanations: Whether to include SHAP explanations
            
        Returns:
            List of PredictionResult objects
            
        Raises:
            ValidationError: If any customer record is invalid
            ValueError: If batch size exceeds maximum or explanations requested without explainer
        """
        if len(customers) > config.MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(customers)} exceeds maximum {config.MAX_BATCH_SIZE}"
            )
        
        if include_explanations and self.explainer is None:
            raise ValueError(
                "Explanations requested but explainer not initialized. "
                "Create ChurnPredictor with enable_explanations=True and background_data."
            )
        
        # Convert all customers to CustomerRecord objects
        customer_records = []
        for customer in customers:
            if isinstance(customer, dict):
                # Extract customer_id and features from dict
                if 'customer_id' not in customer:
                    raise ValidationError("Each customer dict must have 'customer_id'")
                
                customer_id = customer['customer_id']
                # All other fields are features
                features = {k: v for k, v in customer.items() if k != 'customer_id'}
                
                record = CustomerRecord(customer_id=customer_id, features=features)
            else:
                record = customer
            
            # Validate each record
            record.validate()
            customer_records.append(record)
        
        # Create DataFrame from all customer records
        customer_ids = [record.customer_id for record in customer_records]
        features_list = [record.features for record in customer_records]
        df = pd.DataFrame(features_list)
        
        # Validate input data
        self.validator.validate_for_prediction(df)
        
        # Apply feature engineering
        df_engineered = self.feature_engineer.transform(df)
        
        # Transform features
        X_transformed = self.transformer.transform(df_engineered)
        
        # Get predictions
        probabilities = self.model.predict_proba(X_transformed)[:, 1]
        high_risk_flags = probabilities >= self.high_risk_threshold
        
        # Compute SHAP explanations if requested
        shap_explanations = None
        if include_explanations:
            shap_values = self.explainer.explain(X_transformed)
            shap_explanations = []
            for i in range(len(customer_records)):
                top_features = self.explainer.get_top_features(
                    shap_values[i],
                    self.feature_names
                )
                shap_explanations.append(top_features)
        
        # Create PredictionResult objects
        results = []
        prediction_time = datetime.now()
        
        for i, (customer_id, prob, is_high_risk) in enumerate(
            zip(customer_ids, probabilities, high_risk_flags)
        ):
            result = PredictionResult(
                customer_id=customer_id,
                churn_probability=float(prob),
                is_high_risk=bool(is_high_risk),
                prediction_timestamp=prediction_time,
                model_version=self.model_version,
                top_features=shap_explanations[i] if shap_explanations else None
            )
            results.append(result)
        
        return results
    
    def predict_single(
        self,
        customer: Union[Dict[str, Any], CustomerRecord],
        include_explanations: bool = False
    ) -> PredictionResult:
        """
        Generate prediction for a single customer.
        
        Args:
            customer: Customer dictionary or CustomerRecord object
            include_explanations: Whether to include SHAP explanations
            
        Returns:
            PredictionResult object
        """
        results = self.predict_batch([customer], include_explanations=include_explanations)
        return results[0]
