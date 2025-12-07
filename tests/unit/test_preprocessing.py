"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from services.preprocessing import (
    DataValidator,
    DataTransformer,
    FeatureEngineer,
    ValidationError
)


class TestDataValidator:
    """Unit tests for DataValidator class."""
    
    def test_valid_data_passes_validation(self):
        """Test that valid data passes validation."""
        validator = DataValidator(
            required_features=['age', 'income', 'category'],
            numerical_features=['age', 'income'],
            categorical_features=['category']
        )
        
        data = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000],
            'category': ['A', 'B', 'C']
        })
        
        assert validator.validate(data) is True
    
    def test_missing_required_features_raises_error(self):
        """Test that missing required features raises ValidationError."""
        validator = DataValidator(
            required_features=['age', 'income', 'category'],
            numerical_features=['age', 'income'],
            categorical_features=['category']
        )
        
        data = pd.DataFrame({
            'age': [25, 30, 35],
            'income': [50000, 60000, 70000]
            # Missing 'category'
        })
        
        with pytest.raises(ValidationError, match="Missing required features"):
            validator.validate(data)
    
    def test_empty_dataframe_raises_error(self):
        """Test that empty dataframe raises ValidationError."""
        validator = DataValidator(
            required_features=['age'],
            numerical_features=['age'],
            categorical_features=[]
        )
        
        data = pd.DataFrame()
        
        with pytest.raises(ValidationError, match="Data is empty"):
            validator.validate(data)
    
    def test_wrong_type_for_numerical_feature_raises_error(self):
        """Test that wrong type for numerical feature raises ValidationError."""
        validator = DataValidator(
            required_features=['age'],
            numerical_features=['age'],
            categorical_features=[]
        )
        
        data = pd.DataFrame({
            'age': ['twenty', 'thirty', 'forty']  # Should be numeric
        })
        
        with pytest.raises(ValidationError, match="must be numeric"):
            validator.validate(data)
    
    def test_value_out_of_range_raises_error(self):
        """Test that values outside valid range raise ValidationError."""
        validator = DataValidator(
            required_features=['age'],
            numerical_features=['age'],
            categorical_features=[],
            numerical_ranges={'age': (0, 120)}
        )
        
        data = pd.DataFrame({
            'age': [25, 30, 150]  # 150 is out of range
        })
        
        with pytest.raises(ValidationError, match="outside valid range"):
            validator.validate(data)
    
    def test_validate_for_training_requires_target(self):
        """Test that validate_for_training requires target variable."""
        validator = DataValidator(
            required_features=['age'],
            numerical_features=['age'],
            categorical_features=[],
            target_variable='churn'
        )
        
        data = pd.DataFrame({
            'age': [25, 30, 35]
            # Missing 'churn' target
        })
        
        with pytest.raises(ValidationError, match="Missing target variable"):
            validator.validate_for_training(data)


class TestDataTransformer:
    """Unit tests for DataTransformer class."""
    
    def test_fit_transform_numerical_features(self):
        """Test fitting and transforming numerical features."""
        transformer = DataTransformer(
            numerical_features=['age', 'income'],
            categorical_features=[]
        )
        
        data = pd.DataFrame({
            'age': [25, 30, 35, 40],
            'income': [50000, 60000, 70000, 80000]
        })
        
        result = transformer.fit_transform(data)
        
        assert result.shape == (4, 2)
        assert transformer.is_fitted is True
        # Check that scaling was applied (mean should be close to 0)
        assert np.abs(result.mean(axis=0)).max() < 0.1
    
    def test_fit_transform_categorical_features(self):
        """Test fitting and transforming categorical features."""
        transformer = DataTransformer(
            numerical_features=[],
            categorical_features=['category']
        )
        
        data = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C']
        })
        
        result = transformer.fit_transform(data)
        
        assert result.shape[0] == 4
        assert result.shape[1] == 3  # One-hot encoded: A, B, C
        assert transformer.is_fitted is True
    
    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises ValueError."""
        transformer = DataTransformer(
            numerical_features=['age'],
            categorical_features=[]
        )
        
        data = pd.DataFrame({'age': [25, 30, 35]})
        
        with pytest.raises(ValueError, match="must be fitted"):
            transformer.transform(data)
    
    def test_handles_missing_values(self):
        """Test that transformer handles missing values."""
        transformer = DataTransformer(
            numerical_features=['age'],
            categorical_features=['category']
        )
        
        train_data = pd.DataFrame({
            'age': [25, 30, np.nan, 40],
            'category': ['A', 'B', None, 'A']
        })
        
        result = transformer.fit_transform(train_data)
        
        # Should not contain NaN values
        assert not np.isnan(result).any()
    
    def test_unknown_category_handling(self):
        """Test that unknown categories are handled gracefully."""
        transformer = DataTransformer(
            numerical_features=[],
            categorical_features=['category']
        )
        
        train_data = pd.DataFrame({
            'category': ['A', 'B', 'A']
        })
        
        transformer.fit(train_data)
        
        # Test data with unknown category
        test_data = pd.DataFrame({
            'category': ['A', 'C']  # 'C' was not in training
        })
        
        result = transformer.transform(test_data)
        
        # Should not raise error and should return valid result
        assert result.shape[0] == 2
    
    def test_save_and_load(self):
        """Test saving and loading transformer."""
        transformer = DataTransformer(
            numerical_features=['age'],
            categorical_features=['category']
        )
        
        data = pd.DataFrame({
            'age': [25, 30, 35],
            'category': ['A', 'B', 'A']
        })
        
        transformer.fit(data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'transformer.pkl'
            
            # Save
            transformer.save(str(path))
            
            # Load
            loaded_transformer = DataTransformer.load(str(path))
            
            # Verify loaded transformer works
            result_original = transformer.transform(data)
            result_loaded = loaded_transformer.transform(data)
            
            np.testing.assert_array_almost_equal(result_original, result_loaded)


class TestFeatureEngineer:
    """Unit tests for FeatureEngineer class."""
    
    def test_creates_derived_features(self):
        """Test that feature engineer creates derived features."""
        engineer = FeatureEngineer()
        
        data = pd.DataFrame({
            'tenure_months': [12, 24, 36],
            'monthly_charges': [50, 60, 70],
            'total_charges': [600, 1440, 2520]
        })
        
        result = engineer.fit_transform(data)
        
        # Should have original + derived features
        assert result.shape[0] == 3
        assert result.shape[1] > data.shape[1]
        
        # Check specific derived features exist
        assert 'tenure_years' in result.columns
        assert 'total_spend_ratio' in result.columns
    
    def test_consistency_between_training_and_inference(self):
        """Test that feature engineering is consistent."""
        engineer = FeatureEngineer()
        
        train_data = pd.DataFrame({
            'tenure_months': [12, 24, 36],
            'monthly_charges': [50, 60, 70],
            'total_charges': [600, 1440, 2520]
        })
        
        engineer.fit(train_data)
        
        test_data = pd.DataFrame({
            'tenure_months': [18],
            'monthly_charges': [55],
            'total_charges': [990]
        })
        
        result = engineer.transform(test_data)
        
        # Should create same derived features
        assert 'tenure_years' in result.columns
        assert result['tenure_years'].iloc[0] == 18 / 12.0
    
    def test_handles_missing_columns_gracefully(self):
        """Test that feature engineer handles missing columns."""
        engineer = FeatureEngineer()
        
        # Data with only some columns
        data = pd.DataFrame({
            'tenure_months': [12, 24, 36]
        })
        
        result = engineer.fit_transform(data)
        
        # Should still work and create applicable features
        assert 'tenure_years' in result.columns
        assert result.shape[0] == 3
