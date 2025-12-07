"""
Unit tests for prediction service.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, MagicMock

from services.prediction import (
    CustomerRecord,
    PredictionResult,
    ChurnPredictor,
)
from services.preprocessing import ValidationError


class TestCustomerRecord:
    """Unit tests for CustomerRecord class."""
    
    def test_valid_customer_record_passes_validation(self):
        """Test that valid customer record passes validation."""
        record = CustomerRecord(
            customer_id='CUST001',
            features={
                'tenure_months': 12,
                'monthly_charges': 50.0,
                'total_charges': 600.0,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'DSL'
            }
        )
        
        assert record.validate() is True
    
    def test_empty_customer_id_raises_error(self):
        """Test that empty customer_id raises ValidationError."""
        record = CustomerRecord(
            customer_id='',
            features={'tenure_months': 12}
        )
        
        with pytest.raises(ValidationError, match="customer_id must be a non-empty string"):
            record.validate()
    
    def test_missing_required_features_raises_error(self):
        """Test that missing required features raises ValidationError."""
        record = CustomerRecord(
            customer_id='CUST001',
            features={
                'tenure_months': 12
                # Missing other required features
            }
        )
        
        with pytest.raises(ValidationError, match="Missing required features"):
            record.validate()
    
    def test_all_none_features_raises_error(self):
        """Test that all None features raises ValidationError."""
        record = CustomerRecord(
            customer_id='CUST001',
            features={
                'tenure_months': None,
                'monthly_charges': None,
                'total_charges': None,
                'contract_type': None,
                'payment_method': None,
                'internet_service': None
            }
        )
        
        with pytest.raises(ValidationError, match="All feature values cannot be None"):
            record.validate()
    
    def test_to_dataframe_conversion(self):
        """Test conversion to DataFrame."""
        record = CustomerRecord(
            customer_id='CUST001',
            features={
                'tenure_months': 12,
                'monthly_charges': 50.0
            }
        )
        
        df = record.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df['tenure_months'].iloc[0] == 12
        assert df['monthly_charges'].iloc[0] == 50.0


class TestPredictionResult:
    """Unit tests for PredictionResult class."""
    
    def test_prediction_result_creation(self):
        """Test creating a PredictionResult."""
        result = PredictionResult(
            customer_id='CUST001',
            churn_probability=0.75,
            is_high_risk=True,
            prediction_timestamp=datetime.now(),
            model_version='v1.0.0',
            top_features=[('tenure_months', -0.3), ('monthly_charges', 0.5)]
        )
        
        assert result.customer_id == 'CUST001'
        assert result.churn_probability == 0.75
        assert result.is_high_risk is True
        assert result.model_version == 'v1.0.0'
        assert len(result.top_features) == 2
    
    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        timestamp = datetime.now()
        result = PredictionResult(
            customer_id='CUST001',
            churn_probability=0.75,
            is_high_risk=True,
            prediction_timestamp=timestamp,
            model_version='v1.0.0'
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['customer_id'] == 'CUST001'
        assert result_dict['churn_probability'] == 0.75
        assert result_dict['is_high_risk'] is True
        assert result_dict['model_version'] == 'v1.0.0'
        assert 'prediction_timestamp' in result_dict


class TestChurnPredictor:
    """Unit tests for ChurnPredictor class."""
    
    @pytest.fixture
    def mock_repository(self):
        """Create a mock repository with model and transformer."""
        repository = Mock()
        
        # Create mock model
        model = Mock()
        model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.8, 0.2]]))
        
        # Create mock transformer
        transformer = Mock()
        transformer.transform = Mock(return_value=np.array([[1, 2, 3], [4, 5, 6]]))
        
        repository.load_deployed = Mock(return_value=(model, transformer))
        repository.get_deployed_version = Mock(return_value='v1.0.0')
        repository.get_latest_version = Mock(return_value='v1.0.0')
        
        return repository
    
    def test_predict_proba_returns_probabilities(self, mock_repository):
        """Test that predict_proba returns probability values."""
        predictor = ChurnPredictor(repository=mock_repository)
        
        data = pd.DataFrame({
            'tenure_months': [12, 24],
            'monthly_charges': [50.0, 60.0],
            'total_charges': [600.0, 1440.0],
            'contract_type': ['Month-to-month', 'One year'],
            'payment_method': ['Electronic check', 'Credit card'],
            'internet_service': ['DSL', 'Fiber optic']
        })
        
        probabilities = predictor.predict_proba(data)
        
        assert len(probabilities) == 2
        assert all(0 <= p <= 1 for p in probabilities)
    
    def test_predict_high_risk_with_default_threshold(self, mock_repository):
        """Test high-risk classification with default threshold."""
        predictor = ChurnPredictor(repository=mock_repository, high_risk_threshold=0.5)
        
        data = pd.DataFrame({
            'tenure_months': [12, 24],
            'monthly_charges': [50.0, 60.0],
            'total_charges': [600.0, 1440.0],
            'contract_type': ['Month-to-month', 'One year'],
            'payment_method': ['Electronic check', 'Credit card'],
            'internet_service': ['DSL', 'Fiber optic']
        })
        
        high_risk = predictor.predict_high_risk(data)
        
        assert len(high_risk) == 2
        assert isinstance(high_risk, np.ndarray)
        assert high_risk.dtype == bool
    
    def test_predict_high_risk_with_custom_threshold(self, mock_repository):
        """Test high-risk classification with custom threshold."""
        predictor = ChurnPredictor(repository=mock_repository)
        
        data = pd.DataFrame({
            'tenure_months': [12],
            'monthly_charges': [50.0],
            'total_charges': [600.0],
            'contract_type': ['Month-to-month'],
            'payment_method': ['Electronic check'],
            'internet_service': ['DSL']
        })
        
        # Test with different thresholds
        high_risk_low = predictor.predict_high_risk(data, threshold=0.3)
        high_risk_high = predictor.predict_high_risk(data, threshold=0.9)
        
        assert isinstance(high_risk_low, np.ndarray)
        assert isinstance(high_risk_high, np.ndarray)
    
    def test_predict_single_customer(self, mock_repository):
        """Test prediction for single customer."""
        predictor = ChurnPredictor(repository=mock_repository)
        
        customer = {
            'customer_id': 'CUST001',
            'tenure_months': 12,
            'monthly_charges': 50.0,
            'total_charges': 600.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'DSL'
        }
        
        result = predictor.predict_single(customer)
        
        assert isinstance(result, PredictionResult)
        assert result.customer_id == 'CUST001'
        assert 0 <= result.churn_probability <= 1
        assert isinstance(result.is_high_risk, bool)
        assert result.model_version == 'v1.0.0'
    
    def test_predict_batch_returns_correct_count(self, mock_repository):
        """Test that batch prediction returns correct number of results."""
        predictor = ChurnPredictor(repository=mock_repository)
        
        customers = [
            {
                'customer_id': 'CUST001',
                'tenure_months': 12,
                'monthly_charges': 50.0,
                'total_charges': 600.0,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'DSL'
            },
            {
                'customer_id': 'CUST002',
                'tenure_months': 24,
                'monthly_charges': 60.0,
                'total_charges': 1440.0,
                'contract_type': 'One year',
                'payment_method': 'Credit card',
                'internet_service': 'Fiber optic'
            }
        ]
        
        results = predictor.predict_batch(customers)
        
        assert len(results) == 2
        assert all(isinstance(r, PredictionResult) for r in results)
        assert results[0].customer_id == 'CUST001'
        assert results[1].customer_id == 'CUST002'
    
    def test_predict_batch_with_customer_records(self, mock_repository):
        """Test batch prediction with CustomerRecord objects."""
        predictor = ChurnPredictor(repository=mock_repository)
        
        customers = [
            CustomerRecord(
                customer_id='CUST001',
                features={
                    'tenure_months': 12,
                    'monthly_charges': 50.0,
                    'total_charges': 600.0,
                    'contract_type': 'Month-to-month',
                    'payment_method': 'Electronic check',
                    'internet_service': 'DSL'
                }
            )
        ]
        
        results = predictor.predict_batch(customers)
        
        assert len(results) == 1
        assert results[0].customer_id == 'CUST001'
    
    def test_predict_batch_exceeds_max_size_raises_error(self, mock_repository):
        """Test that exceeding max batch size raises ValueError."""
        predictor = ChurnPredictor(repository=mock_repository)
        
        # Create a batch larger than MAX_BATCH_SIZE (1000)
        customers = [
            {
                'customer_id': f'CUST{i:04d}',
                'tenure_months': 12,
                'monthly_charges': 50.0,
                'total_charges': 600.0,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'DSL'
            }
            for i in range(1001)
        ]
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            predictor.predict_batch(customers)
    
    def test_invalid_customer_raises_validation_error(self, mock_repository):
        """Test that invalid customer data raises ValidationError."""
        predictor = ChurnPredictor(repository=mock_repository)
        
        # Missing required features
        customer = {
            'customer_id': 'CUST001',
            'tenure_months': 12
            # Missing other required features
        }
        
        with pytest.raises(ValidationError):
            predictor.predict_single(customer)
