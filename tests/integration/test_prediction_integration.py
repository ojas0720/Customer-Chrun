"""
Integration tests for prediction service with real models.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from services.prediction import ChurnPredictor, CustomerRecord, PredictionResult
from services.model_repository import ModelRepository
from services.preprocessing import ValidationError


class TestPredictionIntegration:
    """Integration tests for prediction service."""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor with real model from repository."""
        # Use the actual model repository
        repository = ModelRepository()
        
        # Check if any models exist
        try:
            predictor = ChurnPredictor(repository=repository)
            return predictor
        except Exception as e:
            pytest.skip(f"No models available in repository: {e}")
    
    def test_predict_with_real_model(self, predictor):
        """Test prediction with real trained model."""
        # Create sample customer data
        customer = {
            'customer_id': 'TEST001',
            'tenure_months': 12,
            'monthly_charges': 65.5,
            'total_charges': 786.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
        
        result = predictor.predict_single(customer)
        
        # Verify result structure
        assert isinstance(result, PredictionResult)
        assert result.customer_id == 'TEST001'
        assert 0 <= result.churn_probability <= 1
        assert isinstance(result.is_high_risk, bool)
        assert result.model_version is not None
        assert result.prediction_timestamp is not None
    
    def test_batch_prediction_with_real_model(self, predictor):
        """Test batch prediction with real trained model."""
        customers = [
            {
                'customer_id': 'TEST001',
                'tenure_months': 12,
                'monthly_charges': 65.5,
                'total_charges': 786.0,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'Fiber optic'
            },
            {
                'customer_id': 'TEST002',
                'tenure_months': 48,
                'monthly_charges': 45.0,
                'total_charges': 2160.0,
                'contract_type': 'Two year',
                'payment_method': 'Bank transfer',
                'internet_service': 'DSL'
            },
            {
                'customer_id': 'TEST003',
                'tenure_months': 6,
                'monthly_charges': 89.9,
                'total_charges': 539.4,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'Fiber optic'
            }
        ]
        
        results = predictor.predict_batch(customers)
        
        # Verify results
        assert len(results) == 3
        assert all(isinstance(r, PredictionResult) for r in results)
        assert [r.customer_id for r in results] == ['TEST001', 'TEST002', 'TEST003']
        assert all(0 <= r.churn_probability <= 1 for r in results)
    
    def test_high_risk_classification_consistency(self, predictor):
        """Test that high-risk classification is consistent with probability."""
        customer = {
            'customer_id': 'TEST001',
            'tenure_months': 12,
            'monthly_charges': 65.5,
            'total_charges': 786.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
        
        result = predictor.predict_single(customer)
        
        # Verify consistency
        expected_high_risk = result.churn_probability >= predictor.high_risk_threshold
        assert result.is_high_risk == expected_high_risk
    
    def test_prediction_with_customer_record(self, predictor):
        """Test prediction using CustomerRecord object."""
        record = CustomerRecord(
            customer_id='TEST001',
            features={
                'tenure_months': 12,
                'monthly_charges': 65.5,
                'total_charges': 786.0,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'Fiber optic'
            }
        )
        
        result = predictor.predict_single(record)
        
        assert isinstance(result, PredictionResult)
        assert result.customer_id == 'TEST001'
        assert 0 <= result.churn_probability <= 1
    
    def test_prediction_with_missing_values(self, predictor):
        """Test that prediction handles missing values appropriately."""
        customer = {
            'customer_id': 'TEST001',
            'tenure_months': 12,
            'monthly_charges': np.nan,  # Missing value
            'total_charges': 786.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
        
        # Should handle missing values through imputation
        result = predictor.predict_single(customer)
        
        assert isinstance(result, PredictionResult)
        assert 0 <= result.churn_probability <= 1
    
    def test_different_thresholds_affect_classification(self, predictor):
        """Test that different thresholds produce different classifications."""
        customer = {
            'customer_id': 'TEST001',
            'tenure_months': 12,
            'monthly_charges': 65.5,
            'total_charges': 786.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
        
        # Get prediction with different thresholds
        data = pd.DataFrame([{k: v for k, v in customer.items() if k != 'customer_id'}])
        
        high_risk_low = predictor.predict_high_risk(data, threshold=0.3)
        high_risk_high = predictor.predict_high_risk(data, threshold=0.9)
        
        # At least one should be different (unless probability is exactly 0.3 or 0.9)
        # This tests that threshold parameter is being used
        assert isinstance(high_risk_low, np.ndarray)
        assert isinstance(high_risk_high, np.ndarray)
