"""
Integration tests for the Streamlit dashboard.
Tests dashboard components and data loading.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import config
from services.model_repository import ModelRepository
from services.prediction import ChurnPredictor
from services.preprocessing import FeatureEngineer


class TestDashboardIntegration:
    """Integration tests for dashboard functionality."""
    
    @pytest.fixture
    def repository(self):
        """Create model repository instance."""
        return ModelRepository()
    
    @pytest.fixture
    def test_data(self):
        """Load test data."""
        if config.TEST_DATA_PATH.exists():
            return pd.read_csv(config.TEST_DATA_PATH)
        return None
    
    @pytest.fixture
    def predictor_with_shap(self, repository):
        """Create predictor with SHAP explanations enabled."""
        # Load model
        model, transformer = repository.load_deployed()
        
        # Load background data
        if config.TRAINING_DATA_PATH.exists():
            train_df = pd.read_csv(config.TRAINING_DATA_PATH)
            X_train = train_df.drop('churn', axis=1)
            
            feature_engineer = FeatureEngineer()
            feature_engineer.fit(X_train)
            X_train_eng = feature_engineer.transform(X_train)
            X_train_transformed = transformer.transform(X_train_eng)
            background_data = X_train_transformed[:config.SHAP_BACKGROUND_SAMPLES]
            
            return ChurnPredictor(
                repository=repository,
                enable_explanations=True,
                background_data=background_data
            )
        
        return ChurnPredictor(repository=repository)
    
    def test_model_loading(self, repository):
        """Test that models can be loaded for dashboard."""
        versions = repository.list_versions()
        assert len(versions) > 0, "No models found in repository"
        
        # Test loading deployed model
        model, transformer = repository.load_deployed()
        assert model is not None
        assert transformer is not None
    
    def test_test_data_available(self, test_data):
        """Test that test data is available."""
        assert test_data is not None, "Test data not found"
        assert len(test_data) > 0, "Test data is empty"
        
        # Check required columns
        required_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        for feature in required_features:
            assert feature in test_data.columns, f"Missing feature: {feature}"
    
    def test_batch_prediction_for_dashboard(self, predictor_with_shap, test_data):
        """Test batch prediction for dashboard overview."""
        if test_data is None:
            pytest.skip("Test data not available")
        
        # Prepare customers
        sample_size = min(10, len(test_data))
        customers = []
        for idx in range(sample_size):
            row = test_data.iloc[idx]
            features = row.drop('churn').to_dict() if 'churn' in row else row.to_dict()
            customers.append({
                'customer_id': f"CUST_{idx:04d}",
                **features
            })
        
        # Generate predictions
        predictions = predictor_with_shap.predict_batch(customers, include_explanations=False)
        
        assert len(predictions) == sample_size
        
        # Verify prediction structure
        for pred in predictions:
            assert hasattr(pred, 'customer_id')
            assert hasattr(pred, 'churn_probability')
            assert hasattr(pred, 'is_high_risk')
            assert 0 <= pred.churn_probability <= 1
            assert isinstance(pred.is_high_risk, bool)
    
    def test_single_prediction_with_explanation(self, predictor_with_shap, test_data):
        """Test single prediction with SHAP explanation for customer detail view."""
        if test_data is None:
            pytest.skip("Test data not available")
        
        # Prepare single customer
        row = test_data.iloc[0]
        features = row.drop('churn').to_dict() if 'churn' in row else row.to_dict()
        customer = {
            'customer_id': 'CUST_0000',
            **features
        }
        
        # Generate prediction with explanation
        try:
            result = predictor_with_shap.predict_single(customer, include_explanations=True)
            
            assert result.customer_id == 'CUST_0000'
            assert 0 <= result.churn_probability <= 1
            assert isinstance(result.is_high_risk, bool)
            
            # Check SHAP explanations if available
            if result.top_features:
                assert len(result.top_features) <= config.TOP_FEATURES_COUNT
                for feature_name, shap_value in result.top_features:
                    assert isinstance(feature_name, str)
                    assert isinstance(shap_value, (int, float))
        except ValueError as e:
            # SHAP might not be available
            if "background_data" not in str(e):
                raise
    
    def test_aggregate_statistics(self, predictor_with_shap, test_data):
        """Test aggregate statistics computation for statistics tab."""
        if test_data is None:
            pytest.skip("Test data not available")
        
        # Generate predictions
        sample_size = min(20, len(test_data))
        customers = []
        for idx in range(sample_size):
            row = test_data.iloc[idx]
            features = row.drop('churn').to_dict() if 'churn' in row else row.to_dict()
            customers.append({
                'customer_id': f"CUST_{idx:04d}",
                **features
            })
        
        predictions = predictor_with_shap.predict_batch(customers, include_explanations=False)
        
        # Compute statistics
        probabilities = [p.churn_probability for p in predictions]
        
        avg_prob = np.mean(probabilities)
        median_prob = np.median(probabilities)
        std_prob = np.std(probabilities)
        
        # Verify statistics are valid
        assert 0 <= avg_prob <= 1
        assert 0 <= median_prob <= 1
        assert std_prob >= 0
        
        # Test high-risk count
        high_risk_count = sum(p.is_high_risk for p in predictions)
        assert 0 <= high_risk_count <= len(predictions)
    
    def test_model_metadata_retrieval(self, repository):
        """Test model metadata retrieval for sidebar."""
        versions = repository.list_versions()
        assert len(versions) > 0
        
        # Get metadata for first version
        version = versions[0]
        metadata = repository.get_metadata(version.version)
        
        assert 'version' in metadata
        assert 'n_features' in metadata
        
        # Check if metrics are present
        if 'metrics' in metadata:
            metrics = metadata['metrics']
            assert 'recall' in metrics or 'precision' in metrics
    
    def test_confusion_matrix_computation(self, test_data):
        """Test confusion matrix computation for statistics tab."""
        if test_data is None or 'churn' not in test_data.columns:
            pytest.skip("Test data with labels not available")
        
        # Simple confusion matrix test
        actual = test_data['churn'].values[:20]
        predicted = (actual > 0.5).astype(int)  # Dummy predictions
        
        tp = sum((a == 1 and p == 1) for a, p in zip(actual, predicted))
        fp = sum((a == 0 and p == 1) for a, p in zip(actual, predicted))
        tn = sum((a == 0 and p == 0) for a, p in zip(actual, predicted))
        fn = sum((a == 1 and p == 0) for a, p in zip(actual, predicted))
        
        # Verify confusion matrix sums to total samples
        assert tp + fp + tn + fn == len(actual)
    
    def test_feature_importance_aggregation(self, predictor_with_shap, test_data):
        """Test feature importance aggregation for statistics tab."""
        if test_data is None:
            pytest.skip("Test data not available")
        
        # Generate predictions with explanations
        sample_size = min(10, len(test_data))
        customers = []
        for idx in range(sample_size):
            row = test_data.iloc[idx]
            features = row.drop('churn').to_dict() if 'churn' in row else row.to_dict()
            customers.append({
                'customer_id': f"CUST_{idx:04d}",
                **features
            })
        
        try:
            predictions = predictor_with_shap.predict_batch(customers, include_explanations=True)
            
            # Aggregate feature importance
            feature_importance = {}
            for pred in predictions:
                if pred.top_features:
                    for feature_name, shap_value in pred.top_features:
                        if feature_name not in feature_importance:
                            feature_importance[feature_name] = []
                        feature_importance[feature_name].append(abs(shap_value))
            
            if feature_importance:
                # Compute average importance
                avg_importance = {
                    feature: np.mean(values)
                    for feature, values in feature_importance.items()
                }
                
                assert len(avg_importance) > 0
                for feature, importance in avg_importance.items():
                    assert importance >= 0
        except ValueError as e:
            # SHAP might not be available
            if "background_data" not in str(e):
                raise
    
    def test_model_version_switching(self, repository):
        """Test switching between model versions."""
        versions = repository.list_versions()
        
        if len(versions) < 2:
            pytest.skip("Need at least 2 model versions for this test")
        
        # Load first version
        model1, transformer1 = repository.load(versions[0].version)
        assert model1 is not None
        
        # Load second version
        model2, transformer2 = repository.load(versions[1].version)
        assert model2 is not None
        
        # Verify they are different versions
        assert versions[0].version != versions[1].version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
