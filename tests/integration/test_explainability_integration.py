"""
Integration tests for SHAP explainability service.
Tests end-to-end functionality of SHAP explanations with prediction service.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from services.prediction import ChurnPredictor, CustomerRecord
from services.model_repository import ModelRepository
from services.preprocessing import DataTransformer, FeatureEngineer
from services.explainability import SHAPExplainer
from xgboost import XGBClassifier
import config


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'churn': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def trained_model_with_background(sample_training_data):
    """Train a model and prepare background data for SHAP."""
    # Prepare data
    X = sample_training_data.drop('churn', axis=1)
    y = sample_training_data['churn']
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    X_engineered = feature_engineer.fit_transform(X)
    
    # Transform data
    transformer = DataTransformer(
        numerical_features=config.NUMERICAL_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES
    )
    X_transformed = transformer.fit_transform(X_engineered)
    
    # Train model
    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_transformed, y)
    
    # Use subset of training data as background
    background_data = X_transformed[:100]
    
    return model, transformer, feature_engineer, background_data


def test_shap_explainer_initialization(trained_model_with_background):
    """Test SHAP explainer can be initialized with model and background data."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    explainer = SHAPExplainer(model, background_data)
    
    assert explainer is not None
    assert explainer.model is model
    assert explainer.background_data.shape == background_data.shape
    assert hasattr(explainer, 'base_value')
    assert isinstance(explainer.base_value, float)


def test_shap_explain_single_prediction(trained_model_with_background):
    """Test SHAP explanation for a single prediction."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    explainer = SHAPExplainer(model, background_data)
    
    # Create test sample
    test_sample = background_data[0:1]
    
    # Get SHAP values
    shap_values = explainer.explain(test_sample)
    
    assert shap_values.shape == test_sample.shape
    assert isinstance(shap_values, np.ndarray)
    assert not np.any(np.isnan(shap_values))


def test_shap_explain_batch(trained_model_with_background):
    """Test SHAP explanation for batch of predictions."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    explainer = SHAPExplainer(model, background_data)
    
    # Create test batch
    test_batch = background_data[:10]
    
    # Get SHAP values
    shap_values = explainer.explain(test_batch)
    
    assert shap_values.shape == test_batch.shape
    assert isinstance(shap_values, np.ndarray)


def test_get_top_features(trained_model_with_background):
    """Test getting top contributing features."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    explainer = SHAPExplainer(model, background_data)
    feature_names = transformer.get_feature_names_out()
    
    # Get SHAP values for one sample
    test_sample = background_data[0:1]
    shap_values = explainer.explain(test_sample)
    
    # Get top features
    top_features = explainer.get_top_features(shap_values[0], feature_names, n=5)
    
    assert len(top_features) == 5
    assert all(isinstance(item, tuple) for item in top_features)
    assert all(len(item) == 2 for item in top_features)
    
    # Check sorted by absolute value
    abs_values = [abs(value) for _, value in top_features]
    assert abs_values == sorted(abs_values, reverse=True)


def test_validate_shap_sum(trained_model_with_background):
    """Test SHAP additivity property validation."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    explainer = SHAPExplainer(model, background_data)
    
    # Get prediction and SHAP values
    test_sample = background_data[0:1]
    shap_values = explainer.explain(test_sample)
    
    # For XGBoost with TreeExplainer, predictions are in log-odds space
    # Get raw prediction (margin/log-odds) from the model
    prediction_margin = model.predict(test_sample, output_margin=True)[0]
    
    # Validate additivity in log-odds space
    # Use tolerance of 0.02 to account for numerical precision in SHAP calculations
    is_valid = explainer.validate_shap_sum(shap_values[0], prediction_margin, tolerance=0.02)
    
    # Also verify the difference is small relative to the magnitude
    shap_sum = np.sum(shap_values[0])
    expected_diff = prediction_margin - explainer.base_value
    relative_error = abs(shap_sum - expected_diff) / (abs(expected_diff) + 1e-10)
    
    assert is_valid, f"SHAP values should sum to prediction - base_value within tolerance"
    assert relative_error < 0.01, f"Relative error {relative_error} should be < 1%"


def test_churn_predictor_with_explanations(trained_model_with_background, tmp_path):
    """Test ChurnPredictor with SHAP explanations enabled."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    # Save model to repository
    repository = ModelRepository(
        repository_dir=tmp_path / "models",
        transformer_dir=tmp_path / "transformers"
    )
    version = repository.save(model, transformer, metadata={'test': True})
    
    # Create predictor with explanations enabled
    predictor = ChurnPredictor(
        model_version=version,
        repository=repository,
        enable_explanations=True,
        background_data=background_data
    )
    
    # Create test customer
    customer = CustomerRecord(
        customer_id="test_001",
        features={
            'tenure_months': 24,
            'monthly_charges': 75.5,
            'total_charges': 1800.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
    )
    
    # Get prediction with explanations
    result = predictor.predict_single(customer, include_explanations=True)
    
    assert result.customer_id == "test_001"
    assert 0 <= result.churn_probability <= 1
    assert isinstance(result.is_high_risk, bool)
    assert result.top_features is not None
    assert len(result.top_features) == config.TOP_FEATURES_COUNT
    assert all(isinstance(item, tuple) for item in result.top_features)


def test_churn_predictor_batch_with_explanations(trained_model_with_background, tmp_path):
    """Test batch predictions with SHAP explanations."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    # Save model to repository
    repository = ModelRepository(
        repository_dir=tmp_path / "models",
        transformer_dir=tmp_path / "transformers"
    )
    version = repository.save(model, transformer, metadata={'test': True})
    
    # Create predictor with explanations enabled
    predictor = ChurnPredictor(
        model_version=version,
        repository=repository,
        enable_explanations=True,
        background_data=background_data
    )
    
    # Create test customers
    customers = [
        {
            'customer_id': f'test_{i:03d}',
            'tenure_months': 12 + i * 6,
            'monthly_charges': 50.0 + i * 10,
            'total_charges': 600.0 + i * 120,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
        for i in range(5)
    ]
    
    # Get batch predictions with explanations
    results = predictor.predict_batch(customers, include_explanations=True)
    
    assert len(results) == 5
    for result in results:
        assert result.top_features is not None
        assert len(result.top_features) == config.TOP_FEATURES_COUNT


def test_churn_predictor_without_explanations(trained_model_with_background, tmp_path):
    """Test that predictions work without explanations (backward compatibility)."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    # Save model to repository
    repository = ModelRepository(
        repository_dir=tmp_path / "models",
        transformer_dir=tmp_path / "transformers"
    )
    version = repository.save(model, transformer, metadata={'test': True})
    
    # Create predictor WITHOUT explanations
    predictor = ChurnPredictor(
        model_version=version,
        repository=repository,
        enable_explanations=False
    )
    
    # Create test customer
    customer = CustomerRecord(
        customer_id="test_001",
        features={
            'tenure_months': 24,
            'monthly_charges': 75.5,
            'total_charges': 1800.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
    )
    
    # Get prediction without explanations
    result = predictor.predict_single(customer, include_explanations=False)
    
    assert result.customer_id == "test_001"
    assert 0 <= result.churn_probability <= 1
    assert result.top_features is None


def test_explanation_error_when_not_enabled(trained_model_with_background, tmp_path):
    """Test that requesting explanations without enabling them raises error."""
    model, transformer, feature_engineer, background_data = trained_model_with_background
    
    # Save model to repository
    repository = ModelRepository(
        repository_dir=tmp_path / "models",
        transformer_dir=tmp_path / "transformers"
    )
    version = repository.save(model, transformer, metadata={'test': True})
    
    # Create predictor WITHOUT explanations
    predictor = ChurnPredictor(
        model_version=version,
        repository=repository,
        enable_explanations=False
    )
    
    # Create test customer
    customer = CustomerRecord(
        customer_id="test_001",
        features={
            'tenure_months': 24,
            'monthly_charges': 75.5,
            'total_charges': 1800.0,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
    )
    
    # Try to get prediction with explanations - should raise error
    with pytest.raises(ValueError, match="Explanations requested but explainer not initialized"):
        predictor.predict_single(customer, include_explanations=True)
