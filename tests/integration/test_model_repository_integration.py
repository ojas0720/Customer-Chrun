"""
Integration tests for ModelRepository with existing training pipeline.
Tests end-to-end workflow of training, saving, loading, and using models.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from services.model_training import ModelTrainer
from services.preprocessing import DataTransformer
from services.model_repository import ModelRepository
from services.parameter_validation import ParameterValidator


@pytest.fixture
def temp_dirs():
    """Create temporary directories for testing."""
    temp_repo = tempfile.mkdtemp()
    temp_transformer = tempfile.mkdtemp()
    yield Path(temp_repo), Path(temp_transformer)
    # Cleanup
    shutil.rmtree(temp_repo, ignore_errors=True)
    shutil.rmtree(temp_transformer, ignore_errors=True)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    
    data = pd.DataFrame({
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(100, 8000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'churn': np.random.randint(0, 2, n_samples)
    })
    
    return data


class TestModelRepositoryIntegration:
    """Integration tests for ModelRepository with training pipeline."""
    
    def test_end_to_end_train_save_load_predict(self, temp_dirs, sample_data):
        """Test complete workflow: train -> save -> load -> predict."""
        repo_dir, trans_dir = temp_dirs
        
        # Split data
        train_data = sample_data.iloc[:150]
        test_data = sample_data.iloc[150:]
        
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        X_test = test_data.drop('churn', axis=1)
        y_test = test_data['churn']
        
        # 1. Train model
        transformer = DataTransformer(
            numerical_features=['tenure_months', 'monthly_charges', 'total_charges'],
            categorical_features=['contract_type', 'payment_method', 'internet_service']
        )
        X_train_transformed = transformer.fit_transform(X_train)
        
        trainer = ModelTrainer()
        model = trainer.train(X_train_transformed, y_train)
        
        # 2. Evaluate model
        X_test_transformed = transformer.transform(X_test)
        metrics = trainer.evaluate(model, X_test_transformed, y_test)
        
        # 3. Save to repository
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        version = repo.save(
            model,
            transformer,
            version="v1.0.0",
            metadata={'metrics': metrics}
        )
        
        assert version == "v1.0.0"
        
        # 4. Load from repository
        loaded_model, loaded_transformer = repo.load("v1.0.0")
        
        # 5. Make predictions with loaded model
        X_new = X_test.iloc[:5]
        X_new_transformed = loaded_transformer.transform(X_new)
        predictions = loaded_model.predict_proba(X_new_transformed)
        
        # Verify predictions are valid probabilities
        assert predictions.shape[0] == 5
        assert predictions.shape[1] == 2
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 1)
        assert np.allclose(predictions.sum(axis=1), 1.0)
    
    def test_parameter_validation_before_training(self, sample_data):
        """Test that parameter validation works before training."""
        train_data = sample_data.iloc[:150]
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        
        # Transform data
        transformer = DataTransformer(
            numerical_features=['tenure_months', 'monthly_charges', 'total_charges'],
            categorical_features=['contract_type', 'payment_method', 'internet_service']
        )
        X_train_transformed = transformer.fit_transform(X_train)
        
        # Validate parameters before training
        validator = ParameterValidator()
        
        # Valid parameters
        valid_params = {
            'learning_rate': 0.1,
            'max_depth': 6,
            'n_estimators': 50,
            'objective': 'binary:logistic'
        }
        assert validator.validate(valid_params) is True
        
        # Train with valid parameters
        trainer = ModelTrainer(params=valid_params)
        model = trainer.train(X_train_transformed, y_train)
        assert model is not None
        
        # Invalid parameters should be caught
        invalid_params = {
            'learning_rate': -0.1,  # Invalid
            'max_depth': 6
        }
        
        with pytest.raises(Exception):  # ParameterValidationError
            validator.validate(invalid_params)
    
    def test_version_management_workflow(self, temp_dirs, sample_data):
        """Test version management: save multiple versions, rollback."""
        repo_dir, trans_dir = temp_dirs
        
        train_data = sample_data.iloc[:150]
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        
        # Transform data
        transformer = DataTransformer(
            numerical_features=['tenure_months', 'monthly_charges', 'total_charges'],
            categorical_features=['contract_type', 'payment_method', 'internet_service']
        )
        X_train_transformed = transformer.fit_transform(X_train)
        
        # Train first model
        trainer1 = ModelTrainer(params={'n_estimators': 50, 'max_depth': 3})
        model1 = trainer1.train(X_train_transformed, y_train)
        
        # Train second model with different params
        trainer2 = ModelTrainer(params={'n_estimators': 100, 'max_depth': 6})
        model2 = trainer2.train(X_train_transformed, y_train)
        
        # Save both versions
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        v1 = repo.save(model1, transformer, version="v1.0.0")
        v2 = repo.save(model2, transformer, version="v2.0.0")
        
        # List versions
        versions = repo.list_versions()
        assert len(versions) == 2
        
        # Get latest version
        latest = repo.get_latest_version()
        assert latest == "v2.0.0"
        
        # Rollback to v1
        repo.rollback("v1.0.0")
        deployed = repo.get_deployed_version()
        assert deployed == "v1.0.0"
        
        # Load deployed version
        deployed_model, deployed_transformer = repo.load_deployed()
        assert deployed_model is not None
        assert deployed_transformer is not None
    
    def test_metadata_persistence(self, temp_dirs, sample_data):
        """Test that metadata is correctly saved and retrieved."""
        repo_dir, trans_dir = temp_dirs
        
        train_data = sample_data.iloc[:150]
        test_data = sample_data.iloc[150:]
        
        X_train = train_data.drop('churn', axis=1)
        y_train = train_data['churn']
        X_test = test_data.drop('churn', axis=1)
        y_test = test_data['churn']
        
        # Train and evaluate
        transformer = DataTransformer(
            numerical_features=['tenure_months', 'monthly_charges', 'total_charges'],
            categorical_features=['contract_type', 'payment_method', 'internet_service']
        )
        X_train_transformed = transformer.fit_transform(X_train)
        X_test_transformed = transformer.transform(X_test)
        
        trainer = ModelTrainer()
        model = trainer.train(X_train_transformed, y_train)
        metrics = trainer.evaluate(model, X_test_transformed, y_test)
        
        # Save with metadata
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        custom_metadata = {
            'metrics': metrics,
            'features': {
                'numerical': ['tenure_months', 'monthly_charges', 'total_charges'],
                'categorical': ['contract_type', 'payment_method', 'internet_service']
            },
            'description': 'Test model for integration testing'
        }
        
        version = repo.save(model, transformer, metadata=custom_metadata)
        
        # Retrieve metadata
        retrieved_metadata = repo.get_metadata(version)
        
        assert 'metrics' in retrieved_metadata
        assert 'features' in retrieved_metadata
        assert 'description' in retrieved_metadata
        assert retrieved_metadata['description'] == 'Test model for integration testing'
        assert 'precision' in retrieved_metadata['metrics']
        assert 'recall' in retrieved_metadata['metrics']
