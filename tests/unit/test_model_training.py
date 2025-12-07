"""
Unit tests for model training service.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json
from xgboost import XGBClassifier

from services.model_training import ModelTrainer


class TestModelTrainer:
    """Unit tests for ModelTrainer class."""
    
    def test_train_creates_fitted_model(self):
        """Test that train method creates a fitted model."""
        trainer = ModelTrainer()
        
        # Create simple training data
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        
        model = trainer.train(X_train, y_train)
        
        assert isinstance(model, XGBClassifier)
        assert hasattr(model, 'n_features_in_')
        assert model.n_features_in_ == 2
    
    def test_train_with_empty_data_raises_error(self):
        """Test that training with empty data raises ValueError."""
        trainer = ModelTrainer()
        
        X_train = np.array([]).reshape(0, 2)
        y_train = np.array([])
        
        with pytest.raises(ValueError, match="Training data is empty"):
            trainer.train(X_train, y_train)
    
    def test_train_with_mismatched_lengths_raises_error(self):
        """Test that mismatched X and y lengths raise ValueError."""
        trainer = ModelTrainer()
        
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1, 0])  # Wrong length
        
        with pytest.raises(ValueError, match="different lengths"):
            trainer.train(X_train, y_train)
    
    def test_train_with_custom_params(self):
        """Test training with custom hyperparameters."""
        custom_params = {
            'max_depth': 3,
            'learning_rate': 0.05,
            'n_estimators': 50,
            'random_state': 42
        }
        trainer = ModelTrainer(params=custom_params)
        
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        
        model = trainer.train(X_train, y_train)
        
        assert model.max_depth == 3
        assert model.learning_rate == 0.05
        assert model.n_estimators == 50
    
    def test_evaluate_returns_all_metrics(self):
        """Test that evaluate returns all required metrics."""
        trainer = ModelTrainer()
        
        # Train a simple model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model = trainer.train(X_train, y_train)
        
        # Evaluate on test data
        X_test = np.array([[2, 3], [4, 5]])
        y_test = np.array([0, 1])
        
        metrics = trainer.evaluate(model, X_test, y_test)
        
        # Check all required metrics are present
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'confusion_matrix' in metrics
        
        # Check metric types and ranges
        assert isinstance(metrics['precision'], float)
        assert isinstance(metrics['recall'], float)
        assert isinstance(metrics['f1_score'], float)
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_evaluate_with_empty_data_raises_error(self):
        """Test that evaluate with empty data raises ValueError."""
        trainer = ModelTrainer()
        
        # Train a model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model = trainer.train(X_train, y_train)
        
        # Try to evaluate with empty data
        X_test = np.array([]).reshape(0, 2)
        y_test = np.array([])
        
        with pytest.raises(ValueError, match="Test data is empty"):
            trainer.evaluate(model, X_test, y_test)
    
    def test_evaluate_with_mismatched_lengths_raises_error(self):
        """Test that evaluate with mismatched lengths raises ValueError."""
        trainer = ModelTrainer()
        
        # Train a model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model = trainer.train(X_train, y_train)
        
        # Try to evaluate with mismatched lengths
        X_test = np.array([[2, 3], [4, 5]])
        y_test = np.array([0, 1, 0])  # Wrong length
        
        with pytest.raises(ValueError, match="different lengths"):
            trainer.evaluate(model, X_test, y_test)
    
    def test_save_model_creates_files(self):
        """Test that save_model creates model and metadata files."""
        trainer = ModelTrainer()
        
        # Train a model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model = trainer.train(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Save model
            version = trainer.save_model(
                model,
                version="test_v1",
                model_dir=model_dir
            )
            
            assert version == "test_v1"
            
            # Check files exist
            version_dir = model_dir / "test_v1"
            assert version_dir.exists()
            assert (version_dir / "model.pkl").exists()
            assert (version_dir / "metadata.json").exists()
    
    def test_save_model_generates_timestamp_version(self):
        """Test that save_model generates timestamp-based version if not provided."""
        trainer = ModelTrainer()
        
        # Train a model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model = trainer.train(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Save model without version
            version = trainer.save_model(model, model_dir=model_dir)
            
            # Check version format (should start with v_)
            assert version.startswith("v_")
            assert len(version) > 2
    
    def test_save_model_stores_metadata(self):
        """Test that save_model stores metadata correctly."""
        trainer = ModelTrainer()
        
        # Train a model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 1, 0, 1])
        model = trainer.train(X_train, y_train)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            # Save model with custom metadata
            custom_metadata = {
                'training_samples': 4,
                'test_recall': 0.92
            }
            version = trainer.save_model(
                model,
                version="test_v1",
                metadata=custom_metadata,
                model_dir=model_dir
            )
            
            # Load and check metadata
            metadata_path = model_dir / "test_v1" / "metadata.json"
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert metadata['version'] == "test_v1"
            assert 'timestamp' in metadata
            assert metadata['n_features'] == 2
            assert metadata['training_samples'] == 4
            assert metadata['test_recall'] == 0.92
    
    def test_save_unfitted_model_raises_error(self):
        """Test that saving unfitted model raises ValueError."""
        trainer = ModelTrainer()
        
        # Create unfitted model
        model = XGBClassifier()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            
            with pytest.raises(ValueError, match="Cannot save unfitted model"):
                trainer.save_model(model, model_dir=model_dir)
    
    def test_confusion_matrix_correctness(self):
        """Test that confusion matrix is computed correctly."""
        trainer = ModelTrainer()
        
        # Create a simple dataset where we can predict outcomes
        X_train = np.array([[1], [2], [3], [4], [5], [6]])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        
        model = trainer.train(X_train, y_train)
        
        # Test with known outcomes
        X_test = np.array([[1], [6]])
        y_test = np.array([0, 1])
        
        metrics = trainer.evaluate(model, X_test, y_test)
        
        # Confusion matrix should be 2x2
        conf_matrix = metrics['confusion_matrix']
        assert len(conf_matrix) == 2
        assert len(conf_matrix[0]) == 2
        
        # Sum of confusion matrix should equal number of test samples
        total = sum(sum(row) for row in conf_matrix)
        assert total == len(y_test)
