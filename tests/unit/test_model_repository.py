"""
Unit tests for ModelRepository class.
Tests model versioning, storage, retrieval, and rollback functionality.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import json
from xgboost import XGBClassifier
import numpy as np
import pandas as pd

from services.model_repository import (
    ModelRepository,
    ModelVersion,
    ModelNotFoundError
)
from services.preprocessing import DataTransformer


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
def sample_model():
    """Create a simple fitted XGBoost model for testing."""
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model = XGBClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def sample_transformer():
    """Create a fitted DataTransformer for testing."""
    X = pd.DataFrame({
        'num1': np.random.rand(100),
        'num2': np.random.rand(100),
        'cat1': np.random.choice(['A', 'B', 'C'], 100)
    })
    transformer = DataTransformer(
        numerical_features=['num1', 'num2'],
        categorical_features=['cat1']
    )
    transformer.fit(X)
    return transformer


class TestModelRepository:
    """Test suite for ModelRepository class."""
    
    def test_save_model_with_auto_version(self, temp_dirs, sample_model, sample_transformer):
        """Test saving model with automatic version generation."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        metadata = {'metrics': {'recall': 0.85}}
        version = repo.save(sample_model, sample_transformer, metadata=metadata)
        
        # Check version format
        assert version.startswith('v_')
        assert len(version) > 2
        
        # Check files exist
        assert (repo_dir / version / "model.pkl").exists()
        assert (repo_dir / version / "metadata.json").exists()
        assert (trans_dir / f"{version}_transformer.pkl").exists()
    
    def test_save_model_with_custom_version(self, temp_dirs, sample_model, sample_transformer):
        """Test saving model with custom version string."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        custom_version = "v1.0.0"
        version = repo.save(sample_model, sample_transformer, version=custom_version)
        
        assert version == custom_version
        assert (repo_dir / custom_version / "model.pkl").exists()
    
    def test_save_unfitted_model_raises_error(self, temp_dirs, sample_transformer):
        """Test that saving unfitted model raises ValueError."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        unfitted_model = XGBClassifier()
        
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            repo.save(unfitted_model, sample_transformer)
    
    def test_save_unfitted_transformer_raises_error(self, temp_dirs, sample_model):
        """Test that saving unfitted transformer raises ValueError."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        unfitted_transformer = DataTransformer(
            numerical_features=['num1'],
            categorical_features=['cat1']
        )
        
        with pytest.raises(ValueError, match="Cannot save unfitted transformer"):
            repo.save(sample_model, unfitted_transformer)
    
    def test_load_existing_version(self, temp_dirs, sample_model, sample_transformer):
        """Test loading an existing model version."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        # Save model
        version = repo.save(sample_model, sample_transformer)
        
        # Load model
        loaded_model, loaded_transformer = repo.load(version)
        
        assert loaded_model is not None
        assert loaded_transformer is not None
        assert loaded_transformer.is_fitted
    
    def test_load_nonexistent_version_raises_error(self, temp_dirs):
        """Test that loading non-existent version raises ModelNotFoundError."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        with pytest.raises(ModelNotFoundError, match="not found in repository"):
            repo.load("nonexistent_version")
    
    def test_list_versions_empty_repository(self, temp_dirs):
        """Test listing versions in empty repository."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        versions = repo.list_versions()
        assert versions == []
    
    def test_list_versions_multiple_models(self, temp_dirs, sample_model, sample_transformer):
        """Test listing multiple model versions."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        # Save multiple versions
        v1 = repo.save(sample_model, sample_transformer, version="v1")
        v2 = repo.save(sample_model, sample_transformer, version="v2")
        v3 = repo.save(sample_model, sample_transformer, version="v3")
        
        versions = repo.list_versions()
        
        assert len(versions) == 3
        version_strings = [v.version for v in versions]
        assert "v1" in version_strings
        assert "v2" in version_strings
        assert "v3" in version_strings
    
    def test_get_latest_version(self, temp_dirs, sample_model, sample_transformer):
        """Test getting the latest model version."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        # Save versions with different timestamps
        import time
        repo.save(sample_model, sample_transformer, version="v1")
        time.sleep(0.1)
        repo.save(sample_model, sample_transformer, version="v2")
        time.sleep(0.1)
        latest_version = repo.save(sample_model, sample_transformer, version="v3")
        
        retrieved_latest = repo.get_latest_version()
        assert retrieved_latest == "v3"
    
    def test_get_latest_version_empty_repository_raises_error(self, temp_dirs):
        """Test that getting latest version from empty repository raises error."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        with pytest.raises(ModelNotFoundError, match="No models found"):
            repo.get_latest_version()
    
    def test_rollback_to_existing_version(self, temp_dirs, sample_model, sample_transformer):
        """Test rollback to a previous version."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        # Save multiple versions
        v1 = repo.save(sample_model, sample_transformer, version="v1")
        v2 = repo.save(sample_model, sample_transformer, version="v2")
        
        # Rollback to v1
        repo.rollback("v1")
        
        # Check deployed version
        deployed = repo.get_deployed_version()
        assert deployed == "v1"
    
    def test_rollback_to_nonexistent_version_raises_error(self, temp_dirs):
        """Test that rollback to non-existent version raises error."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        with pytest.raises(ModelNotFoundError, match="Cannot rollback"):
            repo.rollback("nonexistent_version")
    
    def test_get_deployed_version_none_when_not_set(self, temp_dirs):
        """Test that deployed version is None when not set."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        deployed = repo.get_deployed_version()
        assert deployed is None
    
    def test_load_deployed_falls_back_to_latest(self, temp_dirs, sample_model, sample_transformer):
        """Test that load_deployed falls back to latest when no deployed version set."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        # Save a version
        version = repo.save(sample_model, sample_transformer, version="v1")
        
        # Load deployed (should fall back to latest)
        model, transformer = repo.load_deployed()
        
        assert model is not None
        assert transformer is not None
    
    def test_get_metadata(self, temp_dirs, sample_model, sample_transformer):
        """Test retrieving metadata for a version."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        metadata = {'metrics': {'recall': 0.87, 'precision': 0.75}}
        version = repo.save(sample_model, sample_transformer, metadata=metadata)
        
        retrieved_metadata = repo.get_metadata(version)
        
        assert 'version' in retrieved_metadata
        assert 'timestamp' in retrieved_metadata
        assert 'metrics' in retrieved_metadata
        assert retrieved_metadata['metrics']['recall'] == 0.87
    
    def test_delete_version(self, temp_dirs, sample_model, sample_transformer):
        """Test deleting a model version."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        # Save two versions
        v1 = repo.save(sample_model, sample_transformer, version="v1")
        v2 = repo.save(sample_model, sample_transformer, version="v2")
        
        # Delete v1
        repo.delete_version("v1")
        
        # Check v1 is gone
        with pytest.raises(ModelNotFoundError):
            repo.load("v1")
        
        # Check v2 still exists
        model, transformer = repo.load("v2")
        assert model is not None
    
    def test_delete_deployed_version_raises_error(self, temp_dirs, sample_model, sample_transformer):
        """Test that deleting deployed version raises error."""
        repo_dir, trans_dir = temp_dirs
        repo = ModelRepository(repository_dir=repo_dir, transformer_dir=trans_dir)
        
        # Save and deploy a version
        version = repo.save(sample_model, sample_transformer, version="v1")
        repo.rollback("v1")
        
        # Try to delete deployed version
        with pytest.raises(ValueError, match="Cannot delete currently deployed version"):
            repo.delete_version("v1")


class TestModelVersion:
    """Test suite for ModelVersion dataclass."""
    
    def test_model_version_creation(self):
        """Test creating a ModelVersion instance."""
        version = ModelVersion(
            version="v1.0.0",
            timestamp="2025-12-07T10:00:00",
            n_features=10,
            metrics={'recall': 0.85},
            params={'max_depth': 6},
            features={'numerical': ['age', 'income']}
        )
        
        assert version.version == "v1.0.0"
        assert version.n_features == 10
        assert version.metrics['recall'] == 0.85
    
    def test_model_version_to_dict(self):
        """Test converting ModelVersion to dictionary."""
        version = ModelVersion(
            version="v1.0.0",
            timestamp="2025-12-07T10:00:00",
            n_features=10,
            metrics={'recall': 0.85},
            params={'max_depth': 6},
            features={'numerical': ['age']}
        )
        
        version_dict = version.to_dict()
        
        assert isinstance(version_dict, dict)
        assert version_dict['version'] == "v1.0.0"
        assert version_dict['n_features'] == 10
