"""
Model repository for version management in Customer Churn Prediction System.
Handles storage, retrieval, and management of versioned models and transformers.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import joblib
from xgboost import XGBClassifier

import config
from services.preprocessing import DataTransformer


@dataclass
class ModelVersion:
    """Represents a versioned model with metadata."""
    version: str
    timestamp: str
    n_features: int
    metrics: Dict[str, Any]
    params: Dict[str, Any]
    features: Dict[str, List[str]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ModelNotFoundError(Exception):
    """Raised when requested model version doesn't exist."""
    pass


class ModelRepository:
    """
    Handles model persistence and retrieval with version management.
    Manages model artifacts, transformers, and metadata in a versioned repository.
    """
    
    def __init__(self, repository_dir: Optional[Path] = None, transformer_dir: Optional[Path] = None):
        """
        Initialize ModelRepository.
        
        Args:
            repository_dir: Directory for storing model versions. Defaults to config.MODEL_REPOSITORY_DIR
            transformer_dir: Directory for storing transformers. Defaults to config.TRANSFORMER_DIR
        """
        self.repository_dir = repository_dir if repository_dir is not None else config.MODEL_REPOSITORY_DIR
        self.transformer_dir = transformer_dir if transformer_dir is not None else config.TRANSFORMER_DIR
        
        # Ensure directories exist
        self.repository_dir.mkdir(parents=True, exist_ok=True)
        self.transformer_dir.mkdir(parents=True, exist_ok=True)
        
        # Track currently deployed version
        self._deployed_version_file = self.repository_dir / ".deployed_version"
    
    def save(
        self,
        model: XGBClassifier,
        transformer: DataTransformer,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save model, transformer, and metadata to repository.
        
        Args:
            model: Trained XGBClassifier model
            transformer: Fitted DataTransformer
            version: Optional version string. If None, generates timestamp-based version
            metadata: Optional metadata dictionary (metrics, features, etc.)
            
        Returns:
            Version string of saved model
            
        Raises:
            ValueError: If model or transformer is not fitted
        """
        # Validate inputs
        if not hasattr(model, 'n_features_in_'):
            raise ValueError("Cannot save unfitted model")
        if not transformer.is_fitted:
            raise ValueError("Cannot save unfitted transformer")
        
        # Generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            version = f"v_{timestamp}"
        
        # Create version directory
        version_dir = self.repository_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = version_dir / "model.pkl"
        joblib.dump(model, model_path)
        
        # Save transformer
        transformer_path = self.transformer_dir / f"{version}_transformer.pkl"
        transformer.save(str(transformer_path))
        
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        metadata_complete = {
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'n_features': int(model.n_features_in_),
            'params': model.get_params(),
            **metadata
        }
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_complete, f, indent=2, default=str)
        
        return version
    
    def load(self, version: str) -> Tuple[XGBClassifier, DataTransformer]:
        """
        Load model and transformer for specific version.
        
        Args:
            version: Version string to load
            
        Returns:
            Tuple of (model, transformer)
            
        Raises:
            ModelNotFoundError: If version doesn't exist
            FileNotFoundError: If model or transformer files are missing
        """
        # Check if version exists
        version_dir = self.repository_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(f"Model version '{version}' not found in repository")
        
        # Load model
        model_path = version_dir / "model.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for version '{version}'")
        model = joblib.load(model_path)
        
        # Load transformer
        transformer_path = self.transformer_dir / f"{version}_transformer.pkl"
        if not transformer_path.exists():
            raise FileNotFoundError(f"Transformer file not found for version '{version}'")
        transformer = DataTransformer.load(str(transformer_path))
        
        return model, transformer
    
    def list_versions(self) -> List[ModelVersion]:
        """
        List all available model versions with metadata.
        
        Returns:
            List of ModelVersion objects sorted by timestamp (newest first)
        """
        versions = []
        
        # Iterate through version directories
        for version_dir in self.repository_dir.iterdir():
            if version_dir.is_dir() and not version_dir.name.startswith('.'):
                metadata_path = version_dir / "metadata.json"
                
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Create ModelVersion object
                        model_version = ModelVersion(
                            version=metadata.get('version', version_dir.name),
                            timestamp=metadata.get('timestamp', ''),
                            n_features=metadata.get('n_features', 0),
                            metrics=metadata.get('metrics', {}),
                            params=metadata.get('params', {}),
                            features=metadata.get('features', {})
                        )
                        versions.append(model_version)
                    except (json.JSONDecodeError, KeyError) as e:
                        # Skip invalid metadata files
                        continue
        
        # Sort by timestamp (newest first)
        versions.sort(key=lambda v: v.timestamp, reverse=True)
        
        return versions
    
    def get_latest_version(self) -> str:
        """
        Get the latest model version based on timestamp.
        
        Returns:
            Version string of the latest model
            
        Raises:
            ModelNotFoundError: If no models exist in repository
        """
        versions = self.list_versions()
        
        if not versions:
            raise ModelNotFoundError("No models found in repository")
        
        # Return the first version (already sorted by timestamp, newest first)
        return versions[0].version
    
    def rollback(self, version: str) -> None:
        """
        Rollback to a previous model version by setting it as deployed.
        
        Args:
            version: Version string to rollback to
            
        Raises:
            ModelNotFoundError: If version doesn't exist
        """
        # Verify version exists
        version_dir = self.repository_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(f"Cannot rollback to non-existent version '{version}'")
        
        # Verify model and transformer files exist
        model_path = version_dir / "model.pkl"
        transformer_path = self.transformer_dir / f"{version}_transformer.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for version '{version}'")
        if not transformer_path.exists():
            raise FileNotFoundError(f"Transformer file not found for version '{version}'")
        
        # Set as deployed version
        with open(self._deployed_version_file, 'w') as f:
            f.write(version)
    
    def get_deployed_version(self) -> Optional[str]:
        """
        Get the currently deployed model version.
        
        Returns:
            Version string of deployed model, or None if not set
        """
        if self._deployed_version_file.exists():
            with open(self._deployed_version_file, 'r') as f:
                return f.read().strip()
        return None
    
    def load_deployed(self) -> Tuple[XGBClassifier, DataTransformer]:
        """
        Load the currently deployed model and transformer.
        Falls back to latest version if no deployed version is set.
        
        Returns:
            Tuple of (model, transformer)
            
        Raises:
            ModelNotFoundError: If no models exist in repository
        """
        deployed_version = self.get_deployed_version()
        
        if deployed_version is None:
            # Fall back to latest version
            deployed_version = self.get_latest_version()
        
        return self.load(deployed_version)
    
    def get_metadata(self, version: str) -> Dict[str, Any]:
        """
        Get metadata for a specific version.
        
        Args:
            version: Version string
            
        Returns:
            Metadata dictionary
            
        Raises:
            ModelNotFoundError: If version doesn't exist
        """
        version_dir = self.repository_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(f"Model version '{version}' not found in repository")
        
        metadata_path = version_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found for version '{version}'")
        
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    def delete_version(self, version: str) -> None:
        """
        Delete a specific model version from repository.
        
        Args:
            version: Version string to delete
            
        Raises:
            ModelNotFoundError: If version doesn't exist
            ValueError: If trying to delete the currently deployed version
        """
        # Check if version exists
        version_dir = self.repository_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(f"Model version '{version}' not found in repository")
        
        # Prevent deletion of deployed version
        deployed_version = self.get_deployed_version()
        if deployed_version == version:
            raise ValueError(f"Cannot delete currently deployed version '{version}'")
        
        # Delete model directory
        shutil.rmtree(version_dir)
        
        # Delete transformer file
        transformer_path = self.transformer_dir / f"{version}_transformer.pkl"
        if transformer_path.exists():
            transformer_path.unlink()
