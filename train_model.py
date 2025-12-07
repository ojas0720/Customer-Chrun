"""
Training script for Customer Churn Prediction System.
Loads data, preprocesses it, trains XGBoost model, evaluates performance,
and saves the trained model with versioning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import sys

import config
from services.preprocessing import DataValidator, DataTransformer, FeatureEngineer
from services.model_training import ModelTrainer


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_training_data(data_path: Path) -> pd.DataFrame:
    """
    Load training data from CSV file.
    
    Args:
        data_path: Path to training data CSV
        
    Returns:
        DataFrame containing training data
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Please provide training data or run data generation script."
        )
    
    logger.info(f"Loading training data from {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Loaded {len(data)} records with {len(data.columns)} columns")
    
    return data


def validate_data(data: pd.DataFrame) -> None:
    """
    Validate training data meets requirements.
    
    Args:
        data: Training dataframe to validate
        
    Raises:
        ValidationError: If validation fails
    """
    logger.info("Validating training data")
    
    # Create validator
    all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    validator = DataValidator(
        required_features=all_features,
        numerical_features=config.NUMERICAL_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
        target_variable=config.TARGET_VARIABLE
    )
    
    # Validate
    validator.validate_for_training(data)
    logger.info("Data validation passed")


def preprocess_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, DataTransformer]:
    """
    Preprocess training and test data.
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_transformed, X_test_transformed, fitted_transformer)
    """
    logger.info("Starting data preprocessing")
    
    # Feature engineering
    logger.info("Applying feature engineering")
    feature_engineer = FeatureEngineer()
    X_train_engineered = feature_engineer.fit_transform(X_train)
    X_test_engineered = feature_engineer.transform(X_test)
    
    # Get all feature columns (original + derived)
    all_numerical_features = [
        col for col in X_train_engineered.columns
        if col in config.NUMERICAL_FEATURES or 
        pd.api.types.is_numeric_dtype(X_train_engineered[col])
    ]
    all_categorical_features = [
        col for col in X_train_engineered.columns
        if col in config.CATEGORICAL_FEATURES
    ]
    
    # Transform data
    logger.info("Applying data transformations (imputation, encoding, scaling)")
    transformer = DataTransformer(
        numerical_features=all_numerical_features,
        categorical_features=all_categorical_features
    )
    
    X_train_transformed = transformer.fit_transform(X_train_engineered)
    X_test_transformed = transformer.transform(X_test_engineered)
    
    logger.info(
        f"Preprocessing complete. "
        f"Training shape: {X_train_transformed.shape}, "
        f"Test shape: {X_test_transformed.shape}"
    )
    
    return X_train_transformed, X_test_transformed, transformer


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> tuple[object, dict]:
    """
    Train XGBoost model and evaluate performance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    logger.info("Starting model training")
    
    # Create trainer with hyperparameters optimized for recall
    # Adjust scale_pos_weight to handle class imbalance and boost recall
    params = config.XGBOOST_PARAMS.copy()
    
    # Calculate class imbalance ratio and boost it to favor recall
    n_positive = np.sum(y_train == 1)
    n_negative = np.sum(y_train == 0)
    if n_positive > 0:
        # Multiply by 7.5 to strongly favor recall at the cost of some precision
        scale_pos_weight = (n_negative / n_positive) * 7.5
        params['scale_pos_weight'] = scale_pos_weight
        logger.info(f"Setting scale_pos_weight to {scale_pos_weight:.2f} for high recall")
    
    trainer = ModelTrainer(params=params)
    
    # Train model
    model = trainer.train(X_train, y_train)
    logger.info("Model training complete")
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    metrics = trainer.evaluate(model, X_test, y_test)
    
    # Log metrics
    logger.info(f"Model Performance Metrics:")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
    logger.info(f"  Confusion Matrix: {metrics['confusion_matrix']}")
    
    return model, metrics


def verify_recall_threshold(metrics: dict) -> None:
    """
    Verify that model meets minimum recall threshold.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        
    Raises:
        ValueError: If recall is below threshold
    """
    recall = metrics['recall']
    threshold = config.MIN_RECALL_THRESHOLD
    
    if recall < threshold:
        logger.error(
            f"Model recall {recall:.4f} is below minimum threshold {threshold:.4f}"
        )
        raise ValueError(
            f"Model does not meet minimum recall requirement. "
            f"Got {recall:.4f}, required {threshold:.4f}"
        )
    
    logger.info(f"✓ Model meets recall threshold: {recall:.4f} >= {threshold:.4f}")


def save_artifacts(
    model: object,
    transformer: DataTransformer,
    metrics: dict,
    version: str = None
) -> str:
    """
    Save trained model and transformer with metadata.
    
    Args:
        model: Trained model
        transformer: Fitted data transformer
        metrics: Evaluation metrics
        version: Optional version string
        
    Returns:
        Version string of saved model
    """
    logger.info("Saving model artifacts")
    
    # Save model
    trainer = ModelTrainer()
    model_version = trainer.save_model(
        model,
        version=version,
        metadata={
            'metrics': metrics,
            'features': {
                'numerical': transformer.numerical_features,
                'categorical': transformer.categorical_features
            }
        }
    )
    
    logger.info(f"Model saved with version: {model_version}")
    
    # Save transformer
    transformer_path = config.TRANSFORMER_DIR / f"{model_version}_transformer.pkl"
    transformer.save(str(transformer_path))
    logger.info(f"Transformer saved to: {transformer_path}")
    
    return model_version


def main():
    """Main training pipeline."""
    try:
        logger.info("=" * 60)
        logger.info("Starting Customer Churn Prediction Model Training")
        logger.info("=" * 60)
        
        # Load data
        data = load_training_data(config.TRAINING_DATA_PATH)
        
        # Validate data
        validate_data(data)
        
        # Split features and target
        all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        X = data[all_features]
        y = data[config.TARGET_VARIABLE]
        
        # Split into train and test sets
        logger.info(f"Splitting data (test_size={config.TEST_SIZE})")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.TEST_SIZE,
            random_state=config.RANDOM_STATE,
            stratify=y
        )
        logger.info(
            f"Train set: {len(X_train)} samples, "
            f"Test set: {len(X_test)} samples"
        )
        
        # Preprocess data
        X_train_transformed, X_test_transformed, transformer = preprocess_data(
            X_train, X_test
        )
        
        # Train and evaluate model
        model, metrics = train_and_evaluate(
            X_train_transformed, y_train.values,
            X_test_transformed, y_test.values
        )
        
        # Verify recall threshold
        verify_recall_threshold(metrics)
        
        # Save artifacts
        version = save_artifacts(model, transformer, metrics)
        
        logger.info("=" * 60)
        logger.info(f"Training Complete! Model version: {version}")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        logger.info(
            "To generate synthetic training data, run: "
            "python generate_sample_data.py"
        )
        return 1
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
