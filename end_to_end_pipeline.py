"""
End-to-End Pipeline for Customer Churn Prediction System.

This script demonstrates the complete workflow:
1. Train model on synthetic data
2. Generate predictions for test customers
3. Compute SHAP explanations
4. Store predictions and evaluate performance

This serves as both a demonstration and validation of the entire system.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import sys
from datetime import datetime

import config
from services.preprocessing import DataValidator, DataTransformer, FeatureEngineer
from services.model_training import ModelTrainer
from services.model_repository import ModelRepository
from services.prediction import ChurnPredictor
from services.explainability import SHAPExplainer
from services.monitoring import ModelEvaluator, PredictionStore


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('end_to_end_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EndToEndPipeline:
    """Orchestrates the complete churn prediction pipeline."""
    
    def __init__(self):
        """Initialize pipeline components."""
        self.model = None
        self.transformer = None
        self.model_version = None
        self.predictor = None
        self.explainer = None
        self.evaluator = ModelEvaluator()
        self.prediction_store = PredictionStore()
        
    def step1_load_and_validate_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Step 1: Load and validate training and test data.
        
        Returns:
            Tuple of (training_data, test_data)
        """
        logger.info("=" * 70)
        logger.info("STEP 1: Load and Validate Data")
        logger.info("=" * 70)
        
        # Load training data
        logger.info(f"Loading training data from {config.TRAINING_DATA_PATH}")
        train_data = pd.read_csv(config.TRAINING_DATA_PATH)
        logger.info(f"Loaded {len(train_data)} training records")
        
        # Load test data
        logger.info(f"Loading test data from {config.TEST_DATA_PATH}")
        test_data = pd.read_csv(config.TEST_DATA_PATH)
        logger.info(f"Loaded {len(test_data)} test records")
        
        # Validate training data
        logger.info("Validating training data schema and values")
        all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        validator = DataValidator(
            required_features=all_features,
            numerical_features=config.NUMERICAL_FEATURES,
            categorical_features=config.CATEGORICAL_FEATURES,
            target_variable=config.TARGET_VARIABLE
        )
        validator.validate_for_training(train_data)
        logger.info("[OK] Training data validation passed")
        
        # Validate test data
        validator.validate_for_training(test_data)
        logger.info("[OK] Test data validation passed")
        
        # Log data statistics
        logger.info(f"\nTraining Data Statistics:")
        logger.info(f"  Total samples: {len(train_data)}")
        logger.info(f"  Churn rate: {train_data['churn'].mean():.2%}")
        logger.info(f"  Features: {len(all_features)}")
        
        logger.info(f"\nTest Data Statistics:")
        logger.info(f"  Total samples: {len(test_data)}")
        logger.info(f"  Churn rate: {test_data['churn'].mean():.2%}")
        
        return train_data, test_data
    
    def step2_preprocess_data(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Step 2: Preprocess training and test data.
        
        Args:
            train_data: Training dataframe
            test_data: Test dataframe
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 2: Preprocess Data")
        logger.info("=" * 70)
        
        # Split features and target
        all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        X_train = train_data[all_features]
        y_train = train_data[config.TARGET_VARIABLE]
        X_test = test_data[all_features]
        y_test = test_data[config.TARGET_VARIABLE]
        
        # Feature engineering
        logger.info("Applying feature engineering")
        feature_engineer = FeatureEngineer()
        X_train_engineered = feature_engineer.fit_transform(X_train)
        X_test_engineered = feature_engineer.transform(X_test)
        
        # Get all feature columns
        all_numerical_features = [
            col for col in X_train_engineered.columns
            if col in config.NUMERICAL_FEATURES or 
            pd.api.types.is_numeric_dtype(X_train_engineered[col])
        ]
        all_categorical_features = [
            col for col in X_train_engineered.columns
            if col in config.CATEGORICAL_FEATURES
        ]
        
        logger.info(f"  Numerical features: {len(all_numerical_features)}")
        logger.info(f"  Categorical features: {len(all_categorical_features)}")
        
        # Transform data
        logger.info("Applying transformations (imputation, encoding, scaling)")
        self.transformer = DataTransformer(
            numerical_features=all_numerical_features,
            categorical_features=all_categorical_features
        )
        
        X_train_transformed = self.transformer.fit_transform(X_train_engineered)
        X_test_transformed = self.transformer.transform(X_test_engineered)
        
        logger.info(f"[OK] Preprocessing complete")
        logger.info(f"  Training shape: {X_train_transformed.shape}")
        logger.info(f"  Test shape: {X_test_transformed.shape}")
        
        return X_train_transformed, X_test_transformed, y_train.values, y_test.values
    
    def step3_train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> dict:
        """
        Step 3: Train XGBoost model and evaluate performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: Train Model")
        logger.info("=" * 70)
        
        # Configure hyperparameters for high recall
        params = config.XGBOOST_PARAMS.copy()
        n_positive = np.sum(y_train == 1)
        n_negative = np.sum(y_train == 0)
        if n_positive > 0:
            scale_pos_weight = (n_negative / n_positive) * 7.5
            params['scale_pos_weight'] = scale_pos_weight
            logger.info(f"Setting scale_pos_weight to {scale_pos_weight:.2f} for high recall")
        
        # Train model
        logger.info("Training XGBoost classifier")
        trainer = ModelTrainer(params=params)
        self.model = trainer.train(X_train, y_train)
        logger.info("[OK] Model training complete")
        
        # Evaluate model
        logger.info("\nEvaluating model on test set")
        metrics = trainer.evaluate(self.model, X_test, y_test)
        
        logger.info(f"\nModel Performance Metrics:")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        # Verify recall threshold
        if metrics['recall'] >= config.MIN_RECALL_THRESHOLD:
            logger.info(f"[OK] Model meets recall threshold: {metrics['recall']:.4f} >= {config.MIN_RECALL_THRESHOLD:.4f}")
        else:
            logger.warning(f"[WARNING] Model recall {metrics['recall']:.4f} is below threshold {config.MIN_RECALL_THRESHOLD:.4f}")
        
        return metrics
    
    def step4_save_model(self, metrics: dict) -> str:
        """
        Step 4: Save trained model with versioning.
        
        Args:
            metrics: Evaluation metrics
            
        Returns:
            Model version string
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 4: Save Model with Versioning")
        logger.info("=" * 70)
        
        # Save model
        trainer = ModelTrainer()
        self.model_version = trainer.save_model(
            self.model,
            metadata={
                'metrics': metrics,
                'features': {
                    'numerical': self.transformer.numerical_features,
                    'categorical': self.transformer.categorical_features
                }
            }
        )
        logger.info(f"[OK] Model saved with version: {self.model_version}")
        
        # Save transformer
        transformer_path = config.TRANSFORMER_DIR / f"{self.model_version}_transformer.pkl"
        self.transformer.save(str(transformer_path))
        logger.info(f"[OK] Transformer saved to: {transformer_path}")
        
        return self.model_version
    
    def step5_generate_predictions(
        self,
        test_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Step 5: Generate predictions for test customers.
        
        Args:
            test_data: Test dataframe
            
        Returns:
            DataFrame with predictions
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 5: Generate Predictions")
        logger.info("=" * 70)
        
        # Initialize predictor
        logger.info(f"Loading model version: {self.model_version}")
        self.predictor = ChurnPredictor(model_version=self.model_version)
        
        # Prepare test data
        all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        X_test = test_data[all_features]
        
        # Generate predictions
        logger.info(f"Generating predictions for {len(X_test)} customers")
        predictions = self.predictor.predict_proba(X_test)
        
        # Create results dataframe
        results = pd.DataFrame({
            'customer_id': range(len(predictions)),
            'churn_probability': predictions,
            'is_high_risk': predictions >= config.HIGH_RISK_THRESHOLD,
            'actual_churn': test_data[config.TARGET_VARIABLE].values
        })
        
        # Log statistics
        high_risk_count = results['is_high_risk'].sum()
        logger.info(f"[OK] Predictions generated")
        logger.info(f"  Total predictions: {len(results)}")
        logger.info(f"  High-risk accounts: {high_risk_count} ({high_risk_count/len(results):.1%})")
        logger.info(f"  Average churn probability: {results['churn_probability'].mean():.3f}")
        
        return results
    
    def step6_compute_explanations(
        self,
        test_data: pd.DataFrame,
        predictions: pd.DataFrame,
        X_train_transformed: np.ndarray,
        sample_size: int = 100
    ) -> pd.DataFrame:
        """
        Step 6: Compute SHAP explanations for predictions.
        
        Args:
            test_data: Test dataframe
            predictions: Predictions dataframe
            X_train_transformed: Transformed training data for background
            sample_size: Number of samples to explain (for performance)
            
        Returns:
            DataFrame with explanations
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 6: Compute SHAP Explanations")
        logger.info("=" * 70)
        
        # Prepare data
        all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
        X_test = test_data[all_features]
        
        # Sample for explanation (SHAP can be slow on large datasets)
        if len(X_test) > sample_size:
            logger.info(f"Sampling {sample_size} customers for SHAP explanation")
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            X_sample = test_data.iloc[sample_indices]
            predictions_sample = predictions.iloc[sample_indices].copy()
        else:
            X_sample = test_data
            predictions_sample = predictions.copy()
        
        # Preprocess sample data
        feature_engineer = FeatureEngineer()
        X_sample_features = X_sample[all_features]
        X_sample_engineered = feature_engineer.fit_transform(X_sample_features)
        X_sample_transformed = self.transformer.transform(X_sample_engineered)
        
        # Use a subset of training data as background (for performance)
        background_size = min(100, X_train_transformed.shape[0])
        background_indices = np.random.choice(
            X_train_transformed.shape[0],
            background_size,
            replace=False
        )
        background_data = X_train_transformed[background_indices]
        
        # Initialize SHAP explainer
        logger.info(f"Initializing SHAP explainer with {background_size} background samples")
        self.explainer = SHAPExplainer(
            model=self.model,
            background_data=background_data
        )
        
        # Compute explanations
        logger.info(f"Computing SHAP values for {len(X_sample_transformed)} samples")
        shap_values = self.explainer.explain(X_sample_transformed)
        
        # Get feature names (after transformation)
        feature_names = [f"feature_{i}" for i in range(X_sample_transformed.shape[1])]
        
        # Extract top features for each sample
        explanations = []
        for idx in range(len(X_sample_transformed)):
            if idx % 20 == 0:
                logger.info(f"  Progress: {idx}/{len(X_sample_transformed)}")
            
            sample_shap = shap_values[idx:idx+1]
            top_features = self.explainer.get_top_features(
                sample_shap,
                feature_names=feature_names,
                n=5
            )
            
            explanations.append({
                'customer_id': predictions_sample.iloc[idx]['customer_id'],
                'top_features': str(top_features)
            })
        
        explanations_df = pd.DataFrame(explanations)
        logger.info(f"[OK] SHAP explanations computed for {len(explanations_df)} customers")
        
        # Merge with predictions
        results_with_explanations = predictions_sample.merge(
            explanations_df,
            on='customer_id',
            how='left'
        )
        
        return results_with_explanations
    
    def step7_store_and_evaluate(
        self,
        predictions: pd.DataFrame
    ) -> dict:
        """
        Step 7: Store predictions and evaluate performance.
        
        Args:
            predictions: Predictions dataframe
            
        Returns:
            Evaluation metrics dictionary
        """
        logger.info("\n" + "=" * 70)
        logger.info("STEP 7: Store Predictions and Evaluate")
        logger.info("=" * 70)
        
        # Store predictions
        logger.info("Storing predictions for future validation")
        for _, row in predictions.iterrows():
            self.prediction_store.store(
                customer_id=str(row['customer_id']),
                model_version=self.model_version,
                prediction=float(row['churn_probability']),
                predicted_class=int(row['is_high_risk']),
                actual_outcome=int(row['actual_churn'])
            )
        logger.info(f"[OK] Stored {len(predictions)} predictions")
        
        # Evaluate performance using predictions
        logger.info("\nEvaluating model performance on stored predictions")
        y_true = predictions['actual_churn'].values
        y_pred_proba = predictions['churn_probability'].values
        y_pred = (y_pred_proba >= config.HIGH_RISK_THRESHOLD).astype(int)
        
        # Compute metrics manually
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        evaluation_metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
        
        logger.info(f"\nEvaluation Metrics:")
        logger.info(f"  Precision: {evaluation_metrics['precision']:.4f}")
        logger.info(f"  Recall: {evaluation_metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {evaluation_metrics['f1_score']:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    {evaluation_metrics['confusion_matrix']}")
        
        # Check if performance meets threshold
        if evaluation_metrics['recall'] < config.MIN_RECALL_THRESHOLD:
            logger.warning(f"[WARNING] Performance degradation detected!")
            logger.warning(f"  Recall {evaluation_metrics['recall']:.4f} < {config.MIN_RECALL_THRESHOLD:.4f}")
        else:
            logger.info(f"[OK] Model performance meets requirements")
        
        return evaluation_metrics
    
    def run(self):
        """Execute the complete end-to-end pipeline."""
        try:
            logger.info("\n" + "=" * 70)
            logger.info("CUSTOMER CHURN PREDICTION - END-TO-END PIPELINE")
            logger.info("=" * 70)
            logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Step 1: Load and validate data
            train_data, test_data = self.step1_load_and_validate_data()
            
            # Step 2: Preprocess data
            X_train, X_test, y_train, y_test = self.step2_preprocess_data(
                train_data, test_data
            )
            
            # Step 3: Train model
            training_metrics = self.step3_train_model(
                X_train, y_train, X_test, y_test
            )
            
            # Step 4: Save model
            model_version = self.step4_save_model(training_metrics)
            
            # Step 5: Generate predictions
            predictions = self.step5_generate_predictions(test_data)
            
            # Step 6: Compute explanations
            predictions_with_explanations = self.step6_compute_explanations(
                test_data, predictions, X_train, sample_size=100
            )
            
            # Step 7: Store and evaluate
            evaluation_metrics = self.step7_store_and_evaluate(predictions)
            
            # Final summary
            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE COMPLETE!")
            logger.info("=" * 70)
            logger.info(f"Model Version: {model_version}")
            logger.info(f"Training Recall: {training_metrics['recall']:.4f}")
            logger.info(f"Test Recall: {evaluation_metrics['recall']:.4f}")
            logger.info(f"High-Risk Accounts: {predictions['is_high_risk'].sum()}")
            logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 70)
            
            return 0
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    pipeline = EndToEndPipeline()
    exit_code = pipeline.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
