"""
Example demonstrating SHAP explainability for churn predictions.
Shows how to use the SHAPExplainer to understand model predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from services.prediction import ChurnPredictor, CustomerRecord
from services.model_repository import ModelRepository
import config


def main():
    """Demonstrate SHAP explainability with churn predictions."""
    
    print("=" * 70)
    print("Customer Churn Prediction - SHAP Explainability Example")
    print("=" * 70)
    print()
    
    # Initialize model repository
    repository = ModelRepository()
    
    try:
        # Get deployed or latest model version
        deployed_version = repository.get_deployed_version()
        if deployed_version is None:
            deployed_version = repository.get_latest_version()
        
        print(f"Using model version: {deployed_version}")
        print()
        
        # Load model and transformer to prepare background data
        model, transformer = repository.load(deployed_version)
        
        # Load some training data to use as background for SHAP
        training_data_path = config.TRAINING_DATA_PATH
        if not training_data_path.exists():
            print(f"Error: Training data not found at {training_data_path}")
            print("Please run generate_sample_data.py first.")
            return
        
        df_train = pd.read_csv(training_data_path)
        
        # Prepare background data (sample of training data)
        from services.preprocessing import FeatureEngineer
        feature_engineer = FeatureEngineer()
        
        X_train = df_train.drop(config.TARGET_VARIABLE, axis=1)
        X_train_engineered = feature_engineer.fit_transform(X_train)
        X_train_transformed = transformer.transform(X_train_engineered)
        
        # Use a sample as background data
        background_data = X_train_transformed[:config.SHAP_BACKGROUND_SAMPLES]
        
        print(f"Prepared background data with {background_data.shape[0]} samples")
        print()
        
        # Create predictor with explanations enabled
        predictor = ChurnPredictor(
            model_version=deployed_version,
            repository=repository,
            enable_explanations=True,
            background_data=background_data
        )
        
        print("SHAP explainer initialized successfully!")
        print()
        
        # Example 1: Single customer prediction with explanation
        print("-" * 70)
        print("Example 1: Single Customer Prediction with SHAP Explanation")
        print("-" * 70)
        
        customer1 = CustomerRecord(
            customer_id="CUST_001",
            features={
                'tenure_months': 6,
                'monthly_charges': 85.50,
                'total_charges': 510.00,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'Fiber optic'
            }
        )
        
        result1 = predictor.predict_single(customer1, include_explanations=True)
        
        print(f"Customer ID: {result1.customer_id}")
        print(f"Churn Probability: {result1.churn_probability:.2%}")
        print(f"High Risk: {'Yes' if result1.is_high_risk else 'No'}")
        print(f"Model Version: {result1.model_version}")
        print()
        print("Top Contributing Features:")
        for i, (feature_name, shap_value) in enumerate(result1.top_features, 1):
            direction = "increases" if shap_value > 0 else "decreases"
            print(f"  {i}. {feature_name}: {shap_value:+.4f} ({direction} churn risk)")
        print()
        
        # Example 2: Batch predictions with explanations
        print("-" * 70)
        print("Example 2: Batch Predictions with SHAP Explanations")
        print("-" * 70)
        
        customers = [
            {
                'customer_id': 'CUST_002',
                'tenure_months': 48,
                'monthly_charges': 55.00,
                'total_charges': 2640.00,
                'contract_type': 'Two year',
                'payment_method': 'Bank transfer',
                'internet_service': 'DSL'
            },
            {
                'customer_id': 'CUST_003',
                'tenure_months': 12,
                'monthly_charges': 95.00,
                'total_charges': 1140.00,
                'contract_type': 'Month-to-month',
                'payment_method': 'Electronic check',
                'internet_service': 'Fiber optic'
            },
            {
                'customer_id': 'CUST_004',
                'tenure_months': 36,
                'monthly_charges': 70.00,
                'total_charges': 2520.00,
                'contract_type': 'One year',
                'payment_method': 'Credit card',
                'internet_service': 'Fiber optic'
            }
        ]
        
        results = predictor.predict_batch(customers, include_explanations=True)
        
        for result in results:
            print(f"\nCustomer: {result.customer_id}")
            print(f"  Churn Probability: {result.churn_probability:.2%}")
            print(f"  High Risk: {'Yes' if result.is_high_risk else 'No'}")
            print(f"  Top 3 Features:")
            for feature_name, shap_value in result.top_features[:3]:
                direction = "↑" if shap_value > 0 else "↓"
                print(f"    {direction} {feature_name}: {shap_value:+.4f}")
        
        print()
        print("-" * 70)
        print("Example completed successfully!")
        print("-" * 70)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
