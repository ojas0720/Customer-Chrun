"""
Example script demonstrating dashboard functionality.
This script shows how to prepare data and run the dashboard programmatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from services.model_repository import ModelRepository
from services.prediction import ChurnPredictor


def prepare_dashboard_data():
    """
    Prepare data for the dashboard.
    Ensures test data and trained models are available.
    """
    print("Preparing dashboard data...")
    
    # Check if test data exists
    if not config.TEST_DATA_PATH.exists():
        print(f"Warning: Test data not found at {config.TEST_DATA_PATH}")
        print("Please run generate_sample_data.py first")
        return False
    
    # Check if models exist
    repository = ModelRepository()
    versions = repository.list_versions()
    
    if not versions:
        print("Warning: No trained models found in repository")
        print("Please run train_model.py first")
        return False
    
    print(f"✓ Found {len(versions)} model version(s)")
    print(f"✓ Test data available at {config.TEST_DATA_PATH}")
    
    # Display available versions
    print("\nAvailable model versions:")
    for version in versions:
        print(f"  - {version.version}")
        if version.metrics:
            print(f"    Recall: {version.metrics.get('recall', 0):.3f}")
            print(f"    Precision: {version.metrics.get('precision', 0):.3f}")
    
    # Check deployed version
    deployed = repository.get_deployed_version()
    if deployed:
        print(f"\n✓ Deployed version: {deployed}")
    else:
        print("\nNote: No deployed version set. Dashboard will use latest version.")
    
    return True


def test_dashboard_components():
    """
    Test dashboard components to ensure they work correctly.
    """
    print("\nTesting dashboard components...")
    
    # Initialize repository
    repository = ModelRepository()
    
    # Load model
    try:
        model, transformer = repository.load_deployed()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False
    
    # Load test data
    try:
        test_df = pd.read_csv(config.TEST_DATA_PATH)
        print(f"✓ Test data loaded: {len(test_df)} customers")
    except Exception as e:
        print(f"✗ Error loading test data: {e}")
        return False
    
    # Initialize predictor
    try:
        # Load background data for SHAP
        train_df = pd.read_csv(config.TRAINING_DATA_PATH)
        X_train = train_df.drop('churn', axis=1)
        
        from services.preprocessing import FeatureEngineer
        feature_engineer = FeatureEngineer()
        feature_engineer.fit(X_train)
        X_train_eng = feature_engineer.transform(X_train)
        X_train_transformed = transformer.transform(X_train_eng)
        background_data = X_train_transformed[:config.SHAP_BACKGROUND_SAMPLES]
        
        predictor = ChurnPredictor(
            repository=repository,
            enable_explanations=True,
            background_data=background_data
        )
        print("✓ Predictor initialized with SHAP explanations")
    except Exception as e:
        print(f"Warning: Could not initialize SHAP explanations: {e}")
        predictor = ChurnPredictor(repository=repository)
        print("✓ Predictor initialized without SHAP explanations")
    
    # Test batch prediction
    try:
        # Prepare sample customers
        sample_size = min(5, len(test_df))
        customers = []
        for idx in range(sample_size):
            row = test_df.iloc[idx]
            features = row.drop('churn').to_dict() if 'churn' in row else row.to_dict()
            customers.append({
                'customer_id': f"CUST_{idx:04d}",
                **features
            })
        
        # Generate predictions
        predictions = predictor.predict_batch(customers, include_explanations=False)
        print(f"✓ Batch prediction successful: {len(predictions)} predictions")
        
        # Display sample predictions
        print("\nSample predictions:")
        for pred in predictions[:3]:
            risk_label = "HIGH RISK" if pred.is_high_risk else "LOW RISK"
            print(f"  {pred.customer_id}: {pred.churn_probability:.3f} ({risk_label})")
    
    except Exception as e:
        print(f"✗ Error generating predictions: {e}")
        return False
    
    # Test single prediction with explanation
    try:
        if predictor.explainer:
            result = predictor.predict_single(customers[0], include_explanations=True)
            print(f"\n✓ Single prediction with explanation successful")
            
            if result.top_features:
                print("\nTop contributing features:")
                for feature, shap_value in result.top_features[:3]:
                    impact = "increases" if shap_value > 0 else "decreases"
                    print(f"  {feature}: {shap_value:.4f} ({impact} churn risk)")
    except Exception as e:
        print(f"Warning: Could not generate explanation: {e}")
    
    return True


def display_dashboard_info():
    """
    Display information about accessing the dashboard.
    """
    print("\n" + "=" * 60)
    print("Dashboard Ready!")
    print("=" * 60)
    print("\nTo start the dashboard, run:")
    print("  python run_dashboard.py")
    print("\nOr use streamlit directly:")
    print(f"  streamlit run dashboard/app.py --server.port {config.DASHBOARD_PORT}")
    print("\nThe dashboard will be available at:")
    print(f"  http://localhost:{config.DASHBOARD_PORT}")
    print("\nDashboard features:")
    print("  • Overview: High-risk accounts, probability distribution, metrics")
    print("  • Customer Details: Individual predictions with SHAP explanations")
    print("  • Statistics: Aggregated metrics and feature importance trends")
    print("  • Model Selector: Switch between different model versions")
    print("=" * 60)


def main():
    """Main function."""
    print("=" * 60)
    print("Customer Churn Prediction Dashboard - Setup Check")
    print("=" * 60)
    
    # Prepare data
    if not prepare_dashboard_data():
        print("\n✗ Dashboard data preparation failed")
        print("Please ensure you have:")
        print("  1. Generated sample data (python generate_sample_data.py)")
        print("  2. Trained a model (python train_model.py)")
        return
    
    # Test components
    if not test_dashboard_components():
        print("\n✗ Dashboard component testing failed")
        return
    
    # Display info
    display_dashboard_info()


if __name__ == "__main__":
    main()
