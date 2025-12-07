"""
Example script demonstrating ModelRepository usage.
Shows how to save, load, list, and manage model versions.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from services.model_training import ModelTrainer
from services.preprocessing import DataTransformer
from services.model_repository import ModelRepository
from services.parameter_validation import ParameterValidator
import config


def create_sample_data(n_samples=500):
    """Create sample customer data for demonstration."""
    np.random.seed(42)
    
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


def main():
    """Demonstrate ModelRepository functionality."""
    
    print("=" * 70)
    print("Model Repository Example")
    print("=" * 70)
    
    # 1. Create and prepare data
    print("\n1. Creating sample data...")
    data = create_sample_data()
    train_data = data.iloc[:400]
    test_data = data.iloc[400:]
    
    X_train = train_data.drop('churn', axis=1)
    y_train = train_data['churn']
    X_test = test_data.drop('churn', axis=1)
    y_test = test_data['churn']
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # 2. Validate parameters
    print("\n2. Validating model parameters...")
    validator = ParameterValidator()
    
    params_v1 = {
        'learning_rate': 0.1,
        'max_depth': 3,
        'n_estimators': 50,
        'objective': 'binary:logistic'
    }
    
    try:
        validator.validate(params_v1)
        print("   ✓ Parameters are valid")
    except Exception as e:
        print(f"   ✗ Parameter validation failed: {e}")
        return
    
    # 3. Train first model version
    print("\n3. Training model v1.0.0...")
    transformer = DataTransformer(
        numerical_features=['tenure_months', 'monthly_charges', 'total_charges'],
        categorical_features=['contract_type', 'payment_method', 'internet_service']
    )
    X_train_transformed = transformer.fit_transform(X_train)
    X_test_transformed = transformer.transform(X_test)
    
    trainer_v1 = ModelTrainer(params=params_v1)
    model_v1 = trainer_v1.train(X_train_transformed, y_train)
    metrics_v1 = trainer_v1.evaluate(model_v1, X_test_transformed, y_test)
    
    print(f"   Recall: {metrics_v1['recall']:.3f}")
    print(f"   Precision: {metrics_v1['precision']:.3f}")
    print(f"   F1-Score: {metrics_v1['f1_score']:.3f}")
    
    # 4. Save model v1 to repository
    print("\n4. Saving model v1.0.0 to repository...")
    repo = ModelRepository()
    
    version_v1 = repo.save(
        model_v1,
        transformer,
        version="v1.0.0",
        metadata={
            'metrics': metrics_v1,
            'features': {
                'numerical': ['tenure_months', 'monthly_charges', 'total_charges'],
                'categorical': ['contract_type', 'payment_method', 'internet_service']
            },
            'description': 'Initial model with conservative parameters'
        }
    )
    print(f"   ✓ Saved as version: {version_v1}")
    
    # 5. Train second model version with different parameters
    print("\n5. Training model v2.0.0 with improved parameters...")
    params_v2 = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'n_estimators': 100,
        'objective': 'binary:logistic'
    }
    
    validator.validate(params_v2)
    
    trainer_v2 = ModelTrainer(params=params_v2)
    model_v2 = trainer_v2.train(X_train_transformed, y_train)
    metrics_v2 = trainer_v2.evaluate(model_v2, X_test_transformed, y_test)
    
    print(f"   Recall: {metrics_v2['recall']:.3f}")
    print(f"   Precision: {metrics_v2['precision']:.3f}")
    print(f"   F1-Score: {metrics_v2['f1_score']:.3f}")
    
    # 6. Save model v2
    print("\n6. Saving model v2.0.0 to repository...")
    version_v2 = repo.save(
        model_v2,
        transformer,
        version="v2.0.0",
        metadata={
            'metrics': metrics_v2,
            'features': {
                'numerical': ['tenure_months', 'monthly_charges', 'total_charges'],
                'categorical': ['contract_type', 'payment_method', 'internet_service']
            },
            'description': 'Improved model with deeper trees and more estimators'
        }
    )
    print(f"   ✓ Saved as version: {version_v2}")
    
    # 7. List all versions
    print("\n7. Listing all model versions...")
    versions = repo.list_versions()
    print(f"   Found {len(versions)} versions:")
    for v in versions:
        print(f"   - {v.version} (timestamp: {v.timestamp})")
        if 'metrics' in v.metrics:
            print(f"     Recall: {v.metrics.get('recall', 'N/A'):.3f}")
    
    # 8. Get latest version
    print("\n8. Getting latest version...")
    latest = repo.get_latest_version()
    print(f"   Latest version: {latest}")
    
    # 9. Load specific version
    print("\n9. Loading model v1.0.0...")
    loaded_model, loaded_transformer = repo.load("v1.0.0")
    print("   ✓ Model and transformer loaded successfully")
    
    # 10. Make predictions with loaded model
    print("\n10. Making predictions with loaded model...")
    sample_customer = X_test.iloc[:1]
    sample_transformed = loaded_transformer.transform(sample_customer)
    prediction = loaded_model.predict_proba(sample_transformed)
    
    print(f"   Churn probability: {prediction[0][1]:.3f}")
    print(f"   Retention probability: {prediction[0][0]:.3f}")
    
    # 11. Rollback to v1
    print("\n11. Rolling back to v1.0.0...")
    repo.rollback("v1.0.0")
    deployed = repo.get_deployed_version()
    print(f"   ✓ Deployed version is now: {deployed}")
    
    # 12. Load deployed model
    print("\n12. Loading deployed model...")
    deployed_model, deployed_transformer = repo.load_deployed()
    print("   ✓ Deployed model loaded successfully")
    
    # 13. Get metadata
    print("\n13. Retrieving metadata for v2.0.0...")
    metadata = repo.get_metadata("v2.0.0")
    print(f"   Description: {metadata.get('description', 'N/A')}")
    print(f"   Number of features: {metadata.get('n_features', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
