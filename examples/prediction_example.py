"""
Example usage of the Customer Churn Prediction Service.

This script demonstrates how to:
1. Load a trained model
2. Make single predictions
3. Make batch predictions
4. Use different risk thresholds
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.prediction import ChurnPredictor, CustomerRecord, PredictionResult


def main():
    """Demonstrate prediction service usage."""
    
    print("=" * 70)
    print("Customer Churn Prediction Service - Example Usage")
    print("=" * 70)
    print()
    
    # Initialize predictor (loads deployed or latest model)
    print("Loading model...")
    try:
        predictor = ChurnPredictor()
        print(f"✓ Model loaded: {predictor.model_version}")
        print(f"  High-risk threshold: {predictor.high_risk_threshold}")
        print()
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("  Please ensure a trained model exists in the repository.")
        return
    
    # Example 1: Single prediction using dictionary
    print("-" * 70)
    print("Example 1: Single Prediction (Dictionary)")
    print("-" * 70)
    
    customer1 = {
        'customer_id': 'CUST001',
        'tenure_months': 6,
        'monthly_charges': 89.95,
        'total_charges': 539.70,
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic'
    }
    
    result1 = predictor.predict_single(customer1)
    print(f"Customer ID: {result1.customer_id}")
    print(f"Churn Probability: {result1.churn_probability:.2%}")
    print(f"High Risk: {'YES' if result1.is_high_risk else 'NO'}")
    print(f"Model Version: {result1.model_version}")
    print()
    
    # Example 2: Single prediction using CustomerRecord
    print("-" * 70)
    print("Example 2: Single Prediction (CustomerRecord)")
    print("-" * 70)
    
    customer2 = CustomerRecord(
        customer_id='CUST002',
        features={
            'tenure_months': 48,
            'monthly_charges': 45.00,
            'total_charges': 2160.00,
            'contract_type': 'Two year',
            'payment_method': 'Bank transfer',
            'internet_service': 'DSL'
        }
    )
    
    result2 = predictor.predict_single(customer2)
    print(f"Customer ID: {result2.customer_id}")
    print(f"Churn Probability: {result2.churn_probability:.2%}")
    print(f"High Risk: {'YES' if result2.is_high_risk else 'NO'}")
    print()
    
    # Example 3: Batch prediction
    print("-" * 70)
    print("Example 3: Batch Prediction")
    print("-" * 70)
    
    customers = [
        {
            'customer_id': 'CUST003',
            'tenure_months': 12,
            'monthly_charges': 65.50,
            'total_charges': 786.00,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        },
        {
            'customer_id': 'CUST004',
            'tenure_months': 24,
            'monthly_charges': 55.00,
            'total_charges': 1320.00,
            'contract_type': 'One year',
            'payment_method': 'Credit card',
            'internet_service': 'DSL'
        },
        {
            'customer_id': 'CUST005',
            'tenure_months': 3,
            'monthly_charges': 95.00,
            'total_charges': 285.00,
            'contract_type': 'Month-to-month',
            'payment_method': 'Electronic check',
            'internet_service': 'Fiber optic'
        }
    ]
    
    results = predictor.predict_batch(customers)
    
    print(f"Processed {len(results)} customers:")
    print()
    for result in results:
        risk_indicator = "⚠️ HIGH RISK" if result.is_high_risk else "✓ Low Risk"
        print(f"  {result.customer_id}: {result.churn_probability:.2%} - {risk_indicator}")
    print()
    
    # Example 4: Custom threshold
    print("-" * 70)
    print("Example 4: Custom Risk Threshold")
    print("-" * 70)
    
    import pandas as pd
    
    customer_data = pd.DataFrame([{
        'tenure_months': 18,
        'monthly_charges': 70.00,
        'total_charges': 1260.00,
        'contract_type': 'Month-to-month',
        'payment_method': 'Mailed check',
        'internet_service': 'Fiber optic'
    }])
    
    prob = predictor.predict_proba(customer_data)[0]
    high_risk_30 = predictor.predict_high_risk(customer_data, threshold=0.3)[0]
    high_risk_50 = predictor.predict_high_risk(customer_data, threshold=0.5)[0]
    high_risk_70 = predictor.predict_high_risk(customer_data, threshold=0.7)[0]
    
    print(f"Churn Probability: {prob:.2%}")
    print(f"High Risk (30% threshold): {'YES' if high_risk_30 else 'NO'}")
    print(f"High Risk (50% threshold): {'YES' if high_risk_50 else 'NO'}")
    print(f"High Risk (70% threshold): {'YES' if high_risk_70 else 'NO'}")
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("The prediction service successfully:")
    print("  ✓ Loaded trained model from repository")
    print("  ✓ Made single predictions using dictionaries and CustomerRecord objects")
    print("  ✓ Processed batch predictions efficiently")
    print("  ✓ Applied custom risk thresholds")
    print()
    print("Next steps:")
    print("  - Integrate SHAP explanations for feature importance")
    print("  - Add prediction service to dashboard")
    print("  - Set up monitoring and alerting")
    print()


if __name__ == '__main__':
    main()
