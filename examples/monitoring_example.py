"""
Example usage of the model validation and monitoring service.
Demonstrates ModelEvaluator, AlertService, and PredictionStore.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from services.monitoring import ModelEvaluator, AlertService, PredictionStore
from services.model_repository import ModelRepository
from services.prediction import ChurnPredictor
from services.preprocessing import DataTransformer, FeatureEngineer
import config


def example_model_evaluation():
    """Example: Evaluate model performance on validation data."""
    print("=" * 60)
    print("Example 1: Model Evaluation")
    print("=" * 60)
    
    # Load model from repository
    repository = ModelRepository()
    
    try:
        model, transformer = repository.load_deployed()
        deployed_version = repository.get_deployed_version()
        print(f"Loaded deployed model: {deployed_version}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train a model first using train_model.py")
        return
    
    # Load test data
    try:
        test_data = pd.read_csv(config.TEST_DATA_PATH)
        print(f"Loaded test data: {len(test_data)} samples")
    except FileNotFoundError:
        print(f"Test data not found at {config.TEST_DATA_PATH}")
        print("Please generate sample data first using generate_sample_data.py")
        return
    
    # Prepare data
    X_test = test_data.drop(columns=[config.TARGET_VARIABLE])
    y_test = test_data[config.TARGET_VARIABLE]
    
    # Apply feature engineering
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(X_test)
    X_test_engineered = feature_engineer.transform(X_test)
    
    # Transform features
    X_test_transformed = transformer.transform(X_test_engineered)
    
    # Evaluate model
    evaluator = ModelEvaluator(model_version=deployed_version)
    result = evaluator.evaluate(model, X_test_transformed, y_test.values)
    
    print(f"\nEvaluation Results:")
    print(f"  Model Version: {result.model_version}")
    print(f"  Evaluation Date: {result.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Samples: {result.n_samples}")
    print(f"\nMetrics:")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  F1-Score:  {result.f1_score:.4f}")
    print(f"  Accuracy:  {result.accuracy:.4f}")
    print(f"  ROC AUC:   {result.roc_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {result.confusion_matrix}")
    
    # Check if meets threshold
    threshold = config.MIN_RECALL_THRESHOLD
    if result.meets_threshold(threshold):
        print(f"\n✓ Model meets recall threshold ({threshold:.2f})")
    else:
        print(f"\n✗ Model below recall threshold ({threshold:.2f})")
    
    return result


def example_alert_service(eval_result):
    """Example: Monitor performance and generate alerts."""
    print("\n" + "=" * 60)
    print("Example 2: Alert Service")
    print("=" * 60)
    
    # Initialize alert service
    alert_service = AlertService(
        recall_threshold=config.MIN_RECALL_THRESHOLD,
        alert_log_path=config.BASE_DIR / "alerts.json"
    )
    
    # Check performance
    performance_ok = alert_service.check_performance(eval_result)
    
    if performance_ok:
        print(f"✓ Model performance is acceptable")
        print(f"  Recall: {eval_result.recall:.4f} >= {config.MIN_RECALL_THRESHOLD:.4f}")
    else:
        print(f"✗ Model performance degradation detected!")
        print(f"  Recall: {eval_result.recall:.4f} < {config.MIN_RECALL_THRESHOLD:.4f}")
        
        # Generate alert
        alert = alert_service.generate_alert(
            eval_result,
            alert_type="performance_degradation",
            severity="high"
        )
        
        print(f"\nAlert Generated:")
        print(f"  Alert ID: {alert.alert_id}")
        print(f"  Severity: {alert.severity}")
        print(f"  Message: {alert.message}")
        print(f"  Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Retrieve recent alerts
    recent_alerts = alert_service.get_alerts(limit=5)
    
    if recent_alerts:
        print(f"\nRecent Alerts ({len(recent_alerts)}):")
        for i, alert in enumerate(recent_alerts, 1):
            print(f"  {i}. [{alert.severity}] {alert.model_version} - "
                  f"{alert.timestamp.strftime('%Y-%m-%d %H:%M')}")


def example_prediction_store():
    """Example: Store predictions and track outcomes."""
    print("\n" + "=" * 60)
    print("Example 3: Prediction Store")
    print("=" * 60)
    
    # Initialize prediction store
    pred_store = PredictionStore(
        storage_path=config.BASE_DIR / "predictions_store.csv"
    )
    
    # Load model and make predictions
    repository = ModelRepository()
    
    try:
        deployed_version = repository.get_deployed_version()
        predictor = ChurnPredictor(model_version=deployed_version)
    except Exception as e:
        print(f"Error loading predictor: {e}")
        return
    
    # Create sample customers
    sample_customers = [
        {
            "customer_id": "CUST001",
            "tenure_months": 12,
            "monthly_charges": 50.0,
            "total_charges": 600.0,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic"
        },
        {
            "customer_id": "CUST002",
            "tenure_months": 36,
            "monthly_charges": 75.0,
            "total_charges": 2700.0,
            "contract_type": "Two year",
            "payment_method": "Credit card",
            "internet_service": "DSL"
        },
        {
            "customer_id": "CUST003",
            "tenure_months": 6,
            "monthly_charges": 90.0,
            "total_charges": 540.0,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic"
        }
    ]
    
    print(f"Making predictions for {len(sample_customers)} customers...")
    
    # Make predictions
    try:
        predictions = predictor.predict_batch(sample_customers)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return
    
    # Store predictions (without outcomes initially)
    pred_ids = []
    for pred in predictions:
        pred_id = pred_store.store(
            customer_id=pred.customer_id,
            model_version=pred.model_version,
            prediction=pred.churn_probability,
            predicted_class=1 if pred.is_high_risk else 0,
            actual_outcome=None  # Outcome not yet known
        )
        pred_ids.append(pred_id)
        
        print(f"\nStored prediction for {pred.customer_id}:")
        print(f"  Prediction ID: {pred_id}")
        print(f"  Churn Probability: {pred.churn_probability:.4f}")
        print(f"  High Risk: {pred.is_high_risk}")
    
    # Simulate updating outcomes later
    print("\n" + "-" * 60)
    print("Simulating outcome updates...")
    
    # Update with simulated outcomes
    simulated_outcomes = [1, 0, 1]  # CUST001 churned, CUST002 stayed, CUST003 churned
    
    for pred_id, outcome in zip(pred_ids, simulated_outcomes):
        success = pred_store.update_outcome(pred_id, outcome)
        if success:
            print(f"  Updated {pred_id}: actual_outcome = {outcome}")
    
    # Retrieve predictions with outcomes
    print("\n" + "-" * 60)
    print("Retrieving predictions with outcomes...")
    
    predictions_with_outcomes = pred_store.retrieve(has_outcome=True, limit=10)
    
    print(f"\nFound {len(predictions_with_outcomes)} predictions with outcomes:")
    for pred in predictions_with_outcomes:
        correct = "✓" if pred.predicted_class == pred.actual_outcome else "✗"
        print(f"  {correct} {pred.customer_id}: "
              f"predicted={pred.predicted_class}, actual={pred.actual_outcome}, "
              f"prob={pred.prediction:.4f}")
    
    # Get statistics
    print("\n" + "-" * 60)
    print("Prediction Store Statistics:")
    
    stats = pred_store.get_statistics()
    
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  With outcomes: {stats['predictions_with_outcomes']}")
    print(f"  Without outcomes: {stats['predictions_without_outcomes']}")
    
    if 'accuracy' in stats:
        print(f"  Accuracy: {stats['accuracy']:.4f}")


def example_calibration_curve():
    """Example: Compute and display calibration curve."""
    print("\n" + "=" * 60)
    print("Example 4: Calibration Curve")
    print("=" * 60)
    
    # Load model and data
    repository = ModelRepository()
    
    try:
        model, transformer = repository.load_deployed()
        deployed_version = repository.get_deployed_version()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    try:
        test_data = pd.read_csv(config.TEST_DATA_PATH)
    except FileNotFoundError:
        print(f"Test data not found")
        return
    
    # Prepare data
    X_test = test_data.drop(columns=[config.TARGET_VARIABLE])
    y_test = test_data[config.TARGET_VARIABLE]
    
    # Apply transformations
    feature_engineer = FeatureEngineer()
    feature_engineer.fit(X_test)
    X_test_engineered = feature_engineer.transform(X_test)
    X_test_transformed = transformer.transform(X_test_engineered)
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    
    # Compute calibration curve
    evaluator = ModelEvaluator(model_version=deployed_version)
    prob_true, prob_pred = evaluator.compute_calibration_curve(
        y_test.values,
        y_pred_proba,
        n_bins=10,
        strategy='uniform'
    )
    
    print(f"Calibration Curve (10 bins):")
    print(f"  {'Predicted Prob':<15} {'True Prob':<15} {'Difference':<15}")
    print(f"  {'-'*15} {'-'*15} {'-'*15}")
    
    for pred_prob, true_prob in zip(prob_pred, prob_true):
        diff = abs(pred_prob - true_prob)
        print(f"  {pred_prob:<15.4f} {true_prob:<15.4f} {diff:<15.4f}")
    
    # Calculate mean absolute calibration error
    mean_error = np.mean(np.abs(prob_pred - prob_true))
    print(f"\nMean Absolute Calibration Error: {mean_error:.4f}")
    
    if mean_error < 0.05:
        print("✓ Model is well-calibrated")
    elif mean_error < 0.10:
        print("⚠ Model calibration is acceptable")
    else:
        print("✗ Model calibration needs improvement")


def main():
    """Run all monitoring examples."""
    print("\n" + "=" * 60)
    print("Model Validation and Monitoring Service Examples")
    print("=" * 60)
    
    # Example 1: Model Evaluation
    eval_result = example_model_evaluation()
    
    if eval_result is None:
        print("\nCannot continue without evaluation result.")
        print("Please ensure model and data are available.")
        return
    
    # Example 2: Alert Service
    example_alert_service(eval_result)
    
    # Example 3: Prediction Store
    example_prediction_store()
    
    # Example 4: Calibration Curve
    example_calibration_curve()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
