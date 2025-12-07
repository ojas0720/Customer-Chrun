# Model Validation and Monitoring Service Guide

## Overview

The monitoring service provides comprehensive model validation and monitoring capabilities for the Customer Churn Prediction System. It includes three main components:

1. **ModelEvaluator**: Evaluates model performance on new data
2. **AlertService**: Monitors performance and generates alerts for degradation
3. **PredictionStore**: Stores predictions with actual outcomes for validation

## Components

### ModelEvaluator

Evaluates model performance and computes various metrics.

#### Basic Usage

```python
from services.monitoring import ModelEvaluator
from services.model_repository import ModelRepository

# Load model
repository = ModelRepository()
model, transformer = repository.load("v1.0.0")

# Prepare test data
X_test, y_test = load_test_data()  # Your data loading function

# Evaluate model
evaluator = ModelEvaluator(model_version="v1.0.0")
result = evaluator.evaluate(model, X_test, y_test)

# Access metrics
print(f"Recall: {result.recall:.4f}")
print(f"Precision: {result.precision:.4f}")
print(f"F1-Score: {result.f1_score:.4f}")
print(f"Accuracy: {result.accuracy:.4f}")
print(f"ROC AUC: {result.roc_auc:.4f}")
print(f"Confusion Matrix: {result.confusion_matrix}")

# Check if performance meets threshold
if result.meets_threshold(0.85):
    print("Model performance is acceptable")
else:
    print("Model performance is below threshold")
```

#### Computing Calibration Curves

```python
# Get predicted probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Compute calibration curve
prob_true, prob_pred = evaluator.compute_calibration_curve(
    y_test,
    y_pred_proba,
    n_bins=10,
    strategy='uniform'
)

# Plot calibration curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Model')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curve')
plt.legend()
plt.show()
```

### AlertService

Monitors model performance and generates alerts when performance degrades.

#### Basic Usage

```python
from services.monitoring import AlertService

# Initialize alert service
alert_service = AlertService(
    recall_threshold=0.85,
    alert_log_path=Path("alerts.json")
)

# Check performance
performance_ok = alert_service.check_performance(result)

if not performance_ok:
    # Generate alert
    alert = alert_service.generate_alert(
        result,
        alert_type="performance_degradation",
        severity="high"
    )
    
    print(f"Alert ID: {alert.alert_id}")
    print(f"Message: {alert.message}")
    print(f"Severity: {alert.severity}")
```

#### Retrieving Alerts

```python
# Get all alerts
all_alerts = alert_service.get_alerts()

# Get alerts for specific model version
v1_alerts = alert_service.get_alerts(model_version="v1.0.0")

# Get most recent 10 alerts
recent_alerts = alert_service.get_alerts(limit=10)

# Display alerts
for alert in recent_alerts:
    print(f"{alert.timestamp}: {alert.message}")
```

### PredictionStore

Stores predictions with actual outcomes for validation and monitoring.

#### Storing Predictions

```python
from services.monitoring import PredictionStore

# Initialize prediction store
pred_store = PredictionStore(storage_path=Path("predictions_store.csv"))

# Store prediction with outcome
pred_id = pred_store.store(
    customer_id="CUST001",
    model_version="v1.0.0",
    prediction=0.75,
    predicted_class=1,
    actual_outcome=1,
    features={
        "tenure_months": 12,
        "monthly_charges": 50.0,
        "contract_type": "Month-to-month"
    }
)

print(f"Stored prediction: {pred_id}")
```

#### Storing Predictions Without Outcomes

```python
# Store prediction without outcome (outcome will be added later)
pred_id = pred_store.store(
    customer_id="CUST002",
    model_version="v1.0.0",
    prediction=0.60,
    predicted_class=1,
    actual_outcome=None  # Outcome not yet known
)

# Later, update with actual outcome
success = pred_store.update_outcome(pred_id, actual_outcome=0)
if success:
    print("Outcome updated successfully")
```

#### Retrieving Predictions

```python
# Get all predictions
all_predictions = pred_store.retrieve()

# Filter by customer
customer_preds = pred_store.retrieve(customer_id="CUST001")

# Filter by model version
v1_preds = pred_store.retrieve(model_version="v1.0.0")

# Get predictions with outcomes
with_outcomes = pred_store.retrieve(has_outcome=True)

# Get predictions without outcomes
without_outcomes = pred_store.retrieve(has_outcome=False)

# Get predictions in date range
from datetime import datetime, timedelta

start_date = datetime.now() - timedelta(days=7)
recent_preds = pred_store.retrieve(start_date=start_date)

# Limit results
top_10 = pred_store.retrieve(limit=10)
```

#### Getting Statistics

```python
# Get overall statistics
stats = pred_store.get_statistics()

print(f"Total predictions: {stats['total_predictions']}")
print(f"With outcomes: {stats['predictions_with_outcomes']}")
print(f"Without outcomes: {stats['predictions_without_outcomes']}")

if 'accuracy' in stats:
    print(f"Accuracy: {stats['accuracy']:.4f}")

# Get statistics for specific model version
v1_stats = pred_store.get_statistics(model_version="v1.0.0")
```

## Complete Monitoring Workflow

Here's a complete example of a monitoring workflow:

```python
from services.monitoring import ModelEvaluator, AlertService, PredictionStore
from services.model_repository import ModelRepository
from services.prediction import ChurnPredictor
import pandas as pd

# Initialize services
repository = ModelRepository()
evaluator = ModelEvaluator(model_version="v1.0.0")
alert_service = AlertService(recall_threshold=0.85)
pred_store = PredictionStore()

# Load model
model, transformer = repository.load("v1.0.0")

# Load new validation data
X_val, y_val = load_validation_data()

# Step 1: Evaluate model performance
eval_result = evaluator.evaluate(model, X_val, y_val)

print(f"Model Performance:")
print(f"  Recall: {eval_result.recall:.4f}")
print(f"  Precision: {eval_result.precision:.4f}")
print(f"  F1-Score: {eval_result.f1_score:.4f}")

# Step 2: Check performance and alert if needed
if not alert_service.check_performance(eval_result):
    alert = alert_service.generate_alert(eval_result, severity="high")
    print(f"\n⚠️  ALERT: {alert.message}")
    
    # In production, you might send email, Slack notification, etc.
    send_notification(alert)

# Step 3: Store predictions for future validation
predictor = ChurnPredictor(model_version="v1.0.0")

# Make predictions on new customers
new_customers = load_new_customers()
predictions = predictor.predict_batch(new_customers)

# Store predictions (outcomes will be added later)
for pred in predictions:
    pred_store.store(
        customer_id=pred.customer_id,
        model_version=pred.model_version,
        prediction=pred.churn_probability,
        predicted_class=1 if pred.is_high_risk else 0,
        actual_outcome=None  # Will be updated when outcome is known
    )

print(f"\nStored {len(predictions)} predictions")

# Step 4: Get monitoring statistics
stats = pred_store.get_statistics(model_version="v1.0.0")
print(f"\nPrediction Store Statistics:")
print(f"  Total predictions: {stats['total_predictions']}")
print(f"  With outcomes: {stats['predictions_with_outcomes']}")
if 'accuracy' in stats:
    print(f"  Accuracy: {stats['accuracy']:.4f}")
```

## Scheduled Monitoring

For production systems, you should set up scheduled monitoring:

```python
import schedule
import time
from datetime import datetime

def monitor_model_performance():
    """Scheduled monitoring task."""
    print(f"Running monitoring at {datetime.now()}")
    
    # Load latest validation data
    X_val, y_val = load_latest_validation_data()
    
    # Evaluate model
    evaluator = ModelEvaluator(model_version="v1.0.0")
    result = evaluator.evaluate(model, X_val, y_val)
    
    # Check performance
    alert_service = AlertService(recall_threshold=0.85)
    if not alert_service.check_performance(result):
        alert = alert_service.generate_alert(result)
        send_alert_notification(alert)
    
    print(f"Monitoring complete. Recall: {result.recall:.4f}")

# Schedule monitoring every 24 hours
schedule.every(24).hours.do(monitor_model_performance)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## Best Practices

1. **Regular Evaluation**: Evaluate model performance regularly on new data to detect degradation early

2. **Store All Predictions**: Store predictions with features for future analysis and debugging

3. **Update Outcomes**: Update prediction outcomes as soon as they become available

4. **Monitor Calibration**: Check calibration curves periodically to ensure probability estimates are accurate

5. **Set Appropriate Thresholds**: Configure alert thresholds based on business requirements

6. **Track Multiple Metrics**: Don't rely on a single metric - monitor recall, precision, F1, and calibration

7. **Version Tracking**: Always track which model version generated each prediction

8. **Alert Fatigue**: Avoid too many alerts - set thresholds appropriately and aggregate alerts

## Error Handling

The monitoring service includes comprehensive error handling:

```python
try:
    result = evaluator.evaluate(model, X_test, y_test)
except ValueError as e:
    print(f"Evaluation error: {e}")
    # Handle invalid data

try:
    pred_id = pred_store.store(
        customer_id="CUST001",
        model_version="v1.0.0",
        prediction=1.5,  # Invalid - must be 0-1
        predicted_class=1
    )
except ValueError as e:
    print(f"Storage error: {e}")
    # Handle invalid prediction values
```

## Integration with Other Services

The monitoring service integrates seamlessly with other system components:

```python
from services.model_repository import ModelRepository
from services.prediction import ChurnPredictor
from services.monitoring import ModelEvaluator, AlertService, PredictionStore

# Load model from repository
repository = ModelRepository()
model, transformer = repository.load_deployed()

# Make predictions
predictor = ChurnPredictor()
predictions = predictor.predict_batch(customers)

# Store predictions
pred_store = PredictionStore()
for pred in predictions:
    pred_store.store(
        customer_id=pred.customer_id,
        model_version=pred.model_version,
        prediction=pred.churn_probability,
        predicted_class=1 if pred.is_high_risk else 0
    )

# Evaluate on validation set
evaluator = ModelEvaluator(model_version=predictor.model_version)
result = evaluator.evaluate(model, X_val, y_val)

# Check for alerts
alert_service = AlertService()
if not alert_service.check_performance(result):
    alert = alert_service.generate_alert(result)
    # Trigger rollback if needed
    repository.rollback(previous_version)
```
