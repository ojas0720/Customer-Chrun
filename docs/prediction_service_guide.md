# Prediction Service Guide

## Overview

The Prediction Service provides a comprehensive interface for generating customer churn predictions using trained XGBoost models. It handles model loading, input validation, feature engineering, and batch processing.

## Components

### 1. CustomerRecord

A data structure representing a customer with their features.

```python
from services.prediction import CustomerRecord

record = CustomerRecord(
    customer_id='CUST001',
    features={
        'tenure_months': 12,
        'monthly_charges': 65.5,
        'total_charges': 786.0,
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic'
    }
)

# Validate the record
record.validate()  # Returns True or raises ValidationError

# Convert to DataFrame
df = record.to_dataframe()
```

### 2. PredictionResult

Contains the prediction output with probability, risk classification, and metadata.

```python
from services.prediction import PredictionResult

result = PredictionResult(
    customer_id='CUST001',
    churn_probability=0.75,
    is_high_risk=True,
    prediction_timestamp=datetime.now(),
    model_version='v1.0.0',
    top_features=None  # Will be populated when SHAP is integrated
)

# Convert to dictionary
result_dict = result.to_dict()
```

### 3. ChurnPredictor

Main prediction interface that orchestrates the prediction pipeline.

## Usage Examples

### Initialize Predictor

```python
from services.prediction import ChurnPredictor

# Load deployed or latest model
predictor = ChurnPredictor()

# Load specific model version
predictor = ChurnPredictor(model_version='v1.0.0')

# Use custom high-risk threshold
predictor = ChurnPredictor(high_risk_threshold=0.7)
```

### Single Prediction (Dictionary)

```python
customer = {
    'customer_id': 'CUST001',
    'tenure_months': 12,
    'monthly_charges': 65.5,
    'total_charges': 786.0,
    'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic'
}

result = predictor.predict_single(customer)
print(f"Churn Probability: {result.churn_probability:.2%}")
print(f"High Risk: {result.is_high_risk}")
```

### Single Prediction (CustomerRecord)

```python
from services.prediction import CustomerRecord

record = CustomerRecord(
    customer_id='CUST001',
    features={
        'tenure_months': 12,
        'monthly_charges': 65.5,
        'total_charges': 786.0,
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic'
    }
)

result = predictor.predict_single(record)
```

### Batch Prediction

```python
customers = [
    {
        'customer_id': 'CUST001',
        'tenure_months': 12,
        'monthly_charges': 65.5,
        'total_charges': 786.0,
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic'
    },
    {
        'customer_id': 'CUST002',
        'tenure_months': 48,
        'monthly_charges': 45.0,
        'total_charges': 2160.0,
        'contract_type': 'Two year',
        'payment_method': 'Bank transfer',
        'internet_service': 'DSL'
    }
]

results = predictor.predict_batch(customers)

for result in results:
    print(f"{result.customer_id}: {result.churn_probability:.2%}")
```

### Probability Prediction

```python
import pandas as pd

# Create DataFrame with customer features
data = pd.DataFrame([{
    'tenure_months': 12,
    'monthly_charges': 65.5,
    'total_charges': 786.0,
    'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic'
}])

# Get churn probabilities
probabilities = predictor.predict_proba(data)
print(f"Churn probability: {probabilities[0]:.2%}")
```

### High-Risk Classification

```python
import pandas as pd

data = pd.DataFrame([{
    'tenure_months': 12,
    'monthly_charges': 65.5,
    'total_charges': 786.0,
    'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
    'internet_service': 'Fiber optic'
}])

# Use default threshold (0.5)
high_risk = predictor.predict_high_risk(data)

# Use custom threshold
high_risk_70 = predictor.predict_high_risk(data, threshold=0.7)
```

## Required Features

All predictions require the following features:

**Numerical Features:**
- `tenure_months`: Customer tenure in months
- `monthly_charges`: Current monthly charges
- `total_charges`: Total charges to date

**Categorical Features:**
- `contract_type`: Type of contract (e.g., 'Month-to-month', 'One year', 'Two year')
- `payment_method`: Payment method (e.g., 'Electronic check', 'Credit card', 'Bank transfer')
- `internet_service`: Internet service type (e.g., 'DSL', 'Fiber optic', 'No')

## Feature Engineering

The predictor automatically applies feature engineering to create derived features:

- `tenure_years`: Tenure in years
- `is_new_customer`: Binary flag for customers with tenure <= 12 months
- `is_long_term_customer`: Binary flag for customers with tenure >= 36 months
- `avg_monthly_spend`: Average monthly spend over tenure
- `current_to_avg_ratio`: Ratio of current to average spend
- `total_spend_ratio`: Ratio of monthly to total charges
- `service_count`: Number of services subscribed
- `is_high_value`: Binary flag for high-value customers

## Input Validation

The predictor validates all inputs before making predictions:

1. **Schema Validation**: Checks for required features
2. **Type Checking**: Ensures numerical and categorical features have correct types
3. **Value Validation**: Checks for reasonable value ranges
4. **Missing Values**: Handles missing values through imputation

## Error Handling

The service raises specific exceptions for different error conditions:

- `ValidationError`: Invalid input data (missing features, wrong types, etc.)
- `ModelNotFoundError`: Requested model version doesn't exist
- `ValueError`: Invalid parameters (e.g., batch size exceeds maximum)

## Performance Considerations

- **Single Predictions**: Complete within 2 seconds
- **Batch Predictions**: Process up to 1000 customers within 30 seconds
- **Maximum Batch Size**: 1000 customers per batch

## Integration with Other Services

The prediction service integrates with:

1. **Model Repository**: Loads trained models and transformers
2. **Preprocessing Service**: Applies data validation and transformation
3. **Feature Engineering**: Creates derived features consistently
4. **SHAP Explainer** (coming soon): Will provide feature importance explanations

## Next Steps

1. Integrate SHAP explanations for feature importance
2. Add prediction caching for frequently requested customers
3. Implement prediction logging for monitoring
4. Add API endpoint for remote predictions
5. Create dashboard integration

## Example Script

See `examples/prediction_example.py` for a complete working example demonstrating all features of the prediction service.

## Testing

The prediction service includes comprehensive tests:

- **Unit Tests**: `tests/unit/test_prediction.py`
- **Integration Tests**: `tests/integration/test_prediction_integration.py`

Run tests with:
```bash
pytest tests/unit/test_prediction.py -v
pytest tests/integration/test_prediction_integration.py -v
```
