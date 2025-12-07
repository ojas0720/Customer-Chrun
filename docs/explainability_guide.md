# SHAP Explainability Guide

## Overview

The SHAP (SHapley Additive exPlanations) explainability service provides interpretable explanations for churn predictions. It helps business users understand **why** the model predicts a customer will churn by identifying the most important contributing features.

## Key Concepts

### What is SHAP?

SHAP values are based on game theory and provide a unified measure of feature importance. For each prediction:
- Each feature gets a SHAP value representing its contribution to the prediction
- Positive SHAP values increase churn risk
- Negative SHAP values decrease churn risk
- The sum of all SHAP values equals the difference between the prediction and the base value

### Why Use SHAP?

1. **Interpretability**: Understand which features drive each prediction
2. **Trust**: Build confidence in model decisions with transparent explanations
3. **Actionability**: Identify specific factors to address for customer retention
4. **Fairness**: Detect potential biases in model predictions

## Usage

### Basic Setup

To enable SHAP explanations, you need to provide background data when creating a `ChurnPredictor`:

```python
from services.prediction import ChurnPredictor
from services.model_repository import ModelRepository
import numpy as np

# Load model and prepare background data
repository = ModelRepository()
model, transformer = repository.load_deployed()

# Background data should be a representative sample of training data (preprocessed)
# Typically 50-100 samples is sufficient
background_data = X_train_transformed[:100]  # numpy array

# Create predictor with explanations enabled
predictor = ChurnPredictor(
    enable_explanations=True,
    background_data=background_data
)
```

### Single Prediction with Explanation

```python
from services.prediction import CustomerRecord

customer = CustomerRecord(
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

# Get prediction with explanation
result = predictor.predict_single(customer, include_explanations=True)

print(f"Churn Probability: {result.churn_probability:.2%}")
print(f"Top Contributing Features:")
for feature_name, shap_value in result.top_features:
    direction = "increases" if shap_value > 0 else "decreases"
    print(f"  {feature_name}: {shap_value:+.4f} ({direction} churn risk)")
```

### Batch Predictions with Explanations

```python
customers = [
    {
        'customer_id': 'CUST_001',
        'tenure_months': 6,
        'monthly_charges': 85.50,
        # ... other features
    },
    {
        'customer_id': 'CUST_002',
        'tenure_months': 48,
        'monthly_charges': 55.00,
        # ... other features
    }
]

results = predictor.predict_batch(customers, include_explanations=True)

for result in results:
    print(f"\nCustomer: {result.customer_id}")
    print(f"Churn Probability: {result.churn_probability:.2%}")
    print(f"Top Features: {result.top_features[:3]}")
```

### Using SHAPExplainer Directly

For advanced use cases, you can use the `SHAPExplainer` class directly:

```python
from services.explainability import SHAPExplainer
import numpy as np

# Initialize explainer
explainer = SHAPExplainer(model, background_data)

# Compute SHAP values for preprocessed customer data
customer_data_transformed = transformer.transform(customer_data)
shap_values = explainer.explain(customer_data_transformed)

# Get top contributing features
feature_names = transformer.get_feature_names_out()
top_features = explainer.get_top_features(
    shap_values[0],
    feature_names,
    n=5
)

# Validate SHAP additivity property
prediction_margin = model.predict(customer_data_transformed, output_margin=True)[0]
is_valid = explainer.validate_shap_sum(shap_values[0], prediction_margin)
```

## Interpreting SHAP Values

### Sign and Magnitude

- **Positive SHAP value**: Feature increases churn probability
  - Example: `contract_type_Month-to-month: +0.35` means this contract type increases churn risk
  
- **Negative SHAP value**: Feature decreases churn probability
  - Example: `tenure_months: -0.42` means longer tenure decreases churn risk

- **Magnitude**: Larger absolute values indicate stronger influence
  - `+0.50` has more impact than `+0.10`

### Common Patterns

Based on typical churn prediction models, you might see:

**High Churn Risk Indicators** (positive SHAP values):
- Short tenure (new customers)
- Month-to-month contracts
- Electronic check payment method
- High monthly charges relative to tenure
- Recent price increases

**Low Churn Risk Indicators** (negative SHAP values):
- Long tenure (loyal customers)
- Long-term contracts (1-2 years)
- Automatic payment methods
- Multiple services subscribed
- Consistent payment history

## Configuration

### Background Data Size

The number of background samples affects computation time and explanation quality:

```python
# config.py
SHAP_BACKGROUND_SAMPLES = 100  # Default: 100 samples
```

- **Fewer samples (50)**: Faster computation, less precise
- **More samples (200)**: Slower computation, more precise
- **Recommended**: 100 samples for good balance

### Top Features Count

Control how many top features are returned:

```python
# config.py
TOP_FEATURES_COUNT = 5  # Default: 5 features
```

You can also override this per call:

```python
top_features = explainer.get_top_features(shap_values, feature_names, n=10)
```

## Performance Considerations

### Computation Time

SHAP computation adds overhead to predictions:
- Single prediction: ~50-200ms additional time
- Batch of 100: ~2-5 seconds additional time

For production systems with strict latency requirements:
1. Pre-compute explanations for high-risk customers only
2. Use async processing for batch explanations
3. Cache explanations for repeated queries
4. Consider reducing background sample size

### Memory Usage

SHAP explainer stores background data in memory:
- 100 samples × 20 features × 8 bytes ≈ 16 KB
- Negligible for most applications

## Best Practices

### 1. Use Representative Background Data

```python
# Good: Random sample from training data
background_indices = np.random.choice(len(X_train), size=100, replace=False)
background_data = X_train_transformed[background_indices]

# Bad: Only high-risk customers
# This biases the base value and SHAP values
```

### 2. Enable Explanations Selectively

```python
# For high-risk customers only
if result.is_high_risk:
    result_with_explanation = predictor.predict_single(
        customer,
        include_explanations=True
    )
```

### 3. Validate SHAP Additivity

```python
# Ensure SHAP values are computed correctly
is_valid = explainer.validate_shap_sum(shap_values, prediction_margin)
if not is_valid:
    logger.warning("SHAP additivity check failed - values may be unreliable")
```

### 4. Handle Missing Explanations Gracefully

```python
result = predictor.predict_single(customer, include_explanations=True)

if result.top_features is None:
    print("Explanation not available")
else:
    print(f"Top features: {result.top_features}")
```

## Troubleshooting

### Error: "background_data is required when enable_explanations=True"

**Solution**: Provide background data when creating the predictor:

```python
predictor = ChurnPredictor(
    enable_explanations=True,
    background_data=X_train_transformed[:100]
)
```

### Error: "Explanations requested but explainer not initialized"

**Solution**: Enable explanations when creating the predictor:

```python
# Wrong
predictor = ChurnPredictor(enable_explanations=False)
result = predictor.predict_single(customer, include_explanations=True)  # Error!

# Correct
predictor = ChurnPredictor(enable_explanations=True, background_data=background_data)
result = predictor.predict_single(customer, include_explanations=True)  # Works!
```

### SHAP Values Don't Sum Correctly

**Cause**: SHAP values are computed in log-odds space for XGBoost models.

**Solution**: Use `output_margin=True` when validating:

```python
prediction_margin = model.predict(X, output_margin=True)[0]
is_valid = explainer.validate_shap_sum(shap_values[0], prediction_margin)
```

### Slow Explanation Computation

**Solutions**:
1. Reduce background sample size: `SHAP_BACKGROUND_SAMPLES = 50`
2. Use batch predictions instead of single predictions
3. Pre-compute explanations offline for known customers
4. Consider using SHAP's `approximate=True` option (requires code modification)

## Examples

See `examples/explainability_example.py` for a complete working example demonstrating:
- Single customer prediction with explanation
- Batch predictions with explanations
- Interpreting SHAP values
- Identifying key churn drivers

## API Reference

### SHAPExplainer

```python
class SHAPExplainer:
    def __init__(self, model: XGBClassifier, background_data: np.ndarray)
    def explain(self, customer_data: np.ndarray) -> np.ndarray
    def get_top_features(self, shap_values: np.ndarray, feature_names: List[str], n: int = 5) -> List[Tuple[str, float]]
    def validate_shap_sum(self, shap_values: np.ndarray, prediction: float, tolerance: float = 0.001) -> bool
```

### ChurnPredictor (with explanations)

```python
class ChurnPredictor:
    def __init__(
        self,
        model_version: Optional[str] = None,
        repository: Optional[ModelRepository] = None,
        high_risk_threshold: float = 0.5,
        enable_explanations: bool = False,
        background_data: Optional[np.ndarray] = None
    )
    
    def predict_single(
        self,
        customer: Union[Dict, CustomerRecord],
        include_explanations: bool = False
    ) -> PredictionResult
    
    def predict_batch(
        self,
        customers: List[Union[Dict, CustomerRecord]],
        include_explanations: bool = False
    ) -> List[PredictionResult]
```

### PredictionResult

```python
@dataclass
class PredictionResult:
    customer_id: str
    churn_probability: float
    is_high_risk: bool
    prediction_timestamp: datetime
    model_version: str
    top_features: Optional[List[Tuple[str, float]]]  # (feature_name, shap_value)
```

## Further Reading

- [SHAP Documentation](https://shap.readthedocs.io/)
- [SHAP Paper](https://arxiv.org/abs/1705.07874)
- [TreeExplainer for XGBoost](https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20XGBoost.html)
