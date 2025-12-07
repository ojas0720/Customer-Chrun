# Customer Churn Prediction System

A production-ready machine learning system that predicts customer attrition risk using XGBoost and provides interpretable explanations through SHAP (Shapley Additive exPlanations). The system includes model versioning, monitoring, and an interactive dashboard for business insights.

## Features

- **High-Accuracy Predictions**: XGBoost classifier optimized for 85%+ recall on high-risk accounts
- **Explainable AI**: SHAP values provide transparent feature importance for every prediction
- **Model Versioning**: Complete version control with rollback capabilities
- **Real-time Monitoring**: Track model performance and detect degradation
- **Interactive Dashboard**: Streamlit-based UI for exploring predictions and insights
- **Production-Ready**: Comprehensive testing with unit and property-based tests

## Quick Start

### 1. Installation

```bash
# Clone the repository (if applicable)
# cd customer-churn-prediction

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python verify_setup.py
```

See [INSTALL.md](INSTALL.md) for detailed installation instructions and troubleshooting.

### 2. Run the Complete Pipeline

```bash
# Generate sample data
python generate_sample_data.py

# Run end-to-end pipeline (training, prediction, explanation)
python end_to_end_pipeline.py
```

This will train a model, generate predictions, and compute SHAP explanations. See [END_TO_END_PIPELINE.md](END_TO_END_PIPELINE.md) for details.

### 3. Launch the Dashboard

```bash
python run_dashboard.py
```

Open your browser to http://localhost:8501 to explore predictions and insights.

## Project Structure

```
.
├── data/
│   ├── raw/              # Raw customer data (CSV files)
│   └── processed/        # Preprocessed data (generated)
├── models/
│   ├── repository/       # Versioned model artifacts
│   └── transformers/     # Fitted data transformers
├── services/             # Core ML services
│   ├── preprocessing.py  # Data validation and transformation
│   ├── model_training.py # Model training and evaluation
│   ├── model_repository.py # Version management
│   ├── prediction.py     # Prediction service
│   ├── explainability.py # SHAP explanations
│   └── monitoring.py     # Performance monitoring
├── dashboard/            # Streamlit dashboard
│   └── app.py           # Main dashboard application
├── tests/
│   ├── unit/            # Unit tests
│   ├── property/        # Property-based tests (Hypothesis)
│   └── integration/     # Integration tests
├── examples/            # Usage examples
├── docs/                # Detailed documentation
├── config.py            # Configuration settings
├── train_model.py       # Standalone training script
├── generate_sample_data.py # Sample data generator
└── requirements.txt     # Python dependencies
```

## Usage Guide

### Training a Model

#### Option 1: Quick Training (Standalone Script)

```bash
python train_model.py
```

This trains an XGBoost model on the sample data and saves it with automatic versioning.

#### Option 2: Custom Training (Programmatic)

```python
from services.model_training import ModelTrainer
from services.preprocessing import DataValidator, DataTransformer, FeatureEngineer
from services.model_repository import ModelRepository
import pandas as pd

# Load data
train_data = pd.read_csv('data/raw/training_data.csv')

# Validate data
validator = DataValidator()
validator.validate(train_data)

# Preprocess
engineer = FeatureEngineer()
train_data = engineer.engineer_features(train_data)

transformer = DataTransformer()
X_train = transformer.fit_transform(train_data.drop('churn', axis=1), train_data['churn'])
y_train = train_data['churn']

# Train model
trainer = ModelTrainer()
model = trainer.train(X_train, y_train)

# Evaluate
test_data = pd.read_csv('data/raw/test_data.csv')
test_data = engineer.engineer_features(test_data)
X_test = transformer.transform(test_data.drop('churn', axis=1))
y_test = test_data['churn']

metrics = trainer.evaluate(model, X_test, y_test)
print(f"Recall: {metrics['recall']:.4f}")

# Save with versioning
repo = ModelRepository()
version = repo.save(model, transformer, metrics)
print(f"Model saved as version: {version}")
```

### Making Predictions

#### Single Customer Prediction

```python
from services.prediction import ChurnPredictor, CustomerRecord

# Initialize predictor (loads latest model by default)
predictor = ChurnPredictor()

# Create customer record
customer = CustomerRecord(
    customer_id="CUST_001",
    features={
        'tenure_months': 12,
        'monthly_charges': 65.50,
        'total_charges': 786.00,
        'contract_type': 'Month-to-month',
        'payment_method': 'Electronic check',
        'internet_service': 'Fiber optic',
        'online_security': 'No',
        'tech_support': 'No',
        'streaming_tv': 'Yes',
        'streaming_movies': 'Yes'
    }
)

# Get prediction with explanations
result = predictor.predict_single(customer, include_explanations=True)

print(f"Customer ID: {result.customer_id}")
print(f"Churn Probability: {result.churn_probability:.3f}")
print(f"High Risk: {result.is_high_risk}")
print(f"\nTop Contributing Features:")
for feature, value in result.top_features:
    direction = "increases" if value > 0 else "decreases"
    print(f"  {feature}: {value:.4f} ({direction} churn risk)")
```

#### Batch Predictions

```python
import pandas as pd

# Load customer data
customers_df = pd.read_csv('data/raw/customers_to_score.csv')

# Batch predict
results = predictor.predict_batch(customers_df)

# Convert to DataFrame for analysis
results_df = pd.DataFrame([
    {
        'customer_id': r.customer_id,
        'churn_probability': r.churn_probability,
        'is_high_risk': r.is_high_risk
    }
    for r in results
])

# Save results
results_df.to_csv('predictions_output.csv', index=False)

# Summary statistics
print(f"Total customers: {len(results_df)}")
print(f"High-risk accounts: {results_df['is_high_risk'].sum()}")
print(f"Average churn probability: {results_df['churn_probability'].mean():.3f}")
```

### Understanding Predictions with SHAP

```python
from services.explainability import SHAPExplainer
from services.prediction import ChurnPredictor
import pandas as pd

# Load model and data
predictor = ChurnPredictor()
test_data = pd.read_csv('data/raw/test_data.csv')

# Initialize SHAP explainer
explainer = SHAPExplainer(predictor.model, predictor.transformer)

# Get explanations for a customer
customer_data = test_data.iloc[0:1]
shap_values = explainer.explain(customer_data)

# Get top features
top_features = explainer.get_top_features(
    shap_values[0], 
    predictor.feature_names, 
    n=5
)

print("Top 5 Features Contributing to Churn:")
for feature, shap_value in top_features:
    print(f"  {feature}: {shap_value:.4f}")

# Verify SHAP additivity property
prediction = predictor.model.predict_proba(
    predictor.transformer.transform(customer_data)
)[0, 1]
base_value = explainer.explainer.expected_value
is_valid = explainer.validate_shap_sum(shap_values[0], prediction, base_value)
print(f"\nSHAP additivity verified: {is_valid}")
```

### Model Version Management

```python
from services.model_repository import ModelRepository

repo = ModelRepository()

# List all versions
versions = repo.list_versions()
print("Available model versions:")
for v in versions:
    print(f"  {v.version} - Recall: {v.metadata.get('recall', 'N/A')}")

# Load specific version
model, transformer = repo.load(version="v_20251207_095111")

# Get latest version
latest = repo.get_latest_version()
print(f"Latest version: {latest}")

# Rollback to previous version
repo.rollback(version="v1.0.0")
print("Rolled back to v1.0.0")
```

### Monitoring Model Performance

```python
from services.monitoring import ModelEvaluator, AlertService, PredictionStore
import pandas as pd

# Store predictions with actual outcomes
store = PredictionStore()
store.store(
    customer_id="CUST_001",
    prediction=0.75,
    actual_outcome=1,  # Customer did churn
    model_version="v_20251207_095111"
)

# Evaluate model on new data
evaluator = ModelEvaluator()
test_data = pd.read_csv('data/raw/test_data.csv')
X_test = test_data.drop('churn', axis=1)
y_test = test_data['churn']

metrics = evaluator.evaluate(predictor.model, predictor.transformer, X_test, y_test)
print(f"Current Performance:")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall: {metrics['recall']:.4f}")
print(f"  F1-Score: {metrics['f1_score']:.4f}")

# Check for performance degradation
alert_service = AlertService()
alert = alert_service.check_performance(metrics['recall'], threshold=0.85)
if alert:
    print(f"[ALERT] {alert}")

# Get confusion matrix
cm = evaluator.compute_confusion_matrix(y_test, predictions)
print(f"\nConfusion Matrix:")
print(cm)
```

### Running the Dashboard

#### Quick Start

```bash
python run_dashboard.py
```

The dashboard will open automatically in your browser at http://localhost:8501

#### Manual Start

```bash
streamlit run dashboard/app.py --server.port 8501
```

#### Dashboard Features

The interactive dashboard provides three main views:

**1. Overview Tab**
- Total high-risk account count
- Churn probability distribution histogram
- Model performance metrics (precision, recall, F1-score)
- Model version selector

**2. Customer Details Tab**
- Customer selection dropdown
- Individual churn probability
- Top 5 contributing features with SHAP values
- SHAP waterfall plot visualization
- Feature impact interpretation

**3. Statistics Tab**
- Average churn probability across all customers
- Feature importance trends
- Confusion matrix heatmap
- Retention metrics summary

See [docs/dashboard_guide.md](docs/dashboard_guide.md) for detailed dashboard documentation.

## Testing

The system includes comprehensive testing with both unit tests and property-based tests.

### Run All Tests

```bash
pytest tests/ -v
```

### Run Unit Tests Only

```bash
pytest tests/unit/ -v
```

### Run Property-Based Tests

Property-based tests use Hypothesis to verify correctness properties across randomly generated inputs:

```bash
pytest tests/property/ -v
```

### Run Integration Tests

```bash
pytest tests/integration/ -v
```

### Test Coverage

```bash
pytest tests/ --cov=services --cov-report=html
```

View coverage report by opening `htmlcov/index.html` in your browser.

## Configuration

Edit `config.py` to customize system behavior:

```python
# Model hyperparameters
XGBOOST_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Prediction settings
HIGH_RISK_THRESHOLD = 0.5  # Probability threshold for high-risk classification
MIN_RECALL_THRESHOLD = 0.85  # Minimum acceptable recall for alerts

# File paths
MODEL_REPOSITORY_PATH = 'models/repository'
TRANSFORMER_PATH = 'models/transformers'
DATA_PATH = 'data/raw'

# Dashboard settings
DASHBOARD_PORT = 8501
SHAP_SAMPLE_SIZE = 100  # Number of samples for SHAP background
```

## Examples

The `examples/` directory contains standalone scripts demonstrating each component:

- `prediction_example.py` - Making predictions with the prediction service
- `explainability_example.py` - Computing SHAP explanations
- `model_repository_example.py` - Managing model versions
- `monitoring_example.py` - Evaluating and monitoring model performance
- `dashboard_example.py` - Programmatic dashboard data access

Run any example:

```bash
python examples/prediction_example.py
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Installation Guide](INSTALL.md) - Setup and troubleshooting
- [End-to-End Pipeline](END_TO_END_PIPELINE.md) - Complete workflow walkthrough
- [Prediction Service Guide](docs/prediction_service_guide.md)
- [Explainability Guide](docs/explainability_guide.md)
- [Model Repository Guide](docs/model_repository_guide.md)
- [Monitoring Service Guide](docs/monitoring_service_guide.md)
- [Dashboard Guide](docs/dashboard_guide.md)

## Architecture

The system follows a modular architecture with clear separation of concerns:

- **Data Layer**: Validation, transformation, and feature engineering
- **Model Layer**: Training, versioning, and persistence
- **Prediction Layer**: Inference and batch processing
- **Explainability Layer**: SHAP value computation and interpretation
- **Monitoring Layer**: Performance tracking and alerting
- **Presentation Layer**: Interactive dashboard

See the [Design Document](.kiro/specs/customer-churn-prediction/design.md) for detailed architecture information.

## Requirements

- **Python**: 3.8 or higher
- **Key Dependencies**:
  - XGBoost >= 2.0.0
  - SHAP >= 0.43.0
  - scikit-learn >= 1.3.0
  - pandas >= 2.0.0
  - Streamlit >= 1.28.0
  - pytest >= 7.4.0
  - Hypothesis >= 6.92.0

See [requirements.txt](requirements.txt) for complete dependency list.

## Troubleshooting

### Common Issues

**Model not found error**
```bash
# Ensure you've trained a model first
python train_model.py
```

**Dashboard won't start**
```bash
# Check if port 8501 is already in use
# Try a different port
streamlit run dashboard/app.py --server.port 8502
```

**Low recall warning**
```bash
# Adjust scale_pos_weight in config.py to increase recall
# Retrain the model
python train_model.py
```

**SHAP computation slow**
```bash
# Reduce SHAP_SAMPLE_SIZE in config.py
# Or reduce the number of customers being explained
```

See [INSTALL.md](INSTALL.md) for more troubleshooting guidance.

## Contributing

When adding new features:

1. Follow the existing code structure
2. Add unit tests for specific functionality
3. Add property-based tests for universal properties
4. Update relevant documentation
5. Run all tests before committing
