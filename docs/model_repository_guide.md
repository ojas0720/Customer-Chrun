# Model Repository Guide

## Overview

The Model Repository system provides comprehensive version management for XGBoost models and their associated data transformers. It enables teams to save, load, track, and rollback model versions with full metadata support.

## Components

### 1. ModelRepository Class

Located in `services/model_repository.py`, this class handles all model versioning operations.

**Key Features:**
- Automatic timestamp-based versioning
- Custom version naming support
- Model and transformer persistence
- Metadata storage and retrieval
- Version listing and comparison
- Deployment tracking
- Rollback capabilities

**Basic Usage:**

```python
from services.model_repository import ModelRepository

# Initialize repository
repo = ModelRepository()

# Save a model with metadata
version = repo.save(
    model=trained_model,
    transformer=fitted_transformer,
    version="v1.0.0",  # Optional, auto-generated if not provided
    metadata={
        'metrics': {'recall': 0.87, 'precision': 0.75},
        'description': 'Production model'
    }
)

# Load a specific version
model, transformer = repo.load("v1.0.0")

# List all versions
versions = repo.list_versions()
for v in versions:
    print(f"{v.version}: Recall={v.metrics.get('recall', 'N/A')}")

# Get latest version
latest = repo.get_latest_version()

# Rollback to a previous version
repo.rollback("v1.0.0")

# Load deployed version
deployed_model, deployed_transformer = repo.load_deployed()
```

### 2. ParameterValidator Class

Located in `services/parameter_validation.py`, this class validates XGBoost hyperparameters before training.

**Key Features:**
- Validates numerical parameter ranges
- Validates categorical parameter values
- Provides detailed error messages
- Supports parameter info lookup

**Basic Usage:**

```python
from services.parameter_validation import ParameterValidator

validator = ParameterValidator()

# Validate parameters
params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 100,
    'objective': 'binary:logistic'
}

try:
    validator.validate(params)
    print("Parameters are valid!")
except ParameterValidationError as e:
    print(f"Validation failed: {e}")

# Get validation errors without raising exception
errors = validator.validate_and_get_errors(params)
if errors:
    for error in errors:
        print(f"- {error}")

# Get parameter constraints
info = validator.get_parameter_info('learning_rate')
print(info)
```

## Directory Structure

```
models/
├── repository/           # Model versions
│   ├── v1.0.0/
│   │   ├── model.pkl
│   │   └── metadata.json
│   ├── v2.0.0/
│   │   ├── model.pkl
│   │   └── metadata.json
│   └── .deployed_version  # Tracks current deployment
└── transformers/         # Data transformers
    ├── v1.0.0_transformer.pkl
    └── v2.0.0_transformer.pkl
```

## Metadata Format

Each model version includes a `metadata.json` file with:

```json
{
  "version": "v1.0.0",
  "timestamp": "2025-12-07T09:00:00.000000",
  "n_features": 12,
  "params": {
    "learning_rate": 0.1,
    "max_depth": 6,
    "n_estimators": 100
  },
  "metrics": {
    "precision": 0.75,
    "recall": 0.87,
    "f1_score": 0.80
  },
  "features": {
    "numerical": ["tenure_months", "monthly_charges"],
    "categorical": ["contract_type", "payment_method"]
  },
  "description": "Production model with optimized recall"
}
```

## Workflow Examples

### Training and Saving a New Model

```python
from services.model_training import ModelTrainer
from services.preprocessing import DataTransformer
from services.model_repository import ModelRepository
from services.parameter_validation import ParameterValidator

# 1. Validate parameters
validator = ParameterValidator()
params = {'learning_rate': 0.1, 'max_depth': 6, 'n_estimators': 100}
validator.validate(params)

# 2. Prepare data
transformer = DataTransformer(
    numerical_features=['tenure_months', 'monthly_charges'],
    categorical_features=['contract_type']
)
X_train_transformed = transformer.fit_transform(X_train)

# 3. Train model
trainer = ModelTrainer(params=params)
model = trainer.train(X_train_transformed, y_train)
metrics = trainer.evaluate(model, X_test_transformed, y_test)

# 4. Save to repository
repo = ModelRepository()
version = repo.save(
    model,
    transformer,
    metadata={'metrics': metrics}
)
print(f"Saved as {version}")
```

### Loading and Using a Model

```python
from services.model_repository import ModelRepository

# Load specific version
repo = ModelRepository()
model, transformer = repo.load("v1.0.0")

# Transform new data
X_new_transformed = transformer.transform(X_new)

# Make predictions
predictions = model.predict_proba(X_new_transformed)
```

### Version Management

```python
from services.model_repository import ModelRepository

repo = ModelRepository()

# List all versions
versions = repo.list_versions()
print(f"Available versions: {len(versions)}")

# Compare versions
for v in versions:
    print(f"{v.version}:")
    print(f"  Recall: {v.metrics.get('recall', 'N/A')}")
    print(f"  Timestamp: {v.timestamp}")

# Deploy a specific version
repo.rollback("v2.0.0")
print(f"Deployed: {repo.get_deployed_version()}")

# Load deployed model
model, transformer = repo.load_deployed()
```

## Error Handling

The repository system includes comprehensive error handling:

- `ModelNotFoundError`: Raised when a requested version doesn't exist
- `ParameterValidationError`: Raised when parameters are invalid
- `ValueError`: Raised for invalid operations (e.g., saving unfitted models)
- `FileNotFoundError`: Raised when model/transformer files are missing

## Testing

The implementation includes comprehensive test coverage:

- **Unit Tests**: 51 tests covering all core functionality
  - `tests/unit/test_model_repository.py` (19 tests)
  - `tests/unit/test_parameter_validation.py` (32 tests)

- **Integration Tests**: 4 tests covering end-to-end workflows
  - `tests/integration/test_model_repository_integration.py`

Run tests with:
```bash
pytest tests/unit/test_model_repository.py -v
pytest tests/unit/test_parameter_validation.py -v
pytest tests/integration/test_model_repository_integration.py -v
```

## Example Script

A complete working example is available at `examples/model_repository_example.py`. Run it with:

```bash
python examples/model_repository_example.py
```

This demonstrates:
- Parameter validation
- Training multiple model versions
- Saving models with metadata
- Listing and comparing versions
- Loading specific versions
- Rollback functionality
- Making predictions with loaded models

## Requirements Validation

This implementation satisfies the following requirements from the design document:

- **Requirement 6.1**: Unique version identifiers with timestamps ✓
- **Requirement 6.2**: Version selection for predictions ✓
- **Requirement 6.3**: Parameter validation before training ✓
- **Requirement 6.4**: Versioned repository with artifacts and metadata ✓
- **Requirement 6.5**: Rollback to previous versions ✓

## Best Practices

1. **Always validate parameters** before training
2. **Include meaningful metadata** when saving models (metrics, description, features)
3. **Use semantic versioning** for production models (e.g., v1.0.0, v1.1.0, v2.0.0)
4. **Test rollback procedures** before deploying to production
5. **Monitor deployed version** and track performance metrics
6. **Keep metadata up-to-date** with accurate metrics and descriptions
7. **Document version changes** in metadata descriptions
