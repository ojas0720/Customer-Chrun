# Design Document: Customer Churn Prediction System

## Overview

The Customer Churn Prediction System is a machine learning application that predicts customer attrition risk using XGBoost and provides interpretable explanations through SHAP (Shapley Additive exPlanations). The system consists of three main components: a training pipeline for model development, a prediction pipeline for inference, and a visualization dashboard for business insights.

The architecture follows a modular design with clear separation between data processing, model operations, explainability, and presentation layers. This enables independent testing, maintenance, and evolution of each component.

## Architecture

The system follows a layered architecture:

```
┌─────────────────────────────────────────────────────────┐
│                  Presentation Layer                      │
│              (Retention Dashboard - Streamlit)           │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Training   │  │  Prediction  │  │ Explanation  │  │
│  │   Service    │  │   Service    │  │   Service    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                    Core ML Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   XGBoost    │  │     SHAP     │  │ Preprocessor │  │
│  │    Model     │  │   Explainer  │  │              │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│                     Data Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Model      │  │  Prediction  │  │   Training   │  │
│  │  Repository  │  │    Store     │  │     Data     │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Components and Interfaces

### 1. Data Preprocessing Module

**Responsibilities:**
- Load and validate raw customer data
- Handle missing values through imputation
- Encode categorical features
- Scale numerical features
- Generate derived features

**Key Classes:**
- `DataValidator`: Validates input data schema and value ranges
- `FeatureEngineer`: Creates derived features from raw data
- `DataTransformer`: Applies encoding and scaling transformations

**Interfaces:**
```python
class DataTransformer:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'DataTransformer'
    def transform(self, X: pd.DataFrame) -> np.ndarray
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray
    def save(self, path: str) -> None
    def load(self, path: str) -> 'DataTransformer'
```

### 2. Model Training Service

**Responsibilities:**
- Train XGBoost classifier on preprocessed data
- Optimize hyperparameters
- Evaluate model performance
- Persist trained models with versioning

**Key Classes:**
- `ModelTrainer`: Orchestrates the training process
- `HyperparameterTuner`: Performs hyperparameter optimization
- `ModelEvaluator`: Computes performance metrics

**Interfaces:**
```python
class ModelTrainer:
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              params: dict) -> XGBClassifier
    def evaluate(self, model: XGBClassifier, X_test: np.ndarray, 
                 y_test: np.ndarray) -> dict
    def save_model(self, model: XGBClassifier, version: str, 
                   metadata: dict) -> None
```

### 3. Prediction Service

**Responsibilities:**
- Load trained models
- Generate churn probability predictions
- Classify accounts as high-risk based on threshold
- Handle batch predictions

**Key Classes:**
- `ChurnPredictor`: Main prediction interface
- `ModelLoader`: Loads versioned models from storage

**Interfaces:**
```python
class ChurnPredictor:
    def predict_proba(self, customer_data: pd.DataFrame) -> np.ndarray
    def predict_high_risk(self, customer_data: pd.DataFrame, 
                          threshold: float = 0.5) -> np.ndarray
    def predict_batch(self, customers: List[dict]) -> List[dict]
```

### 4. Explainability Service

**Responsibilities:**
- Compute SHAP values for predictions
- Identify top contributing features
- Generate feature importance visualizations
- Ensure explanation consistency

**Key Classes:**
- `SHAPExplainer`: Computes Shapley values
- `FeatureImportanceAnalyzer`: Analyzes and ranks feature contributions

**Interfaces:**
```python
class SHAPExplainer:
    def __init__(self, model: XGBClassifier, background_data: np.ndarray)
    def explain(self, customer_data: np.ndarray) -> np.ndarray
    def get_top_features(self, shap_values: np.ndarray, 
                         feature_names: List[str], n: int = 5) -> List[tuple]
    def validate_shap_sum(self, shap_values: np.ndarray, 
                          prediction: float, base_value: float) -> bool
```

### 5. Model Repository

**Responsibilities:**
- Store and retrieve model artifacts
- Manage model versions
- Track model metadata and performance

**Key Classes:**
- `ModelRepository`: Handles model persistence and retrieval
- `ModelVersion`: Represents a versioned model with metadata

**Interfaces:**
```python
class ModelRepository:
    def save(self, model: XGBClassifier, transformer: DataTransformer,
             version: str, metadata: dict) -> None
    def load(self, version: str) -> tuple[XGBClassifier, DataTransformer]
    def list_versions(self) -> List[ModelVersion]
    def get_latest_version(self) -> str
    def rollback(self, version: str) -> None
```

### 6. Retention Dashboard

**Responsibilities:**
- Display churn predictions and metrics
- Visualize SHAP explanations
- Show customer-level details
- Present aggregated statistics

**Technology:** Streamlit for rapid dashboard development

**Key Components:**
- High-risk account counter
- Churn probability distribution chart
- Customer detail view with SHAP waterfall plot
- Feature importance summary
- Model performance metrics display

## Data Models

### CustomerRecord
```python
@dataclass
class CustomerRecord:
    customer_id: str
    features: dict  # Raw feature values
    
    def validate(self) -> bool
    def to_dataframe(self) -> pd.DataFrame
```

### PredictionResult
```python
@dataclass
class PredictionResult:
    customer_id: str
    churn_probability: float
    is_high_risk: bool
    top_features: List[tuple[str, float]]  # (feature_name, shap_value)
    prediction_timestamp: datetime
    model_version: str
```

### ModelMetadata
```python
@dataclass
class ModelMetadata:
    version: str
    timestamp: datetime
    performance_metrics: dict  # precision, recall, f1, etc.
    hyperparameters: dict
    feature_names: List[str]
    training_data_size: int
```

### ValidationResult
```python
@dataclass
class ValidationResult:
    model_version: str
    evaluation_date: datetime
    recall: float
    precision: float
    f1_score: float
    confusion_matrix: np.ndarray
    meets_threshold: bool  # recall >= 0.85
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Training Pipeline Properties

**Property 1: Model persistence round-trip**
*For any* trained XGBoost model, saving and then loading the model should produce a model that makes identical predictions on the same input data.
**Validates: Requirements 1.1**

**Property 2: Missing value handling robustness**
*For any* training dataset with missing values, the preprocessing pipeline should complete without errors and produce a valid transformed dataset with no missing values.
**Validates: Requirements 1.3**

**Property 3: Training data validation**
*For any* dataset missing required features or the target variable, the validation function should reject it and raise an appropriate error.
**Validates: Requirements 1.4**

**Property 4: Metrics completeness**
*For any* completed training run, the returned metrics dictionary should contain keys for 'precision', 'recall', 'f1_score', and all values should be between 0 and 1.
**Validates: Requirements 1.5**

### Prediction Pipeline Properties

**Property 5: Probability bounds**
*For any* customer record, the predicted churn probability should be a float value between 0.0 and 1.0 inclusive.
**Validates: Requirements 2.1**

**Property 6: High-risk classification consistency**
*For any* prediction with probability p and threshold t, the high-risk classification should be True if and only if p >= t.
**Validates: Requirements 2.2**

**Property 7: Batch prediction completeness**
*For any* batch of n customer records, the prediction function should return exactly n prediction results.
**Validates: Requirements 2.3**

**Property 8: Invalid input rejection**
*For any* customer record with missing required features or out-of-range values, the prediction function should raise a validation error rather than returning a prediction.
**Validates: Requirements 2.4**

### Explainability Properties

**Property 9: SHAP completeness**
*For any* prediction, the SHAP explainer should return Shapley values for every input feature used by the model.
**Validates: Requirements 3.1**

**Property 10: Top features ranking**
*For any* set of SHAP values, the top 5 features returned should be sorted by absolute value in descending order and contain exactly 5 features (or fewer if the model has fewer features).
**Validates: Requirements 3.2**

**Property 11: SHAP sign interpretation**
*For any* SHAP value, a positive value should be marked as increasing churn risk and a negative value should be marked as decreasing churn risk.
**Validates: Requirements 3.3**

**Property 12: SHAP additivity**
*For any* prediction, the sum of all SHAP values should equal the difference between the prediction and the base value (within numerical tolerance of 0.001).
**Validates: Requirements 3.5**

### Dashboard Properties

**Property 13: Aggregate statistics correctness**
*For any* set of predictions, the computed average churn probability should equal the mean of all individual probabilities.
**Validates: Requirements 4.5**

### Validation Properties

**Property 14: Evaluation metrics computation**
*For any* set of predictions and true labels, the evaluation function should return a dictionary containing 'precision', 'recall', 'f1_score', and 'confusion_matrix'.
**Validates: Requirements 5.1**

**Property 15: Alert triggering logic**
*For any* evaluation result with recall below 0.85, an alert should be generated; for recall >= 0.85, no alert should be generated.
**Validates: Requirements 5.2**

**Property 16: Prediction storage completeness**
*For any* stored prediction record, it should contain both the 'prediction' field and the 'actual_outcome' field.
**Validates: Requirements 5.3**

**Property 17: Confusion matrix correctness**
*For any* set of binary predictions and labels, the sum of all confusion matrix elements should equal the total number of samples.
**Validates: Requirements 5.4**

**Property 18: Calibration curve computation**
*For any* set of predicted probabilities and true labels, the calibration function should return probability bins and corresponding true frequencies.
**Validates: Requirements 5.5**

### Model Management Properties

**Property 19: Version uniqueness**
*For any* two models trained at different times, they should receive different version identifiers.
**Validates: Requirements 6.1**

**Property 20: Version selection**
*For any* selected model version, predictions should use the model artifacts associated with that specific version.
**Validates: Requirements 6.2**

**Property 21: Parameter validation**
*For any* invalid hyperparameter configuration (e.g., negative learning rate, invalid max_depth), the validation function should reject it before training begins.
**Validates: Requirements 6.3**

**Property 22: Deployment completeness**
*For any* deployed model version, both the model artifact file and metadata file should exist in the repository.
**Validates: Requirements 6.4**

**Property 23: Rollback restoration**
*For any* model version V1, if we deploy V2 and then rollback, the active model should be V1 and predictions should match V1's behavior.
**Validates: Requirements 6.5**

### Preprocessing Properties

**Property 24: Categorical encoding consistency**
*For any* categorical value seen during training, encoding it during inference should produce the same encoded value.
**Validates: Requirements 7.2**

**Property 25: Numerical scaling consistency**
*For any* numerical value, scaling it with the fitted scaler should produce the same result whether done during training or inference.
**Validates: Requirements 7.3**

**Property 26: Unknown category handling**
*For any* categorical value not seen during training, the encoder should assign it to a default category without raising an error.
**Validates: Requirements 7.4**

**Property 27: Feature engineering consistency**
*For any* customer record, derived features should be computed identically whether during training or inference preprocessing.
**Validates: Requirements 7.5**

## Error Handling

The system implements comprehensive error handling across all components:

### Input Validation Errors
- **MissingFeatureError**: Raised when required features are missing from input data
- **InvalidValueError**: Raised when feature values are out of expected ranges
- **SchemaValidationError**: Raised when data doesn't match expected schema

### Model Errors
- **ModelNotFoundError**: Raised when requested model version doesn't exist
- **ModelLoadError**: Raised when model artifacts are corrupted or incompatible
- **PredictionError**: Raised when prediction fails due to model issues

### Data Processing Errors
- **TransformationError**: Raised when preprocessing transformations fail
- **EncodingError**: Raised when categorical encoding encounters issues
- **ScalingError**: Raised when numerical scaling fails

### Explainability Errors
- **SHAPComputationError**: Raised when SHAP value calculation fails
- **ExplanationError**: Raised when feature importance analysis encounters issues

### Error Recovery Strategies
1. **Graceful Degradation**: Return partial results when possible (e.g., predictions without explanations if SHAP fails)
2. **Logging**: All errors are logged with full context for debugging
3. **User-Friendly Messages**: Error messages provide actionable guidance
4. **Validation First**: Validate inputs before expensive operations to fail fast

## Testing Strategy

The system employs a dual testing approach combining unit tests and property-based tests to ensure comprehensive correctness validation.

### Unit Testing Approach

Unit tests verify specific examples, edge cases, and integration points:

**Key Test Areas:**
- **Data Validation**: Test specific invalid inputs (empty dataframes, wrong column names, null values)
- **Model Loading**: Test loading specific model versions and handling missing files
- **Dashboard Components**: Test rendering with specific customer data examples
- **Error Conditions**: Test specific error scenarios (division by zero, empty predictions)
- **Integration Points**: Test component interactions with known data

**Example Unit Tests:**
- Test that empty customer record raises ValidationError
- Test that model version "v1.0.0" loads correctly
- Test that high-risk threshold of 0.7 correctly classifies probability 0.75
- Test that SHAP explainer handles single-feature models

### Property-Based Testing Approach

Property-based tests verify universal properties across randomly generated inputs:

**Testing Framework:** We will use **Hypothesis** for Python, which provides powerful property-based testing capabilities with built-in strategies for generating test data.

**Configuration:**
- Each property test should run a minimum of **100 iterations** to ensure thorough coverage
- Tests should use appropriate Hypothesis strategies to generate valid customer data, model parameters, and predictions
- Each property test MUST include a comment tag referencing the design document property

**Tag Format:** `# Feature: customer-churn-prediction, Property X: [property description]`

**Key Property Tests:**
- **Property 1**: Generate random models, save/load, verify identical predictions
- **Property 5**: Generate random customer records, verify all probabilities in [0,1]
- **Property 7**: Generate random batch sizes, verify output count matches input count
- **Property 12**: Generate random predictions, verify SHAP values sum correctly
- **Property 23**: Generate random model versions, verify rollback restores correct version

**Test Data Generation Strategies:**
- Use Hypothesis strategies to generate valid customer records with realistic feature distributions
- Generate edge cases automatically (empty strings, boundary values, extreme numbers)
- Create invalid inputs to test error handling
- Generate various batch sizes (1, 10, 100, 1000) for performance testing

### Test Organization

```
tests/
├── unit/
│   ├── test_data_validation.py
│   ├── test_model_training.py
│   ├── test_prediction.py
│   ├── test_explainability.py
│   └── test_dashboard.py
├── property/
│   ├── test_training_properties.py
│   ├── test_prediction_properties.py
│   ├── test_explainability_properties.py
│   ├── test_validation_properties.py
│   ├── test_model_management_properties.py
│   └── test_preprocessing_properties.py
└── integration/
    ├── test_end_to_end_pipeline.py
    └── test_dashboard_integration.py
```

### Continuous Validation

- Run unit tests on every code change
- Run property tests nightly or before releases
- Monitor test coverage (target: >85% line coverage)
- Track property test failure rates to identify flaky tests
- Validate model performance metrics against requirements in CI/CD pipeline

## Implementation Notes

### Technology Stack
- **ML Framework**: XGBoost 2.0+
- **Explainability**: SHAP 0.43+
- **Data Processing**: pandas 2.0+, numpy 1.24+, scikit-learn 1.3+
- **Dashboard**: Streamlit 1.28+
- **Testing**: pytest 7.4+, Hypothesis 6.92+
- **Model Storage**: joblib for serialization

### Performance Considerations
- Use XGBoost's native handling of missing values when possible
- Cache SHAP explainer background data to avoid recomputation
- Implement batch prediction with vectorized operations
- Use lazy loading for dashboard data to improve responsiveness
- Consider model quantization for faster inference if needed

### Security Considerations
- Validate all user inputs to prevent injection attacks
- Sanitize file paths for model loading to prevent directory traversal
- Implement rate limiting on prediction API if exposed
- Log all model version changes for audit trail
- Encrypt sensitive customer data at rest and in transit

### Scalability Considerations
- Design prediction service to be stateless for horizontal scaling
- Use connection pooling for database access
- Implement caching layer for frequently accessed predictions
- Consider async processing for batch predictions
- Plan for model serving infrastructure (e.g., separate prediction service)

