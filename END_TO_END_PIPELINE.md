# End-to-End Pipeline Documentation

## Overview

The `end_to_end_pipeline.py` script demonstrates the complete Customer Churn Prediction System workflow from training to deployment. It serves as both a validation tool and a reference implementation for the entire system.

## What It Does

The pipeline executes 7 sequential steps:

### Step 1: Load and Validate Data
- Loads training data (2000 samples) and test data (500 samples)
- Validates data schema and value ranges
- Reports data statistics including churn rates

### Step 2: Preprocess Data
- Applies feature engineering (creates derived features)
- Handles missing values through imputation
- Encodes categorical features
- Scales numerical features
- Transforms data for model input

### Step 3: Train Model
- Trains XGBoost classifier with optimized hyperparameters
- Configures scale_pos_weight for high recall (targeting 85%+)
- Evaluates model on test set
- Reports precision, recall, and F1-score

### Step 4: Save Model with Versioning
- Saves trained model with timestamp-based version
- Stores model metadata including performance metrics
- Saves fitted transformer for consistent preprocessing

### Step 5: Generate Predictions
- Loads the trained model
- Generates churn probability predictions for all test customers
- Classifies high-risk accounts (probability >= 0.5)
- Reports prediction statistics

### Step 6: Compute SHAP Explanations
- Samples 100 customers for explanation (configurable)
- Initializes SHAP explainer with background data
- Computes Shapley values for feature importance
- Extracts top 5 contributing features per customer

### Step 7: Store and Evaluate
- Stores all predictions with actual outcomes
- Evaluates final model performance
- Computes confusion matrix
- Checks if recall meets 85% threshold
- Alerts if performance degradation detected

## Usage

### Basic Execution

```bash
python end_to_end_pipeline.py
```

### Prerequisites

1. Ensure synthetic data exists:
   ```bash
   python generate_sample_data.py
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Output

The pipeline generates:

1. **Console Output**: Real-time progress and metrics
2. **Log File**: `end_to_end_pipeline.log` with detailed execution logs
3. **Model Artifacts**: 
   - Model file: `models/repository/v_YYYYMMDD_HHMMSS/model.pkl`
   - Transformer: `models/transformers/v_YYYYMMDD_HHMMSS_transformer.pkl`
   - Metadata: `models/repository/v_YYYYMMDD_HHMMSS/metadata.json`
4. **Predictions**: Stored in `predictions.csv` for future validation

### Example Output

```
======================================================================
CUSTOMER CHURN PREDICTION - END-TO-END PIPELINE
======================================================================
Started at: 2025-12-07 09:51:10

STEP 1: Load and Validate Data
  Training: 2000 samples, 35.25% churn
  Test: 500 samples, 37.00% churn

STEP 2: Preprocess Data
  Training shape: (2000, 21)
  Test shape: (500, 21)

STEP 3: Train Model
  Precision: 0.5068
  Recall: 0.8108
  F1-Score: 0.6237

STEP 4: Save Model with Versioning
  Model version: v_20251207_095111

STEP 5: Generate Predictions
  Total predictions: 500
  High-risk accounts: 296 (59.2%)

STEP 6: Compute SHAP Explanations
  SHAP explanations computed for 100 customers

STEP 7: Store Predictions and Evaluate
  Stored 500 predictions
  Final Recall: 0.8108

PIPELINE COMPLETE!
======================================================================
```

## Configuration

Key parameters can be adjusted in `config.py`:

- `MIN_RECALL_THRESHOLD`: Minimum acceptable recall (default: 0.85)
- `HIGH_RISK_THRESHOLD`: Probability threshold for high-risk classification (default: 0.5)
- `XGBOOST_PARAMS`: Model hyperparameters
- `TEST_SIZE`: Train/test split ratio

## Integration with Other Components

The pipeline demonstrates integration with:

- **Data Preprocessing** (`services/preprocessing.py`)
- **Model Training** (`services/model_training.py`)
- **Model Repository** (`services/model_repository.py`)
- **Prediction Service** (`services/prediction.py`)
- **Explainability Service** (`services/explainability.py`)
- **Monitoring Service** (`services/monitoring.py`)

## Troubleshooting

### Low Recall Warning

If you see `[WARNING] Model recall X.XXXX is below threshold 0.8500`:

1. Adjust `scale_pos_weight` in Step 3 (increase the multiplier from 7.5)
2. Tune other XGBoost hyperparameters in `config.py`
3. Review feature engineering in `services/preprocessing.py`
4. Check data quality and class balance

### SHAP Computation Slow

SHAP explanation can be slow on large datasets:

1. Reduce `sample_size` parameter in Step 6 (default: 100)
2. Reduce `background_size` for SHAP explainer (default: 100)
3. Consider using SHAP's GPU acceleration if available

### Memory Issues

For large datasets:

1. Process data in batches
2. Reduce background sample size for SHAP
3. Use data sampling for training

## Next Steps

After running the pipeline:

1. **Review Results**: Check `end_to_end_pipeline.log` for detailed metrics
2. **Inspect Model**: Use `examples/model_repository_example.py` to explore saved models
3. **Test Predictions**: Use `examples/prediction_example.py` for individual predictions
4. **View Dashboard**: Run `python run_dashboard.py` to visualize results
5. **Monitor Performance**: Use `examples/monitoring_example.py` to track model health

## Related Documentation

- [Model Training Guide](docs/model_training_guide.md)
- [Prediction Service Guide](docs/prediction_service_guide.md)
- [Explainability Guide](docs/explainability_guide.md)
- [Monitoring Service Guide](docs/monitoring_service_guide.md)
- [Dashboard Guide](docs/dashboard_guide.md)
