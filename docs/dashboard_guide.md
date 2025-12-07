# Customer Churn Prediction Dashboard Guide

## Overview

The Customer Churn Prediction Dashboard is a Streamlit-based web application that provides an interactive interface for viewing churn predictions, model performance metrics, and SHAP explanations.

## Features

### 1. Overview Tab
- **High-Risk Account Count**: Displays the total number of customers classified as high-risk
- **Churn Probability Distribution**: Histogram showing the distribution of churn probabilities across all customers
- **Model Performance Metrics**: Shows recall, precision, F1-score, and accuracy
- **Recent Predictions Table**: Lists recent predictions with customer IDs and risk classifications

### 2. Customer Details Tab
- **Customer Selection**: Dropdown to select individual customers
- **Customer Information**: Displays customer attributes (tenure, charges, contract type, etc.)
- **Churn Probability Gauge**: Visual gauge showing the customer's churn probability
- **Top Contributing Features**: Bar chart showing the top 5 features with highest SHAP values
- **Feature Interpretation**: Explains whether each feature increases or decreases churn risk

### 3. Statistics Tab
- **Summary Statistics**: Average, median, and standard deviation of churn probabilities
- **Feature Importance Trends**: Shows the most important features across all customers
- **Retention Metrics**: Displays confusion matrix and retention rates
- **Confusion Matrix Visualization**: Heatmap showing prediction accuracy

### 4. Model Version Selector (Sidebar)
- **Version Selection**: Dropdown to switch between different model versions
- **Model Metadata**: Displays version information, feature count, and performance metrics

## Running the Dashboard

### Method 1: Using the Helper Script

```bash
python run_dashboard.py
```

### Method 2: Direct Streamlit Command

```bash
streamlit run dashboard/app.py --server.port 8501
```

### Method 3: Custom Port

```bash
streamlit run dashboard/app.py --server.port 8080
```

## Accessing the Dashboard

Once running, the dashboard will be available at:
- Local: http://localhost:8501
- Network: http://[your-ip]:8501

## Requirements

The dashboard requires the following data to be available:

1. **Trained Model**: At least one model version in the model repository
2. **Test Data**: Customer data in `data/raw/test_data.csv`
3. **Training Data**: (Optional) For SHAP background data in `data/raw/training_data.csv`

## Dashboard Components

### Model Loading

The dashboard automatically:
- Loads the deployed model version (or latest if no deployed version)
- Initializes the ChurnPredictor with SHAP explanations enabled
- Caches the model to avoid reloading on every interaction

### Prediction Generation

Predictions are generated:
- In batch for all customers in the Overview tab
- Individually with explanations in the Customer Details tab
- With caching to improve performance

### SHAP Explanations

SHAP explanations are computed when:
- Background data is available from training data
- Individual customer predictions are requested
- Feature importance trends are calculated

## Customization

### Changing the High-Risk Threshold

Edit `config.py`:

```python
HIGH_RISK_THRESHOLD = 0.5  # Change to desired threshold
```

### Adjusting SHAP Background Samples

Edit `config.py`:

```python
SHAP_BACKGROUND_SAMPLES = 100  # Change to desired sample size
```

### Modifying Dashboard Port

Edit `config.py`:

```python
DASHBOARD_PORT = 8501  # Change to desired port
```

## Troubleshooting

### Dashboard Won't Start

**Issue**: Error message "No models found in repository"
**Solution**: Train a model first using `python train_model.py`

**Issue**: Error message "No test data available"
**Solution**: Ensure `data/raw/test_data.csv` exists and contains customer data

### SHAP Explanations Not Available

**Issue**: "SHAP explanations not available" message
**Solution**: Ensure `data/raw/training_data.csv` exists for background data

### Slow Performance

**Issue**: Dashboard is slow to load or update
**Solution**: 
- Reduce the number of customers in test data
- Decrease `SHAP_BACKGROUND_SAMPLES` in config
- Use a smaller model or fewer features

### Model Version Not Found

**Issue**: Selected model version fails to load
**Solution**: 
- Check that model files exist in `models/repository/[version]/`
- Verify transformer file exists in `models/transformers/[version]_transformer.pkl`
- Try selecting a different model version

## Best Practices

1. **Data Preparation**: Ensure test data has the same features as training data
2. **Model Deployment**: Set a deployed version using the model repository
3. **Regular Updates**: Retrain models periodically and update the dashboard
4. **Performance Monitoring**: Use the Statistics tab to monitor model performance
5. **Feature Analysis**: Review feature importance trends to understand churn drivers

## API Reference

### Main Functions

#### `load_model_and_data(model_version=None)`
Loads model, data, and initializes predictor. Cached for performance.

**Parameters:**
- `model_version` (str, optional): Specific model version to load

**Returns:**
- `predictor` (ChurnPredictor): Initialized predictor
- `df` (pd.DataFrame): Test data
- `metadata` (dict): Model metadata
- `repository` (ModelRepository): Model repository instance

### Dashboard Tabs

#### Overview Tab
Displays high-level metrics and predictions for all customers.

#### Customer Details Tab
Shows detailed information and explanations for individual customers.

#### Statistics Tab
Presents aggregated statistics and feature importance trends.

## Examples

### Viewing High-Risk Customers

1. Navigate to the Overview tab
2. Check the "High-Risk Accounts" metric
3. Scroll down to the predictions table
4. Filter for customers with "🚨 Yes" in the is_high_risk column

### Understanding Feature Impact

1. Navigate to the Customer Details tab
2. Select a customer from the dropdown
3. View the "Top Contributing Features" chart
4. Red bars indicate features increasing churn risk
5. Green bars indicate features decreasing churn risk

### Comparing Model Versions

1. Use the sidebar to select a model version
2. Note the performance metrics in the sidebar
3. Switch to a different version
4. Compare the metrics and predictions

## Integration

The dashboard integrates with:
- **Prediction Service**: For generating churn predictions
- **Model Repository**: For loading and managing model versions
- **Explainability Service**: For computing SHAP values
- **Monitoring Service**: For tracking predictions and performance

## Future Enhancements

Potential improvements:
- Real-time prediction updates
- Export predictions to CSV
- Custom threshold adjustment in UI
- Historical trend analysis
- Alert notifications for high-risk customers
- Batch customer upload
- Model comparison view
