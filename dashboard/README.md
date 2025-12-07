# Customer Churn Prediction Dashboard

## Overview

The Customer Churn Prediction Dashboard is an interactive Streamlit web application that provides comprehensive visualization and analysis of customer churn predictions.

## Quick Start

### 1. Ensure Prerequisites

Before running the dashboard, make sure you have:

- Trained at least one model (run `python train_model.py`)
- Generated test data (run `python generate_sample_data.py`)
- Installed all dependencies (run `pip install -r requirements.txt`)

### 2. Run the Dashboard

```bash
python run_dashboard.py
```

Or directly with Streamlit:

```bash
streamlit run dashboard/app.py
```

### 3. Access the Dashboard

Open your browser and navigate to:
- http://localhost:8501

## Dashboard Structure

### Overview Tab

The Overview tab provides a high-level view of churn predictions:

- **Metrics Cards**:
  - High-Risk Accounts: Count and percentage of customers at risk
  - Average Churn Probability: Mean probability across all customers
  - Total Customers: Number of customers analyzed

- **Churn Probability Distribution**: 
  - Histogram showing the distribution of churn probabilities
  - Red dashed line indicates the high-risk threshold

- **Model Performance Metrics**:
  - Recall, Precision, F1-Score, and Accuracy
  - Visual indicator if recall meets the minimum threshold

- **Recent Predictions Table**:
  - Customer IDs with their churn probabilities
  - High-risk classification
  - Actual churn status (if available)

### Customer Details Tab

The Customer Details tab provides in-depth analysis for individual customers:

- **Customer Selection**: Dropdown to select any customer from the test dataset

- **Customer Information Panel**:
  - Customer ID
  - Tenure (months)
  - Monthly and total charges
  - Contract type
  - Payment method
  - Internet service type

- **Churn Probability Gauge**:
  - Visual gauge showing churn probability (0-100%)
  - Color-coded: green for low risk, red for high risk
  - Threshold indicator

- **Top Contributing Features**:
  - Horizontal bar chart showing top 5 features
  - Color-coded by SHAP value (green = decreases risk, red = increases risk)
  - Sorted by absolute impact

- **Feature Interpretation**:
  - Explanation of SHAP values
  - Table showing feature values and their impact

### Statistics Tab

The Statistics tab provides aggregated insights:

- **Summary Statistics**:
  - Average churn probability
  - Median churn probability
  - Standard deviation

- **Feature Importance Trends**:
  - Bar chart showing top 10 most important features
  - Averaged across all customers
  - Helps identify global churn drivers

- **Retention Metrics**:
  - True Positives, False Positives, True Negatives, False Negatives
  - Confusion matrix heatmap
  - Actual retention rate
  - Predicted high-risk rate

### Model Version Selector (Sidebar)

The sidebar provides model management:

- **Version Dropdown**: Select from available model versions
- **Model Information**:
  - Version identifier
  - Number of features
  - Performance metrics (recall, precision, F1-score)

## Features

### Real-Time Predictions

The dashboard generates predictions on-demand:
- Batch predictions for all customers in the Overview tab
- Individual predictions with explanations in the Customer Details tab

### SHAP Explanations

When background data is available, the dashboard provides:
- Feature-level explanations for individual predictions
- Aggregated feature importance across all customers
- Visual interpretation of feature impacts

### Model Comparison

Switch between model versions to:
- Compare performance metrics
- Analyze prediction differences
- Evaluate model improvements

### Interactive Visualizations

All charts are interactive:
- Hover for detailed information
- Zoom and pan capabilities
- Export as images

## Configuration

### Adjusting the High-Risk Threshold

Edit `config.py`:

```python
HIGH_RISK_THRESHOLD = 0.5  # Change to desired value (0.0 to 1.0)
```

### Changing the Dashboard Port

Edit `config.py`:

```python
DASHBOARD_PORT = 8501  # Change to desired port
```

### Adjusting SHAP Background Samples

Edit `config.py`:

```python
SHAP_BACKGROUND_SAMPLES = 100  # Increase for more accurate SHAP, decrease for speed
```

## Troubleshooting

### Dashboard Won't Start

**Error**: "No models found in repository"

**Solution**: Train a model first:
```bash
python train_model.py
```

### No Test Data Available

**Error**: "No test data available"

**Solution**: Generate sample data:
```bash
python generate_sample_data.py
```

### SHAP Explanations Not Working

**Warning**: "SHAP explanations not available"

**Solution**: Ensure training data exists at `data/raw/training_data.csv`

### Slow Performance

**Issue**: Dashboard is slow to load or update

**Solutions**:
- Reduce the number of customers in test data
- Decrease `SHAP_BACKGROUND_SAMPLES` in config
- Disable SHAP explanations for faster predictions

### Port Already in Use

**Error**: "Address already in use"

**Solution**: Use a different port:
```bash
streamlit run dashboard/app.py --server.port 8080
```

## Technical Details

### Caching

The dashboard uses Streamlit's caching to improve performance:
- `@st.cache_resource` for model loading
- Models and data are loaded once and reused

### Data Flow

1. User selects model version in sidebar
2. Dashboard loads model and test data
3. Predictions are generated on-demand
4. SHAP explanations computed when requested
5. Visualizations updated automatically

### Dependencies

- **streamlit**: Web application framework
- **plotly**: Interactive visualizations
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **xgboost**: Model inference
- **shap**: Explainability

## Best Practices

1. **Regular Model Updates**: Retrain models periodically with new data
2. **Monitor Performance**: Check metrics in the Overview tab regularly
3. **Analyze Feature Importance**: Use Statistics tab to understand churn drivers
4. **Review Individual Cases**: Use Customer Details tab for targeted interventions
5. **Compare Versions**: Test new models before deployment

## API Integration

The dashboard can be extended to integrate with:
- Real-time data sources
- Customer databases
- CRM systems
- Alert notification systems

See `docs/dashboard_guide.md` for detailed integration examples.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `docs/dashboard_guide.md` for detailed documentation
3. Run `python examples/dashboard_example.py` to verify setup

## Version History

- **v1.0.0**: Initial dashboard release
  - Overview, Customer Details, and Statistics tabs
  - Model version selector
  - SHAP explanations
  - Interactive visualizations
