# Customer Churn Prediction - Jupyter Notebooks

This directory contains interactive Jupyter notebooks that demonstrate the complete workflow of the Customer Churn Prediction System.

## Notebooks Overview

### 1. Model Training (`01_model_training.ipynb`)

**Purpose**: Learn how to train an XGBoost model for churn prediction

**Topics Covered**:
- Loading and exploring customer data
- Data validation and quality checks
- Feature engineering
- Data preprocessing (encoding, scaling)
- Training XGBoost classifier
- Evaluating model performance
- Saving models with versioning

**Prerequisites**: 
- Sample data generated (`python generate_sample_data.py`)
- All dependencies installed

**Estimated Time**: 15-20 minutes

---

### 2. Prediction and Explanation (`02_prediction_and_explanation.ipynb`)

**Purpose**: Make predictions and understand model decisions using SHAP

**Topics Covered**:
- Loading trained models
- Single customer predictions
- Batch predictions
- Computing SHAP explanations
- Visualizing feature contributions
- Interpreting SHAP values
- Verifying SHAP additivity

**Prerequisites**:
- At least one trained model (run notebook 1 first)

**Estimated Time**: 20-25 minutes

---

### 3. Model Evaluation and Monitoring (`03_model_evaluation_and_monitoring.ipynb`)

**Purpose**: Evaluate model performance and set up monitoring

**Topics Covered**:
- Computing performance metrics
- Confusion matrix analysis
- ROC curves and AUC
- Model calibration
- Probability distribution analysis
- Storing predictions for validation
- Performance monitoring and alerts
- Model version management
- Rollback procedures

**Prerequisites**:
- Trained model and test data

**Estimated Time**: 25-30 minutes

---

## Getting Started

### 1. Install Jupyter

If you haven't already, install Jupyter:

```bash
pip install jupyter notebook
```

Or use JupyterLab:

```bash
pip install jupyterlab
```

### 2. Launch Jupyter

From the project root directory:

```bash
# For Jupyter Notebook
jupyter notebook notebooks/

# For JupyterLab
jupyter lab notebooks/
```

### 3. Run Notebooks in Order

For the best learning experience, run the notebooks in sequence:

1. Start with `01_model_training.ipynb` to train your first model
2. Move to `02_prediction_and_explanation.ipynb` to make predictions
3. Finish with `03_model_evaluation_and_monitoring.ipynb` to monitor performance

### 4. Experiment and Modify

Feel free to:
- Modify hyperparameters
- Try different feature engineering approaches
- Experiment with different thresholds
- Add your own visualizations
- Test with your own data

## Tips for Using the Notebooks

### Running Cells

- **Shift + Enter**: Run current cell and move to next
- **Ctrl + Enter**: Run current cell and stay
- **Alt + Enter**: Run current cell and insert new cell below

### Kernel Management

- **Restart Kernel**: Use if you encounter errors or want a fresh start
- **Restart & Run All**: Useful for verifying the entire notebook works

### Saving Work

- Notebooks auto-save periodically
- Use **File > Save and Checkpoint** to manually save
- Export to HTML/PDF for sharing: **File > Download as**

## Common Issues and Solutions

### Issue: Import Errors

**Solution**: Ensure you're in the correct virtual environment and all packages are installed:

```bash
# Activate virtual environment
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Issue: Data Not Found

**Solution**: Generate sample data first:

```bash
python generate_sample_data.py
```

### Issue: Model Not Found

**Solution**: Run the training notebook first to create a model:

```bash
jupyter notebook notebooks/01_model_training.ipynb
```

### Issue: SHAP Visualization Not Showing

**Solution**: Ensure you have the latest version of SHAP and run `shap.initjs()` in a cell:

```python
import shap
shap.initjs()
```

### Issue: Slow SHAP Computation

**Solution**: Reduce the sample size in the notebook:

```python
# Instead of all customers
sample_customers = test_data.head(10)  # Use only 10 customers
```

## Customization Ideas

### Beginner Level
- Change the high-risk threshold (default: 0.5)
- Modify the number of top features displayed
- Adjust visualization colors and styles

### Intermediate Level
- Add new feature engineering steps
- Experiment with different XGBoost hyperparameters
- Create custom SHAP visualizations
- Add additional performance metrics

### Advanced Level
- Implement cross-validation
- Add hyperparameter tuning with GridSearchCV
- Create ensemble models
- Implement custom evaluation metrics
- Add data drift detection

## Additional Resources

### Documentation
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

### Project Documentation
- [Main README](../README.md) - Project overview and setup
- [Installation Guide](../INSTALL.md) - Detailed installation instructions
- [End-to-End Pipeline](../END_TO_END_PIPELINE.md) - Complete workflow documentation
- [Service Guides](../docs/) - Detailed service documentation

### Jupyter Resources
- [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/)
- [JupyterLab Documentation](https://jupyterlab.readthedocs.io/)
- [Jupyter Keyboard Shortcuts](https://towardsdatascience.com/jypyter-notebook-shortcuts-bf0101a98330)

## Contributing

If you create additional notebooks or improvements:

1. Follow the existing naming convention (`XX_descriptive_name.ipynb`)
2. Include clear markdown explanations
3. Add comments to code cells
4. Test the notebook from start to finish
5. Update this README with the new notebook information

## Feedback

If you encounter issues or have suggestions for improving these notebooks, please:

1. Check the troubleshooting section above
2. Review the main project documentation
3. Open an issue in the project repository (if applicable)

## Next Steps

After completing these notebooks:

1. **Explore the Dashboard**: Run `python run_dashboard.py` to see the interactive UI
2. **Review Examples**: Check the `examples/` directory for standalone scripts
3. **Read Documentation**: Dive deeper into specific services in the `docs/` directory
4. **Customize**: Adapt the system for your own churn prediction use case
5. **Deploy**: Consider productionizing the system for real-world use

Happy Learning! 🚀
