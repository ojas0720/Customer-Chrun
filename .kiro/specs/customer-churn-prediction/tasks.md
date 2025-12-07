# Implementation Plan: Customer Churn Prediction System

- [x] 1. Set up project structure and dependencies





  - Create directory structure for data, models, services, dashboard, and tests
  - Set up virtual environment and install required packages (XGBoost, SHAP, pandas, scikit-learn, Streamlit, pytest, Hypothesis)
  - Create configuration file for model parameters and paths
  - _Requirements: All_

- [x] 2. Implement data validation and preprocessing module





  - _Requirements: 1.3, 1.4, 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 2.1 Create DataValidator class


  - Implement schema validation to check for required features and target variable
  - Implement value range validation for numerical features
  - Implement type checking for all features
  - _Requirements: 1.4_

- [ ]* 2.2 Write property test for data validation
  - **Property 3: Training data validation**
  - **Validates: Requirements 1.4**

- [x] 2.3 Create DataTransformer class with fit/transform methods


  - Implement missing value imputation using SimpleImputer
  - Implement categorical encoding using OneHotEncoder with handle_unknown='ignore'
  - Implement numerical scaling using StandardScaler
  - Implement save/load methods for fitted transformers using joblib
  - _Requirements: 1.3, 7.2, 7.3, 7.4_

- [ ]* 2.4 Write property test for missing value handling
  - **Property 2: Missing value handling robustness**
  - **Validates: Requirements 1.3**

- [ ]* 2.5 Write property test for categorical encoding consistency
  - **Property 24: Categorical encoding consistency**
  - **Validates: Requirements 7.2**

- [ ]* 2.6 Write property test for numerical scaling consistency
  - **Property 25: Numerical scaling consistency**
  - **Validates: Requirements 7.3**

- [ ]* 2.7 Write property test for unknown category handling
  - **Property 26: Unknown category handling**
  - **Validates: Requirements 7.4**

- [x] 2.8 Create FeatureEngineer class for derived features


  - Implement feature engineering logic (e.g., tenure_months, total_spend_ratio)
  - Ensure consistency between training and inference
  - _Requirements: 7.5_

- [ ]* 2.9 Write property test for feature engineering consistency
  - **Property 27: Feature engineering consistency**
  - **Validates: Requirements 7.5**

- [x] 3. Implement model training service





  - _Requirements: 1.1, 1.2, 1.5_

- [x] 3.1 Create ModelTrainer class


  - Implement train method that fits XGBoost classifier
  - Implement evaluate method that computes precision, recall, F1-score
  - Implement save_model method with versioning (timestamp-based)
  - _Requirements: 1.1, 1.5_

- [ ]* 3.2 Write property test for model persistence
  - **Property 1: Model persistence round-trip**
  - **Validates: Requirements 1.1**

- [ ]* 3.3 Write property test for metrics completeness
  - **Property 4: Metrics completeness**
  - **Validates: Requirements 1.5**

- [x] 3.2 Create training script


  - Load and preprocess training data
  - Train XGBoost model with hyperparameters optimized for recall
  - Evaluate model on test set and verify recall >= 85%
  - Save trained model and transformer
  - _Requirements: 1.1, 1.2, 1.5_

- [x] 4. Implement model repository for version management





  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 4.1 Create ModelRepository class


  - Implement save method to store model, transformer, and metadata
  - Implement load method to retrieve specific version
  - Implement list_versions method to show all available versions
  - Implement get_latest_version method
  - Implement rollback method to restore previous version
  - _Requirements: 6.1, 6.2, 6.4, 6.5_

- [ ]* 4.2 Write property test for version uniqueness
  - **Property 19: Version uniqueness**
  - **Validates: Requirements 6.1**

- [ ]* 4.3 Write property test for version selection
  - **Property 20: Version selection**
  - **Validates: Requirements 6.2**

- [ ]* 4.4 Write property test for deployment completeness
  - **Property 22: Deployment completeness**
  - **Validates: Requirements 6.4**

- [ ]* 4.5 Write property test for rollback restoration
  - **Property 23: Rollback restoration**
  - **Validates: Requirements 6.5**

- [x] 4.6 Create parameter validation utility


  - Implement validation for XGBoost hyperparameters
  - Check for valid ranges (learning_rate > 0, max_depth > 0, etc.)
  - _Requirements: 6.3_

- [ ]* 4.7 Write property test for parameter validation
  - **Property 21: Parameter validation**
  - **Validates: Requirements 6.3**

- [x] 5. Implement prediction service





  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 5.1 Create ChurnPredictor class

  - Implement predict_proba method for single and batch predictions
  - Implement predict_high_risk method with configurable threshold
  - Implement input validation before prediction
  - Load model and transformer from repository
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [ ]* 5.2 Write property test for probability bounds
  - **Property 5: Probability bounds**
  - **Validates: Requirements 2.1**

- [ ]* 5.3 Write property test for high-risk classification
  - **Property 6: High-risk classification consistency**
  - **Validates: Requirements 2.2**

- [ ]* 5.4 Write property test for batch prediction completeness
  - **Property 7: Batch prediction completeness**
  - **Validates: Requirements 2.3**

- [ ]* 5.5 Write property test for invalid input rejection
  - **Property 8: Invalid input rejection**
  - **Validates: Requirements 2.4**

- [x] 5.6 Create CustomerRecord and PredictionResult data models


  - Implement CustomerRecord with validation method
  - Implement PredictionResult with all required fields
  - _Requirements: 2.1, 2.2_

- [x] 6. Checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement SHAP explainability service





  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 7.1 Create SHAPExplainer class


  - Initialize with trained model and background data
  - Implement explain method to compute SHAP values
  - Implement get_top_features method to rank features by absolute SHAP value
  - Implement validate_shap_sum method to verify additivity property
  - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [ ]* 7.2 Write property test for SHAP completeness
  - **Property 9: SHAP completeness**
  - **Validates: Requirements 3.1**

- [ ]* 7.3 Write property test for top features ranking
  - **Property 10: Top features ranking**
  - **Validates: Requirements 3.2**

- [ ]* 7.4 Write property test for SHAP sign interpretation
  - **Property 11: SHAP sign interpretation**
  - **Validates: Requirements 3.3**

- [ ]* 7.5 Write property test for SHAP additivity
  - **Property 12: SHAP additivity**
  - **Validates: Requirements 3.5**

- [x] 7.6 Integrate SHAP explainer with prediction service


  - Modify PredictionResult to include top_features field
  - Update ChurnPredictor to optionally compute explanations
  - _Requirements: 3.1, 3.2, 3.3_

- [ ] 8. Implement model validation and monitoring service





  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 8.1 Create ModelEvaluator class


  - Implement evaluate method to compute metrics on new data
  - Implement confusion matrix calculation
  - Implement calibration curve computation
  - _Requirements: 5.1, 5.4, 5.5_

- [ ]* 8.2 Write property test for evaluation metrics computation
  - **Property 14: Evaluation metrics computation**
  - **Validates: Requirements 5.1**

- [ ]* 8.3 Write property test for confusion matrix correctness
  - **Property 17: Confusion matrix correctness**
  - **Validates: Requirements 5.4**

- [ ]* 8.4 Write property test for calibration curve computation
  - **Property 18: Calibration curve computation**
  - **Validates: Requirements 5.5**

- [x] 8.5 Create AlertService class


  - Implement check_performance method to compare recall against threshold
  - Implement generate_alert method for performance degradation
  - _Requirements: 5.2_

- [ ]* 8.6 Write property test for alert triggering logic
  - **Property 15: Alert triggering logic**
  - **Validates: Requirements 5.2**

- [x] 8.7 Create PredictionStore class


  - Implement store method to save predictions with actual outcomes
  - Implement retrieve method to get historical predictions
  - Use CSV or SQLite for simple storage
  - _Requirements: 5.3_

- [ ]* 8.8 Write property test for prediction storage completeness
  - **Property 16: Prediction storage completeness**
  - **Validates: Requirements 5.3**

- [x] 9. Implement Streamlit dashboard





  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [x] 9.1 Create dashboard main page


  - Display high-risk account count
  - Show churn probability distribution histogram
  - Display model performance metrics
  - _Requirements: 4.1, 4.2_


- [x] 9.2 Create customer detail view


  - Add customer selection dropdown
  - Display selected customer's churn probability
  - Show top 5 contributing features with SHAP values
  - Create SHAP waterfall plot visualization
  - _Requirements: 4.3, 4.4_


- [x] 9.3 Create aggregated statistics view


  - Compute and display average churn probability
  - Show feature importance trends across all customers
  - Display retention metrics summary
  - _Requirements: 4.5_

- [ ]* 9.4 Write property test for aggregate statistics correctness
  - **Property 13: Aggregate statistics correctness**
  - **Validates: Requirements 4.5**


- [x] 9.5 Add model version selector to dashboard


  - Display available model versions
  - Allow switching between versions
  - Show metadata for selected version
  - _Requirements: 6.2_

- [x] 10. Create end-to-end integration and sample data





  - _Requirements: All_

- [x] 10.1 Generate synthetic customer dataset


  - Create realistic customer features (tenure, monthly_charges, contract_type, etc.)
  - Include both churned and retained customers
  - Split into training and test sets
  - _Requirements: 1.1_


- [x] 10.2 Create end-to-end pipeline script

  - Train model on synthetic data
  - Generate predictions for test customers
  - Compute SHAP explanations
  - Store predictions and evaluate performance
  - _Requirements: All_

- [ ]* 10.3 Write integration tests for complete pipeline
  - Test training → prediction → explanation flow
  - Test model versioning and rollback
  - Test dashboard data loading
  - _Requirements: All_

- [x] 11. Final checkpoint - Ensure all tests pass





  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Create documentation and usage examples





  - _Requirements: All_



- [x] 12.1 Write README with setup instructions

  - Document installation steps
  - Provide usage examples for training and prediction
  - Explain how to run the dashboard
  - _Requirements: All_


- [x] 12.2 Create example notebooks

  - Create Jupyter notebook demonstrating model training
  - Create notebook showing prediction and explanation workflow
  - Create notebook for model evaluation and monitoring
  - _Requirements: All_
