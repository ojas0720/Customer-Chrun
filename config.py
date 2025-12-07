"""
Configuration file for Customer Churn Prediction System.
Contains model parameters, file paths, and system settings.
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAINING_DATA_PATH = RAW_DATA_DIR / "training_data.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test_data.csv"

# Model paths
MODEL_REPOSITORY_DIR = MODELS_DIR / "repository"
TRANSFORMER_DIR = MODELS_DIR / "transformers"

# XGBoost hyperparameters (optimized for recall)
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.03,
    "n_estimators": 300,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 1,
    "gamma": 0,
    "reg_alpha": 0.05,
    "reg_lambda": 0.5,
    "scale_pos_weight": 1,  # Will be adjusted based on class imbalance
    "random_state": 42,
}

# Model training settings
MIN_RECALL_THRESHOLD = 0.85
HIGH_RISK_THRESHOLD = 0.5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Preprocessing settings
NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
]

TARGET_VARIABLE = "churn"

# SHAP settings
SHAP_BACKGROUND_SAMPLES = 100
TOP_FEATURES_COUNT = 5

# Prediction settings
BATCH_PREDICTION_TIMEOUT = 30  # seconds
SINGLE_PREDICTION_TIMEOUT = 2  # seconds
MAX_BATCH_SIZE = 1000

# Dashboard settings
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Customer Churn Prediction Dashboard"

# Monitoring settings
PERFORMANCE_CHECK_INTERVAL = 86400  # 24 hours in seconds
ALERT_EMAIL_ENABLED = False
ALERT_EMAIL_RECIPIENTS = []

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "churn_prediction.log"

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                  MODELS_DIR, MODEL_REPOSITORY_DIR, TRANSFORMER_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
