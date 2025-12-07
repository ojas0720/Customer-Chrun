# Installation Guide

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### 2. Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- XGBoost (>=2.0.0) - Gradient boosting framework
- SHAP (>=0.43.0) - Model explainability
- pandas (>=2.0.0) - Data manipulation
- numpy (>=1.24.0) - Numerical computing
- scikit-learn (>=1.3.0) - Machine learning utilities
- Streamlit (>=1.28.0) - Dashboard framework
- pytest (>=7.4.0) - Testing framework
- Hypothesis (>=6.92.0) - Property-based testing

### 4. Verify Installation

```bash
python verify_setup.py
```

You should see all packages marked with ✓ if installation was successful.

## Troubleshooting

### XGBoost Installation Issues

If XGBoost fails to install, try:

```bash
pip install --upgrade pip
pip install xgboost
```

### SHAP Installation Issues

SHAP may require additional build tools. If installation fails:

**Windows:**
- Install Microsoft C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/

**Linux:**
```bash
sudo apt-get install build-essential
```

**Mac:**
```bash
xcode-select --install
```

Then retry:
```bash
pip install shap
```

### Alternative: Install from conda

If pip installation continues to fail, you can use conda:

```bash
conda create -n churn-prediction python=3.10
conda activate churn-prediction
conda install -c conda-forge xgboost shap pandas numpy scikit-learn streamlit pytest hypothesis joblib
```

## Next Steps

After successful installation:

1. Review the project structure in README.md
2. Check configuration settings in config.py
3. Start implementing the data preprocessing module (Task 2)

## Getting Help

If you encounter issues:
1. Check that Python version is 3.8+: `python --version`
2. Ensure pip is up to date: `pip install --upgrade pip`
3. Try installing packages individually to identify which one fails
4. Check package documentation for platform-specific requirements
