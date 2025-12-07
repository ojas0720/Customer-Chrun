"""
Verification script to check if all dependencies are installed correctly.
"""

import sys

def verify_imports():
    """Verify that all required packages can be imported."""
    required_packages = {
        'xgboost': 'XGBoost',
        'shap': 'SHAP',
        'pandas': 'pandas',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'streamlit': 'Streamlit',
        'pytest': 'pytest',
        'hypothesis': 'Hypothesis',
        'joblib': 'joblib',
    }
    
    print("Verifying package installations...\n")
    
    all_installed = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed successfully")
        except ImportError:
            print(f"✗ {name} NOT installed")
            all_installed = False
    
    print("\n" + "="*50)
    if all_installed:
        print("✓ All packages installed successfully!")
        print("You're ready to start developing the churn prediction system.")
        return 0
    else:
        print("✗ Some packages are missing.")
        print("Please run: pip install -r requirements.txt")
        return 1

def verify_directories():
    """Verify that all required directories exist."""
    from pathlib import Path
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'models/repository',
        'models/transformers',
        'services',
        'dashboard',
        'tests/unit',
        'tests/property',
        'tests/integration',
    ]
    
    print("\nVerifying directory structure...\n")
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} NOT found")
            all_exist = False
    
    print("\n" + "="*50)
    if all_exist:
        print("✓ All directories exist!")
        return 0
    else:
        print("✗ Some directories are missing.")
        return 1

if __name__ == "__main__":
    print("="*50)
    print("Customer Churn Prediction System - Setup Verification")
    print("="*50 + "\n")
    
    dir_result = verify_directories()
    import_result = verify_imports()
    
    sys.exit(max(dir_result, import_result))
