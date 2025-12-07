"""
Generate synthetic customer data for training and testing the churn prediction model.
Creates realistic customer features with appropriate distributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_customer_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer data with realistic distributions.
    
    Args:
        n_samples: Number of customer records to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with customer features and churn target
    """
    np.random.seed(random_state)
    
    logger.info(f"Generating {n_samples} synthetic customer records")
    
    # Generate tenure (months with customer)
    # Bimodal distribution: new customers and long-term customers
    tenure_months = np.concatenate([
        np.random.exponential(12, size=n_samples // 2),
        np.random.normal(48, 20, size=n_samples // 2)
    ])
    tenure_months = np.clip(tenure_months, 1, 72).astype(int)
    np.random.shuffle(tenure_months)
    
    # Generate monthly charges
    # Higher charges for customers with more services
    base_charges = np.random.normal(50, 20, size=n_samples)
    monthly_charges = np.clip(base_charges, 20, 120).round(2)
    
    # Generate total charges (tenure * average monthly charge with some variation)
    total_charges = (tenure_months * monthly_charges * 
                     np.random.uniform(0.9, 1.1, size=n_samples)).round(2)
    
    # Generate contract types
    # Month-to-month, One year, Two year
    contract_types = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'],
        size=n_samples,
        p=[0.5, 0.3, 0.2]  # More month-to-month contracts
    )
    
    # Generate payment methods
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        size=n_samples,
        p=[0.35, 0.25, 0.2, 0.2]
    )
    
    # Generate internet service types
    internet_services = np.random.choice(
        ['DSL', 'Fiber optic', 'No'],
        size=n_samples,
        p=[0.4, 0.4, 0.2]
    )
    
    # Create DataFrame
    data = pd.DataFrame({
        'tenure_months': tenure_months,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'internet_service': internet_services
    })
    
    # Generate churn target based on realistic patterns
    # Higher churn probability for:
    # - Short tenure
    # - Month-to-month contracts
    # - Electronic check payment
    # - High monthly charges relative to tenure
    
    churn_prob = np.zeros(n_samples)
    
    # Base churn rate
    churn_prob += 0.15
    
    # Tenure effect (shorter tenure = higher churn)
    churn_prob += np.where(tenure_months < 6, 0.3, 0)
    churn_prob += np.where((tenure_months >= 6) & (tenure_months < 12), 0.2, 0)
    churn_prob += np.where(tenure_months >= 36, -0.15, 0)
    
    # Contract type effect
    churn_prob += np.where(contract_types == 'Month-to-month', 0.25, 0)
    churn_prob += np.where(contract_types == 'Two year', -0.2, 0)
    
    # Payment method effect
    churn_prob += np.where(payment_methods == 'Electronic check', 0.15, 0)
    churn_prob += np.where(payment_methods == 'Credit card', -0.1, 0)
    
    # Monthly charges effect (high charges = higher churn)
    churn_prob += np.where(monthly_charges > 80, 0.15, 0)
    
    # Internet service effect
    churn_prob += np.where(internet_services == 'Fiber optic', 0.1, 0)
    
    # Clip probabilities to valid range
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    
    # Generate binary churn outcome
    data['churn'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    # Log statistics
    churn_rate = data['churn'].mean()
    logger.info(f"Generated data statistics:")
    logger.info(f"  Total samples: {len(data)}")
    logger.info(f"  Churn rate: {churn_rate:.2%}")
    logger.info(f"  Tenure range: {data['tenure_months'].min()}-{data['tenure_months'].max()} months")
    logger.info(f"  Monthly charges range: ${data['monthly_charges'].min():.2f}-${data['monthly_charges'].max():.2f}")
    
    return data


def save_data(data: pd.DataFrame, output_path: Path) -> None:
    """
    Save generated data to CSV file.
    
    Args:
        data: DataFrame to save
        output_path: Path to save CSV file
    """
    # Ensure directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    data.to_csv(output_path, index=False)
    logger.info(f"Data saved to: {output_path}")


def main():
    """Generate and save training data."""
    logger.info("=" * 60)
    logger.info("Generating Synthetic Customer Churn Data")
    logger.info("=" * 60)
    
    # Generate training data
    training_data = generate_customer_data(
        n_samples=2000,
        random_state=config.RANDOM_STATE
    )
    
    # Save training data
    save_data(training_data, config.TRAINING_DATA_PATH)
    
    # Generate separate test data (optional, for final evaluation)
    test_data = generate_customer_data(
        n_samples=500,
        random_state=config.RANDOM_STATE + 1
    )
    
    # Save test data
    save_data(test_data, config.TEST_DATA_PATH)
    
    logger.info("=" * 60)
    logger.info("Data generation complete!")
    logger.info("=" * 60)
    logger.info(f"Training data: {config.TRAINING_DATA_PATH}")
    logger.info(f"Test data: {config.TEST_DATA_PATH}")
    logger.info("\nYou can now run: python train_model.py")


if __name__ == "__main__":
    main()
