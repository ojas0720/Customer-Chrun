"""
Script to run the Customer Churn Prediction Dashboard.
"""

import subprocess
import sys
from pathlib import Path

import config


def main():
    """Run the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    
    if not dashboard_path.exists():
        print(f"Error: Dashboard file not found at {dashboard_path}")
        sys.exit(1)
    
    print(f"Starting {config.DASHBOARD_TITLE}...")
    print(f"Dashboard will be available at http://localhost:{config.DASHBOARD_PORT}")
    print("Press Ctrl+C to stop the dashboard")
    print("-" * 60)
    
    # Run streamlit
    subprocess.run([
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port",
        str(config.DASHBOARD_PORT),
        "--server.headless",
        "true"
    ])


if __name__ == "__main__":
    main()
