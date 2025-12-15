"""
Entry point script for EDS Factor Model
Run this script from the project root to execute the main workflow

Usage:
    python run_main.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.main import main

if __name__ == "__main__":
    # Configure parameters here or pass as command-line arguments
    main(
        start_date='2020-01-01',
        end_date=None,
        neutralize=False,
        output_dir='results',
        data_dir='data',
        skip_download=True  # Set to False to download data from Snowflake
    )

