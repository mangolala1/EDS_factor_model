"""
Configuration file for EDS Factor Model
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Snowflake Connection Configuration
SNOWFLAKE_CONFIG = {
    'account': os.getenv('SNOWFLAKE_ACCOUNT', ''),
    'user': os.getenv('SNOWFLAKE_USER', ''),
    'password': os.getenv('SNOWFLAKE_PASSWORD', ''),
    'warehouse': os.getenv('SNOWFLAKE_WAREHOUSE', ''),
    'database': os.getenv('SNOWFLAKE_DATABASE', ''),
    'schema': os.getenv('SNOWFLAKE_SCHEMA', ''),
    'role': os.getenv('SNOWFLAKE_ROLE', '')
}

# Model Parameters
TARGET_CORRELATION = 0.70  # Minimum correlation with S&P500
LOOKBACK_PERIOD = 252  # Trading days (1 year)
MIN_DATA_POINTS = 60  # Minimum data points required for factor calculation

# S&P500 Ticker
SP500_TICKER = '^GSPC'

# Factor Construction Parameters
FACTOR_PARAMS = {
    'momentum_lookback': 63,  # 3 months
    'volatility_lookback': 63,  # 3 months
    'value_metrics': ['pe_ratio', 'pb_ratio', 'ev_ebitda'],
    'quality_metrics': ['roe', 'roa', 'debt_to_equity'],
    'size_metric': 'market_cap'
}



