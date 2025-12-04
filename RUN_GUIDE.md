# How to Run the EDS Factor Model

## Prerequisites

1. **Python Environment**: Python 3.8 or higher
2. **Dependencies**: Install required packages
3. **Snowflake Access**: Valid credentials and access to the database
4. **Environment Variables**: Configured `.env` file

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with your Snowflake credentials:

```bash
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
SNOWFLAKE_ROLE=your_role  # Optional
```

### 3. Verify Configuration

Check that your table names in `config.py` match your Snowflake schema:

```python
SNOWFLAKE_TABLES = {
    'fundamentals': 'EDS_FACTORS_FUNDAMENTALS_NTM_LTM',
    'universe': 'EDS_FACTORS_UNIVERSE',
    # Returns are calculated from prices, no separate table needed
    'prices': 'SPLIT_ADJUSTED_PRICES_EDS_FACTORS',
    'exchange_rates': 'EXCHANGERATES',
    'market_value': 'MARKET_VALUE_HISTORY',
    'enterprise_value': 'ENTERPRISEVALUE_HISTORY'
}
```

## Running the Model

### Basic Run (Default Parameters)

```bash
python factors.py
```

This will:
- Use default date range (252 trading days from today)
- Run without neutralization
- Require minimum 50 stocks per date

### Custom Run (Python Script)

Create a custom script or modify `factors.py`:

```python
from factors import main

# Run with custom parameters
results = main(
    start_date='2023-01-01',
    end_date='2024-12-31',
    neutralize=False,  # Set to True to neutralize style factors
    min_stocks_per_date=50
)

# Access results
exposures = results['exposures']
factor_returns = results['factor_returns']
specific_returns = results['specific_returns']
```

### Command Line with Custom Dates

Modify the `if __name__ == "__main__":` section in `factors.py`:

```python
if __name__ == "__main__":
    results = main(
        start_date='2023-01-01',  # Change this
        end_date='2024-12-31',    # Change this
        neutralize=False,
        min_stocks_per_date=50
    )
```

## Output Files

All outputs are automatically saved to the `outputs/` directory with timestamps.

### Output Files Generated

1. **`factor_exposures_YYYYMMDD_YYYYMMDD_TIMESTAMP.csv`**
   - Factor exposures for each stock-date
   - Columns: FACTSET_ID, DATE, VALUE, PROFITABILITY, GROWTH, MOMENTUM, VOLATILITY, LIQUIDITY, plus continent and sector dummies

2. **`factor_returns_YYYYMMDD_YYYYMMDD_TIMESTAMP.csv`**
   - Daily factor returns
   - Columns: DATE, FACTOR_NAME, RETURN

3. **`specific_returns_YYYYMMDD_YYYYMMDD_TIMESTAMP.csv`**
   - Idiosyncratic returns not explained by factors
   - Columns: FACTSET_ID, DATE, SPECIFIC_RETURN

4. **`factor_comparisons_YYYYMMDD_YYYYMMDD_TIMESTAMP.csv`** (if S&P500 comparison enabled)
   - Factor comparison metrics with S&P500
   - Columns: correlation, correlation_squared, sharpe_ratio, etc.

5. **`sp500_returns_YYYYMMDD_YYYYMMDD_TIMESTAMP.csv`** (if retrieved)
   - S&P500 benchmark returns

6. **`summary_YYYYMMDD_YYYYMMDD_TIMESTAMP.txt`**
   - Text summary of the run with key statistics

### Output Directory Structure

```
EDS_factor_model/
├── outputs/
│   ├── factor_exposures_20230101_20241231_20241215_143022.csv
│   ├── factor_returns_20230101_20241231_20241215_143022.csv
│   ├── specific_returns_20230101_20241231_20241215_143022.csv
│   ├── summary_20230101_20241231_20241215_143022.txt
│   └── ...
```

## Accessing Results in Python

After running, you can access results programmatically:

```python
from factors import main
import pandas as pd

# Run the model
results = main(
    start_date='2023-01-01',
    end_date='2024-12-31',
    neutralize=False
)

# Access DataFrames
exposures_df = results['exposures']
factor_returns_df = results['factor_returns']
specific_returns_df = results['specific_returns']

# Example: Analyze factor returns
factor_returns_wide = factor_returns_df.pivot_table(
    index='DATE',
    columns='FACTOR_NAME',
    values='RETURN'
)

# Calculate cumulative returns
cumulative_returns = (1 + factor_returns_wide).cumprod()

# Example: Analyze exposures
exposures_by_date = exposures_df.groupby('DATE').mean()
```

## Parameters Explained

### `start_date` and `end_date`
- Format: `'YYYY-MM-DD'` (e.g., `'2023-01-01'`)
- Minimum enforced: `'2015-01-01'` (all data must be >= this date)
- Default: 252 trading days ago to today

### `neutralize`
- `False`: Style factors are standardized but not neutralized
- `True`: Style factors are regressed on country/sector dummies, residuals are z-scored
- Country and sector factors are always 0/1 dummies (not neutralized)

### `min_stocks_per_date`
- Minimum number of stocks required per date for factor return calculation
- Default: 50
- Dates with fewer stocks will be skipped

## Troubleshooting

### Connection Errors
- Verify `.env` file exists and has correct credentials
- Test connection: `python test_connection.py`
- Check network/firewall settings

### Data Errors
- Verify table names in `config.py` match your Snowflake schema
- Check that date range has sufficient data
- Ensure minimum date constraint: `DATE >= '2015-01-01'`

### Memory Issues
- Reduce date range if processing large datasets
- Process in smaller chunks if needed

### Output Directory
- The `outputs/` directory is created automatically
- If you get permission errors, check write permissions
- Files are named with timestamps to avoid overwriting

## Next Steps

After running the model:

1. **Review Outputs**: Check CSV files in `outputs/` directory
2. **Analyze Factor Returns**: Examine which factors are most significant
3. **Review Exposures**: Understand factor loadings across stocks
4. **Specific Returns**: Analyze idiosyncratic risk
5. **Iterate**: Adjust parameters (neutralization, date ranges) as needed

## Example Workflow

```python
# 1. Run the model
from factors import main

results = main(
    start_date='2023-01-01',
    end_date='2024-12-31',
    neutralize=True,  # Try with neutralization
    min_stocks_per_date=50
)

# 2. Load saved outputs
import pandas as pd
exposures = pd.read_csv('outputs/factor_exposures_20230101_20241231_*.csv')
factor_returns = pd.read_csv('outputs/factor_returns_20230101_20241231_*.csv')

# 3. Analyze results
factor_returns_wide = factor_returns.pivot_table(
    index='DATE',
    columns='FACTOR_NAME',
    values='RETURN'
)

print(factor_returns_wide.describe())
print(factor_returns_wide.corr())
```

