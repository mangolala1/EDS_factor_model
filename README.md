# EDS Factor Model

A factor model system that calculates factor exposures, factor returns, specific returns, and risk metrics for equity securities.

## What Does This Project Do?

This project builds a **multi-factor risk model** that:

1. **Calculates factor exposures** - How much each stock is exposed to different risk factors (Value, Growth, Momentum, etc.)
2. **Estimates factor returns** - Daily returns for each risk factor
3. **Computes specific returns** - Stock-specific (idiosyncratic) returns not explained by factors
4. **Calculates risk metrics** - Specific risk (variance) and factor covariance matrices

**Output**: CSV files with all factor model components ready for risk analysis and portfolio optimization.

---

## Project Structure

```
EDS_factor_model/
├── src/                    # Main source code
│   ├── main.py            # Main workflow orchestrator
│   ├── bulk_download.py   # Downloads data from Snowflake
│   ├── quarter_processor.py  # Processes quarters in parallel
│   ├── model_builder.py   # Factor calculations (returns, risk)
│   ├── data_retrieval.py  # Snowflake connection utilities
│   ├── ring_buffers.py    # Efficient rolling window calculations
│   ├── config.py          # Configuration settings
│   └── continent_mapping.py  # Geographic mapping utilities
│
├── tests/                 # Test scripts
│   ├── test_quarter.py   # Test for a single quarter (Q1 2020)
│   └── test_single_day.py # Test for a single day
│
├── data/                  # Input data (Parquet files)
│   ├── prices.parquet
│   ├── returns.parquet
│   ├── fundamentals.parquet
│   └── universe.parquet
│
├── results/               # Output CSV files
│   ├── exposures.csv
│   ├── factor_returns.csv
│   ├── specific_returns.csv
│   ├── specific_risk.csv
│   ├── factor_covariance.csv
│   └── factor_model_factor_names.csv
│
├── run_main.py           # Entry point - run this to start!
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

---

## Quick Start Guide

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Configure Snowflake Connection

Create a `.env` file in the project root with your Snowflake credentials:

```env
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

### Step 3: Run the Model

**Option A: Run the main workflow**
```bash
python run_main.py
```

**Option B: Run a test (recommended for first-time users)**
```bash
# Test with a single quarter (Q1 2020)
python tests/test_quarter.py

# Or test with a single day
python tests/test_single_day.py
```

---

## What Happens When You Run?

The workflow executes in 6 stages:

1. **Stage 1: Download Data** (one-time, ~30-60 min)
   - Downloads prices, returns, fundamentals from Snowflake
   - Saves to local Parquet files (`data/*.parquet`)
   - Skip this step on subsequent runs (set `skip_download=True`)

2. **Stage 2: Process Quarters** (~10 min per quarter)
   - Calculates factor exposures for each stock on each date
   - Processes quarters in parallel for speed
   - Output: `exposures.csv`

3. **Stage 3: Calculate Factor Returns** (~5 min)
   - Runs cross-sectional regressions to estimate factor returns
   - Output: `factor_returns.csv`

4. **Stage 4: Calculate Specific Returns** (~5 min)
   - Computes residual returns (actual - predicted)
   - Output: `specific_returns.csv`

5. **Stage 5: Calculate Specific Risk** (~10-20 min)
   - Calculates specific variance using 60-day rolling window
   - Output: `specific_risk.csv`

6. **Stage 6: Calculate Factor Covariance** (~5 min)
   - Computes factor covariance matrix using 60-day rolling window
   - Output: `factor_covariance.csv`

**Total time**: ~70-90 minutes for a full year (first run), ~30-40 minutes for subsequent runs

---

## Output Files Explained

All output files are saved as CSV in the `results/` directory:

| File | Description | Columns |
|------|-------------|---------|
| `exposures.csv` | Factor exposures for each stock | MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE |
| `factor_returns.csv` | Daily returns for each factor | MODEL, DATE, FACTOR_NAME, RETURN |
| `specific_returns.csv` | Stock-specific returns | MODEL, DATE, SECURITY_ID, SPECIFIC_RETURN |
| `specific_risk.csv` | Specific variance for each stock | MODEL, DATE, SECURITY_ID, SPECIFIC_VAR |
| `factor_covariance.csv` | Factor covariance matrix | MODEL, DATE, FACTOR_NAME_1, FACTOR_NAME_2, COVARIANCE |
| `factor_model_factor_names.csv` | Factor metadata | MODEL, FACTOR_DISPLAY_NAME, FACTOR_GROUP |

---

## Customizing the Run

Edit `run_main.py` to customize parameters:

```python
main(
    start_date='2020-01-01',      # Start date
    end_date='2020-12-31',        # End date (None = today)
    neutralize=False,              # Whether to neutralize factors
    output_dir='results',         # Output directory
    data_dir='data',              # Data directory
    skip_download=True            # Skip download if data exists
)
```

Or import and run programmatically:

```python
from src.main import main

results = main(
    start_date='2024-01-01',
    end_date='2024-12-31',
    skip_download=False  # Download new data
)
```

---

## Running Tests

**Test a single quarter (Q1 2020):**
```bash
python tests/test_quarter.py
```
Outputs: `test_results_q1_2020/*.csv`

**Test a single day:**
```bash
python tests/test_single_day.py
```
Outputs: `test_results/*.csv`

---

## Key Features

- ✅ **Fast**: Uses Parquet files (10-50x faster than SQLite)
- ✅ **Parallel Processing**: Processes quarters in parallel
- ✅ **Optimized**: Specific risk calculation optimized (~40x speedup)
- ✅ **No Database**: Pure file-based workflow (CSV + Parquet)
- ✅ **One-Time Download**: Download data once, process many times

---

## Troubleshooting

**"No module named 'src'"**
- Make sure you're running from the project root directory
- Use `python run_main.py` (not `python src/main.py`)

**"Snowflake connection failed"**
- Check your `.env` file exists and has correct credentials
- Verify Snowflake account, warehouse, database, and schema names

**"Insufficient data"**
- Ensure you have at least 60 trading days of data for specific risk calculation
- Check that `data/*.parquet` files exist and contain data

**Memory issues**
- Reduce `MAX_STOCKS` in `src/config.py` if processing too many stocks
- Process smaller date ranges

---

## For More Details

See `FACTOR_MODEL_WORKFLOW.md` for:
- Detailed mathematical formulation
- Step-by-step algorithm descriptions
- Performance optimization notes
- Factor construction methodology
