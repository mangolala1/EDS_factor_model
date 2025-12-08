# EDS Factor Model

Streamlined factor model workflow that downloads all data upfront to **local Parquet files**, then processes quarters in parallel for speed. **No Snowflake queries after initial download!**

## Quick Start

### 1. Install Dependencies

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Snowflake Connection

Create `.env` file with your Snowflake credentials:
```
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=your_user
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=your_database
SNOWFLAKE_SCHEMA=your_schema
```

### 3. Run the Workflow

```bash
python main.py
```

This will:
1. **Download all data** from Snowflake → save to `data/*.parquet` (one-time, ~30-60 minutes)
2. **Process quarters in parallel** using local Parquet files (~10 minutes per quarter)
3. **Calculate factor returns, specific returns, risk, covariance** (~20-30 minutes total)

## Workflow Overview

### Stage 1: Bulk Download (One-Time)
- Downloads prices, returns, fundamentals from Snowflake
- **Saves to local Parquet files** (`data/prices.parquet`, `data/returns.parquet`, etc.)
- **Much faster to read than SQLite** for large datasets
- **One-time operation** - subsequent runs skip this

### Stage 2: Quarter Processing (Local Files Only!)
- Loads data from Parquet files (fast!)
- Processes quarters in parallel (uses all CPU cores)
- Each quarter takes ~10 minutes
- **No Snowflake queries** - works entirely locally

### Stages 3-6: Factor Calculations
- Factor returns (cross-sectional OLS)
- Specific returns (idiosyncratic)
- Specific risk (60-day rolling window)
- Factor covariance (60-day rolling window)

## File Structure

- `main.py` - Main entry point
- `bulk_download.py` - Downloads all data from Snowflake → Parquet files
- `quarter_processor.py` - Processes quarters in parallel (reads Parquet files, writes CSV)
- `model_builder.py` - Factor return/risk calculations
- `ring_buffers.py` - Memory-efficient rolling calculations
- `data_retrieval.py` - Snowflake connection and queries
- `config.py` - Configuration parameters
- `continent_mapping.py` - Continent mapping utility

## Data Storage

### Input Data (Parquet Files - Fast!)
- `data/prices.parquet` - All price data
- `data/returns.parquet` - All return data
- `data/fundamentals.parquet` - All fundamental data
- `data/universe.parquet` - Universe metadata

### Output (CSV Files)
All results are stored as CSV files in `results/` directory:
- `results/exposures.csv` - Factor exposures per stock per date
- `results/factor_returns.csv` - Daily factor returns
- `results/specific_returns.csv` - Stock-specific returns
- `results/specific_risk.csv` - Specific risk metrics (60-day window)
- `results/factor_covariance.csv` - Factor covariance matrix (60-day window)

## Performance

- **Bulk download**: ~30-60 minutes (one-time)
- **Parquet file reads**: **10-50x faster than SQLite** for large datasets
- **Quarter processing**: ~10 minutes per quarter
- **Factor calculations**: ~20-30 minutes total

For 2025 (4 quarters): ~70-90 minutes total

## Re-running

If you've already downloaded data, set `skip_download=True` in `main.py`:

```python
results = main(
    start_date='2020-01-01',
    end_date=None,
    skip_download=True  # Use local Parquet files (no Snowflake!)
)
```

## Processing Previous Years

Just change the date range in `main.py`:

```python
results = main(
    start_date='2024-01-01',
    end_date='2024-12-31',
    skip_download=False  # Download new data
)
```

The workflow will automatically:
- Download data for the new date range
- Append to existing Parquet files (or create new ones)
- Process all quarters in parallel
- Skip already-processed dates

## Why Parquet for Input & CSV for Output?

**Input (Parquet):**
- **10-50x faster reads** for large datasets
- **Columnar storage** - only reads columns you need
- **Compression** - smaller file sizes
- **Perfect for analytical workloads** like factor models

**Output (CSV):**
- **Universal format** - easy to open in Excel, Python, R, etc.
- **No database overhead** - simple file-based storage
- **Easy to share and analyze** - standard format for results
