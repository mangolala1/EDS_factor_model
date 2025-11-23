# EDS Factor Model - MSCI-Styled Factor Model

A comprehensive factor model system that constructs, tests, and optimizes factors to achieve high correlation with S&P500 returns.

## Overview

This project implements an MSCI-styled factor model that:
1. Retrieves data from Snowflake
2. Constructs multiple factors (momentum, value, quality, size, volatility, reversal)
3. Tests factor effectiveness using Information Coefficient (IC) and quantile analysis
4. Builds an optimized factor model targeting >70% correlation with S&P500 returns

## Project Structure

```
EDS_factor_model/
├── factors.py              # Main workflow script
├── config.py               # Configuration settings
├── data_retrieval.py       # Snowflake data retrieval module
├── factor_construction.py  # Factor construction logic
├── factor_testing.py       # Factor testing and validation
├── model_builder.py        # Factor model optimization
├── requirements.txt        # Python dependencies
├── .env.example           # Example environment variables
└── README.md              # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Snowflake Connection

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

Edit `.env` with your Snowflake credentials:
- `SNOWFLAKE_ACCOUNT`: Your Snowflake account identifier
- `SNOWFLAKE_USER`: Your Snowflake username
- `SNOWFLAKE_PASSWORD`: Your Snowflake password
- `SNOWFLAKE_WAREHOUSE`: Your warehouse name
- `SNOWFLAKE_DATABASE`: Your database name
- `SNOWFLAKE_SCHEMA`: Your schema name
- `SNOWFLAKE_ROLE`: Your role (optional)

### 3. Prepare Your Snowflake Data

Your Snowflake database should have two tables:

**Price Table** (`stock_prices`):
- `ticker` (VARCHAR)
- `date` (DATE)
- `open`, `high`, `low`, `close` (NUMERIC)
- `volume` (NUMERIC)
- `adj_close` (NUMERIC)

**Fundamental Table** (`stock_fundamentals`):
- `ticker` (VARCHAR)
- `date` (DATE)
- `market_cap` (NUMERIC)
- `pe_ratio`, `pb_ratio`, `ev_ebitda` (NUMERIC)
- `roe`, `roa` (NUMERIC)
- `debt_to_equity` (NUMERIC)
- `revenue`, `earnings` (NUMERIC)

## Usage

### Basic Workflow

1. **Update the main script** (`factors.py`) with your actual:
   - Table names
   - Ticker list (or query to retrieve tickers)
   - Date ranges

2. **Run the model**:

```bash
python factors.py
```

### Customization

#### Modify Factor Parameters

Edit `config.py` to adjust:
- `FACTOR_PARAMS`: Factor construction parameters
- `TARGET_CORRELATION`: Target correlation threshold (default: 0.70)
- `LOOKBACK_PERIOD`: Historical data period (default: 252 days)

#### Add Custom Factors

Extend `FactorConstructor` in `factor_construction.py` to add new factors:

```python
def construct_custom_factor(self) -> pd.DataFrame:
    # Your custom factor logic
    pass
```

## Factor Model Components

### Factors Constructed

1. **Momentum**: Price momentum over 3-month lookback period
2. **Value**: Composite of PE, PB, and EV/EBITDA ratios
3. **Quality**: Composite of ROE, ROA, and debt-to-equity
4. **Size**: Market capitalization (inverse log rank)
5. **Volatility**: Low volatility factor (inverse of rolling volatility)
6. **Reversal**: Short-term mean reversion factor

### Model Building Methods

1. **Ridge Regression**: L2 regularization
2. **Lasso Regression**: L1 regularization with feature selection
3. **Optimization**: Direct weight optimization to maximize correlation

## Output

The model provides:
- Factor Information Coefficient (IC) statistics
- Quantile portfolio returns
- Factor weights
- Model correlation with S&P500
- R² score and other performance metrics

## Target Correlation

The model is optimized to achieve **>70% correlation** with historical S&P500 returns. The target can be adjusted in `config.py`.

## Notes

- The model uses equal-weighted portfolio returns by default. For production use, consider implementing market-cap weighting.
- Factor construction handles missing data by forward-filling where appropriate.
- The model includes time-series cross-validation considerations.

## Troubleshooting

### Connection Issues
- Verify Snowflake credentials in `.env`
- Check network connectivity and firewall settings
- Ensure your Snowflake user has appropriate permissions

### Data Issues
- Verify table names match your Snowflake schema
- Check date formats and data types
- Ensure sufficient historical data (minimum 60 data points)

### Model Performance
- If correlation is below target, try:
  - Increasing lookback period
  - Adding more factors
  - Adjusting regularization parameters
  - Using optimization method instead of regression

## License

This project is for internal use.

