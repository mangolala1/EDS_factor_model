# EDS Factor Model Pipeline (BERKELEY)

This project builds a **daily cross-sectional factor model** from Snowflake-sourced market and fundamentals data, then writes the model outputs as Parquet tables.

It produces the core artifacts you typically need for a Barra-like workflow:
- daily **exposures** (betas)
- daily **factor returns** and **specific returns**
- rolling **factor covariance**
- rolling **specific variance / volatility**
- **periodic** (monthly/quarterly) factor returns
- **peer group** statistics (sector and region×sector)
- **factor metadata**
- **factor-mimicking portfolios** for **style factors** (true mimicking weights)

---

## What does this project do?

### Inputs (from Snowflake → local Parquet)
The pipeline expects three local Parquet inputs under `data/`:

- `prices.parquet` (split-adjusted price + volume)
- `fundamentals.parquet` (NTM/LTM fundamentals)
- `universe.parquet` (security attributes, including `SECTOR` and `COUNTRY`)

A helper download step (`scripts/bulk_download.py`) pulls these from a **read-only Snowflake account** and saves them locally, including a derived `returns.parquet`.

### Model (daily cross-sectional regression)
For each trading day \(t\), the model runs a cross-sectional OLS regression:

\[
r_{i,t} = \alpha_t + \sum_{k \in \text{style}} \beta_{i,k,t} f_{k,t} + \sum_{g \in \text{dummies}} d_{i,g,t} f_{g,t} + \varepsilon_{i,t}
\]

Where:
- \(r_{i,t}\) is the daily stock return
- \(\alpha_t\) is a global **intercept** (“market”/baseline)
- \(\beta_{i,k,t}\) are **style** exposures (z-scored cross-sectionally each day)
- \(d_{i,g,t}\) are **sector** and **continent** dummy exposures (mean-centered per day)
- \(f_{k,t}\) are daily factor returns
- \(\varepsilon_{i,t}\) are daily **specific returns** (residuals)

### Risk (factor covariance + specific variance)
The pipeline builds:
- **Factor covariance** \(\Sigma_{f,t}\): rolling covariance of factor returns over `cov_window` days.
- **Specific variance** \(\sigma^2_{\varepsilon,i,t}\) using either:
  - `total_minus_explained` (default): rolling total variance minus model-implied explained variance
  - `residual_rolling`: rolling variance of residuals

Specific variance is floored for stability and also reported as specific volatility.

---

## Project structure

```
EDS_factor_model/
  data/                         # local inputs (Parquet)
  outputs/                      # model outputs (Parquet), written by the pipeline

  factor_model/                 # reusable library code
    config.py                   # paths + model defaults + Snowflake env loader
    logging_utils.py            # timestamped logging setup

    data_retrieval.py           # Snowflake read-only retriever (tables/views)
    bulk_download.py            # chunked downloads -> data/*.parquet (+ returns.parquet)

    features_raw.py             # raw panel + characteristics (value/growth/etc.)
    exposures.py                # winsorize + zscore + dummies + composites
    regression.py               # daily OLS -> factor returns + residuals
    risk.py                     # factor covariance + specific variance
    tables.py                   # RETURNS, BETA, PERIODIC_FACTOR_RETURNS, PEER_GROUP_DATA, FACTOR_METADATA
    portfolios.py               # style factor-mimicking portfolios (streamed parquet)
    pipeline.py                 # orchestrates quarterly chunks + writes outputs
    combine.py                  # combines quarterly shards into full-history tables

  scripts/
    bulk_download.py            # entry point: Snowflake -> local Parquet
    run_pipeline.py             # entry point: run quarterly pipeline + combine

  notebooks/
    run_pipeline.py             # same as scripts/run_pipeline.py, convenient for notebooks

  tests/
    test_quarterly.py           # placeholder for automated checks

  requirements.txt
  .env.example
```

---

## Quick start guide

### 1) Create and activate an environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows (PowerShell)
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Create your `.env`
Copy `.env.example` to `.env` and fill in Snowflake credentials:

```env
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_WAREHOUSE=...
SNOWFLAKE_ROLE=...
SNOWFLAKE_DATABASE=EDS_DEV_BERKELEY
SNOWFLAKE_SCHEMA=BERKELEY
```

### 4) Download input data from Snowflake
```bash
python scripts/bulk_download.py
```

This writes:
- `data/prices.parquet`
- `data/returns.parquet` (computed from prices)
- `data/fundamentals.parquet`
- `data/universe.parquet` (+ `data/universe.csv`)

### 5) Run the factor model pipeline
```bash
python scripts/run_pipeline.py
```

Outputs are written under:
- `outputs/NEW_FY2023_25/` (default in code)

---

## What are the stages of a run?

A full run (via `scripts/run_pipeline.py`) does the following per quarter:

1. **Load inputs** (from `data/*.parquet`) with a lookback window for rolling features.
2. **Build raw features** (`features_raw.py`):
   - join prices + universe + fundamentals
   - build characteristics (value yields, margins, growth, momentum, volatility, liquidity)
   - map `COUNTRY -> CONTINENT` via `continent_mapping.py`
3. **Build exposures** (`exposures.py`):
   - winsorize each characteristic cross-sectionally by date (default 1%/99%)
   - z-score each characteristic cross-sectionally by date
   - build composite style factors (VALUE/PROFITABILITY/GROWTH) and re-zscore them
   - build sector and continent dummies, mean-center per date, drop one dummy per group
4. **Daily regression** (`regression.py`):
   - OLS per date with an intercept
   - outputs: factor returns, specific returns (residuals), long exposures
5. **Risk estimation** (`risk.py`):
   - rolling factor covariance from factor returns (`cov_window`)
   - rolling specific variance (`var_window`) using the chosen method
6. **Additional tables** (`tables.py`):
   - `RETURNS` table (security returns)
   - `BETA` table (wide exposures per security/date)
   - `PERIODIC_FACTOR_RETURNS` (M and Q geometric compounding)
   - `PEER_GROUP_DATA` (sector and region×sector stats per day)
   - `FACTOR_METADATA` (factor groups/types/descriptions)
7. **Factor-mimicking portfolios** (`portfolios.py`):
   - true style factor-mimicking weights per day, streamed to Parquet
8. **Write quarterly shards**, then **combine** shards into full-history Parquets.

---

## Factor model workflow (how the math maps to code)

### Raw characteristics → standardized exposures
For each day \(t\) and characteristic \(x\):

**Winsorization**
\[
\tilde{x}_{i,t} = \min(\max(x_{i,t}, Q_{q_{low},t}), Q_{q_{high},t})
\]

**Z-score**
\[
z_{i,t} = \frac{\tilde{x}_{i,t} - \mu_t}{\sigma_t}
\]

This is done **cross-sectionally by date** (not time-series).

### Style factors currently implemented
From `features_raw.py` and `exposures.py`:

- **VALUE**: average of z-scored yields (`EY_*`, `SY_*`, `EBITDA_Y_NTM`), then re-zscored
- **PROFITABILITY**: average of z-scored margins (`EBITDA_MARGIN`, `GROSS_MARGIN`), then re-zscored
- **GROWTH**: average of z-scored growth metrics (`EPS_GROWTH`, `SALES_GROWTH`), then re-zscored
- **MOMENTUM**: \(12-1\) using log-return sums:
  \[
  \text{MOM} = (\exp(\sum_{252}\log(1+r)) - 1) - (\exp(\sum_{21}\log(1+r)) - 1)
  \]
- **VOLATILITY**: rolling 60d std of daily returns
- **LIQUIDITY**: \(\log(\text{20d avg}(P \times V))\)

### Sector/continent dummies
Dummies are built with `pd.get_dummies`, then **mean-centered per date** (so their cross-sectional mean is ~0 each day). One dummy per group is dropped for identifiability.

### Daily OLS (factor returns + specific returns)
For each day, OLS solves:
\[
\hat{f}_t = \arg\min_f \|y_t - X_t f\|_2^2
\]

Implementation uses the standard normal equations (`solve(X'X, X'y)`) with a fallback to `lstsq` when needed.

### Risk decomposition
For a given day \(t\) and security \(i\):

- Factor covariance estimated as rolling covariance of factor returns:
  \[
  \Sigma_{f,t} = \text{Cov}(f_{t-cov\_window+1:t})
  \]
- Explained variance:
  \[
  \sigma^2_{\text{explained},i,t} = b_{i,t}^\top \Sigma_{f,t} b_{i,t}
  \]
- Total variance is rolling variance of returns.
- Specific variance (default):
  \[
  \sigma^2_{\varepsilon,i,t} = \max(\sigma^2_{\text{total},i,t} - \sigma^2_{\text{explained},i,t}, \text{floor})
  \]

---

## Output files explained

All outputs are written as **quarterly shards** first:

```
outputs/NEW_FY2023_25/
  EDS_BERKELEY_<TABLE>_YYYY-MM-DD_YYYY-MM-DD.parquet
```

Then combined into a full table:

```
outputs/NEW_FY2023_25/
  EDS_BERKELEY_<TABLE>.parquet
```

### Core tables

- **EXPOSURE** (long)
  - Columns: `MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE`
  - Includes style factors, dummies, and `INTERCEPT`.

- **BETA** (wide)
  - Columns: `MODEL, DATE, SECURITY_ID, INTERCEPT, <style factors>, <dummies>`
  - Convenient for portfolio risk calculations.

- **FACTOR_RETURNS**
  - Columns: `MODEL, DATE, FACTOR_NAME, FACTOR_RETURN`

- **SPECIFIC_RETURNS**
  - Columns: `MODEL, DATE, SECURITY_ID, SPECIFIC_RETURN`

- **COVARIANCE**
  - Columns: `MODEL, DATE, FACTOR1, FACTOR2, COVARIANCE`
  - Rolling covariance of factor returns (default 60 trading days).

- **SPECIFIC_VARIANCE**
  - Columns: `MODEL, DATE, SECURITY_ID, SPECIFIC_VAR, SPECIFIC_VOL`

### “Convenience” tables

- **RETURNS**
  - Columns: `MODEL, DATE, SECURITY_ID, RETURN`
  - Security returns used in the regression.

- **PERIODIC_FACTOR_RETURNS**
  - Columns: `MODEL, FREQUENCY, PERIOD_END_DATE, FACTOR_NAME, PERIODIC_RETURN`
  - Compounded from daily factor returns for `FREQUENCY in {M, Q}`.

- **PEER_GROUP_DATA**
  - Sector and region×sector peer statistics per date.
  - Includes peer group labels and mean/std returns.

- **FACTOR_METADATA**
  - Factor descriptions, groups, and types (STYLE / DUMMY / MARKET).

- **FACTOR_NAMES**
  - A lightweight “factor display list” derived from the regression setup.

- **FACTOR_PORTFOLIO**
  - True factor-mimicking portfolios for style factors.
  - Columns: `MODEL, DATE, FACTOR_NAME, SECURITY_ID, WEIGHT`
  - Written in a streaming fashion because it can be very large.

---

## Customizing the run

### Change the date range
Edit the entry points:

- `scripts/bulk_download.py` controls the Snowflake pull range:
  - `start_date="2020-01-01", end_date=None`
- `scripts/run_pipeline.py` controls the model range:
  - `overall_start="2023-01-01", overall_end=None`

Set explicit end dates as `YYYY-MM-DD` to make runs deterministic.

### Change windows and model options
Defaults live in `factor_model/config.py` (`ModelConfig`):

- `cov_window` (factor covariance window)
- `var_window` (specific variance window)
- `lookback_days` (must be large enough for momentum + rolling windows)
- `winsor_low`, `winsor_high`

### Choose specific variance method
In `scripts/run_pipeline.py`:
- `specific_var_method="total_minus_explained"` (default)
- `specific_var_method="residual_rolling"` (alternative)

### Output directory
`run_quarterly()` defaults to:
- `outputs/NEW_FY2023_25/`

You can pass a different `out_dir` in `factor_model/pipeline.py` or modify `scripts/run_pipeline.py`.

---

## Snowflake data sources

The Snowflake retriever (`factor_model/data_retrieval.py`) is wired to these objects by default:

- `EDS_DEV_BERKELEY.BERKELEY.EDS_FACTORS_FUNDAMENTALS_NTM_LTM`
- `EDS_DEV_BERKELEY.BERKELEY.EDS_FACTORS_UNIVERSE`
- `EDS_DEV_BERKELEY.BERKELEY.SPLIT_ADJUSTED_PRICES_EDS_FACTORS`

Optional fetchers exist for:
- `ENTERPRISEVALUE_HISTORY`
- `EXCHANGERATES`
- `MARKET_VALUE_HISTORY`

Reader accounts are supported (no writes are required).

---

## Performance notes (what matters in practice)

- **Run fewer years first.** Most runtime scales with number of dates × number of securities.
- **Factor portfolios are huge.** `FACTOR_PORTFOLIO` can dominate runtime and disk; it writes per day per style factor.
- **Lookback matters.** If you start at 2020-01-01, the pipeline still queries/loads earlier dates due to `lookback_days` (momentum needs ~252 trading days).
- **Parquet + pyarrow helps.** Outputs are written with `snappy` compression.

---

## Troubleshooting

- **“Missing Snowflake env vars …”**
  - Ensure `.env` exists at repo root and includes `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`.

- **No output / slow output**
  - The pipeline logs with timestamps (`factor_model/logging_utils.py`). Ensure you run via `scripts/run_pipeline.py` (it calls `setup_logging()`).

- **Memory pressure**
  - Reduce the date range, or run fewer quarters.
  - The long exposure table and factor portfolio weights are the largest artifacts.

- **Different Snowflake database/schema**
  - Set `SNOWFLAKE_DATABASE` and `SNOWFLAKE_SCHEMA` in `.env`.

---

## Development / testing

The `tests/` folder is a starting point. High-value tests to add:
- z-scores are ~mean 0 / std 1 per date (for raw and composites)
- regression intercept is present and stable
- factor covariance dimensions match factor list
- specific variance is non-negative after flooring
