# EDS Factor Model - Complete Workflow Explanation

## Overview
This document explains the complete workflow of the EDS Factor Model, from data retrieval to factor return calculation and S&P500 comparison.

---

## PART 1: DATA RETRIEVAL

### Step 1.1: Universe Data Retrieval
**Location**: `data_retrieval.py` → `get_universe_data()`

**Table**: `EDS_FACTORS_UNIVERSE`

**What happens**:
1. Retrieves all stocks from the universe table (no ticker filtering)
2. Extracts only these columns:
   - `FACTSET_ID` (unique identifier for each security)
   - `COUNTRY`
   - `SECTOR`

**Output**: DataFrame with FACTSET_IDs, countries, and sectors for all stocks in universe

**Purpose**: This creates our universe of stocks to analyze and provides the FACTSET_IDs needed to filter all other tables.

**Note**: We use all stocks in the universe, not filtered by S&P500 ticker list.

---

### Step 1.2: Extract FACTSET_IDs for Filtering
**Location**: `data_retrieval.py` → `get_all_data()`

**What happens**:
1. Takes the unique `FACTSET_ID` values from the universe
2. Uses these IDs to filter all subsequent data queries

**Output**: List of FACTSET_IDs for all stocks in universe

---

### Step 1.3: Fundamentals Data Retrieval
**Location**: `data_retrieval.py` → `get_fundamentals_data()`

**Table**: `EDS_FACTORS_FUNDAMENTALS_NTM_LTM`

**What happens**:
1. Filters by `FACTSET_ID` (only stocks in our ticker list)
2. Filters by `DATE >= '2015-01-01'` (minimum date constraint)
3. Retrieves these columns:
   - `FACTSET_ID`
   - `DATE`
   - `EPS_NTM` (Earnings Per Share, Next Twelve Months)
   - `EPS_LTM` (Earnings Per Share, Last Twelve Months)
   - `SALES_NTM` (Sales, Next Twelve Months)
   - `SALES_LTM` (Sales, Last Twelve Months)
   - `EBITDA_NTM` (EBITDA, Next Twelve Months)
   - `EBITDA_LTM` (EBITDA, Last Twelve Months)
   - `COGS_NTM` (Cost of Goods Sold, Next Twelve Months)
   - `COGS_LTM` (Cost of Goods Sold, Last Twelve Months)

**Output**: DataFrame with fundamental metrics for each stock-date combination

**Purpose**: These fundamentals are used to construct Value, Profitability, and Growth factors.

---

### Step 1.4: Prices Data Retrieval
**Location**: `data_retrieval.py` → `get_prices_data()`

**Table**: `SPLIT_ADJUSTED_PRICES_EDS_FACTORS`

**What happens**:
1. Filters by `FACTSET_ID`
2. Filters by `DATE >= '2015-01-01'`
3. Retrieves all columns from the table:
   - `FACTSET_ID`
   - `DATE`
   - `SPLIT_FACTOR`
   - `SPECIAL_DIVS_FACTOR`
   - `UNADJUSTED_PRICE`
   - `ADJUSTED_PRICE` ⭐ (used for Value factors)
   - `ADJUSTED_VOLUME` ⭐ (used for Liquidity factor)
   - `ADJUSTED_PRICE_DAY_HIGH`
   - `ADJUSTED_PRICE_DAY_LOW`
   - `CURRENCY` ⭐ (used for currency conversion)
   - `P_DIVS_PD`
   - `P_SPLIT_FACTOR`
   - `IS_HOLIDAY`

**Output**: DataFrame with price and volume data

**Purpose**: Used in Value factors (price in denominator) and Liquidity factor (price × volume).

---

### Step 1.5: Calculate Returns from Prices
**Location**: `data_retrieval.py` → `calculate_returns_from_prices()`

**What happens**:
1. Takes the prices DataFrame (with ADJUSTED_PRICE)
2. Calculates daily returns: `ONE_DAY_PCT = (P_t / P_{t-1}) - 1`
3. Groups by FACTSET_ID and calculates percentage change
4. Removes first row for each stock (no previous price available)

**Output**: DataFrame with returns data (FSYM_ID, P_DATE, ONE_DAY_PCT)

**Purpose**: Used to calculate Momentum and Volatility factors, and for factor return regression.

**Note**: Returns are calculated from ADJUSTED_PRICE, which already accounts for splits and dividends.

---

### Step 1.6: Exchange Rates Retrieval
**Location**: `data_retrieval.py` → `get_exchange_rates()`

**Table**: Exchange rate table (configurable, defaults to common names)

**What happens**:
1. Retrieves exchange rates for currency conversion
2. Filters by `DATE >= '2015-01-01'`
3. Columns:
   - `DATE`
   - `CURRENCY`
   - `EXCHANGE_RATE_TO_USD` (number of USD per unit of currency)

**Output**: DataFrame with exchange rates

**Purpose**: Converts market values from local currency to USD (enterprise value is already in USD).

---

### Step 1.7: Market Value and Enterprise Value Retrieval

**Location**: `data_retrieval.py` → `get_market_value_data()`, `get_enterprise_value_data()`

**Tables**: `MARKET_VALUE_HISTORY`, `ENTERPRISEVALUE_HISTORY`

**What happens**:
1. Retrieves market value (market cap) from `MARKET_VALUE_HISTORY`
   - Market value is in local currency (needs conversion to USD)
2. Retrieves enterprise value from `ENTERPRISEVALUE_HISTORY`
   - Enterprise value is already in USD

**Output**: DataFrames with market value and enterprise value data

**Purpose**: Used for market value calculations and currency conversion

---

### Step 1.8: S&P500 Returns Retrieval (Optional)
**Location**: `data_retrieval.py` → `get_sp500_returns()`

**Source**: Yahoo Finance API

**What happens**:
1. Downloads S&P500 historical prices
2. Calculates daily returns: `(Price_today - Price_yesterday) / Price_yesterday`
3. Returns a time series of daily returns

**Output**: Pandas Series with S&P500 daily returns

**Purpose**: Benchmark for comparing factor returns (currently optional/skipped).

**Note**: S&P500 comparison is kept in code but skipped for now since we're using all stocks in the universe, not just S&P500 stocks.

---

## PART 2: FACTOR CONSTRUCTION

### Step 2.1: Data Preparation
**Location**: `factor_construction.py` → `_prepare_data()`

**What happens**:
1. Converts all DATE columns to datetime format
2. Merges universe data with fundamentals data (adds COUNTRY and SECTOR)

**Output**: Prepared DataFrames ready for factor calculation

---

### Step 2.2: Merge Prices and Fundamentals
**Location**: `factor_construction.py` → `_merge_price_fundamentals()`

**What happens**:
1. Merges fundamentals DataFrame with prices DataFrame
2. Join keys: `FACTSET_ID` and `DATE`
3. Creates a unified dataset with both fundamental metrics and prices

**Output**: Merged DataFrame with all data needed for factor calculations

---

### Step 2.3: Construct Individual Factor Characteristics

#### 2.3.1: Value Factor
**Location**: `factor_construction.py` → `construct_value_factor()`

**Characteristics Calculated** (5 total):
1. **Forward Earnings Yield**: `EY_NTM = EPS_NTM / ADJUSTED_PRICE`
2. **Forward Sales Yield**: `SY_NTM = SALES_NTM / ADJUSTED_PRICE`
3. **Forward EBITDA Yield**: `EBITDA_Y_NTM = EBITDA_NTM / ADJUSTED_PRICE`
4. **Earnings-to-Price LTM**: `EY_LTM = EPS_LTM / ADJUSTED_PRICE`
5. **Sales-to-Price LTM**: `SY_LTM = SALES_LTM / ADJUSTED_PRICE`

**Qualitative Rules**:
- If `EPS_NTM <= 0`, ignore earnings yield (EY_NTM)
- If `EBITDA_NTM <= 0`, ignore EBITDA yield
- If `EPS_LTM <= 0`, ignore LTM earnings yield (EY_LTM)

**Processing Steps**:
1. Winsorize each characteristic cross-sectionally (1st-99th percentile)
2. Standardize each characteristic cross-sectionally (z-scores)
3. Average available standardized characteristics: `c_Value = (1/M) * Σ z_i,m`
4. Standardize the composite again: `β_Value = zscore(c_Value)`

**Output**: Single composite VALUE factor exposure per stock-date

---

#### 2.3.2: Profitability Factor
**Location**: `factor_construction.py` → `construct_profitability_factor()`

**Characteristics Calculated** (2 total):
1. **EBITDA Margin**: `EBITDA_M = EBITDA_LTM / SALES_LTM`
2. **Gross Margin**: `GM = 1 - (COGS_LTM / SALES_LTM)`

**Qualitative Rules**:
- If `SALES_LTM <= 0`, skip both margins
- If `COGS_LTM <= 0`, exclude gross margin

**Processing Steps**:
1. Winsorize each characteristic cross-sectionally
2. Standardize each characteristic cross-sectionally
3. Average available standardized characteristics
4. Standardize the composite again

**Output**: Single composite PROFITABILITY factor exposure per stock-date

---

#### 2.3.3: Growth Factor
**Location**: `factor_construction.py` → `construct_growth_factor()`

**Characteristics Calculated** (2 total):
1. **EPS Growth**: `EPSG = (EPS_NTM / EPS_LTM) - 1`
2. **Sales Growth**: `SG = (SALES_NTM / SALES_LTM) - 1`

**Qualitative Rules**:
- If `EPS_LTM <= 0`, ignore EPS growth
- If `SALES_LTM <= 0`, ignore sales growth

**Processing Steps**:
1. Winsorize each characteristic cross-sectionally
2. Standardize each characteristic cross-sectionally
3. Average available standardized characteristics
4. Standardize the composite again

**Output**: Single composite GROWTH factor exposure per stock-date

---

#### 2.3.4: Momentum Factor
**Location**: `factor_construction.py` → `construct_momentum_factor()`

**Characteristic Calculated** (1 total):
1. **12-1 Month Momentum**: Cumulative return from month -12 to month -1

**Formula**: `MOM12_1_i,t = exp(Σ(s=t-252 to t-21) ln(1 + r_i,s)) - 1`
- Uses 252 trading days (12 months)
- Excludes last 21 days (1 month) to avoid short-term reversal

**Qualitative Rules**:
- If price history < 252 days → no momentum
- Always exclude last 21 days to avoid short-term reversal effect

**Processing Steps**:
1. Winsorize cross-sectionally
2. Standardize cross-sectionally
(No composite needed - single characteristic)

**Output**: Single MOMENTUM factor exposure per stock-date

---

#### 2.3.5: Volatility Factor
**Location**: `factor_construction.py` → `construct_volatility_factor()`

**Characteristic Calculated** (1 total):
1. **60-Day Rolling Volatility**: Standard deviation of daily returns over past 60 days

**Formula**: `VOL60_i,t = sqrt((1/(60-1)) * Σ(s=t-59 to t) (r_i,s - r̄_i,t^(60))^2)`

**Qualitative Rules**:
- If returns < 30 days → skip

**Processing Steps**:
1. Winsorize cross-sectionally
2. Standardize cross-sectionally
(No composite needed - single characteristic)

**Output**: Single VOLATILITY factor exposure per stock-date

---

#### 2.3.6: Liquidity Factor
**Location**: `factor_construction.py` → `construct_liquidity_factor()`

**Characteristic Calculated** (1 total):
1. **Log Dollar Volume**: `LIQ = ln(mean(DV_{t-19:t}))`
   where `DV = ADJUSTED_PRICE × ADJUSTED_VOLUME`

**Qualitative Rules**:
- If volume missing → skip
- If price missing → skip

**Processing Steps**:
1. Calculate dollar volume
2. Calculate 20-day average dollar volume
3. Take natural logarithm
4. Winsorize cross-sectionally
5. Standardize cross-sectionally
(No composite needed - single characteristic)

**Output**: Single LIQUIDITY factor exposure per stock-date

---

### Step 2.4: Combine All Style Factors
**Location**: `factor_construction.py` → `construct_all_factors()`

**What happens**:
1. Each factor is already a single composite exposure (not multiple characteristics)
2. Combines all 6 style factor DataFrames:
   - VALUE (composite)
   - PROFITABILITY (composite)
   - GROWTH (composite)
   - MOMENTUM (single characteristic)
   - VOLATILITY (single characteristic)
   - LIQUIDITY (single characteristic)
3. Concatenates along columns (axis=1)

**Output**: DataFrame with 6 style factor exposures per stock-date

---

### Step 2.4: Composite Factor Construction Process

**For Each Multi-Characteristic Factor** (Value, Profitability, Growth):
1. **Winsorize each characteristic** cross-sectionally by date (1st-99th percentile)
2. **Standardize each characteristic** cross-sectionally by date (z-scores)
3. **Average available standardized characteristics**: `c_Factor = (1/M) * Σ z_i,m`
   - Only averages non-NaN values (respects qualitative rules)
4. **Standardize the composite again**: `β_Factor = zscore(c_Factor)`

**For Single-Characteristic Factors** (Momentum, Volatility, Liquidity):
1. **Winsorize** cross-sectionally by date
2. **Standardize** cross-sectionally by date

**Output**: Each factor is now a single standardized composite exposure

---

### Step 2.5: Neutralization (Optional)
**Location**: `factor_construction.py` → `neutralize()`

**What happens** (for each style factor):
1. Regress standardized composite factor on country and sector dummies
2. Extract residual `u`: `u_i,k,t = β_i,k,t - Σ_j D_i,j * γ_j,k,t`
3. Apply z-score to residuals: `β̃_i,k,t = z-score(u_i,k,t)`

**Note**: Neutralization is applied to style factors only. Country and sector exposures themselves are built as 0/1 dummies.

**Formula**: `β_i,k,t = Σ_j D_i,j * γ_j,k,t + u_i,k,t`
          `β̃_i,k,t = z-score(u_i,k,t)` (residual, then z-scored)

---

### Step 2.6: Add Country and Sector Dummies
**Location**: `factor_construction.py` → `_create_country_sector_dummies()`

**What happens**:
1. Maps countries to 7 continents using continent mapping
2. Creates 0/1 dummy variables for each continent (e.g., `CONTINENT_North America`)
3. Creates 0/1 dummy variables for each sector (e.g., `SECTOR_Technology`)
4. Drops reference categories to avoid multicollinearity:
   - First continent is dropped (reference)
   - First sector is dropped (reference)

**Output**: DataFrame with continent and sector dummy exposures (0/1) per stock-date

**Final Exposures Table Contains**:
- 6 style factor exposures (VALUE, PROFITABILITY, GROWTH, MOMENTUM, VOLATILITY, LIQUIDITY)
- Up to 6 continent dummy exposures (7 continents minus 1 reference)
- All sector dummy exposures (minus 1 reference)

---

## PART 3: FACTOR RETURN CALCULATION

### Step 3.1: Prepare Data for Regression
**Location**: `factors.py` → Step 3

**What happens**:
1. Renames returns DataFrame columns: `FSYM_ID → FACTSET_ID`, `P_DATE → DATE`
2. Merges factor exposures with stock returns
3. Join keys: `FACTSET_ID` and `DATE`
4. Creates aligned dataset with exposures and returns

**Output**: DataFrame with exposures and returns for each stock-date

---

### Step 3.2: Calculate Daily Factor Returns via OLS
**Location**: `model_builder.py` → `calculate_daily_factor_returns()`

**Method**: Cross-sectional OLS regression for each date

**What happens**:
For each trading date:
1. Collects all stock returns on that date
2. Collects all factor exposures on that date
3. Runs OLS regression: `r_i,t = Σ(k) β_i,k,t * f_k,t + ε_i,t`
4. Solves: `f_t = (B_t^T * B_t)^-1 * B_t^T * r_t`
   - `f_t` = vector of factor returns for date t
   - `B_t` = matrix of factor exposures for date t
   - `r_t` = vector of stock returns for date t

**Output**: DataFrame with factor returns for each date:
- `DATE`
- `FACTOR_NAME` (includes: VALUE, PROFITABILITY, GROWTH, MOMENTUM, VOLATILITY, LIQUIDITY, plus all continent and sector factors)
- `RETURN` (factor return for that date)

**Note**: All factors (style + country + sector) are included in the regression simultaneously

**Purpose**: Quantifies how much each factor contributed to returns on each day

---

### Step 3.3: Calculate Specific Returns
**Location**: `model_builder.py` → `calculate_specific_returns()`

**What happens**:
For each stock-date:
1. Calculates explained return: `Σ(k) β_i,k * f_k` (sum of factor contributions)
2. Calculates specific return: `Actual Return - Explained Return`
3. Specific return = residual (idiosyncratic return not explained by factors)

**Formula**: `ε̂_i,t = r_i,t - Σ(k=1 to K) β_i,k,t * f̂_k,t`

**Output**: DataFrame with specific returns per stock-date

**Purpose**: Measures stock-specific performance not captured by factors

---

## PART 4: ANALYSIS AND COMPARISON

### Step 4.1: Compare Factor Returns with S&P500
**Location**: `model_builder.py` → `compare_factor_returns_with_sp500()`

**What happens**:
For each factor:
1. Aligns factor return time series with S&P500 returns
2. Calculates correlation coefficient
3. Calculates R² (correlation squared)
4. Calculates Sharpe ratio: `mean(returns) / std(returns)`
5. Calculates cumulative return over the period

**Output**: Dictionary with metrics for each factor:
- Correlation with S&P500
- R²
- Sharpe Ratio
- Cumulative Return

**Purpose**: Identifies which factors best explain S&P500 performance

---

### Step 4.2: Calculate Factor Covariance Matrix
**Location**: `model_builder.py` → `calculate_factor_covariance()`

**What happens**:
1. Calculates rolling covariance matrix between factors
2. Uses 252-day (1 year) rolling window
3. For each date, computes covariance matrix of factor returns over past 252 days

**Output**: DataFrame with pairwise factor covariances over time

**Purpose**: Measures how factors move together (risk assessment)

---

## SUMMARY: COMPLETE WORKFLOW

```
1. DATA RETRIEVAL
   ├── Retrieve universe data (all stocks) → Get FACTSET_IDs
   ├── Retrieve fundamentals (NTM/LTM metrics)
   ├── Retrieve returns (daily returns)
   ├── Retrieve prices (adjusted prices and volumes)
   ├── Retrieve exchange rates (for currency conversion)
   ├── Retrieve market value (in local currency)
   ├── Retrieve enterprise value (already in USD)
   └── Retrieve S&P500 returns (optional, for future comparison)

2. FACTOR CONSTRUCTION
   ├── Merge prices and fundamentals
   ├── Calculate Value factors (yields)
   ├── Calculate Profitability factors (margins)
   ├── Calculate Growth factors (growth rates)
   ├── Calculate Momentum (12-1 month returns)
   ├── Calculate Volatility (60-day rolling std)
   ├── Calculate Liquidity (20-day avg dollar volume)
   ├── Combine all characteristics
   ├── Winsorize (clip outliers)
   ├── Standardize (z-scores)
   └── Neutralize (optional, by industry/country)

3. FACTOR RETURN CALCULATION
   ├── Merge exposures with returns
   ├── Run daily OLS regression → Factor returns
   └── Calculate specific returns (residuals)

4. ANALYSIS
   ├── Compare factor returns with S&P500
   └── Calculate factor covariance matrix
```

---

## KEY DATA TABLES AND THEIR PURPOSE

| Table | Purpose | Key Columns |
|-------|---------|-------------|
| `EDS_FACTORS_UNIVERSE` | Stock universe metadata | FACTSET_ID, BLOOMBERG_TICKER, COUNTRY, SECTOR |
| `EDS_FACTORS_FUNDAMENTALS_NTM_LTM` | Fundamental metrics | EPS, SALES, EBITDA, COGS (NTM and LTM) |
| `SPLIT_ADJUSTED_PRICES_EDS_FACTORS` | Price and volume data | ADJUSTED_PRICE, ADJUSTED_VOLUME, CURRENCY |
| Returns (calculated) | Daily returns from prices | ONE_DAY_PCT |
| Exchange Rates Table | Currency conversion | EXCHANGE_RATE_TO_USD |

---

## FACTORS CONSTRUCTED

| Factor Category | Characteristics | Final Exposure |
|----------------|----------------|----------------|
| Value | 5 characteristics (EY_NTM, SY_NTM, EBITDA_Y_NTM, EY_LTM, SY_LTM) | VALUE (composite) |
| Profitability | 2 characteristics (EBITDA_MARG, GROSS_MARG) | PROFITABILITY (composite) |
| Growth | 2 characteristics (EPS_GROWTH, SALES_GROWTH) | GROWTH (composite) |
| Momentum | 1 characteristic (12-1 Month Momentum) | MOMENTUM |
| Volatility | 1 characteristic (60-Day Rolling Volatility) | VOLATILITY |
| Liquidity | 1 characteristic (Log Dollar Volume) | LIQUIDITY |
| Country | 0/1 dummies for 7 continents | CONTINENT_* (up to 6, minus reference) |
| Sector | 0/1 dummies for all sectors | SECTOR_* (all sectors, minus reference) |
| **Total Style Factors** | | **6** |
| **Total Exposures** | | **6 + continent dummies + sector dummies** |

---

## CONSTRAINTS AND FILTERS

1. **Date Filter**: All data must have `DATE >= '2015-01-01'`
2. **Universe**: All stocks in the `EDS_FACTORS_UNIVERSE` table are used (no ticker filtering)
3. **Universe Columns**: Only `FACTSET_ID`, `COUNTRY`, `SECTOR` are kept from universe table

---

## OUTPUTS

The workflow produces:
- **Factor Exposures**: Normalized factor characteristics for each stock-date
- **Factor Returns**: Time series of daily returns for each factor
- **Specific Returns**: Idiosyncratic returns not explained by factors
- **Factor Comparisons**: Correlation and performance metrics vs S&P500
- **Factor Covariance**: Risk relationships between factors

