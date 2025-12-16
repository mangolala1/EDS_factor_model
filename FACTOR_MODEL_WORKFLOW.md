# EDS Factor Model - Complete Workflow Documentation

## Overview

This document describes the complete workflow for constructing the EDS factor model and generating the required output tables as specified in the data dictionary. The workflow transforms raw stock characteristics into standardized factor exposures, then uses these exposures to estimate factor returns and calculate risk metrics.

## Required Output Tables

Based on the specification, we need to generate the following tables:

1. **`<client>_<model>_EXPOSURE`** - Factor exposures for each stock on each date
2. **`<client>_<model>_FACTOR_RETURNS`** - Daily factor returns
3. **`<client>_<model>_SPECIFIC_RETURNS`** - Stock-specific (idiosyncratic) returns
4. **`<client>_<model>_FACTOR_MODEL_FACTOR_NAMES`** - Factor metadata (display names and groups)
5. **`<client>_<model>_COVARIANCE`** - Factor covariance matrix
6. **`<client>_<model>_VARIANCE`** - Specific risk/variance table (calculated as total variance - factor variance)

---

## Workflow Steps

### STEP 1: Construct All Characteristics

**Objective**: Calculate raw stock characteristics from fundamental and market data for each stock on each trading date.

#### 1.1 Value Characteristics
Calculate earnings and sales yields from fundamentals and prices:

```python
# For each stock i on date t:
EY_NTM[i,t] = EPS_NTM[i,t] / PRICE[i,t]
SY_NTM[i,t] = SALES_NTM[i,t] / PRICE[i,t]
EBITDA_Y_NTM[i,t] = EBITDA_NTM[i,t] / PRICE[i,t]
EY_LTM[i,t] = EPS_LTM[i,t] / PRICE[i,t]
SY_LTM[i,t] = SALES_LTM[i,t] / PRICE[i,t]

# Handle invalid values:
IF EPS_NTM[i,t] <= 0 THEN EY_NTM[i,t] = NaN
IF EBITDA_NTM[i,t] <= 0 THEN EBITDA_Y_NTM[i,t] = NaN
IF EPS_LTM[i,t] <= 0 THEN EY_LTM[i,t] = NaN
```

**Pseudocode**:
```
FOR each trading date t:
    FOR each stock i in universe:
        Load fundamentals[i,t]: EPS_NTM, SALES_NTM, EBITDA_NTM, EPS_LTM, SALES_LTM
        Load price[i,t]
        
        Calculate value characteristics:
            EY_NTM[i,t] = EPS_NTM[i,t] / price[i,t]
            SY_NTM[i,t] = SALES_NTM[i,t] / price[i,t]
            EBITDA_Y_NTM[i,t] = EBITDA_NTM[i,t] / price[i,t]
            EY_LTM[i,t] = EPS_LTM[i,t] / price[i,t]
            SY_LTM[i,t] = SALES_LTM[i,t] / price[i,t]
        
        Apply validity checks (set to NaN if denominator <= 0)
```

#### 1.2 Profitability Characteristics
Calculate margin metrics:

```python
EBITDA_MARGIN[i,t] = EBITDA_LTM[i,t] / SALES_LTM[i,t]
GROSS_MARGIN[i,t] = 1 - (COGS_LTM[i,t] / SALES_LTM[i,t])

# Handle invalid values:
IF SALES_LTM[i,t] <= 0 THEN EBITDA_MARGIN[i,t] = NaN, GROSS_MARGIN[i,t] = NaN
IF COGS_LTM[i,t] <= 0 THEN GROSS_MARGIN[i,t] = NaN
```

#### 1.3 Growth Characteristics
Calculate growth rates:

```python
EPS_GROWTH[i,t] = (EPS_NTM[i,t] / EPS_LTM[i,t]) - 1
SALES_GROWTH[i,t] = (SALES_NTM[i,t] / SALES_LTM[i,t]) - 1

# Handle invalid values:
IF EPS_LTM[i,t] <= 0 THEN EPS_GROWTH[i,t] = NaN
IF SALES_LTM[i,t] <= 0 THEN SALES_GROWTH[i,t] = NaN
```

#### 1.4 Market-Based Characteristics
Calculate momentum, volatility, and liquidity using rolling windows:

```python
# Momentum: 12-month return excluding last month
# Lookback: 252 trading days, exclude last 21 days
MOMENTUM[i,t] = (PRICE[i,t-21] / PRICE[i,t-252]) - 1

# Volatility: 60-day rolling standard deviation of returns
VOLATILITY[i,t] = STDDEV(RETURNS[i,t-59:t])

# Liquidity: 20-day average dollar volume
LIQUIDITY[i,t] = MEAN(PRICE[i,t-19:t] * VOLUME[i,t-19:t])
```

**Pseudocode**:
```
FOR each trading date t:
    FOR each stock i:
        # Update rolling window buffers
        Add return[i,t] and dollar_volume[i,t] to stock i's ring buffer
        
        # Calculate momentum (252-day lookback, exclude last 21 days)
        IF buffer has >= 252 days:
            price_t_minus_21 = buffer.price[t-21]
            price_t_minus_252 = buffer.price[t-252]
            MOMENTUM[i,t] = (price_t_minus_21 / price_t_minus_252) - 1
        
        # Calculate volatility (60-day rolling std dev)
        IF buffer has >= 60 days:
            returns_window = buffer.returns[t-59:t]
            VOLATILITY[i,t] = std(returns_window)
        
        # Calculate liquidity (20-day average dollar volume)
        IF buffer has >= 20 days:
            dollar_volumes = buffer.dollar_volume[t-19:t]
            LIQUIDITY[i,t] = mean(dollar_volumes)
```

**Current Implementation**: Uses `StockCalculatorManager` class with ring buffers to efficiently maintain rolling windows for each stock.

---

### STEP 2: Combine Characteristics into Style Factors

**Objective**: Aggregate related characteristics into composite style factors using equal-weighted averaging.

#### 2.1 Factor Combination Logic

```python
# Value factor: average of all value characteristics
VALUE[i,t] = MEAN(EY_NTM[i,t], SY_NTM[i,t], EBITDA_Y_NTM[i,t], EY_LTM[i,t], SY_LTM[i,t])

# Profitability factor: average of profitability characteristics
PROFITABILITY[i,t] = MEAN(EBITDA_MARGIN[i,t], GROSS_MARGIN[i,t])

# Growth factor: average of growth characteristics
GROWTH[i,t] = MEAN(EPS_GROWTH[i,t], SALES_GROWTH[i,t])

# Momentum, Volatility, Liquidity are already single characteristics
# They become factors directly: MOMENTUM, VOLATILITY, LIQUIDITY
```

**Pseudocode**:
```
FOR each trading date t:
    FOR each stock i:
        # Value factor
        value_chars = [EY_NTM[i,t], SY_NTM[i,t], EBITDA_Y_NTM[i,t], EY_LTM[i,t], SY_LTM[i,t]]
        VALUE[i,t] = mean(value_chars, skipna=True)  # Skip NaN values
        
        # Profitability factor
        profitability_chars = [EBITDA_MARGIN[i,t], GROSS_MARGIN[i,t]]
        PROFITABILITY[i,t] = mean(profitability_chars, skipna=True)
        
        # Growth factor
        growth_chars = [EPS_GROWTH[i,t], SALES_GROWTH[i,t]]
        GROWTH[i,t] = mean(growth_chars, skipna=True)
        
        # Individual factors (no combination needed)
        MOMENTUM[i,t] = MOMENTUM[i,t]  # Already calculated
        VOLATILITY[i,t] = VOLATILITY[i,t]  # Already calculated
        LIQUIDITY[i,t] = LIQUIDITY[i,t]  # Already calculated
```

**Note**: The combination logic uses equal-weighted averaging. This can be modified in the future to use other weighting schemes (e.g., factor loadings, economic significance).

---

### STEP 3: Calculate Z-Scores (Cross-Sectional Standardization)

**Objective**: Standardize all characteristics and factors to z-scores on each date across all stocks. This creates the exposure table.

#### 3.1 Winsorization (Outlier Treatment)

Before standardization, winsorize extreme values to prevent outliers from dominating:

```python
# For each characteristic/factor on each date t:
FOR each char in [all characteristics + style factors]:
    values = [char[i,t] for all stocks i on date t]
    lower_bound = quantile(values, 0.01)  # 1st percentile
    upper_bound = quantile(values, 0.99)  # 99th percentile
    
    FOR each stock i:
        char[i,t] = clip(char[i,t], lower_bound, upper_bound)
```

**Pseudocode**:
```
FOR each trading date t:
    FOR each characteristic char in [EY_NTM, SY_NTM, ..., VALUE, PROFITABILITY, GROWTH, MOMENTUM, VOLATILITY, LIQUIDITY]:
        # Get all values for this date
        values = [char[i,t] for all stocks i where char[i,t] is not NaN]
        
        IF len(values) > 0:
            # Calculate winsorization bounds
            lower_bound = quantile(values, 0.01)  # 1% lower tail
            upper_bound = quantile(values, 0.99)  # 99% upper tail
            
            # Clip extreme values
            FOR each stock i:
                IF char[i,t] < lower_bound:
                    char[i,t] = lower_bound
                ELSE IF char[i,t] > upper_bound:
                    char[i,t] = upper_bound
```

#### 3.2 Cross-Sectional Z-Score Standardization

Standardize each characteristic/factor to have mean=0 and std=1 across all stocks on each date:

```python
# For each characteristic/factor on each date t:
FOR each char in [all characteristics + style factors]:
    values = [char[i,t] for all stocks i on date t]
    mean_t = mean(values)
    std_t = std(values)
    
    IF std_t > 0:
        FOR each stock i:
            char[i,t] = (char[i,t] - mean_t) / std_t
    ELSE:
        # All values are the same, set to 0
        FOR each stock i:
            char[i,t] = 0.0
```

**Pseudocode**:
```
FOR each trading date t:
    FOR each characteristic char in [all characteristics + style factors]:
        # Get all non-NaN values for this date
        values = [char[i,t] for all stocks i where char[i,t] is not NaN]
        
        IF len(values) > 1:
            mean_t = mean(values)
            std_t = std(values)
            
            IF std_t > 0:
                # Standardize to z-scores
                FOR each stock i:
                    IF char[i,t] is not NaN:
                        char[i,t] = (char[i,t] - mean_t) / std_t
            ELSE:
                # All values identical, set to 0
                FOR each stock i:
                    IF char[i,t] is not NaN:
                        char[i,t] = 0.0
        ELSE:
            # Not enough data, leave as NaN
            pass
```

**Note**: After this step, we have z-scores for all style factors (VALUE, PROFITABILITY, GROWTH, MOMENTUM, VOLATILITY, LIQUIDITY). These z-scores represent the **exposures** to each style factor.

---

### STEP 4: Create Dummy Variables for Sector and Continent

**Objective**: Create binary dummy variables for sector and continent classifications, then apply sum-to-zero constraints to avoid multicollinearity.

#### 4.1 Sector Dummy Variables

```python
# Create one-hot encoded dummy variables for each sector
sector_dummies = pd.get_dummies(df['SECTOR'], prefix='SECTOR')
# Example: SECTOR_Technology, SECTOR_Financials, SECTOR_Healthcare, etc.
```

#### 4.2 Continent Dummy Variables

```python
# Map country to continent (using continent_mapping.py)
FOR each stock i:
    continent[i] = get_continent(country[i])

# Create one-hot encoded dummy variables for each continent
continent_dummies = pd.get_dummies(df['CONTINENT'], prefix='CONTINENT')
# Example: CONTINENT_North_America, CONTINENT_Europe, CONTINENT_Asia, etc.
```

**TODO**: The user requested **continent + developed/developing country mapping**. Currently, we only map to continents. We need to add:

1. **Developed/Developing Classification**: Create a mapping of countries to "Developed" or "Developing" status (e.g., using MSCI classification)
2. **Combined Dummy Variables**: Create dummies like `CONTINENT_North_America_Developed`, `CONTINENT_Asia_Developing`, etc.

**Pseudocode for Enhanced Mapping**:
```
FOR each stock i:
    country = stock.country
    continent = get_continent(country)
    dev_status = get_developed_status(country)  # "Developed" or "Developing"
    
    # Create combined dummy variable name
    dummy_name = f"CONTINENT_{continent}_{dev_status}"
    # Example: CONTINENT_Asia_Developed, CONTINENT_Asia_Developing
```

#### 4.3 Apply Sum-to-Zero Constraint

To avoid multicollinearity (perfect linear dependence), we apply a sum-to-zero constraint by subtracting the mean from each dummy variable:

```python
# For sector dummies:
FOR each sector dummy column col:
    col_mean = mean(col)  # Mean across all stocks on date t
    col = col - col_mean  # Center around zero

# For continent dummies:
FOR each continent dummy column col:
    col_mean = mean(col)
    col = col - col_mean
```

**Why Sum-to-Zero?**
- Without constraint: If we have N sectors, we only need N-1 dummies (one is redundant)
- With sum-to-zero: All N dummies sum to zero, so we can include all of them without perfect multicollinearity
- The intercept in regression will represent the average return across all sectors/continents

**Pseudocode**:
```
FOR each trading date t:
    # Sector dummies
    sector_dummies = get_dummies(sectors[t], prefix='SECTOR')
    
    FOR each sector column col in sector_dummies:
        col_mean = mean(col)  # Mean across all stocks on date t
        col = col - col_mean  # Subtract mean (sum-to-zero constraint)
    
    # Continent dummies (with developed/developing)
    FOR each stock i:
        continent_dev = f"{continent[i]}_{developed_status[i]}"
    
    continent_dummies = get_dummies(continent_dev_values, prefix='CONTINENT')
    
    FOR each continent column col in continent_dummies:
        col_mean = mean(col)
        col = col - col_mean  # Sum-to-zero constraint
```

---

### STEP 5: Construct Exposure Table

**Objective**: Combine all style factors and dummy variables into the final exposure table.

#### 5.1 Combine All Exposures

```python
# Style factors (already z-scores)
style_factors = ['VALUE', 'PROFITABILITY', 'GROWTH', 'MOMENTUM', 'VOLATILITY', 'LIQUIDITY']

# Dummy variables (sum-to-zero constrained)
sector_dummies = [all SECTOR_* columns]
continent_dummies = [all CONTINENT_* columns]

# Combine into exposure table
exposure_table = [
    MODEL,           # Model identifier (e.g., 'EDS_MODEL')
    DATE,            # Trading date
    SECURITY_ID,     # FACTSET_ID
    FACTOR_NAME,     # Factor name (VALUE, PROFITABILITY, ..., SECTOR_Technology, CONTINENT_North_America_Developed, etc.)
    EXPOSURE         # Exposure value (z-score for style factors, centered dummy value for dummies)
]
```

**Table Structure** (Long Format):
```
MODEL | DATE       | SECURITY_ID | FACTOR_NAME                    | EXPOSURE
------|------------|-------------|--------------------------------|----------
EDS   | 2020-01-02 | ABC123      | VALUE                          | 0.85
EDS   | 2020-01-02 | ABC123      | PROFITABILITY                  | -0.42
EDS   | 2020-01-02 | ABC123      | GROWTH                         | 1.23
EDS   | 2020-01-02 | ABC123      | MOMENTUM                       | 0.15
EDS   | 2020-01-02 | ABC123      | VOLATILITY                     | -0.67
EDS   | 2020-01-02 | ABC123      | LIQUIDITY                      | 0.33
EDS   | 2020-01-02 | ABC123      | SECTOR_Technology              | 0.12
EDS   | 2020-01-02 | ABC123      | CONTINENT_North_America_Developed | 0.08
...
```

**Pseudocode**:
```
FOR each trading date t:
    FOR each stock i:
        # Style factor exposures (z-scores)
        FOR each style_factor in [VALUE, PROFITABILITY, GROWTH, MOMENTUM, VOLATILITY, LIQUIDITY]:
            exposure_table.append({
                'MODEL': 'EDS_MODEL',
                'DATE': t,
                'SECURITY_ID': stock_id[i],
                'FACTOR_NAME': style_factor,
                'EXPOSURE': style_factor[i,t]
            })
        
        # Sector dummy exposures (sum-to-zero)
        FOR each sector_dummy in sector_dummies:
            IF sector_dummy[i,t] != 0:  # Only store non-zero exposures
                exposure_table.append({
                    'MODEL': 'EDS_MODEL',
                    'DATE': t,
                    'SECURITY_ID': stock_id[i],
                    'FACTOR_NAME': sector_dummy.name,
                    'EXPOSURE': sector_dummy[i,t]
                })
        
        # Continent dummy exposures (sum-to-zero)
        FOR each continent_dummy in continent_dummies:
            IF continent_dummy[i,t] != 0:
                exposure_table.append({
                    'MODEL': 'EDS_MODEL',
                    'DATE': t,
                    'SECURITY_ID': stock_id[i],
                    'FACTOR_NAME': continent_dummy.name,
                    'EXPOSURE': continent_dummy[i,t]
                })
```

**Note**: The exposure table can be stored in wide format (one row per stock-date, one column per factor) or long format (one row per stock-date-factor combination). The specification shows long format.

---

### STEP 6: Fit Regressions to Obtain Factor Returns

**Objective**: For each trading date, run a cross-sectional regression of stock returns on factor exposures to estimate factor returns.

#### 6.1 Regression Model

```python
# For each date t:
# r_i,t = α_t + Σ(k=1 to K) β_i,k,t * f_k,t + ε_i,t

# Where:
# - r_i,t = return of stock i on date t
# - α_t = intercept (average return on date t)
# - β_i,k,t = exposure of stock i to factor k on date t (from exposure table)
# - f_k,t = factor return for factor k on date t (what we're estimating)
# - ε_i,t = specific return (residual)
```

#### 6.2 OLS Regression (Ordinary Least Squares)

For each date t, we solve:

```python
# Prepare data for date t
X_t = [exposures for all stocks on date t]  # N stocks × K factors matrix
y_t = [returns for all stocks on date t]    # N stocks × 1 vector

# Add intercept column (if configured)
IF include_intercept:
    X_t = [ones_column, X_t]  # Add column of ones for intercept
    factor_names = ['INTERCEPT'] + factor_names

# OLS solution: f_t = (X_t^T * X_t)^(-1) * X_t^T * y_t
XtX = X_t.T @ X_t
XtX_inv = pinv(XtX)  # Use pseudo-inverse for numerical stability
factor_returns_t = XtX_inv @ X_t.T @ y_t
```

**Pseudocode**:
```
FOR each trading date t:
    # Get all stocks with both exposures and returns on date t
    stocks_t = stocks where (exposure exists AND return exists on date t)
    
    IF len(stocks_t) < (num_factors + 1):
        SKIP date t  # Need more observations than factors
    
    # Prepare regression matrices
    X_t = []  # Exposure matrix (N stocks × K factors)
    y_t = []  # Return vector (N stocks × 1)
    
    FOR each stock i in stocks_t:
        exposures_i = [exposure[i,t,k] for each factor k]
        return_i = return[i,t]
        
        X_t.append(exposures_i)
        y_t.append(return_i)
    
    # Convert to numpy arrays
    X_t = np.array(X_t)  # Shape: (N, K)
    y_t = np.array(y_t)  # Shape: (N,)
    
    # Add intercept if configured
    IF include_intercept:
        intercept_col = np.ones((len(X_t), 1))
        X_t = np.column_stack([intercept_col, X_t])
        factor_names_t = ['INTERCEPT'] + factor_names
    
    # Remove rows with NaN values
    valid_mask = ~(np.isnan(X_t).any(axis=1) | np.isnan(y_t))
    X_t = X_t[valid_mask]
    y_t = y_t[valid_mask]
    
    IF len(X_t) < len(factor_names_t):
        SKIP date t
    
    # Solve OLS: f_t = (X^T * X)^(-1) * X^T * y
    TRY:
        XtX = X_t.T @ X_t
        XtX_inv = np.linalg.pinv(XtX)  # Pseudo-inverse for stability
        factor_returns_t = XtX_inv @ X_t.T @ y_t
        
        # Store factor returns
        FOR each factor k in factor_names_t:
            factor_returns_table.append({
                'MODEL': 'EDS_MODEL',
                'DATE': t,
                'FACTOR_NAME': factor_names_t[k],
                'RETURN': factor_returns_t[k]
            })
    EXCEPT LinAlgError:
        SKIP date t  # Matrix is singular (multicollinearity)
```

**Output Table**: `FACTOR_RETURNS`
```
MODEL | DATE       | FACTOR_NAME                    | RETURN
------|------------|--------------------------------|--------
EDS   | 2020-01-02 | INTERCEPT                      | 0.0012
EDS   | 2020-01-02 | VALUE                          | 0.0005
EDS   | 2020-01-02 | PROFITABILITY                  | -0.0003
EDS   | 2020-01-02 | GROWTH                         | 0.0008
EDS   | 2020-01-02 | MOMENTUM                       | 0.0011
EDS   | 2020-01-02 | VOLATILITY                     | -0.0002
EDS   | 2020-01-02 | LIQUIDITY                      | 0.0001
EDS   | 2020-01-02 | SECTOR_Technology              | 0.0004
...
```

---

### STEP 7: Calculate Specific Returns

**Objective**: Calculate the residual (idiosyncratic) return for each stock, which is the portion of return not explained by factors.

#### 7.1 Specific Return Formula

```python
# For each stock i on date t:
# ε_i,t = r_i,t - [α_t + Σ(k) β_i,k,t * f_k,t]

# Where:
# - r_i,t = actual return of stock i on date t
# - α_t = intercept (if included)
# - β_i,k,t = exposure of stock i to factor k on date t
# - f_k,t = factor return for factor k on date t
# - ε_i,t = specific return (residual)
```

**Pseudocode**:
```
FOR each trading date t:
    # Get factor returns for date t
    factor_returns_t = {factor_name: return_value for each factor on date t}
    
    # Get all stocks with exposures and returns on date t
    stocks_t = stocks where (exposure exists AND return exists on date t)
    
    FOR each stock i in stocks_t:
        actual_return = return[i,t]
        
        # Calculate explained return: α_t + Σ(k) β_i,k * f_k
        explained_return = 0.0
        
        IF include_intercept:
            explained_return += factor_returns_t['INTERCEPT']
        
        FOR each factor k:
            exposure_i_k = exposure[i,t,k]
            factor_return_k = factor_returns_t[k]
            
            IF exposure_i_k is not NaN:
                explained_return += exposure_i_k * factor_return_k
        
        # Specific return = actual - explained
        specific_return = actual_return - explained_return
        
        specific_returns_table.append({
            'MODEL': 'EDS_MODEL',
            'DATE': t,
            'SECURITY_ID': stock_id[i],
            'SPECIFIC_RETURN': specific_return
        })
```

**Output Table**: `SPECIFIC_RETURNS`
```
MODEL | DATE       | SECURITY_ID | SPECIFIC_RETURN
------|------------|-------------|----------------
EDS   | 2020-01-02 | ABC123      | 0.0023
EDS   | 2020-01-02 | DEF456      | -0.0015
EDS   | 2020-01-02 | GHI789      | 0.0008
...
```

---

### STEP 8: Calculate Factor Covariance Matrix

**Objective**: Calculate the covariance matrix of factor returns using a rolling window.

#### 8.1 Rolling Window Covariance

```python
# For each as-of date T:
# Use rolling window of factor returns: [T - window + 1, T]
# window = 60 trading days (default)

# Calculate covariance matrix:
Σ_f(T) = Cov(factor_returns[T-window+1:T])

# Where Σ_f(T) is a K×K matrix (K = number of factors)
```

**Pseudocode**:
```
# Pivot factor returns to wide format (one column per factor)
factor_returns_wide = pivot_table(
    factor_returns_df,
    index='DATE',
    columns='FACTOR_NAME',
    values='RETURN'
)

# Sort by date
factor_returns_wide = factor_returns_wide.sort_index()

FOR each as_of_date T in sorted_dates:
    # OPTIMIZATION: Use efficient window indexing instead of filtering all dates
    window_start_idx = max(0, date_idx - window + 1)
    window_data = factor_returns_wide.iloc[window_start_idx:date_idx + 1]
    
    IF len(window_data) < (window / 2):
        SKIP date T  # Not enough history
    
    # Calculate covariance matrix
    cov_matrix = window_data.cov()  # K×K matrix
    
    # Store pairwise covariances
    FOR each factor pair (k, l) where k <= l:
        covariance_table.append({
            'MODEL': 'EDS_MODEL',
            'DATE': T,
            'FACTOR_NAME_1': factor_names[k],
            'FACTOR_NAME_2': factor_names[l],
            'COVARIANCE': cov_matrix[k, l]
        })
```

**Output Table**: `COVARIANCE`
```
MODEL | DATE       | FACTOR_NAME_1 | FACTOR_NAME_2        | COVARIANCE
------|------------|---------------|---------------------|-----------
EDS   | 2020-03-31 | VALUE         | VALUE               | 0.00015
EDS   | 2020-03-31 | VALUE         | PROFITABILITY       | 0.00008
EDS   | 2020-03-31 | VALUE         | GROWTH              | -0.00005
EDS   | 2020-03-31 | PROFITABILITY | PROFITABILITY       | 0.00012
...
```

**Note**: 
- The covariance matrix is calculated and saved as an output table
- It is also used internally in Step 9 for specific risk calculation
- The calculation uses optimized window indexing for better performance

---

### STEP 9: Calculate Specific Risk

**Objective**: Calculate specific risk for each stock, which is the portion of total variance not explained by factors.

#### 9.1 Specific Risk Calculation

For each stock i and as-of date T:

```python
# Step 1: Calculate total variance of stock returns (rolling window)
total_var[i,T] = Var(returns[i, T-window+1:T])
# window = 60 trading days (default)

# Step 2: Calculate factor variance
# Factor variance = β_i,T^T * Σ_f(T) * β_i,T
# Where:
# - β_i,T = exposure vector for stock i on date T
# - Σ_f(T) = factor covariance matrix on date T (calculated internally in Step 8)

factor_var[i,T] = β_i,T^T @ Σ_f(T) @ β_i,T

# Step 3: Calculate specific variance
# Specific variance = Total variance - Factor variance
specific_var[i,T] = max(total_var[i,T] - factor_var[i,T], epsilon)
# epsilon = 1e-8 (small positive value to avoid negative variance)
```

#### 9.2 Vectorized Implementation (Optimized)

The specific risk calculation uses **fully vectorized operations** to process all stocks simultaneously, providing massive performance improvements.

**Key Optimizations:**

1. **Vectorized Total Variance Calculation**:
   - Pivot returns to wide format (stocks × dates) once per date
   - Calculate variance for all stocks simultaneously using `var(axis=1)`
   - Replaces N individual variance calculations with one vectorized operation

2. **Vectorized Factor Variance Calculation**:
   - Build exposure matrix B (N stocks × K factors) for all stocks on date T
   - Fill missing exposures as 0 (as per standard practice)
   - Compute all factor variances at once using:
     ```python
     tmp = B @ Σ_f  # (N, K) matrix multiplication
     factor_vars = np.einsum('nk,nk->n', tmp, B)  # All N factor variances
     ```
   - This replaces N individual quadratic forms (`β^T * Σ * β`) with a single BLAS-backed matrix operation

3. **Cached Covariance Matrix**:
   - Build `Σ_f(T)` once per date and reuse for all stocks
   - Use efficient window indexing instead of filtering all dates

4. **Pre-filtering and Grouping**:
   - Pre-filter returns by date range once (not per stock)
   - Group exposures by date for O(1) lookups

**Performance Improvement:**
- **Before**: O(N) Python loops with per-stock matrix operations
  - For each stock: extract submatrix, compute `β^T * Σ * β`
  - For each stock: calculate individual variance
- **After**: O(1) vectorized operations for all N stocks simultaneously
  - Single matrix multiplication for all factor variances
  - Single vectorized variance calculation for all stocks
- **Speedup**: **10-100x faster** for large datasets (depends on N and K)

**Implementation Details:**
```python
# For each date T:
# 1. Calculate factor covariance matrix once (cached)
factor_cov_matrix = factor_returns_window[factor_names_for_cov].cov().values

# 2. Vectorize total variance for all stocks
returns_pivot = returns_window.pivot_table(
    index='SECURITY_ID',
    columns='DATE',
    values='ONE_DAY_PCT'
)
total_vars = returns_pivot.var(axis=1, ddof=0)  # All stocks at once

# 3. Build exposure matrix B (N stocks × K factors)
B = np.zeros((N, K))
# Fill B with exposures (missing = 0)

# 4. Vectorize factor variance calculation
tmp = B @ factor_cov_matrix  # (N, K)
factor_vars = np.einsum('nk,nk->n', tmp, B)  # All N factor variances

# 5. Calculate specific variance for all stocks (vectorized)
specific_vars = np.maximum(
    total_vars - factor_vars_capped,
    min_specific_vars
)
```

**Output Table**: `SPECIFIC_RISK`
```
MODEL | DATE       | SECURITY_ID | SPECIFIC_VAR
------|------------|-------------|--------------
EDS   | 2020-03-31 | ABC123      | 0.00013
EDS   | 2020-03-31 | DEF456      | 0.00010
...
```

**Note**: 
- The specific risk table contains `SPECIFIC_VAR`: Specific variance = TOTAL_VAR - FACTOR_VAR
- TOTAL_VAR: Total variance of stock returns (60-day rolling)
- FACTOR_VAR: Variance explained by factors (β^T * Σ_f * β)
- Factor covariance is calculated internally but not saved as a separate output table

---

### STEP 10: Create Factor Names Metadata Table

**Objective**: Create a reference table mapping factor names to display names and factor groups.

#### 10.1 Factor Grouping

```python
factor_groups = {
    'Style Factors': ['VALUE', 'PROFITABILITY', 'GROWTH', 'MOMENTUM', 'VOLATILITY', 'LIQUIDITY'],
    'Sector Factors': [all SECTOR_* factors],
    'Geographic Factors': [all CONTINENT_* factors],
    'Market Factor': ['INTERCEPT']  # If intercept is included
}
```

**Pseudocode**:
```
factor_names_table = []

# Style factors
FOR each factor in [VALUE, PROFITABILITY, GROWTH, MOMENTUM, VOLATILITY, LIQUIDITY]:
    factor_names_table.append({
        'MODEL': 'EDS_MODEL',
        'FACTOR_DISPLAY_NAME': factor,  # Or a more descriptive name
        'FACTOR_GROUP': 'Style Factors'
    })

# Sector factors
FOR each sector in unique_sectors:
    factor_name = f'SECTOR_{sector}'
    factor_names_table.append({
        'MODEL': 'EDS_MODEL',
        'FACTOR_DISPLAY_NAME': f'{sector} Sector',
        'FACTOR_GROUP': 'Sector Factors'
    })

# Geographic factors
FOR each continent_dev in unique_continent_dev_combinations:
    factor_name = f'CONTINENT_{continent_dev}'
    factor_names_table.append({
        'MODEL': 'EDS_MODEL',
        'FACTOR_DISPLAY_NAME': f'{continent_dev}',
        'FACTOR_GROUP': 'Geographic Factors'
    })

# Intercept (if included)
IF include_intercept:
    factor_names_table.append({
        'MODEL': 'EDS_MODEL',
        'FACTOR_DISPLAY_NAME': 'Intercept',
        'FACTOR_GROUP': 'Market Factor'
    })
```

**Output Table**: `FACTOR_MODEL_FACTOR_NAMES`
```
MODEL | FACTOR_DISPLAY_NAME              | FACTOR_GROUP
------|----------------------------------|------------------
EDS   | VALUE                            | Style Factors
EDS   | PROFITABILITY                     | Style Factors
EDS   | GROWTH                           | Style Factors
EDS   | MOMENTUM                         | Style Factors
EDS   | VOLATILITY                       | Style Factors
EDS   | LIQUIDITY                        | Style Factors
EDS   | Technology Sector                | Sector Factors
EDS   | Financials Sector                | Sector Factors
EDS   | North America Developed          | Geographic Factors
EDS   | Asia Developing                  | Geographic Factors
EDS   | Intercept                        | Market Factor
```

---

## Summary of Output Tables

### 1. EXPOSURE Table
- **Columns**: MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE
- **Content**: Factor exposures (z-scores for style factors, centered dummies for sector/continent)
- **Format**: Long format (one row per stock-date-factor combination)
- **Output File**: `exposures.csv`

### 2. FACTOR_RETURNS Table
- **Columns**: MODEL, DATE, FACTOR_NAME, RETURN
- **Content**: Daily factor returns estimated via cross-sectional OLS regression
- **Note**: Includes INTERCEPT if configured
- **Output File**: `factor_returns.csv`

### 3. SPECIFIC_RETURNS Table
- **Columns**: MODEL, DATE, SECURITY_ID, SPECIFIC_RETURN
- **Content**: Idiosyncratic returns (residuals from factor model)
- **Output File**: `specific_returns.csv`

### 4. SPECIFIC_RISK Table
- **Columns**: MODEL, DATE, SECURITY_ID, SPECIFIC_VAR
- **Content**: Specific variance for each stock (calculated as total variance minus factor variance)
- **Output File**: `specific_risk.csv`

### 5. COVARIANCE Table
- **Columns**: MODEL, DATE, FACTOR_NAME_1, FACTOR_NAME_2, COVARIANCE
- **Content**: Factor covariance matrix (60-day rolling window)
- **Output File**: `factor_covariance.csv`

### 6. FACTOR_MODEL_FACTOR_NAMES Table
- **Columns**: MODEL, FACTOR_DISPLAY_NAME, FACTOR_GROUP
- **Content**: Metadata mapping factor names to display names and groups
- **Output File**: `factor_model_factor_names.csv`

**Note**: Factor covariance matrix (Step 8) is saved as an output table and is also used internally in Step 9 for specific risk calculation.

---

## Implementation Notes

### Current Status
- ✅ Steps 1-3: Characteristics construction and z-score standardization (implemented)
- ✅ Step 4: Sector and continent dummies with sum-to-zero (implemented, but needs developed/developing enhancement)
- ✅ Step 5: Exposure table construction (implemented)
- ✅ Step 6: Factor returns via OLS regression (implemented)
- ✅ Step 7: Specific returns calculation (implemented)
- ✅ Step 8: Factor covariance matrix (implemented and saved as output)
  - **Optimization**: Uses efficient window indexing instead of filtering all dates
  - **Output**: `factor_covariance.csv`
- ✅ Step 9: Specific risk calculation (implemented with major vectorization optimizations)
  - **Performance**: Fully vectorized operations for 10-100x speedup
  - **Key optimizations**: 
    - Vectorized total variance calculation (all stocks at once)
    - Vectorized factor variance calculation using matrix operations (`B @ Σ_f @ B^T` diagonal)
    - Cached covariance matrix (calculated once per date)
    - Pre-filtering and grouping for O(1) lookups
- ⚠️ Step 4 Enhancement: Need to add developed/developing country classification

### TODO Items
1. **Developed/Developing Classification**: Create mapping of countries to "Developed" or "Developing" status
2. **Enhanced Continent Dummies**: Combine continent and developed/developing status into single dummy variables
3. **Table Output Format**: Ensure all tables match the exact column names and data types from the specification
4. **Total Risk Factor**: Add "Total Risk" as a FACTOR_NAME in exposure tables (as per specification comment)

---

## Mathematical Formulation

### Factor Model Equation
```
r_i,t = α_t + Σ(k=1 to K) β_i,k,t * f_k,t + ε_i,t

Where:
- r_i,t = return of stock i on date t
- α_t = intercept (average return on date t)
- β_i,k,t = exposure of stock i to factor k on date t
- f_k,t = factor return for factor k on date t
- ε_i,t = specific return (idiosyncratic, uncorrelated with factors)
```

### Variance Decomposition
```
Var(r_i) = Var(Σ(k) β_i,k * f_k) + Var(ε_i)
         = β_i^T * Σ_f * β_i + σ²_ε,i

Where:
- Var(r_i) = total variance of stock i returns
- β_i^T * Σ_f * β_i = factor variance (explained by factors)
- σ²_ε,i = specific variance (idiosyncratic risk)
```

### Sum-to-Zero Constraint
For dummy variables with N categories:
```
Σ(n=1 to N) dummy_n = 0

This ensures:
- No perfect multicollinearity
- Intercept represents average return across all categories
- All dummy coefficients are relative to the average
```

---

## References

- MSCI Factor Model Methodology
- Barra Risk Model Handbook
- Fama-French Factor Models

