# Optimization Opportunities Analysis

This document identifies additional optimization opportunities in the EDS Factor Model codebase, beyond the vectorized optimizations already implemented in stages 8 and 9.

## Summary of Current Optimizations

✅ **Already Optimized:**
- Stage 8 (Factor Covariance): Efficient window indexing
- Stage 9 (Specific Risk): Fully vectorized factor variance and total variance calculations (10-100x speedup)
- Quarter Processing: Parallel processing, indexed lookups, batch writes

---

## High-Impact Optimization Opportunities

### 1. **Specific Returns Calculation (Stage 4)** - **HIGHEST PRIORITY**

**Current Implementation:**
- Nested loops: `for date in dates` → `for row in date_data.iterrows()`
- Per-stock calculation: `explained_return = sum(row[factor] * factor_return for factor in factors)`
- Complexity: O(dates × stocks × factors)

**Optimization Strategy (BEST APPROACH):**
**Vectorize per date-block** (not all dates at once - memory constraint):

```python
# Process dates in blocks (e.g., 10 days at a time)
# This avoids loading 18.9M rows (75k × 252 days) = ~9 GB

factor_returns_wide = factor_returns_df.pivot_table(
    index='DATE', columns='FACTOR_NAME', values='RETURN'
)

# Process in date blocks
block_size = 10  # days per block
all_dates = sorted(merged['DATE'].unique())
specific_returns_list = []

for block_start in range(0, len(all_dates), block_size):
    date_block = all_dates[block_start:block_start + block_size]
    block_data = merged[merged['DATE'].isin(date_block)].copy()
    
    # Get exposures block (B)
    factor_names_ordered = [f for f in factor_returns_wide.columns 
                           if f in exposures_reset.columns]
    B_block = block_data[factor_names_ordered].fillna(0).values  # (N_block, K)
    
    # Get factor returns per date - avoid materializing full F matrix
    # Instead, use indexing to map each row's date to factor returns
    date_to_factors = {date: factor_returns_wide.loc[date, factor_names_ordered].values
                      for date in date_block}
    
    # Create f_by_row using date mapping (more memory efficient)
    f_by_row = np.array([date_to_factors[date] for date in block_data['DATE']])
    
    # Vectorized calculation per block
    explained_returns = np.einsum("nk,nk->n", B_block, f_by_row)
    # Or equivalently: explained_returns = (B_block * f_by_row).sum(axis=1)
    
    specific_returns = block_data['ONE_DAY_PCT'].values - explained_returns
    
    # Store results
    specific_returns_list.append(pd.DataFrame({
        'MODEL': config.MODEL_NAME,
        'DATE': block_data['DATE'].values,
        'SECURITY_ID': block_data['SECURITY_ID'].values,
        'SPECIFIC_RETURN': specific_returns
    }))

# Concatenate all blocks
specific_returns_df = pd.concat(specific_returns_list, ignore_index=True)
```

**Key Correctness Check:**
⚠️ **CRITICAL**: Ensure factor column ordering matches exactly between B and F (including intercept handling, dropped dummies, sum-to-zero constraints). Most vectorization bugs are column-order mismatches.

**Memory Efficiency:**
- Per block (10 days): ~340 MB (manageable)
- Avoids loading all dates at once (~9 GB)

**Expected Speedup:** 20-100x (eliminates iterrows overhead, processes in blocks)

**Implementation Difficulty:** Medium (requires careful column alignment)

---

### 2. **Factor Returns Calculation (Stage 3)** - **MEDIUM PRIORITY**

**Current Implementation:**
- Already uses groupby, but could optimize further
- Filters merged data for each date: `date_data = merged[merged['DATE'] == date]`
- Multiple conversions: DataFrame → numpy → calculations

**Optimization Strategy:**
1. **Pre-group merged data by date** (avoid repeated filtering):
   ```python
   merged_by_date = merged.groupby('DATE')
   for date, date_data in merged_by_date:
       # Process date_data directly
   ```

2. **Batch process dates** where possible (if dates have similar structure)

3. **Use `np.linalg.lstsq` or `scipy.linalg.lstsq`** instead of manual `pinv` for better performance:
   ```python
   # Instead of:
   XtX_inv = np.linalg.pinv(XtX)
   factor_returns = XtX_inv @ X.T @ y
   
   # Use:
   factor_returns, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
   ```

**Expected Speedup:** 2-5x

**Implementation Difficulty:** Low-Medium

---

### 3. **Factor Covariance Calculation (Stage 6)** - **HIGH PRIORITY** ⬆️

**Current Implementation:**
- Loops over dates and calculates covariance for each
- Uses `window_data.cov()` which materializes full covariance matrix
- Nested loops for factor pairs

**Optimization Strategy (BEST APPROACH):**
**DO NOT use `rolling().cov()`** - it's slow and memory-heavy. Instead, use **incremental rolling sums**:

Maintain rolling sums S₁ and S₂:
- S₁(t) = Σ(s=t-W+1 to t) f_s  (sum of factor returns)
- S₂(t) = Σ(s=t-W+1 to t) f_s f_s^T  (sum of outer products)

Then:
- Mean: μ(t) = S₁(t) / W
- Covariance: Σ_f(t) = (S₂(t) - W·μ(t)μ(t)^T) / (W-1)

Update incrementally each day:
- S₁ ← S₁ + f_new - f_old
- S₂ ← S₂ + f_new f_new^T - f_old f_old^T

```python
# Initialize rolling sums
window = 60
S1 = np.zeros(K)  # Sum of factor returns
S2 = np.zeros((K, K))  # Sum of outer products
window_buffer = []  # Keep last W factor return vectors

for date_idx, date in enumerate(factor_returns_wide.index):
    f_t = factor_returns_wide.iloc[date_idx].values  # Current factor returns (K,)
    
    # Add new observation
    S1 += f_t
    S2 += np.outer(f_t, f_t)
    window_buffer.append(f_t)
    
    # Remove old observation if window is full
    if len(window_buffer) > window:
        f_old = window_buffer.pop(0)
        S1 -= f_old
        S2 -= np.outer(f_old, f_old)
    
    # Calculate covariance if we have enough data
    if len(window_buffer) >= window // 2:
        mu = S1 / len(window_buffer)
        cov_matrix = (S2 - len(window_buffer) * np.outer(mu, mu)) / (len(window_buffer) - 1)
        
        # Store pairwise covariances (VECTORIZED - don't loop in Python!)
        # Use upper triangle indices
        i, j = np.triu_indices(K)
        
        # Extract covariances as arrays (vectorized)
        cov_values = cov_matrix[i, j]  # (num_pairs,) array
        
        # Build DataFrame columns as arrays (much faster than appending dicts)
        num_pairs = len(i)
        date_array = np.full(num_pairs, date, dtype=object)
        factor1_array = np.array([factor_names[i_idx] for i_idx in i])
        factor2_array = np.array([factor_names[j_idx] for j_idx in j])
        
        # Create DataFrame for this date
        date_cov_df = pd.DataFrame({
            'MODEL': config.MODEL_NAME,
            'DATE': date_array,
            'FACTOR_NAME_1': factor1_array,
            'FACTOR_NAME_2': factor2_array,
            'COVARIANCE': cov_values
        })
        
        # Append to list (will concatenate at end)
        covariances_list.append(date_cov_df)

# Concatenate all date DataFrames at once (much faster than appending dicts)
covariance_df = pd.concat(covariances_list, ignore_index=True)
```

**Why This is Better:**
- O(K²) per day (with K=29, it's very cheap: ~841 operations)
- No pandas rolling overhead
- No materialization of huge MultiIndex structures
- **Vectorized output writing** (no Python loop over factor pairs)
- Works great on laptops with limited memory

**Expected Speedup:** 5-20x (much better than 2-3x if currently using pandas rolling)

**Implementation Difficulty:** Low-Medium

---

### 4. **Quarter Processing - Winsorization/Standardization** - **MEDIUM PRIORITY**

**Current Implementation:**
- Loops over each characteristic individually:
  ```python
  for char in all_chars:
      values = df_T[char].dropna()
      lower_bound = values.quantile(winsorize_lower)
      upper_bound = values.quantile(winsorize_upper)
      df_T[char] = df_T[char].clip(lower=lower_bound, upper=upper_bound)
  ```

**Optimization Strategy (BEST APPROACH):**
**Use `np.nanpercentile` on NumPy arrays** - faster than pandas quantile for repeated calls:

```python
# Convert to NumPy array for faster operations
char_data = df_T[all_chars].values  # (N stocks × K chars)

# Compute percentiles across axis=0 (all factors at once)
# This is much faster than pandas quantile in a loop
lower_bounds = np.nanpercentile(char_data, winsorize_lower * 100, axis=0)
upper_bounds = np.nanpercentile(char_data, winsorize_upper * 100, axis=0)

# Clip in one pass (vectorized)
char_data_clipped = np.clip(char_data, lower_bounds, upper_bounds, axis=1)

# Standardize in one pass
char_means = np.nanmean(char_data_clipped, axis=0)
char_stds = np.nanstd(char_data_clipped, axis=0, ddof=0)
char_stds = np.where(char_stds > 0, char_stds, 1.0)  # Avoid division by zero
char_data_standardized = (char_data_clipped - char_means) / char_stds

# Write back to DataFrame
df_T[all_chars] = char_data_standardized
```

**Why This is Better:**
- `np.nanpercentile` is faster than `pd.quantile()` for repeated calls
- All operations vectorized across columns
- Single pass for winsorization + standardization
- Works on NumPy arrays (no pandas overhead)

**Expected Speedup:** 3-5x (better than 2-3x with pandas quantile)

**Implementation Difficulty:** Low

---

### 5. **Quarter Processing - Momentum/Volatility/Liquidity** - **LOW PRIORITY**

**Current Implementation:**
- List comprehension calling calculator methods:
  ```python
  df_T['MOMENTUM'] = [calculator_manager.get_momentum(fid) for fid in factset_ids]
  ```

**Optimization Strategy:**
1. **Batch get operations** if calculator_manager supports it:
   ```python
   # If possible, add batch methods:
   momentums = calculator_manager.get_momentum_batch(factset_ids)
   ```

2. **Use vectorized operations** if the underlying data structure supports it

**Expected Speedup:** 1.5-2x

**Implementation Difficulty:** Medium (requires changes to ring_buffers.py)

---

### 6. **Data Loading Optimizations** - **MEDIUM PRIORITY** ⬆️

**Current Implementation:**
- Reads CSV files with `pd.read_csv()`
- Pivot operations on large DataFrames
- Repeated reads/merges per quarter

**Optimization Strategy:**
1. **Use Parquet everywhere** (already done for main data, but check exposures CSV):
   - Consider saving intermediate exposures as Parquet instead of CSV
   - Parquet is 10-50x faster for reads

2. **Minimize merges**:
   - Pre-join data where possible
   - Use indexed lookups instead of repeated merges

3. **Use faster CSV engines** (if CSV is necessary):
   ```python
   # Use pyarrow or c engine:
   df = pd.read_csv(path, engine='c')  # or 'pyarrow' if available
   ```

4. **Cache pivot results** if same pivot is used multiple times

**Why This Matters:**
Over multi-year runs (2020→now), I/O + merges add up significantly. Parquet everywhere + fewer merges can matter a lot.

**Expected Speedup:** 2-5x for I/O operations (higher impact over long date ranges)

**Implementation Difficulty:** Low-Medium

---

### 7. **Memory Optimizations** - **ONGOING**

**Current Optimizations:**
- Per-quarter data loading (already implemented)
- Limited parallel workers to avoid memory issues

**Additional Strategies:**
1. **Use `dtype` optimization** when reading CSVs:
   ```python
   df = pd.read_csv(path, dtype={'SECURITY_ID': 'string', 'DATE': 'datetime64[ns]'})
   ```

2. **Use categorical dtypes** for repeated string values:
   ```python
   df['FACTOR_NAME'] = df['FACTOR_NAME'].astype('category')
   ```

3. **Delete intermediate DataFrames** explicitly:
   ```python
   del large_df
   import gc
   gc.collect()
   ```

**Expected Benefit:** Reduced memory usage, potentially faster operations

**Implementation Difficulty:** Low

---

## Priority Ranking (FINAL - CORRECTED)

1. **Specific Returns (Stage 4)** - Highest impact, medium difficulty
   - Vectorize with full-row einsum, but **process in date-blocks** (5-20 days)
   - Avoid materializing full F matrix - use indexing instead
   - ⚠️ **Critical**: Watch for column ordering alignment

2. **Factor Covariance (Stage 6)** - **HIGH impact** ⬆️, low-medium difficulty
   - Use incremental rolling sums (S₁/S₂) instead of pandas rolling
   - **Vectorize output writing** - don't loop over factor pairs in Python
   - Much faster than previously estimated (5-20x vs 2-3x)

3. **Factor Returns (Stage 3)** - Medium impact, low-medium difficulty
   - groupby-date iteration + `lstsq` + minimize pandas→numpy conversions
   - Process dates sequentially (one at a time)

4. **Winsorization/Standardization** - Medium impact ⬆️, low difficulty
   - Use `np.nanpercentile` vectorized across columns
   - Better than pandas quantile (3-5x vs 2-3x)

5. **Data Loading** - Medium impact ⬆️, low-medium difficulty
   - Parquet everywhere + minimize merges
   - Higher impact over multi-year runs

6. **Momentum/Volatility/Liquidity** - Low impact, medium difficulty

---

## Implementation Notes

### When to Optimize

- **Optimize Specific Returns first** - This is the biggest remaining bottleneck
  - ⚠️ **Important**: Process in date-blocks, not all dates at once
- **Factor Covariance** - High impact, relatively easy
  - ⚠️ **Important**: Vectorize output writing (no Python loops over pairs)
- **Factor Returns optimization** - Quick win with good ROI
- **Quarter processing optimizations** - Incremental improvements

### Testing Strategy

1. Create benchmark tests to measure current performance
2. Implement optimizations one at a time
3. Verify correctness (results should match exactly)
4. Measure speedup for each optimization
5. Document performance improvements

### Code Quality Considerations

- Maintain readability - vectorized code should be well-commented
- Add type hints where helpful
- Keep backward compatibility
- Add unit tests for optimized functions

---

## Expected Overall Speedup (UPDATED)

If all optimizations are implemented:
- **Specific Returns**: 20-100x faster (eliminates both date loop AND iterrows)
- **Factor Covariance**: 5-20x faster (incremental rolling sums vs pandas rolling)
- **Factor Returns**: 2-5x faster
- **Quarter Processing**: 3-5x faster (np.nanpercentile vs pandas quantile)
- **Data Loading**: 2-5x faster (Parquet + fewer merges)

**Total workflow speedup estimate: 5-15x faster** (depending on dataset size and which stages dominate runtime)

**On a laptop with limited CPU**, this combo usually gives the biggest "2020→now" speedup without needing fancy parallelism.

---

## Recommended Implementation Plan (BEST ORDER)

Based on expert feedback, here's the optimal sequence:

1. **Stage 4 Specific Returns**: Full-row `einsum("nk,nk->n")` after aligning F to each exposure row by date
   - Eliminates per-date loop entirely
   - ⚠️ **Critical**: Ensure factor column ordering matches exactly

2. **Factor Covariance**: Replace rolling.cov / factor-pair loops with S₁/S₂ rolling update
   - Incremental algorithm, O(K²) per day
   - Much faster than pandas rolling operations

3. **Stage 3 Factor Returns**: groupby-date iteration + `lstsq` + minimize pandas→numpy conversions
   - Quick win with good ROI

4. **Winsorization/Z-score**: `np.nanpercentile` vectorized across columns per date
   - Single pass for both operations

This combo usually gives the biggest "2020→now" speedup on limited CPU without needing fancy parallelism.

---

## Specific Guidance for Your Dataset

**Parameters:**
- **K = 29 factors** (style factors + sector dummies + continent dummies + intercept)
- **N ≈ 75,000 stocks per day**

### Memory and Performance Analysis

#### 1. Specific Returns (Stage 4) - Full-Row Einsum

**Memory Requirements (PER DATE):**
- B matrix (exposures): 75k × 29 × 8 bytes = ~17 MB ✅
- F matrix (factor returns): 75k × 29 × 8 bytes = ~17 MB ✅
- Total per date: ~34 MB (very manageable)

**⚠️ CRITICAL: Do NOT process all dates at once!**

If processing 252 days together:
- Total rows: 75k × 252 = 18.9 million rows
- B matrix: 18.9M × 29 × 8 bytes ≈ **4.4 GB**
- F matrix: 18.9M × 29 × 8 bytes ≈ **4.4 GB**
- **Total: ~9 GB just for B+F** (before DataFrames, indices, merges)
- This will cause paging/slowness or crash on a laptop

**Recommendation:** Process in date-blocks (5-20 days per block) or per-date
- You still get 95% of the speedup by eliminating `iterrows()`
- The vectorized math is correct, just apply it per-block
- Memory stays manageable (~34 MB per date, or ~170-680 MB per 5-20 day block)

**Optimized Implementation (avoid materializing full F matrix):**
```python
# Process dates in blocks (e.g., 10 days at a time)
for date_block in date_blocks:
    # Get exposures block (B) for these dates
    B_block = exposures_block[factor_names].fillna(0).values  # (N_block, 29)
    
    # Get factor returns per date (f_t) - don't materialize full F matrix
    # Instead, map each row's date to factor returns
    date_to_factors = {date: factor_returns_wide.loc[date].values 
                       for date in date_block}
    
    # Create f_by_row using indexing (more memory efficient)
    date_indices = exposures_block['DATE'].map(date_to_factors.index.get_loc)
    f_by_row = factor_returns_wide.iloc[date_indices].values  # (N_block, 29)
    
    # Vectorized calculation per block
    explained_returns = np.einsum("nk,nk->n", B_block, f_by_row)  # (N_block,)
    # Or: explained_returns = (B_block * f_by_row).sum(axis=1)
    
    specific_returns = returns_block - explained_returns
```

**Expected Speedup:** 20-100x (eliminates iterrows, but process in blocks)

#### 2. Factor Covariance (Stage 6) - Rolling S₁/S₂

**Performance:**
- S₁ update: O(K) = O(29) operations per day
- S₂ update: O(K²) = O(841) operations per day
- Total: ~870 operations per day (trivial)

**Recommendation:** Perfect use case - extremely fast
- No memory concerns (S₁ is 29 floats, S₂ is 29×29 = 841 floats)
- Window buffer: 60 × 29 × 8 bytes = ~14 KB (negligible)

#### 3. Factor Returns (Stage 3) - OLS Regression

**Performance per date:**
- X matrix: 75k × 29 = ~17 MB
- OLS solve: O(N×K²) = O(75k × 841) = O(63M) operations
- Using `np.linalg.lstsq`: Fast with BLAS, but includes overhead:
  - Building X and y from pandas
  - Handling NaNs/masks
  - Converting pandas → NumPy
  - Any weights/constraints/dummies

**Realistic timing:** 0.1-0.5 seconds per date (depends on data prep overhead)

**Recommendation:** Process dates sequentially with groupby
- Memory per date is manageable (~17 MB)
- No need for batching dates (process one at a time)
- Use `groupby('DATE')` to avoid repeated filtering

#### 4. Winsorization/Standardization

**Performance:**
- char_data: 75k × ~10 chars = ~6 MB
- `np.nanpercentile`: O(N×C) = O(750k) operations per date
- Very fast (< 0.01 seconds per date)

**Recommendation:** Process all characteristics at once
- No memory concerns
- Single-pass vectorized operations

### Overall Memory Profile

**Peak Memory Usage (per date):**
- Exposures: ~17 MB
- Returns: ~0.6 MB (75k × 8 bytes)
- Factor returns: ~2.3 KB (29 × 8 bytes)
- Intermediate matrices: ~34 MB (B + F for specific returns, per date)
- **Total per date: ~52 MB** (very manageable)

**For date-block processing (10 days):**
- Peak memory: ~520 MB (still manageable on laptops)
- Allows vectorization benefits while staying within memory limits

**For full workflow:**
- Process dates sequentially or in small blocks (5-20 days)
- Peak memory: ~52 MB (per date) or ~520 MB (per 10-day block)
- **Do NOT load all dates at once** (would be ~9 GB for B+F alone)

### Recommended Chunk Sizes

**Date-block processing:**
- **Specific Returns**: Process in blocks of 5-20 days (not all dates at once)
  - Block size: 5-20 days = 375k-1.5M rows per block
  - Memory: ~170-680 MB per block (manageable)
- **Factor Returns**: Process dates sequentially (one at a time)
  - No batching needed (only ~17 MB per date)
- **Factor Covariance**: Incremental algorithm (no chunking needed)
  - Processes one date at a time, maintains rolling sums

**Row-level chunking (if needed):**
- Only for loading very large CSV files (> 10M rows)
- Chunk size: 100k-500k rows per chunk
- Usually not needed if processing in date blocks

### Optimal Implementation Strategy

1. **Specific Returns**: 
   - Process dates in blocks (5-20 days per block)
   - Align factor returns to each row within block
   - Vectorized einsum operation per block
   - **Avoid materializing full F matrix** - use indexing instead
   - **Date-block processing** (not all dates at once)

2. **Factor Covariance**:
   - Use incremental S₁/S₂ rolling update
   - Process dates sequentially
   - **Vectorize output writing** - don't loop over factor pairs in Python
   - Build columns as arrays and write in one DataFrame per date/block
   - **Extremely fast, no concerns**

3. **Factor Returns**:
   - Use groupby('DATE') iteration
   - Process dates sequentially (one at a time)
   - Use `np.linalg.lstsq` for OLS
   - **No batching needed** (only ~17 MB per date)

4. **Quarter Processing**:
   - Process dates sequentially (already done)
   - Vectorize winsorization/standardization
   - **No changes needed to date loop**

### Expected Performance (with all optimizations)

**Per-date timings (realistic estimates):**
- Factor Returns: 0.1-0.5 seconds (includes data prep overhead)
- Specific Returns: ~0.05-0.1 seconds per date (vectorized, per-block)
- Factor Covariance: ~0.001 seconds (incremental)
- Specific Risk: ~0.5 seconds (already optimized)

**Per-date total: ~0.65-1.1 seconds**

**For 252 trading days:**
- Core calculations: ~2.7-4.6 minutes
- **End-to-end with I/O + merges + writing tables: 5-15 minutes**
  - I/O overhead (reading CSVs, Parquet files)
  - Merges and data alignment
  - Writing output tables
  - These add significant overhead in practice

**Overall workflow speedup: 5-15x** (more realistic estimate)
- Down from ~30-60 minutes to ~5-15 minutes
- Actual speedup depends on I/O characteristics and data size

**⚠️ Important:** Don't promise absolute runtimes like "2.7 minutes" until you benchmark 1 quarter end-to-end with logging. The vectorized math is fast, but I/O and data prep overhead can dominate.

