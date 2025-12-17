"""
Factor model construction module
Calculates factor returns, specific returns, specific risk, and factor covariance
"""
import pandas as pd
import numpy as np
from typing import Dict
from . import config


class FactorModelBuilder:
    """Class to build factor model and calculate risk metrics"""
    
    def __init__(self, factors_df: pd.DataFrame = None, sp500_returns: pd.Series = None):
        """
        Initialize factor model builder
        
        Args:
            factors_df: Optional DataFrame with factor exposures (not directly used, kept for compatibility)
            sp500_returns: Optional Series with S&P500 returns (not used, kept for compatibility)
        """
        # Parameters kept for backward compatibility but not used in current implementation
        pass
    
    def calculate_daily_factor_returns(self, exposures_df: pd.DataFrame, 
                                       returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily factor returns via cross-sectional OLS regression
        f_t = (B_t^T * B_t)^-1 * B_t^T * r_t
        
        OPTIMIZED: Uses groupby('DATE') iteration and np.linalg.lstsq instead of 
        filtering merged data and manual pseudo-inverse. Faster and more numerically stable.
        
        Args:
            exposures_df: DataFrame with factor exposures (index: FACTSET_ID, DATE)
            returns_df: DataFrame with stock returns (columns: FACTSET_ID, DATE, ONE_DAY_PCT)
            
        Returns:
            DataFrame with factor returns (columns: DATE, FACTOR_NAME, RETURN)
        """
        # Merge exposures and returns
        merged = exposures_df.reset_index().merge(
            returns_df.rename(columns={'P_DATE': 'DATE', 'FSYM_ID': 'FACTSET_ID'}),
            on=['FACTSET_ID', 'DATE'],
            how='inner'
        )
        
        if len(merged) == 0:
            return pd.DataFrame(columns=['MODEL', 'DATE', 'FACTOR_NAME', 'RETURN'])
        
        # Get factor names (preserve order from exposures_df)
        factor_names = [col for col in exposures_df.columns]
        
        # OPTIMIZATION: Pre-group by date to avoid repeated filtering
        merged_by_date = merged.groupby('DATE')
        
        factor_returns_list = []
        
        # Calculate factor returns for each date via OLS
        for date, date_data in merged_by_date:
            if len(date_data) < len(factor_names) + 1:
                continue  # Need more observations than factors
            
            # Prepare X (exposures) and y (returns)
            # Ensure all factor columns are numeric and convert to float64
            X_df = date_data[factor_names].copy()
            for col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
            
            # Convert to numpy array with explicit float64 dtype
            X = X_df.astype(float).values
            y = pd.to_numeric(date_data['ONE_DAY_PCT'], errors='coerce').astype(float).values
            
            # Remove NaN values
            X_nan_mask = np.isnan(X).any(axis=1)
            y_nan_mask = np.isnan(y)
            valid_mask = ~(X_nan_mask | y_nan_mask)
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Ensure arrays are float64 (not object dtype)
            if X.dtype == 'object' or X.dtype.kind == 'O':
                X = X.astype(float)
            if y.dtype == 'object' or y.dtype.kind == 'O':
                y = y.astype(float)
            
            if len(X) < len(factor_names) + 1:
                continue
            
            # Add intercept if configured
            include_intercept = config.FACTOR_RETURN_PARAMS.get('include_intercept', False)
            if include_intercept:
                # Add column of ones for intercept
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                factor_names_with_intercept = ['INTERCEPT'] + factor_names
            else:
                X_with_intercept = X
                factor_names_with_intercept = factor_names
            
            if len(X_with_intercept) < len(factor_names_with_intercept) + 1:
                continue
            
            # OLS: f_t = (X^T * X)^-1 * X^T * y
            # OPTIMIZATION: Use lstsq instead of manual pinv (faster and more stable)
            try:
                factor_returns, residuals, rank, s = np.linalg.lstsq(
                    X_with_intercept, y, rcond=None
                )
                
                # Store factor returns
                for i, factor_name in enumerate(factor_names_with_intercept):
                    factor_returns_list.append({
                        'MODEL': config.MODEL_NAME,
                        'DATE': date,
                        'FACTOR_NAME': factor_name,
                        'RETURN': factor_returns[i]
                    })
            except np.linalg.LinAlgError:
                # Skip if matrix is singular
                continue
        
        factor_returns_df = pd.DataFrame(factor_returns_list)
        return factor_returns_df
    
    def calculate_specific_returns(self, exposures_df: pd.DataFrame, 
                                   returns_df: pd.DataFrame,
                                   factor_returns_df: pd.DataFrame,
                                   block_size: int = 10) -> pd.DataFrame:
        """
        Calculate specific (idiosyncratic) returns as regression residuals
        ε̂_i,t = r_i,t - Σ(k=1 to K) β_i,k,t * f̂_k,t
        
        OPTIMIZED: Uses vectorized einsum with date-block processing to eliminate iterrows loops.
        Processes dates in blocks (default 10 days) to avoid memory issues with large datasets.
        
        Args:
            exposures_df: DataFrame with factor exposures (index: FACTSET_ID or SECURITY_ID, DATE)
            returns_df: DataFrame with stock returns
            factor_returns_df: DataFrame with factor returns
            block_size: Number of dates to process per block (default: 10)
            
        Returns:
            DataFrame with specific returns (columns: MODEL, DATE, SECURITY_ID, SPECIFIC_RETURN)
        """
        # Reset index and ensure consistent column names
        exposures_reset = exposures_df.reset_index()
        
        # Handle both SECURITY_ID and FACTSET_ID column names
        if 'SECURITY_ID' in exposures_reset.columns:
            id_col = 'SECURITY_ID'
        elif 'FACTSET_ID' in exposures_reset.columns:
            id_col = 'FACTSET_ID'
            exposures_reset = exposures_reset.rename(columns={'FACTSET_ID': 'SECURITY_ID'})
        else:
            raise ValueError("Exposures DataFrame must have SECURITY_ID or FACTSET_ID column")
        
        # Prepare returns_df with consistent column names
        returns_prep = returns_df.copy()
        if 'FSYM_ID' in returns_prep.columns:
            returns_prep = returns_prep.rename(columns={'FSYM_ID': 'SECURITY_ID'})
        elif 'FACTSET_ID' in returns_prep.columns:
            returns_prep = returns_prep.rename(columns={'FACTSET_ID': 'SECURITY_ID'})
        
        if 'P_DATE' in returns_prep.columns:
            returns_prep = returns_prep.rename(columns={'P_DATE': 'DATE'})
        
        # Ensure DATE columns are datetime
        exposures_reset['DATE'] = pd.to_datetime(exposures_reset['DATE'])
        returns_prep['DATE'] = pd.to_datetime(returns_prep['DATE'])
        factor_returns_df = factor_returns_df.copy()
        factor_returns_df['DATE'] = pd.to_datetime(factor_returns_df['DATE'])
        
        # Merge exposures with returns
        merged = exposures_reset.merge(
            returns_prep[['SECURITY_ID', 'DATE', 'ONE_DAY_PCT']],
            on=['SECURITY_ID', 'DATE'],
            how='inner'  # Only keep stocks with both exposures and returns
        )
        
        if len(merged) == 0:
            print("   ⚠ WARNING: No stocks with both exposures and returns found")
            return pd.DataFrame(columns=['MODEL', 'DATE', 'SECURITY_ID', 'SPECIFIC_RETURN'])
        
        # Get factor names (exclude metadata columns) - CRITICAL: preserve order
        factor_names = [col for col in exposures_reset.columns 
                       if col not in ['SECURITY_ID', 'DATE', 'MODEL', 'ONE_DAY_PCT']]
        
        # Pivot factor returns to wide format for efficient lookup
        factor_returns_wide = factor_returns_df.pivot_table(
            index='DATE',
            columns='FACTOR_NAME',
            values='RETURN'
        ).sort_index()
        
        # Ensure factor column order matches between exposures and factor returns
        # Get intersection of factor names (some factors might not be in factor_returns)
        factor_names_ordered = [f for f in factor_names if f in factor_returns_wide.columns]
        
        if len(factor_names_ordered) == 0:
            print("   ⚠ WARNING: No matching factors between exposures and factor returns")
            return pd.DataFrame(columns=['MODEL', 'DATE', 'SECURITY_ID', 'SPECIFIC_RETURN'])
        
        # Get all unique dates (sorted)
        all_dates = sorted(merged['DATE'].unique())
        all_dates = [d for d in all_dates if d in factor_returns_wide.index]
        
        if len(all_dates) == 0:
            print("   ⚠ WARNING: No matching dates between exposures and factor returns")
            return pd.DataFrame(columns=['MODEL', 'DATE', 'SECURITY_ID', 'SPECIFIC_RETURN'])
        
        # Process dates in blocks to avoid memory issues
        specific_returns_list = []
        
        print(f"  Processing {len(all_dates)} dates in blocks of {block_size}...")
        
        for block_start in range(0, len(all_dates), block_size):
            date_block = all_dates[block_start:block_start + block_size]
            block_data = merged[merged['DATE'].isin(date_block)].copy()
            
            if len(block_data) == 0:
                continue
            
            # Get exposures matrix B for this block
            # Ensure columns are in the same order as factor_returns_wide
            B_block = block_data[factor_names_ordered].fillna(0).values  # (N_block, K)
            
            # Get factor returns per date - avoid materializing full F matrix
            # Create date-to-factor-returns mapping
            date_to_factors = {}
            for date in date_block:
                if date in factor_returns_wide.index:
                    date_to_factors[date] = factor_returns_wide.loc[date, factor_names_ordered].fillna(0).values
            
            # Create f_by_row: map each row's date to its factor returns
            # More memory efficient than creating full F matrix
            f_by_row = np.array([date_to_factors.get(date, np.zeros(len(factor_names_ordered))) 
                                for date in block_data['DATE']])
            
            # Vectorized calculation: explained returns for all rows in block
            # explained = Σ(k) B[n,k] * F[n,k] for each row n
            explained_returns = np.einsum("nk,nk->n", B_block, f_by_row)
            # Alternative (slightly slower but equivalent): 
            # explained_returns = (B_block * f_by_row).sum(axis=1)
            
            # Specific returns = actual returns - explained returns
            specific_returns = block_data['ONE_DAY_PCT'].values - explained_returns
            
            # Store results for this block
            block_results = pd.DataFrame({
                'MODEL': config.MODEL_NAME,
                'DATE': block_data['DATE'].values,
                'SECURITY_ID': block_data['SECURITY_ID'].values,
                'SPECIFIC_RETURN': specific_returns
            })
            specific_returns_list.append(block_results)
        
        # Concatenate all blocks
        if specific_returns_list:
            specific_returns_df = pd.concat(specific_returns_list, ignore_index=True)
        else:
            specific_returns_df = pd.DataFrame(columns=['MODEL', 'DATE', 'SECURITY_ID', 'SPECIFIC_RETURN'])
        
        return specific_returns_df
    
    def calculate_factor_covariance(self, factor_returns_df: pd.DataFrame,
                                    lookback_window: int = 60) -> pd.DataFrame:
        """
        Calculate factor covariance matrix using rolling window
        F_kl = Cov_t(f_k,t, f_l,t)
        
        OPTIMIZED: Uses incremental rolling sums (S₁/S₂) instead of pandas rolling.cov().
        Much faster and more memory efficient. Output writing is vectorized.
        
        Algorithm:
        - Maintain rolling sums: S₁ = Σ f_s, S₂ = Σ f_s f_s^T
        - Update incrementally: S₁ ← S₁ + f_new - f_old
        - Covariance: Σ = (S₂ - W·μμ^T) / (W-1) where μ = S₁/W
        
        Args:
            factor_returns_df: DataFrame with factor returns
            lookback_window: Rolling window size (default: 60 trading days)
            
        Returns:
            DataFrame with factor covariance matrix
        """
        # Pivot factor returns to wide format
        factor_returns_wide = factor_returns_df.pivot_table(
            index='DATE',
            columns='FACTOR_NAME',
            values='RETURN'
        ).sort_index()
        
        factor_names = factor_returns_wide.columns.tolist()
        K = len(factor_names)
        
        # Initialize rolling sums
        S1 = np.zeros(K)  # Sum of factor returns: S₁ = Σ f_s
        S2 = np.zeros((K, K))  # Sum of outer products: S₂ = Σ f_s f_s^T
        window_buffer = []  # Keep last W factor return vectors for removing old observations
        
        # List to store DataFrames (one per date) - faster than appending dicts
        covariances_list = []
        
        for date_idx, date in enumerate(factor_returns_wide.index):
            f_t = factor_returns_wide.iloc[date_idx].values  # Current factor returns (K,)
            
            # Handle NaN values - skip dates with too many NaNs
            if np.isnan(f_t).sum() > K // 2:
                continue
            
            # Fill NaN with 0 for calculation (or could use forward fill, but 0 is safer)
            f_t = np.nan_to_num(f_t, nan=0.0)
            
            # Add new observation to rolling sums
            S1 += f_t
            S2 += np.outer(f_t, f_t)
            window_buffer.append(f_t.copy())
            
            # Remove old observation if window is full
            if len(window_buffer) > lookback_window:
                f_old = window_buffer.pop(0)
                S1 -= f_old
                S2 -= np.outer(f_old, f_old)
            
            # Calculate covariance if we have enough data
            W = len(window_buffer)
            if W < lookback_window // 2:
                continue
            
            # Calculate mean: μ = S₁ / W
            mu = S1 / W
            
            # Calculate covariance: Σ = (S₂ - W·μμ^T) / (W-1)
            cov_matrix = (S2 - W * np.outer(mu, mu)) / (W - 1)
            
            # Store pairwise covariances (VECTORIZED - don't loop in Python!)
            # Use upper triangle indices (symmetric matrix)
            i, j = np.triu_indices(K)
            
            # Extract covariances as arrays (vectorized)
            cov_values = cov_matrix[i, j]  # (num_pairs,) array
            
            # Build DataFrame columns as arrays (much faster than appending dicts)
            num_pairs = len(i)
            # Use list comprehension for date array to preserve datetime type
            date_array = [date] * num_pairs
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
        if covariances_list:
            covariance_df = pd.concat(covariances_list, ignore_index=True)
        else:
            covariance_df = pd.DataFrame(columns=['MODEL', 'DATE', 'FACTOR_NAME_1', 'FACTOR_NAME_2', 'COVARIANCE'])
        
        return covariance_df
    
    def calculate_specific_risk(self, exposures_df: pd.DataFrame,
                                returns_df: pd.DataFrame,
                                factor_returns_df: pd.DataFrame,
                                window: int = 60,
                                epsilon: float = 1e-8) -> pd.DataFrame:
        """
        Calculate specific risk (specific variance) using rolling window
        
        For each as-of date T:
        1. Factor covariance: Σ_f(T) = Cov_60(f_{.,t}) for t = T-59, ..., T
        2. Total variance: Var(r_i)_T = Var_60(r_{i,t}) for t = T-59, ..., T
        3. Factor variance: σ²_factor,i(T) = β_{i,T}^T * Σ_f(T) * β_{i,T}
        4. Specific variance: SPECIFIC_VAR_{i,T} = max(Var(r_i)_T - σ²_factor,i(T), min_floor)
           where min_floor = max(1% of total_var, epsilon)
        
        Edge case handling:
        - If factor_var >= total_var: Cap factor_var to 95% of total_var
        - Minimum specific_var is 1% of total_var (prevents artificially low values like 1e-8)
        
        Args:
            exposures_df: DataFrame with factor exposures (index: FACTSET_ID, DATE)
            returns_df: DataFrame with stock returns (columns: FACTSET_ID, DATE, ONE_DAY_PCT)
            factor_returns_df: DataFrame with factor returns (columns: DATE, FACTOR_NAME, RETURN)
            window: Rolling window size (default: 60 days)
            epsilon: Minimum floor value for specific variance (default: 1e-8)
            
        Returns:
            DataFrame with columns: MODEL, DATE, SECURITY_ID, SPECIFIC_VAR
            - SPECIFIC_VAR: Specific variance = TOTAL_VAR - FACTOR_VAR (with reasonable floor)
        """
        print(f"  Calculating specific risk with {window}-day rolling window...")
        
        # Prepare data
        exposures_reset = exposures_df.reset_index()
        
        # Handle both SECURITY_ID and FACTSET_ID column names
        if 'SECURITY_ID' in exposures_reset.columns:
            id_col = 'SECURITY_ID'
        elif 'FACTSET_ID' in exposures_reset.columns:
            id_col = 'FACTSET_ID'
            exposures_reset = exposures_reset.rename(columns={'FACTSET_ID': 'SECURITY_ID'})
        else:
            raise ValueError("Exposures DataFrame must have SECURITY_ID or FACTSET_ID column")
        
        exposures_reset['DATE'] = pd.to_datetime(exposures_reset['DATE'])
        
        returns_prep = returns_df.copy()
        if 'FSYM_ID' in returns_prep.columns:
            returns_prep = returns_prep.rename(columns={'FSYM_ID': 'SECURITY_ID'})
        elif 'FACTSET_ID' in returns_prep.columns:
            returns_prep = returns_prep.rename(columns={'FACTSET_ID': 'SECURITY_ID'})
        
        if 'P_DATE' in returns_prep.columns:
            returns_prep = returns_prep.rename(columns={'P_DATE': 'DATE'})
        returns_prep['DATE'] = pd.to_datetime(returns_prep['DATE'])
        
        factor_returns_prep = factor_returns_df.copy()
        factor_returns_prep['DATE'] = pd.to_datetime(factor_returns_prep['DATE'])
        
        # Pivot factor returns to wide format
        factor_returns_wide = factor_returns_prep.pivot_table(
            index='DATE',
            columns='FACTOR_NAME',
            values='RETURN'
        ).sort_index()
        
        # Get all dates (sorted)
        all_dates = sorted(exposures_reset['DATE'].unique())
        all_dates = [d for d in all_dates if d in factor_returns_wide.index]
        
        # Get factor names (exclude INTERCEPT if present for covariance calculation)
        factor_names = [col for col in exposures_df.columns]
        factor_names_for_cov = [f for f in factor_returns_wide.columns if f != 'INTERCEPT']
        
        # OPTIMIZATION: Pre-group exposures by date to avoid repeated filtering
        exposures_by_date = exposures_reset.groupby('DATE')
        
        specific_risk_list = []
        
        print(f"  Processing {len(all_dates)} dates...")
        
        for i, date_T in enumerate(all_dates):
            if (i + 1) % 10 == 0 or i == len(all_dates) - 1:
                print(f"    Processed {i+1}/{len(all_dates)} dates...")
            
            # Get exposures for date T (using groupby is faster than filtering)
            try:
                exposures_T = exposures_by_date.get_group(date_T)
            except KeyError:
                continue
            
            if len(exposures_T) == 0:
                continue
            
            # Get factor returns window: T-59 to T
            # Use available data even if less than full window (for early dates)
            date_idx = factor_returns_wide.index.get_loc(date_T)
            window_start_idx = max(0, date_idx - window + 1)
            factor_returns_window = factor_returns_wide.iloc[window_start_idx:date_idx+1]
            
            # Require at least window//2 observations (30 days) for meaningful calculation
            # This allows us to calculate specific risk for dates starting from day 30
            if len(factor_returns_window) < window // 2:
                # Not enough data - skip this date
                continue
            
            # 1. Calculate factor covariance matrix Σ_f(T) (once per date, not per stock)
            # Use only non-intercept factors for covariance
            factor_returns_for_cov = factor_returns_window[factor_names_for_cov]
            factor_cov_matrix = factor_returns_for_cov.cov().values
            
            # OPTIMIZATION: Pre-filter returns by date range once (not per stock)
            window_start_date = factor_returns_window.index[0]
            returns_window = returns_prep[
                (returns_prep['DATE'] >= window_start_date) & 
                (returns_prep['DATE'] <= date_T)
            ].copy()
            
            # MAJOR OPTIMIZATION: Vectorize total variance calculation for all stocks
            # Pivot returns to wide format (stocks × dates) for efficient rolling variance
            if len(returns_window) == 0:
                continue
            
            returns_pivot = returns_window.pivot_table(
                index='SECURITY_ID',
                columns='DATE',
                values='ONE_DAY_PCT'
            )
            
            # Get stocks that have both exposures and returns
            stocks_with_exposures = set(exposures_T['SECURITY_ID'].values)
            stocks_with_returns = set(returns_pivot.index)
            stocks_to_process = list(stocks_with_exposures & stocks_with_returns)
            
            if len(stocks_to_process) == 0:
                continue
            
            # Filter to stocks we'll process
            exposures_T_filtered = exposures_T[exposures_T['SECURITY_ID'].isin(stocks_to_process)].copy()
            returns_pivot_filtered = returns_pivot.loc[stocks_to_process]
            
            # Calculate total variance for all stocks at once (vectorized)
            # Use rolling window variance: need at least window//2 observations
            total_vars = returns_pivot_filtered.var(axis=1, ddof=0)  # Population variance
            
            # Filter stocks with sufficient data and valid variance
            valid_stocks = total_vars[
                (total_vars.notna()) & 
                (total_vars > 0) &
                (returns_pivot_filtered.count(axis=1) >= window // 2)
            ].index.tolist()
            
            if len(valid_stocks) == 0:
                continue
            
            # Filter exposures and total_vars to valid stocks
            # IMPORTANT: Align exposures with total_vars by using total_vars index order
            total_vars_valid = total_vars.loc[valid_stocks]
            
            # Set SECURITY_ID as index for exposures to enable alignment
            exposures_T_indexed = exposures_T_filtered.set_index('SECURITY_ID')
            exposures_T_valid = exposures_T_indexed.loc[valid_stocks].reset_index()
            
            # MAJOR OPTIMIZATION: Vectorize factor variance calculation for all stocks
            # Build exposure matrix B (N stocks × K factors) for all stocks on date T
            # Fill missing exposures as 0 (as per user's suggestion)
            # Stock order now matches total_vars_valid.index
            stock_ids_ordered = valid_stocks  # Use same order as total_vars_valid
            B = np.zeros((len(exposures_T_valid), len(factor_names_for_cov)))
            
            # Map factor names to column indices in B
            factor_to_col = {f: idx for idx, f in enumerate(factor_names_for_cov)}
            
            # Fill exposure matrix (preserve order by iterating in order)
            for stock_idx, (_, stock_row) in enumerate(exposures_T_valid.iterrows()):
                for f in factor_names_for_cov:
                    if f in stock_row.index and not pd.isna(stock_row[f]):
                        col_idx = factor_to_col[f]
                        B[stock_idx, col_idx] = stock_row[f]
            
            # Compute factor variance for all stocks at once: diag(B @ Σ_f @ B^T)
            # Equivalent to: (B @ Σ_f @ B.T).diagonal()
            # More efficient: tmp = B @ Σ_f, then factor_var = (tmp * B).sum(axis=1)
            tmp = B @ factor_cov_matrix  # (N, K)
            factor_vars = np.einsum('nk,nk->n', tmp, B)  # Length N vector
            
            # Handle NaN and negative values
            factor_vars = np.where(np.isnan(factor_vars) | (factor_vars < 0), 0, factor_vars)
            
            # Convert to pandas Series with same order as exposures_T_valid
            factor_vars_series = pd.Series(factor_vars, index=stock_ids_ordered)
            total_vars_series = total_vars_valid
            
            # Align indices (should already match, but ensure)
            common_stocks = factor_vars_series.index.intersection(total_vars_series.index)
            factor_vars_aligned = factor_vars_series.loc[common_stocks]
            total_vars_aligned = total_vars_series.loc[common_stocks]
            
            # Calculate specific variance for all stocks at once (vectorized)
            # Handle edge cases where factor variance might exceed total variance
            factor_vars_capped = np.where(
                factor_vars_aligned >= total_vars_aligned,
                total_vars_aligned * 0.95,  # Cap to 95% of total variance
                factor_vars_aligned
            )
            
            # Calculate minimum floor: 1% of total variance or epsilon
            min_specific_vars = np.maximum(total_vars_aligned * 0.01, epsilon)
            
            # Specific variance = max(total_var - factor_var, min_floor)
            specific_vars = np.maximum(
                total_vars_aligned - factor_vars_capped,
                min_specific_vars
            )
            
            # Store results for all stocks at once
            for security_id, specific_var in zip(common_stocks, specific_vars):
                specific_risk_list.append({
                    'MODEL': config.MODEL_NAME,
                    'DATE': date_T,
                    'SECURITY_ID': security_id,
                    'SPECIFIC_VAR': specific_var  # Specific variance = TOTAL_VAR - FACTOR_VAR
                })
        
        specific_risk_df = pd.DataFrame(specific_risk_list)
        
        if len(specific_risk_df) > 0:
            # Ensure DATE is datetime
            specific_risk_df['DATE'] = pd.to_datetime(specific_risk_df['DATE'])
            # Sort by DATE and SECURITY_ID
            specific_risk_df = specific_risk_df.sort_values(['DATE', 'SECURITY_ID'])
        
        return specific_risk_df
    
    def create_factor_names_table(self, exposures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create factor names metadata table with display names and factor groups
        
        Args:
            exposures_df: DataFrame with factor exposures (to extract factor names)
            
        Returns:
            DataFrame with columns: MODEL, FACTOR_DISPLAY_NAME, FACTOR_GROUP
        """
        factor_names_list = []
        
        # Get all factor names from exposures (exclude non-factor columns)
        exclude_cols = ['FACTSET_ID', 'DATE', 'MODEL', 'SECURITY_ID']
        factor_names = [col for col in exposures_df.columns if col not in exclude_cols]
        
        # Style factors
        style_factors = ['VALUE', 'PROFITABILITY', 'GROWTH', 'MOMENTUM', 'VOLATILITY', 'LIQUIDITY']
        for factor in style_factors:
            if factor in factor_names:
                factor_names_list.append({
                    'MODEL': config.MODEL_NAME,
                    'FACTOR_DISPLAY_NAME': factor,
                    'FACTOR_GROUP': 'Style Factors'
                })
        
        # Sector factors
        sector_factors = [f for f in factor_names if f.startswith('SECTOR_')]
        for factor in sector_factors:
            # Convert SECTOR_Technology to "Technology Sector"
            sector_name = factor.replace('SECTOR_', '').replace('_', ' ')
            factor_names_list.append({
                'MODEL': config.MODEL_NAME,
                'FACTOR_DISPLAY_NAME': f'{sector_name} Sector',
                'FACTOR_GROUP': 'Sector Factors'
            })
        
        # Geographic factors (continent + developed/developing)
        continent_factors = [f for f in factor_names if f.startswith('CONTINENT_')]
        for factor in continent_factors:
            # Convert CONTINENT_North_America_Developed to "North America Developed"
            continent_name = factor.replace('CONTINENT_', '').replace('_', ' ')
            factor_names_list.append({
                'MODEL': config.MODEL_NAME,
                'FACTOR_DISPLAY_NAME': continent_name,
                'FACTOR_GROUP': 'Geographic Factors'
            })
        
        # Intercept (if present in factor returns)
        if 'INTERCEPT' in factor_names:
            factor_names_list.append({
                'MODEL': config.MODEL_NAME,
                'FACTOR_DISPLAY_NAME': 'Intercept',
                'FACTOR_GROUP': 'Market Factor'
            })
        
        factor_names_df = pd.DataFrame(factor_names_list)
        return factor_names_df
    
    @staticmethod
    def convert_exposures_to_long_format(exposures_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert exposure table from wide format to long format
        Wide: one row per stock-date, columns are factors
        Long: one row per stock-date-factor, columns are MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE
        
        Args:
            exposures_df: DataFrame in wide format with columns: FACTSET_ID, DATE, [factor columns]
            
        Returns:
            DataFrame in long format with columns: MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE
        """
        # Identify factor columns (exclude metadata columns)
        exclude_cols = ['FACTSET_ID', 'DATE', 'MODEL', 'SECURITY_ID']
        factor_cols = [col for col in exposures_df.columns if col not in exclude_cols]
        
        # Ensure we have FACTSET_ID or SECURITY_ID
        if 'FACTSET_ID' in exposures_df.columns:
            id_col = 'FACTSET_ID'
        elif 'SECURITY_ID' in exposures_df.columns:
            id_col = 'SECURITY_ID'
        else:
            raise ValueError("Exposures DataFrame must have FACTSET_ID or SECURITY_ID column")
        
        # Melt to long format
        id_vars = [id_col, 'DATE'] if 'DATE' in exposures_df.columns else [id_col]
        exposures_long = exposures_df.melt(
            id_vars=id_vars,
            value_vars=factor_cols,
            var_name='FACTOR_NAME',
            value_name='EXPOSURE'
        )
        
        # Rename ID column to SECURITY_ID
        if id_col == 'FACTSET_ID':
            exposures_long = exposures_long.rename(columns={'FACTSET_ID': 'SECURITY_ID'})
        
        # Add MODEL column
        exposures_long.insert(0, 'MODEL', config.MODEL_NAME)
        
        # Reorder columns: MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE
        column_order = ['MODEL', 'DATE', 'SECURITY_ID', 'FACTOR_NAME', 'EXPOSURE']
        exposures_long = exposures_long[column_order]
        
        # Remove rows with NaN exposures (efficient)
        exposures_long = exposures_long[exposures_long['EXPOSURE'].notna()]
        
        return exposures_long



