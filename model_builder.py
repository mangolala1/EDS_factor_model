"""
Factor model construction module
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import config


class FactorModelBuilder:
    """Class to build and optimize factor model"""
    
    def __init__(self, factors_df: pd.DataFrame, sp500_returns: pd.Series = None):
        """
        Initialize with factors and optional benchmark returns
        
        Args:
            factors_df: DataFrame with factors and stock returns
            sp500_returns: Optional Series with S&P500 returns (for comparison)
        """
        self.factors_df = factors_df.copy()
        self.sp500_returns = sp500_returns.copy() if sp500_returns is not None else pd.Series(dtype=float)
        self.model = None
        self.scaler = StandardScaler()
        self.factor_weights = {}
        self.correlation = 0.0
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training
        
        Returns:
            Tuple of (X, y) where X is factors and y is market returns
        """
        # Calculate market-cap weighted portfolio returns
        # For simplicity, we'll use equal-weighted portfolio
        # In production, you'd want to use market-cap weights
        
        dates = self.factors_df.index.get_level_values('date').unique()
        market_returns = []
        
        for date in dates:
            date_data = self.factors_df.loc[slice(None), date]
            if 'returns' in date_data.columns and len(date_data) > 0:
                # Equal-weighted portfolio return
                market_return = date_data['returns'].mean()
                market_returns.append(market_return)
            else:
                market_returns.append(np.nan)
        
        market_returns_series = pd.Series(market_returns, index=dates)
        market_returns_series = market_returns_series.dropna()
        
        # Align factors with market returns
        factor_names = [col for col in self.factors_df.columns if col != 'returns']
        
        # Aggregate factors by date (average across stocks)
        factor_data = []
        for date in market_returns_series.index:
            date_data = self.factors_df.loc[slice(None), date]
            if len(date_data) > 0:
                factor_row = date_data[factor_names].mean()
                factor_data.append(factor_row)
            else:
                factor_data.append(pd.Series(index=factor_names, data=np.nan))
        
        factor_df = pd.DataFrame(factor_data, index=market_returns_series.index)
        
        # Align indices
        common_dates = factor_df.index.intersection(market_returns_series.index)
        factor_df = factor_df.loc[common_dates]
        market_returns_series = market_returns_series.loc[common_dates]
        
        # Drop any remaining NaN values
        valid_mask = ~(factor_df.isna().any(axis=1) | market_returns_series.isna())
        factor_df = factor_df[valid_mask]
        market_returns_series = market_returns_series[valid_mask]
        
        return factor_df, market_returns_series
    
    def build_regression_model(self, method: str = 'ridge', alpha: float = 1.0) -> Dict:
        """
        Build factor model using regression
        
        Args:
            method: 'ridge' or 'lasso'
            alpha: Regularization parameter
            
        Returns:
            Dictionary with model results
        """
        X, y = self.prepare_data()
        
        if len(X) < config.MIN_DATA_POINTS:
            return {'error': f'Insufficient data points: {len(X)} < {config.MIN_DATA_POINTS}'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Choose model
        if method == 'ridge':
            model = Ridge(alpha=alpha)
        elif method == 'lasso':
            model = Lasso(alpha=alpha)
        else:
            model = Ridge(alpha=alpha)
        
        # Fit model
        model.fit(X_scaled, y)
        
        # Predictions
        y_pred = model.predict(X_scaled)
        
        # Calculate correlation
        correlation = np.corrcoef(y, y_pred)[0, 1]
        
        # Store model
        self.model = model
        self.factor_weights = dict(zip(X.columns, model.coef_))
        
        return {
            'method': method,
            'alpha': alpha,
            'correlation': correlation,
            'r2_score': np.corrcoef(y, y_pred)[0, 1]**2,
            'factor_weights': self.factor_weights,
            'intercept': model.intercept_
        }
    
    def optimize_weights(self, target_correlation: float = None) -> Dict:
        """
        Optimize factor weights to achieve target correlation
        
        Args:
            target_correlation: Target correlation (default from config)
            
        Returns:
            Dictionary with optimization results
        """
        target_corr = target_correlation or config.TARGET_CORRELATION
        
        X, y = self.prepare_data()
        
        if len(X) < config.MIN_DATA_POINTS:
            return {'error': f'Insufficient data points: {len(X)} < {config.MIN_DATA_POINTS}'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        def objective(weights):
            """Objective function: minimize negative correlation"""
            y_pred = X_scaled @ weights
            corr = np.corrcoef(y, y_pred)[0, 1]
            return -corr  # Minimize negative correlation
        
        def constraint_sum(weights):
            """Constraint: weights sum to 1"""
            return np.sum(weights) - 1.0
        
        # Initial guess
        n_factors = X_scaled.shape[1]
        initial_weights = np.ones(n_factors) / n_factors
        
        # Bounds: weights between -1 and 1
        bounds = [(-1, 1) for _ in range(n_factors)]
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'eq', 'fun': constraint_sum},
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            y_pred = X_scaled @ optimal_weights
            correlation = np.corrcoef(y, y_pred)[0, 1]
            
            self.factor_weights = dict(zip(X.columns, optimal_weights))
            self.correlation = correlation
            
            return {
                'success': True,
                'correlation': correlation,
                'factor_weights': self.factor_weights,
                'target_achieved': correlation >= target_corr
            }
        else:
            return {
                'success': False,
                'error': result.message
            }
    
    def build_optimized_model(self, max_iterations: int = 10) -> Dict:
        """
        Build optimized model with iterative approach to achieve target correlation
        
        Args:
            max_iterations: Maximum number of optimization iterations
            
        Returns:
            Dictionary with final model results
        """
        target_corr = config.TARGET_CORRELATION
        
        # Try different regularization strengths
        alphas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        best_result = None
        best_correlation = -1
        
        # First try regression methods
        for alpha in alphas:
            for method in ['ridge', 'lasso']:
                result = self.build_regression_model(method=method, alpha=alpha)
                if 'error' not in result:
                    if result['correlation'] > best_correlation:
                        best_correlation = result['correlation']
                        best_result = result
        
        # If regression doesn't achieve target, try optimization
        if best_correlation < target_corr:
            opt_result = self.optimize_weights(target_corr)
            if opt_result.get('success', False):
                if opt_result['correlation'] > best_correlation:
                    best_result = opt_result
        
        # Final check
        if best_result and 'correlation' in best_result:
            self.correlation = best_result['correlation']
            if 'factor_weights' in best_result:
                self.factor_weights = best_result['factor_weights']
        
        return best_result or {'error': 'Failed to build model'}
    
    def predict_returns(self, factors_df: pd.DataFrame) -> pd.Series:
        """
        Predict returns using the trained model
        
        Args:
            factors_df: DataFrame with factors
            
        Returns:
            Series with predicted returns
        """
        if self.model is None:
            raise ValueError("Model not trained. Call build_optimized_model() first.")
        
        factor_names = list(self.factor_weights.keys())
        X = factors_df[factor_names]
        X_scaled = self.scaler.transform(X)
        
        if hasattr(self.model, 'predict'):
            predictions = self.model.predict(X_scaled)
        else:
            # Use optimized weights
            weights = np.array([self.factor_weights[f] for f in factor_names])
            predictions = X_scaled @ weights
        
        return pd.Series(predictions, index=factors_df.index)
    
    def compare_with_sp500(self) -> Dict:
        """
        Compare model predictions with S&P500 returns
        
        Returns:
            Dictionary with comparison metrics
        """
        X, y = self.prepare_data()
        
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Get predictions
        X_scaled = self.scaler.transform(X)
        if hasattr(self.model, 'predict'):
            y_pred = self.model.predict(X_scaled)
        else:
            weights = np.array([self.factor_weights[f] for f in X.columns])
            y_pred = X_scaled @ weights
        
        # Align with S&P500 returns
        common_dates = X.index.intersection(self.sp500_returns.index)
        if len(common_dates) == 0:
            return {'error': 'No overlapping dates with S&P500'}
        
        y_pred_aligned = pd.Series(y_pred, index=X.index).loc[common_dates]
        sp500_aligned = self.sp500_returns.loc[common_dates]
        
        # Calculate correlation
        correlation = np.corrcoef(y_pred_aligned, sp500_aligned)[0, 1]
        
        return {
            'correlation': correlation,
            'target_achieved': correlation >= config.TARGET_CORRELATION,
            'r2_score': correlation ** 2,
            'mse': np.mean((y_pred_aligned - sp500_aligned) ** 2),
            'mae': np.mean(np.abs(y_pred_aligned - sp500_aligned))
        }
    
    def calculate_daily_factor_returns(self, exposures_df: pd.DataFrame, 
                                       returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily factor returns via cross-sectional OLS regression
        f_t = (B_t^T * B_t)^-1 * B_t^T * r_t
        
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
        
        # Get factor names
        factor_names = [col for col in exposures_df.columns]
        
        factor_returns_list = []
        
        # Calculate factor returns for each date via OLS
        for date in merged['DATE'].unique():
            date_data = merged[merged['DATE'] == date].copy()
            
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
            try:
                XtX = X_with_intercept.T @ X_with_intercept
                XtX_inv = np.linalg.pinv(XtX)  # Use pseudo-inverse for numerical stability
                factor_returns = XtX_inv @ X_with_intercept.T @ y
                
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
                                   factor_returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate specific (idiosyncratic) returns as regression residuals
        ε̂_i,t = r_i,t - Σ(k=1 to K) β_i,k,t * f̂_k,t
        
        Args:
            exposures_df: DataFrame with factor exposures (index: FACTSET_ID or SECURITY_ID, DATE)
            returns_df: DataFrame with stock returns
            factor_returns_df: DataFrame with factor returns
            
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
        
        # Use LEFT merge to keep all stocks with exposures, then filter to those with returns
        # This ensures we don't lose stocks that have exposures but might have missing returns
        merged = exposures_reset.merge(
            returns_prep[['SECURITY_ID', 'DATE', 'ONE_DAY_PCT']],
            on=['SECURITY_ID', 'DATE'],
            how='left'  # Keep all exposures
        )
        
        # Filter to only stocks that have returns (can't calculate specific return without return)
        merged = merged[merged['ONE_DAY_PCT'].notna()]
        
        if len(merged) == 0:
            print("   ⚠ WARNING: No stocks with both exposures and returns found")
            return pd.DataFrame(columns=['MODEL', 'DATE', 'SECURITY_ID', 'SPECIFIC_RETURN'])
        
        # Get factor names (exclude metadata columns)
        factor_names = [col for col in exposures_reset.columns 
                       if col not in ['SECURITY_ID', 'DATE', 'MODEL', 'ONE_DAY_PCT']]
        specific_returns_list = []
        
        # Calculate specific returns for each date and stock
        for date in merged['DATE'].unique():
            date_data = merged[merged['DATE'] == date].copy()
            date_factor_returns = factor_returns_df[factor_returns_df['DATE'] == date]
            
            if len(date_factor_returns) == 0:
                continue
            
            # Create factor returns dictionary
            factor_returns_dict = dict(zip(
                date_factor_returns['FACTOR_NAME'],
                date_factor_returns['RETURN']
            ))
            
            # Calculate explained return: Σ(k) β_i,k * f_k
            for idx, row in date_data.iterrows():
                explained_return = sum(
                    row[factor] * factor_returns_dict.get(factor, 0)
                    for factor in factor_names
                    if not pd.isna(row[factor])
                )
                
                # Specific return = actual return - explained return
                specific_return = row['ONE_DAY_PCT'] - explained_return
                
                specific_returns_list.append({
                    'MODEL': config.MODEL_NAME,
                    'DATE': date,
                    'SECURITY_ID': row['SECURITY_ID'],
                    'SPECIFIC_RETURN': specific_return
                })
        
        specific_returns_df = pd.DataFrame(specific_returns_list)
        return specific_returns_df
    
    def calculate_factor_covariance(self, factor_returns_df: pd.DataFrame,
                                    lookback_window: int = 60) -> pd.DataFrame:
        """
        Calculate factor covariance matrix using rolling window
        F_kl = Cov_t(f_k,t, f_l,t)
        
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
        
        # Calculate rolling covariance
        covariances = []
        factor_names = factor_returns_wide.columns.tolist()
        
        for date in factor_returns_wide.index:
            # Get rolling window
            window_data = factor_returns_wide.loc[
                factor_returns_wide.index <= date
            ].tail(lookback_window)
            
            if len(window_data) < lookback_window // 2:
                continue
            
            # Calculate covariance matrix
            cov_matrix = window_data.cov()
            
            # Store pairwise covariances
            for i, factor1 in enumerate(factor_names):
                for j, factor2 in enumerate(factor_names[i:], start=i):
                    covariances.append({
                        'MODEL': config.MODEL_NAME,
                        'DATE': date,
                        'FACTOR_NAME_1': factor1,
                        'FACTOR_NAME_2': factor2,
                        'COVARIANCE': cov_matrix.loc[factor1, factor2]
                    })
        
        covariance_df = pd.DataFrame(covariances)
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
        4. Specific variance: SPECIFIC_VAR_{i,T} = max(Var(r_i)_T - σ²_factor,i(T), ε)
        
        Args:
            exposures_df: DataFrame with factor exposures (index: FACTSET_ID, DATE)
            returns_df: DataFrame with stock returns (columns: FACTSET_ID, DATE, ONE_DAY_PCT)
            factor_returns_df: DataFrame with factor returns (columns: DATE, FACTOR_NAME, RETURN)
            window: Rolling window size (default: 60 days)
            epsilon: Floor value for specific variance to avoid negatives (default: 1e-8)
            
        Returns:
            DataFrame with columns: MODEL, DATE, SECURITY_ID, SPECIFIC_VAR
            - SPECIFIC_VAR: Specific variance = TOTAL_VAR - FACTOR_VAR
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
            date_idx = factor_returns_wide.index.get_loc(date_T)
            if date_idx < window - 1:
                # Not enough history
                continue
            
            window_start_idx = max(0, date_idx - window + 1)
            factor_returns_window = factor_returns_wide.iloc[window_start_idx:date_idx+1]
            
            if len(factor_returns_window) < window // 2:
                # Not enough data
                continue
            
            # 1. Calculate factor covariance matrix Σ_f(T) (once per date, not per stock)
            # Use only non-intercept factors for covariance
            factor_returns_for_cov = factor_returns_window[factor_names_for_cov]
            factor_cov_matrix = factor_returns_for_cov.cov().values
            
            # Create mapping from factor name to index
            factor_idx_map = {f: idx for idx, f in enumerate(factor_names_for_cov)}
            
            # OPTIMIZATION: Pre-filter returns by date range once (not per stock)
            window_start_date = factor_returns_window.index[0]
            returns_window = returns_prep[
                (returns_prep['DATE'] >= window_start_date) & 
                (returns_prep['DATE'] <= date_T)
            ].copy()
            
            # OPTIMIZATION: Group returns by security_id for faster lookups
            returns_by_security = returns_window.groupby('SECURITY_ID')['ONE_DAY_PCT']
            
            # 2. For each stock, calculate total variance and factor variance
            # OPTIMIZATION: Iterate over rows but use pre-filtered returns
            for _, stock_row in exposures_T.iterrows():
                security_id = stock_row['SECURITY_ID']
                
                # OPTIMIZATION: Get stock returns from pre-filtered and grouped data (O(1) lookup)
                try:
                    stock_returns_series = returns_by_security.get_group(security_id).dropna()
                except KeyError:
                    # Stock not in returns for this window
                    continue
                
                if len(stock_returns_series) < window // 2:
                    # Not enough return history
                    continue
                
                # Calculate total variance: Var_60(r_{i,t})
                total_var = stock_returns_series.var()
                
                if pd.isna(total_var) or total_var <= 0:
                    continue
                
                # 3. Calculate factor variance: β_{i,T}^T * Σ_f(T) * β_{i,T}
                # Get exposure vector for this stock (only non-intercept factors)
                beta_vector = []
                beta_indices = []
                for f in factor_names_for_cov:
                    if f in stock_row.index and not pd.isna(stock_row[f]):
                        beta_vector.append(stock_row[f])
                        beta_indices.append(factor_idx_map[f])
                
                if len(beta_vector) == 0:
                    continue
                
                beta_vector = np.array(beta_vector)
                beta_indices = np.array(beta_indices)
                
                # Extract relevant submatrix of covariance
                factor_cov_submatrix = factor_cov_matrix[np.ix_(beta_indices, beta_indices)]
                
                # Calculate factor variance: β^T * Σ_f * β
                factor_var = beta_vector.T @ factor_cov_submatrix @ beta_vector
                
                if pd.isna(factor_var) or factor_var < 0:
                    factor_var = 0
                
                # 4. Calculate specific variance: max(Var(r_i)_T - σ²_factor,i(T), ε)
                specific_var = max(total_var - factor_var, epsilon)
                
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
    
    def compare_factor_returns_with_sp500(self, factor_returns_df: pd.DataFrame) -> Dict:
        """
        Compare factor returns with S&P500 returns
        
        Args:
            factor_returns_df: DataFrame with factor returns
            
        Returns:
            Dictionary with comparison metrics for each factor
        """
        # Pivot factor returns to wide format
        factor_returns_wide = factor_returns_df.pivot_table(
            index='DATE',
            columns='FACTOR_NAME',
            values='RETURN'
        ).sort_index()
        
        # Align with S&P500
        common_dates = factor_returns_wide.index.intersection(self.sp500_returns.index)
        if len(common_dates) == 0:
            return {'error': 'No overlapping dates with S&P500'}
        
        factor_returns_aligned = factor_returns_wide.loc[common_dates]
        sp500_aligned = self.sp500_returns.loc[common_dates]
        
        # Calculate metrics for each factor
        comparisons = {}
        for factor_name in factor_returns_wide.columns:
            factor_returns = factor_returns_aligned[factor_name].dropna()
            
            if len(factor_returns) == 0:
                continue
            
            # Align dates
            common_dates_factor = factor_returns.index.intersection(sp500_aligned.index)
            if len(common_dates_factor) == 0:
                continue
            
            factor_aligned = factor_returns.loc[common_dates_factor]
            sp500_aligned_factor = sp500_aligned.loc[common_dates_factor]
            
            # Calculate correlation
            correlation = np.corrcoef(factor_aligned, sp500_aligned_factor)[0, 1]
            
            comparisons[factor_name] = {
                'correlation': correlation,
                'correlation_squared': correlation ** 2,
                'mse': np.mean((factor_aligned - sp500_aligned_factor) ** 2),
                'mae': np.mean(np.abs(factor_aligned - sp500_aligned_factor)),
                'sharpe_ratio': np.mean(factor_aligned) / np.std(factor_aligned) if np.std(factor_aligned) > 0 else 0,
                'cumulative_return': (1 + factor_aligned).prod() - 1,
                'sp500_cumulative_return': (1 + sp500_aligned_factor).prod() - 1
            }
        
        return comparisons
    
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



