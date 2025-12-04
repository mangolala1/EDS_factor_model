"""
Factor construction module for MSCI-styled factor model
Implements the complete workflow: Data Retrieval -> Factor Construction -> Factor Returns -> S&P500 Comparison
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
import config
from continent_mapping import get_continent
from sklearn.linear_model import LinearRegression


class FactorConstructor:
    """Class to construct factors according to MSCI methodology"""
    
    def __init__(self, fundamentals_df: pd.DataFrame, prices_df: pd.DataFrame, 
                 returns_df: pd.DataFrame, universe_df: pd.DataFrame):
        """
        Initialize with required data tables
        
        Args:
            fundamentals_df: DataFrame with NTM/LTM fundamental data
            prices_df: DataFrame with split-adjusted prices
            returns_df: DataFrame with daily returns
            universe_df: DataFrame with universe metadata (continent, sector, industry)
        """
        self.fundamentals_df = fundamentals_df.copy()
        self.prices_df = prices_df.copy()
        self.returns_df = returns_df.copy()
        self.universe_df = universe_df.copy()
        
        # Prepare and align data
        self._prepare_data()
        
        # Store constructed exposures
        self.exposures = {}
    
    def _prepare_data(self):
        """Prepare and align all data tables"""
        # Convert DATE columns to datetime
        for df_name in ['fundamentals_df', 'prices_df', 'returns_df']:
            df = getattr(self, df_name)
            if 'DATE' in df.columns:
                df['DATE'] = pd.to_datetime(df['DATE'])
            elif 'P_DATE' in df.columns:
                df['P_DATE'] = pd.to_datetime(df['P_DATE'])
        
        # Merge universe data with fundamentals
        if 'FACTSET_ID' in self.fundamentals_df.columns:
            self.fundamentals_df = self.fundamentals_df.merge(
                self.universe_df, on='FACTSET_ID', how='left'
            )
        
            # Map countries to continents
            if 'COUNTRY' in self.fundamentals_df.columns:
                self.fundamentals_df['CONTINENT'] = self.fundamentals_df['COUNTRY'].apply(get_continent)
        
        # Also add continent to universe_df for later use
        if 'COUNTRY' in self.universe_df.columns:
            self.universe_df['CONTINENT'] = self.universe_df['COUNTRY'].apply(get_continent)
        
        print("Data preparation complete")
    
    def winsorize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize characteristics at chosen percentiles (e.g., 1% and 99%)
        X_i,k,t^win = min(max(X_i,k,t, q_k,t^1%), q_k,t^99%)
        """
        lower_pct = config.FACTOR_PARAMS['winsorize_lower']
        upper_pct = config.FACTOR_PARAMS['winsorize_upper']
        
        winsorized_df = df.copy()
        
        # Winsorize cross-sectionally for each date
        for date in df.index.get_level_values('DATE').unique():
            date_mask = df.index.get_level_values('DATE') == date
            date_data = df[date_mask]
            
            for col in date_data.columns:
                values = date_data[col].dropna()
                if len(values) > 0:
                    lower_bound = values.quantile(lower_pct)
                    upper_bound = values.quantile(upper_pct)
                    
                    # Clip values
                    winsorized_df.loc[date_mask, col] = winsorized_df.loc[date_mask, col].clip(
                        lower=lower_bound, upper=upper_bound
                    )
        
        return winsorized_df
    
    def standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize characteristics cross-sectionally (z-scores)
        β_i,k,t = (X_i,k,t^win - μ_k,t) / σ_k,t
        """
        standardized_df = df.copy()
        
        # Standardize cross-sectionally for each date
        for date in df.index.get_level_values('DATE').unique():
            date_mask = df.index.get_level_values('DATE') == date
            date_data = df[date_mask]
            
            for col in date_data.columns:
                values = date_data[col].dropna()
                if len(values) > 1:
                    mean = values.mean()
                    std = values.std()
                    
                    if std > 0:
                        standardized_df.loc[date_mask, col] = (
                            (standardized_df.loc[date_mask, col] - mean) / std
                        )
                    else:
                        standardized_df.loc[date_mask, col] = 0.0
        
        return standardized_df
    
    def _merge_price_fundamentals(self) -> pd.DataFrame:
        """Merge price and fundamental data"""
        # Merge on FACTSET_ID and DATE
        merged = self.fundamentals_df.merge(
            self.prices_df,
            on=['FACTSET_ID', 'DATE'],
            how='inner'
        )
        merged = merged.sort_values(['FACTSET_ID', 'DATE'])
        return merged
    
    # ==================== VALUE FACTOR ====================
    
    def construct_value_factor(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct Value factor with composite exposure
        
        Characteristics:
        1. Forward Earnings Yield: EY_NTM = EPS_NTM / ADJUSTED_PRICE
        2. Forward Sales Yield: SY_NTM = SALES_NTM / ADJUSTED_PRICE
        3. Forward EBITDA Yield: EBITDA_Y_NTM = EBITDA_NTM / ADJUSTED_PRICE
        4. Earnings-to-Price LTM: EY_LTM = EPS_LTM / ADJUSTED_PRICE
        5. Sales-to-Price LTM: SY_LTM = SALES_LTM / ADJUSTED_PRICE
        
        Qualitative Rules:
        - If EPS_NTM <= 0, ignore earnings yield
        - If EBITDA_NTM <= 0, ignore EBITDA yield
        - If EPS_LTM <= 0, ignore LTM earnings yield
        
        Process: Winsorize each → Standardize each → Average available → Standardize composite
        """
        df = merged_df[['FACTSET_ID', 'DATE', 'EPS_NTM', 'EPS_LTM', 'SALES_NTM', 'SALES_LTM',
                        'EBITDA_NTM', 'ADJUSTED_PRICE']].copy()
        
        # Calculate all 5 characteristics
        df['EY_NTM'] = df['EPS_NTM'] / df['ADJUSTED_PRICE']
        df['SY_NTM'] = df['SALES_NTM'] / df['ADJUSTED_PRICE']
        df['EBITDA_Y_NTM'] = df['EBITDA_NTM'] / df['ADJUSTED_PRICE']
        df['EY_LTM'] = df['EPS_LTM'] / df['ADJUSTED_PRICE']
        df['SY_LTM'] = df['SALES_LTM'] / df['ADJUSTED_PRICE']
        
        # Apply qualitative rules: Set to NaN if conditions not met
        df.loc[df['EPS_NTM'] <= 0, 'EY_NTM'] = np.nan
        df.loc[df['EBITDA_NTM'] <= 0, 'EBITDA_Y_NTM'] = np.nan
        df.loc[df['EPS_LTM'] <= 0, 'EY_LTM'] = np.nan
        
        # Handle division by zero
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Set index
        df = df.set_index(['FACTSET_ID', 'DATE'])
        value_chars = ['EY_NTM', 'SY_NTM', 'EBITDA_Y_NTM', 'EY_LTM', 'SY_LTM']
        
        # Winsorize each characteristic
        df_winsorized = self.winsorize(df[value_chars])
        
        # Standardize each characteristic
        df_standardized = self.standardize(df_winsorized)
        
        # Average available standardized characteristics for each stock-date
        composite = df_standardized.mean(axis=1, skipna=True)
        
        # Standardize the composite again
        composite_df = pd.DataFrame({'VALUE': composite})
        composite_standardized = self.standardize(composite_df)
        
        return composite_standardized
    
    # ==================== PROFITABILITY FACTOR ====================
    
    def construct_profitability_factor(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct Profitability factor with composite exposure
        
        Characteristics:
        1. EBITDA Margin: EBITDA_M = EBITDA_LTM / SALES_LTM
        2. Gross Margin: GM = 1 - (COGS_LTM / SALES_LTM)
        
        Qualitative Rules:
        - If SALES_LTM <= 0, skip both margins
        - If COGS_LTM <= 0, exclude gross margin
        
        Process: Winsorize each → Standardize each → Average available → Standardize composite
        """
        df = merged_df[['FACTSET_ID', 'DATE', 'EBITDA_LTM', 'SALES_LTM', 'COGS_LTM']].copy()
        
        # Calculate margins
        df['EBITDA_MARG'] = df['EBITDA_LTM'] / df['SALES_LTM']
        df['GROSS_MARG'] = 1 - (df['COGS_LTM'] / df['SALES_LTM'])
        
        # Apply qualitative rules
        invalid_sales = df['SALES_LTM'] <= 0
        df.loc[invalid_sales, 'EBITDA_MARG'] = np.nan
        df.loc[invalid_sales, 'GROSS_MARG'] = np.nan
        
        # If COGS_LTM <= 0, exclude gross margin
        df.loc[df['COGS_LTM'] <= 0, 'GROSS_MARG'] = np.nan
        
        # Handle division by zero
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Set index
        df = df.set_index(['FACTSET_ID', 'DATE'])
        profitability_chars = ['EBITDA_MARG', 'GROSS_MARG']
        
        # Winsorize each characteristic
        df_winsorized = self.winsorize(df[profitability_chars])
        
        # Standardize each characteristic
        df_standardized = self.standardize(df_winsorized)
        
        # Average available standardized characteristics
        composite = df_standardized.mean(axis=1, skipna=True)
        
        # Standardize the composite again
        composite_df = pd.DataFrame({'PROFITABILITY': composite})
        composite_standardized = self.standardize(composite_df)
        
        return composite_standardized
    
    # ==================== GROWTH FACTOR ====================
    
    def construct_growth_factor(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct Growth factor with composite exposure
        
        Characteristics:
        1. EPS Growth: EPSG = (EPS_NTM / EPS_LTM) - 1
        2. Sales Growth: SG = (SALES_NTM / SALES_LTM) - 1
        
        Qualitative Rules:
        - If EPS_LTM <= 0, ignore EPS growth
        - If SALES_LTM <= 0, ignore sales growth
        
        Process: Winsorize each → Standardize each → Average available → Standardize composite
        """
        df = merged_df[['FACTSET_ID', 'DATE', 'EPS_NTM', 'EPS_LTM', 
                        'SALES_NTM', 'SALES_LTM']].copy()
        
        # Calculate growth rates
        df['EPS_GROWTH'] = (df['EPS_NTM'] / df['EPS_LTM']) - 1
        df['SALES_GROWTH'] = (df['SALES_NTM'] / df['SALES_LTM']) - 1
        
        # Apply qualitative rules
        df.loc[df['EPS_LTM'] <= 0, 'EPS_GROWTH'] = np.nan
        df.loc[df['SALES_LTM'] <= 0, 'SALES_GROWTH'] = np.nan
        
        # Handle division by zero
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Set index
        df = df.set_index(['FACTSET_ID', 'DATE'])
        growth_chars = ['EPS_GROWTH', 'SALES_GROWTH']
        
        # Winsorize each characteristic
        df_winsorized = self.winsorize(df[growth_chars])
        
        # Standardize each characteristic
        df_standardized = self.standardize(df_winsorized)
        
        # Average available standardized characteristics
        composite = df_standardized.mean(axis=1, skipna=True)
        
        # Standardize the composite again
        composite_df = pd.DataFrame({'GROWTH': composite})
        composite_standardized = self.standardize(composite_df)
        
        return composite_standardized
    
    # ==================== MOMENTUM FACTOR ====================
    
    def construct_momentum_factor(self) -> pd.DataFrame:
        """
        Construct Momentum factor (single characteristic)
        
        Characteristic:
        1. 12-1 Month Momentum: Rolling cumulative return from month -12 to month -1
        
        Rules:
        - If price history < 252 days → no momentum
        - Always exclude last 21 days to avoid short-term reversal
        
        Process: Winsorize → Standardize (no composite needed, single characteristic)
        """
        # Prepare returns data
        returns_wide = self.returns_df.pivot_table(
            index='P_DATE',
            columns='FSYM_ID',
            values='ONE_DAY_PCT',
            aggfunc='first'
        )
        
        # Align with FACTSET_ID if needed (FSYM_ID might need mapping)
        # For now, assume FSYM_ID maps to FACTSET_ID
        returns_wide = returns_wide.sort_index()
        
        lookback = config.FACTOR_PARAMS['momentum_lookback']  # 252 days
        exclude = config.FACTOR_PARAMS['momentum_exclude']  # 21 days
        
        momentum_values = []
        
        for date in returns_wide.index:
            # Get returns from t-252 to t-21
            date_idx = returns_wide.index.get_loc(date)
            start_idx = max(0, date_idx - lookback)
            end_idx = max(0, date_idx - exclude)
            
            if end_idx > start_idx and (end_idx - start_idx) >= 200:  # Need at least ~200 days
                window_returns = returns_wide.iloc[start_idx:end_idx]
                # Calculate cumulative return using log returns
                log_returns = np.log1p(window_returns)
                cumulative_log = log_returns.sum()
                momentum = np.expm1(cumulative_log)
                
                momentum_series = pd.Series(momentum, index=window_returns.columns)
                momentum_series.name = date
                momentum_values.append(momentum_series)
            else:
                # Not enough data
                momentum_series = pd.Series(np.nan, index=returns_wide.columns)
                momentum_series.name = date
                momentum_values.append(momentum_series)
        
        momentum_df = pd.DataFrame(momentum_values)
        momentum_df.index.name = 'DATE'
        momentum_df = momentum_df.stack().reset_index()
        momentum_df.columns = ['DATE', 'FSYM_ID', 'MOM12_1']
        
        # Convert to FACTSET_ID if mapping needed
        momentum_df = momentum_df.rename(columns={'FSYM_ID': 'FACTSET_ID'})
        momentum_df = momentum_df.set_index(['FACTSET_ID', 'DATE'])
        
        # Winsorize
        momentum_winsorized = self.winsorize(momentum_df[['MOM12_1']])
        
        # Standardize
        momentum_standardized = self.standardize(momentum_winsorized)
        
        # Rename to MOMENTUM
        momentum_standardized = momentum_standardized.rename(columns={'MOM12_1': 'MOMENTUM'})
        
        return momentum_standardized
    
    # ==================== VOLATILITY FACTOR ====================
    
    def construct_volatility_factor(self) -> pd.DataFrame:
        """
        Construct Volatility factor (single characteristic)
        
        Characteristic:
        1. 60-Day Rolling Volatility: Standard deviation of daily returns over past 60 days
        
        Rules:
        - If returns < 30 days → skip
        
        Process: Winsorize → Standardize (no composite needed, single characteristic)
        """
        returns_wide = self.returns_df.pivot_table(
            index='P_DATE',
            columns='FSYM_ID',
            values='ONE_DAY_PCT',
            aggfunc='first'
        )
        returns_wide = returns_wide.sort_index()
        
        lookback = config.FACTOR_PARAMS['volatility_lookback']  # 60 days
        
        # Calculate rolling volatility
        rolling_std = returns_wide.rolling(window=lookback).std()
        
        # Stack to long format
        volatility_df = rolling_std.stack().reset_index()
        volatility_df.columns = ['DATE', 'FSYM_ID', 'VOL60']
        volatility_df = volatility_df.rename(columns={'FSYM_ID': 'FACTSET_ID'})
        volatility_df['DATE'] = pd.to_datetime(volatility_df['DATE'])
        volatility_df = volatility_df.set_index(['FACTSET_ID', 'DATE'])
        
        # Remove if less than 30 days of data (set to NaN)
        # This is handled by rolling which returns NaN for insufficient windows
        
        # Winsorize
        volatility_winsorized = self.winsorize(volatility_df[['VOL60']])
        
        # Standardize
        volatility_standardized = self.standardize(volatility_winsorized)
        
        # Rename to VOLATILITY
        volatility_standardized = volatility_standardized.rename(columns={'VOL60': 'VOLATILITY'})
        
        return volatility_standardized
    
    # ==================== LIQUIDITY FACTOR ====================
    
    def construct_liquidity_factor(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construct Liquidity factor (single characteristic)
        
        Characteristic:
        1. Log Dollar Volume: LIQ = ln(mean(DV_{t-19:t}))
           where DV = ADJUSTED_PRICE * ADJUSTED_VOLUME
        
        Rules:
        - If volume missing → skip
        - If price missing → skip
        
        Process: Winsorize → Standardize (no composite needed, single characteristic)
        """
        df = merged_df[['FACTSET_ID', 'DATE', 'ADJUSTED_PRICE', 'ADJUSTED_VOLUME']].copy()
        
        df = df.sort_values(['FACTSET_ID', 'DATE'])
        
        # Calculate dollar volume
        df['DOLLARVOL'] = df['ADJUSTED_PRICE'] * df['ADJUSTED_VOLUME']
        
        # Set missing values to NaN (price or volume missing)
        df.loc[df['ADJUSTED_PRICE'].isna(), 'DOLLARVOL'] = np.nan
        df.loc[df['ADJUSTED_VOLUME'].isna(), 'DOLLARVOL'] = np.nan
        
        # Calculate 20-day average dollar volume
        df = df.set_index(['FACTSET_ID', 'DATE'])
        lookback = config.FACTOR_PARAMS['liquidity_lookback']  # 20 days
        
        dollarvol_wide = df['DOLLARVOL'].unstack(level=0)
        dollarvol_avg = dollarvol_wide.rolling(window=lookback).mean()
        
        # Take natural logarithm (replace 0 with NaN first)
        liquidity = np.log(dollarvol_avg.replace(0, np.nan))
        
        # Stack to long format
        liquidity_df = liquidity.stack().reset_index()
        liquidity_df.columns = ['DATE', 'FACTSET_ID', 'LIQ']
        liquidity_df = liquidity_df.set_index(['FACTSET_ID', 'DATE'])
        
        # Winsorize
        liquidity_winsorized = self.winsorize(liquidity_df[['LIQ']])
        
        # Standardize
        liquidity_standardized = self.standardize(liquidity_winsorized)
        
        # Rename to LIQUIDITY
        liquidity_standardized = liquidity_standardized.rename(columns={'LIQ': 'LIQUIDITY'})
        
        return liquidity_standardized
    
    # ==================== CONTINENT AND SECTOR DUMMIES ====================
    
    def _create_continent_sector_dummies(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create continent and sector dummy variables with sum-to-zero constraint
        For each date, dummies are centered (mean = 0) cross-sectionally
        
        Args:
            merged_df: DataFrame with FACTSET_ID, DATE, CONTINENT, SECTOR
            
        Returns:
            DataFrame with continent and sector dummies (sum-to-zero), indexed by (FACTSET_ID, DATE)
        """
        # Create unique combinations of FACTSET_ID, DATE, CONTINENT, SECTOR
        dummy_data = merged_df[['FACTSET_ID', 'DATE', 'CONTINENT', 'SECTOR']].drop_duplicates()
        dummy_data = dummy_data.set_index(['FACTSET_ID', 'DATE'])
        
        # Create continent dummies (0/1)
        continent_dummies = pd.get_dummies(dummy_data['CONTINENT'], prefix='CONTINENT')
        
        # Create sector dummies (0/1)
        sector_dummies = pd.get_dummies(dummy_data['SECTOR'], prefix='SECTOR')
        
        # Combine
        dummies_df = pd.concat([continent_dummies, sector_dummies], axis=1)
        
        # Convert to float to avoid dtype warnings when applying sum-to-zero transform
        dummies_df = dummies_df.astype(float)
        
        # Apply sum-to-zero constraint: subtract mean cross-sectionally for each date
        # This ensures dummies sum to zero within each date
        for date in dummies_df.index.get_level_values('DATE').unique():
            date_mask = dummies_df.index.get_level_values('DATE') == date
            date_data = dummies_df[date_mask]
            
            # Get sector and continent dummy columns
            sector_cols = [c for c in date_data.columns if c.startswith('SECTOR_')]
            continent_cols = [c for c in date_data.columns if c.startswith('CONTINENT_')]
            
            # Subtract mean for each column (sum-to-zero)
            for col in sector_cols + continent_cols:
                if col in date_data.columns:
                    col_mean = date_data[col].mean()
                    dummies_df.loc[date_mask, col] = date_data[col] - col_mean
        
        return dummies_df
    
    # ==================== NEUTRALIZATION ====================
    
    def neutralize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Neutralize style factor exposures by continent and sector
        For each style factor (already standardized composite):
        1. Regress factor exposure on continent and sector dummies
        2. Extract residual u
        3. Apply z-score to residuals
        
        Note: This neutralization is applied to style factors only.
        Continent and sector exposures themselves are built as 0/1 dummies.
        
        Formula: β_i,k,t = Σ_j D_i,j * γ_j,k,t + u_i,k,t
                 β̃_i,k,t = z-score(u_i,k,t)
        """
        neutralized_df = df.copy()
        
        # Get continent and sector dummies from merged data
        merged_data = self.fundamentals_df[['FACTSET_ID', 'CONTINENT', 'SECTOR']].drop_duplicates()
        merged_data = merged_data.set_index('FACTSET_ID')
        
        # Identify style factors (exclude continent and sector dummies if they exist)
        style_factor_cols = [col for col in df.columns 
                            if not col.startswith('CONTINENT_') and not col.startswith('SECTOR_')]
        
        # Create continent and sector dummy matrices
        dates = df.index.get_level_values('DATE').unique()
        
        for date in dates:
            date_mask = df.index.get_level_values('DATE') == date
            date_data = df[date_mask].copy()
            
            if len(date_data) == 0:
                continue
            
            # Get FACTEST_IDs for this date (from the index)
            date_factset_ids = date_data.index.get_level_values('FACTSET_ID').tolist()
            
            # Get continents and sectors for these stocks
            continents = []
            sectors = []
            for fsid in date_factset_ids:
                if fsid in merged_data.index:
                    continents.append(merged_data.loc[fsid, 'CONTINENT'])
                    sectors.append(merged_data.loc[fsid, 'SECTOR'])
                else:
                    continents.append('Unknown')
                    sectors.append('Unknown')
            
            # Create dummy variables DataFrame aligned with date_data index
            continent_series = pd.Series(continents, index=date_data.index)
            sector_series = pd.Series(sectors, index=date_data.index)
            
            continent_dummies = pd.get_dummies(continent_series, prefix='CONTINENT')
            sector_dummies = pd.get_dummies(sector_series, prefix='SECTOR')
            
            # Combine dummies
            dummy_matrix = pd.concat([continent_dummies, sector_dummies], axis=1)
            
            # Drop reference category (first continent and first sector) to avoid multicollinearity
            if len(dummy_matrix.columns) > 0:
                # Keep all but first continent and first sector as reference
                continent_cols = [c for c in dummy_matrix.columns if c.startswith('CONTINENT_')]
                sector_cols = [c for c in dummy_matrix.columns if c.startswith('SECTOR_')]
                
                if len(continent_cols) > 1:
                    dummy_matrix = dummy_matrix.drop(columns=[continent_cols[0]])
                if len(sector_cols) > 1:
                    dummy_matrix = dummy_matrix.drop(columns=[sector_cols[0]])
            
            # For each style factor, neutralize
            for col in style_factor_cols:
                if col not in date_data.columns:
                    continue
                    
                values = date_data[col].values
                valid_mask = ~pd.isna(values)
                
                if valid_mask.sum() < max(3, len(dummy_matrix.columns) + 2):
                    # Not enough data for regression
                    continue
                
                if len(dummy_matrix.columns) == 0:
                    # No dummies available
                    continue
                
                y = values[valid_mask]
                X = dummy_matrix.iloc[valid_mask].values
                
                if len(X) == 0 or X.shape[1] == 0:
                    # No dummies available
                    continue
                
                try:
                    # Regress factor exposure on dummies
                    reg = LinearRegression(fit_intercept=True)
                    reg.fit(X, y)
                    
                    # Get predicted values
                    y_pred = reg.predict(X)
                    
                    # Get residuals
                    residuals = y - y_pred
                    
                    # Create array for all values (keeping NaN for invalid)
                    neutralized_values = date_data[col].values.copy()
                    neutralized_values[valid_mask] = residuals
                    
                    # Update in neutralized_df
                    neutralized_df.loc[date_mask, col] = neutralized_values
                    
                except Exception as e:
                    # If regression fails, keep original values
                    continue
        
        # Now standardize the neutralized residuals (z-score)
        neutralized_df = self.standardize(neutralized_df)
        
        return neutralized_df
    
    # ==================== CONSTRUCT ALL FACTORS ====================
    
    def construct_all_factors(self, neutralize: bool = False) -> pd.DataFrame:
        """
        Construct all factor exposures following MSCI methodology
        Includes style factors and continent/sector dummy factors
        
        Args:
            neutralize: Whether to neutralize style factors by continent/sector
        
        Returns:
            DataFrame with all factor exposures (style factors + continent/sector dummies)
        """
        print("\n=== Constructing Factor Exposures ===")
        
        # Merge prices with fundamentals
        merged_df = self._merge_price_fundamentals()
        
        # 1. Value Factor
        print("Constructing Value factor...")
        value_df = self.construct_value_factor(merged_df)
        
        # 2. Profitability Factor
        print("Constructing Profitability factor...")
        profitability_df = self.construct_profitability_factor(merged_df)
        
        # 3. Growth Factor
        print("Constructing Growth factor...")
        growth_df = self.construct_growth_factor(merged_df)
        
        # 4. Momentum Factor
        print("Constructing Momentum factor...")
        momentum_df = self.construct_momentum_factor()
        
        # 5. Volatility Factor
        print("Constructing Volatility factor...")
        volatility_df = self.construct_volatility_factor()
        
        # 6. Liquidity Factor
        print("Constructing Liquidity factor...")
        liquidity_df = self.construct_liquidity_factor(merged_df)
        
        # Combine all style factors (each is already a single composite exposure)
        all_factors = pd.concat([
            value_df, profitability_df, growth_df, 
            momentum_df, volatility_df, liquidity_df
        ], axis=1)
        
        # Apply neutralization if requested (only to style factors)
        if neutralize:
            print("\n=== Neutralizing Style Factors ===")
            processed_exposures = self.neutralize(all_factors)
        else:
            processed_exposures = all_factors
        
        # Create continent and sector dummy variables (0/1)
        print("\n=== Creating Continent and Sector Dummies ===")
        dummies_df = self._create_continent_sector_dummies(merged_df)
        
        # Align dummies with processed exposures (same index structure)
        aligned_dummies = dummies_df.reindex(processed_exposures.index, fill_value=0)
        
        # Combine style factors with continent/sector dummies
        all_exposures = pd.concat([processed_exposures, aligned_dummies], axis=1)
        
        print(f"✓ Created {len([c for c in aligned_dummies.columns if c.startswith('CONTINENT_')])} continent factors")
        print(f"✓ Created {len([c for c in aligned_dummies.columns if c.startswith('SECTOR_')])} sector factors")
        
        return all_exposures
    
    def construct_factors_for_date(self, date: str, neutralize: bool = False) -> pd.DataFrame:
        """
        Construct factor exposures for a single date
        
        This method uses the full historical data stored in the class (needed for 
        lookback calculations like momentum, volatility, liquidity) but only 
        returns exposures for the specified date.
        
        Args:
            date: Date in 'YYYY-MM-DD' format
            neutralize: Whether to neutralize style factors by continent/sector
            
        Returns:
            DataFrame with factor exposures for the specified date, indexed by (FACTSET_ID, DATE)
            Returns empty DataFrame if date has no data or is invalid
        """
        # Convert date to datetime
        target_date = pd.to_datetime(date)
        
        # Check if we have data for the target date
        if 'DATE' in self.fundamentals_df.columns:
            date_fundamentals = self.fundamentals_df[self.fundamentals_df['DATE'] == target_date]
        else:
            date_fundamentals = pd.DataFrame()
        
        if 'DATE' in self.prices_df.columns:
            date_prices = self.prices_df[self.prices_df['DATE'] == target_date]
        else:
            date_prices = pd.DataFrame()
        
        if len(date_fundamentals) == 0 or len(date_prices) == 0:
            return pd.DataFrame()
        
        # Construct all factors using full historical data (needed for lookback)
        # The factor construction methods work cross-sectionally by date, so they
        # will calculate exposures for all dates, but we'll filter to only the target date
        all_exposures = self.construct_all_factors(neutralize=neutralize)
        
        # Filter to only the target date
        if 'DATE' in all_exposures.index.names:
            date_exposures = all_exposures[all_exposures.index.get_level_values('DATE') == target_date]
        else:
            # If DATE is not in index, try to filter by resetting index
            all_exposures_reset = all_exposures.reset_index()
            if 'DATE' in all_exposures_reset.columns:
                all_exposures_reset['DATE'] = pd.to_datetime(all_exposures_reset['DATE'])
                date_exposures = all_exposures_reset[all_exposures_reset['DATE'] == target_date]
                if len(date_exposures) > 0:
                    date_exposures = date_exposures.set_index(['FACTSET_ID', 'DATE'])
                else:
                    date_exposures = pd.DataFrame()
            else:
                date_exposures = pd.DataFrame()
        
        return date_exposures
