"""
Exposure Pipeline Module - Stage B: Per-Date Processing
Computes factor exposures for a single date using preloaded data
"""
import pandas as pd
import numpy as np
from typing import Optional
from datetime import timedelta
import config
from continent_mapping import get_continent


def compute_exposures_for_date(
    current_date: pd.Timestamp,
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    neutralize: bool = False
) -> Optional[pd.DataFrame]:
    """
    For a given date T, slice preloaded DataFrames,
    construct factor exposures for the full universe on T.
    
    Args:
        current_date: Target date as pd.Timestamp
        universe: DataFrame with FACTSET_ID, SECTOR, CONTINENT/COUNTRY
        fundamentals: Preloaded fundamentals DataFrame
        prices: Preloaded prices DataFrame (must include IS_HOLIDAY)
        returns: Preloaded returns DataFrame
        neutralize: Whether to neutralize style factors
        
    Returns:
        DataFrame with columns: FACTSET_ID, DATE, and all factor exposures
        Returns None if date is a holiday or has insufficient data
    """
    # (a) FILTER OUT HOLIDAYS
    prices_T = prices[prices['DATE'] == current_date].copy()
    
    if len(prices_T) == 0:
        return None
    
    # Check if all rows are holidays
    if 'IS_HOLIDAY' in prices_T.columns:
        if prices_T['IS_HOLIDAY'].all():
            return None  # Skip holiday dates
    
    # (b) SLICE DATA FOR DATE T AND LOOKBACK WINDOWS
    lookback_mom = timedelta(days=365)  # Safe buffer for 12-1 momentum
    lookback_vol = timedelta(days=90)   # Buffer for 60-day volatility
    lookback_liq = timedelta(days=40)   # Buffer for 20-day liquidity
    
    lookback_start = current_date - max(lookback_mom, lookback_vol, lookback_liq)
    
    # Slice fundamentals for current date
    fundamentals_T = fundamentals[
        (fundamentals['DATE'] == current_date)
    ].copy()
    
    if len(fundamentals_T) == 0:
        return None
    
    # Slice returns and prices for lookback window
    returns_window = returns[
        (pd.to_datetime(returns['P_DATE']) >= lookback_start) &
        (pd.to_datetime(returns['P_DATE']) <= current_date)
    ].copy()
    
    prices_window = prices[
        (prices['DATE'] >= lookback_start) &
        (prices['DATE'] <= current_date)
    ].copy()
    
    # (c) MERGE UNIVERSE + FUNDAMENTALS + PRICES FOR DATE T
    # Merge universe with fundamentals
    df_T = fundamentals_T.merge(
        universe[['FACTSET_ID', 'SECTOR', 'CONTINENT']],
        on='FACTSET_ID',
        how='inner'
    )
    
    # Merge with prices for current date
    prices_T_clean = prices_T[['FACTSET_ID', 'ADJUSTED_PRICE', 'ADJUSTED_VOLUME']].copy()
    df_T = df_T.merge(
        prices_T_clean,
        on='FACTSET_ID',
        how='inner'
    )
    
    if len(df_T) == 0:
        return None
    
    # (d) CONSTRUCT RAW CHARACTERISTICS
    # Value characteristics
    df_T['EY_NTM'] = df_T['EPS_NTM'] / df_T['ADJUSTED_PRICE']
    df_T['SY_NTM'] = df_T['SALES_NTM'] / df_T['ADJUSTED_PRICE']
    df_T['EBITDA_Y_NTM'] = df_T['EBITDA_NTM'] / df_T['ADJUSTED_PRICE']
    df_T['EY_LTM'] = df_T['EPS_LTM'] / df_T['ADJUSTED_PRICE']
    df_T['SY_LTM'] = df_T['SALES_LTM'] / df_T['ADJUSTED_PRICE']
    
    # Apply qualitative rules
    df_T.loc[df_T['EPS_NTM'] <= 0, 'EY_NTM'] = np.nan
    df_T.loc[df_T['EBITDA_NTM'] <= 0, 'EBITDA_Y_NTM'] = np.nan
    df_T.loc[df_T['EPS_LTM'] <= 0, 'EY_LTM'] = np.nan
    
    # Profitability characteristics
    df_T['EBITDA_MARGIN'] = df_T['EBITDA_LTM'] / df_T['SALES_LTM']
    df_T['GROSS_MARGIN'] = 1 - (df_T['COGS_LTM'] / df_T['SALES_LTM'])
    
    df_T.loc[df_T['SALES_LTM'] <= 0, 'EBITDA_MARGIN'] = np.nan
    df_T.loc[df_T['SALES_LTM'] <= 0, 'GROSS_MARGIN'] = np.nan
    df_T.loc[df_T['COGS_LTM'] <= 0, 'GROSS_MARGIN'] = np.nan
    
    # Growth characteristics
    df_T['EPS_GROWTH'] = (df_T['EPS_NTM'] / df_T['EPS_LTM']) - 1
    df_T['SALES_GROWTH'] = (df_T['SALES_NTM'] / df_T['SALES_LTM']) - 1
    
    df_T.loc[df_T['EPS_LTM'] <= 0, 'EPS_GROWTH'] = np.nan
    df_T.loc[df_T['SALES_LTM'] <= 0, 'SALES_GROWTH'] = np.nan
    
    # Handle division by zero
    df_T = df_T.replace([np.inf, -np.inf], np.nan)
    
    # Momentum (12-1 month)
    df_T['MOMENTUM'] = np.nan
    for factset_id in df_T['FACTSET_ID'].unique():
        stock_returns = returns_window[returns_window['FSYM_ID'] == factset_id].copy()
        if len(stock_returns) < 252:
            continue
        
        stock_returns = stock_returns.sort_values('P_DATE')
        stock_returns['DATE'] = pd.to_datetime(stock_returns['P_DATE'])
        
        # Calculate cumulative return excluding last 21 days
        if len(stock_returns) >= 273:  # 252 + 21
            ret_12m = (1 + stock_returns.iloc[:-21]['ONE_DAY_PCT']).prod() - 1
            ret_1m = (1 + stock_returns.iloc[-21:]['ONE_DAY_PCT']).prod() - 1
            df_T.loc[df_T['FACTSET_ID'] == factset_id, 'MOMENTUM'] = ret_12m - ret_1m
    
    # Volatility (60-day)
    df_T['VOLATILITY'] = np.nan
    for factset_id in df_T['FACTSET_ID'].unique():
        stock_returns = returns_window[returns_window['FSYM_ID'] == factset_id].copy()
        if len(stock_returns) < 60:
            continue
        
        stock_returns = stock_returns.sort_values('P_DATE').tail(60)
        df_T.loc[df_T['FACTSET_ID'] == factset_id, 'VOLATILITY'] = stock_returns['ONE_DAY_PCT'].std()
    
    # Liquidity (log 20-day avg dollar volume)
    df_T['LIQUIDITY'] = np.nan
    for factset_id in df_T['FACTSET_ID'].unique():
        stock_prices = prices_window[
            (prices_window['FACTSET_ID'] == factset_id) &
            (~prices_window['ADJUSTED_PRICE'].isna()) &
            (~prices_window['ADJUSTED_VOLUME'].isna())
        ].copy()
        
        if len(stock_prices) < 20:
            continue
        
        stock_prices = stock_prices.sort_values('DATE').tail(20)
        dollar_vol = (stock_prices['ADJUSTED_PRICE'] * stock_prices['ADJUSTED_VOLUME']).mean()
        if dollar_vol > 0:
            df_T.loc[df_T['FACTSET_ID'] == factset_id, 'LIQUIDITY'] = np.log(dollar_vol)
    
    # (e) WINSORIZE CHARACTERISTICS CROSS-SECTIONALLY
    winsorize_lower = config.FACTOR_PARAMS['winsorize_lower']
    winsorize_upper = config.FACTOR_PARAMS['winsorize_upper']
    
    value_chars = ['EY_NTM', 'SY_NTM', 'EBITDA_Y_NTM', 'EY_LTM', 'SY_LTM']
    profitability_chars = ['EBITDA_MARGIN', 'GROSS_MARGIN']
    growth_chars = ['EPS_GROWTH', 'SALES_GROWTH']
    
    all_chars = value_chars + profitability_chars + growth_chars + ['MOMENTUM', 'VOLATILITY', 'LIQUIDITY']
    
    for char in all_chars:
        if char not in df_T.columns:
            continue
        values = df_T[char].dropna()
        if len(values) > 0:
            lower_bound = values.quantile(winsorize_lower)
            upper_bound = values.quantile(winsorize_upper)
            df_T[char] = df_T[char].clip(lower=lower_bound, upper=upper_bound)
    
    # (f) Z-SCORE (STANDARDIZE) CHARACTERISTICS CROSS-SECTIONALLY
    for char in all_chars:
        if char not in df_T.columns:
            continue
        values = df_T[char].dropna()
        if len(values) > 1:
            mean_val = values.mean()
            std_val = values.std()
            if std_val > 0:
                df_T[char] = (df_T[char] - mean_val) / std_val
    
    # (g) COMBINE CHARACTERISTICS INTO STYLE FACTORS
    # Value Factor
    value_avail = df_T[value_chars].notna().sum(axis=1)
    df_T['VALUE'] = df_T[value_chars].mean(axis=1, skipna=True)
    if df_T['VALUE'].notna().sum() > 1:
        val_mean = df_T['VALUE'].mean()
        val_std = df_T['VALUE'].std()
        if val_std > 0:
            df_T['VALUE'] = (df_T['VALUE'] - val_mean) / val_std
    
    # Profitability Factor
    prof_avail = df_T[profitability_chars].notna().sum(axis=1)
    df_T['PROFITABILITY'] = df_T[profitability_chars].mean(axis=1, skipna=True)
    if df_T['PROFITABILITY'].notna().sum() > 1:
        prof_mean = df_T['PROFITABILITY'].mean()
        prof_std = df_T['PROFITABILITY'].std()
        if prof_std > 0:
            df_T['PROFITABILITY'] = (df_T['PROFITABILITY'] - prof_mean) / prof_std
    
    # Growth Factor
    growth_avail = df_T[growth_chars].notna().sum(axis=1)
    df_T['GROWTH'] = df_T[growth_chars].mean(axis=1, skipna=True)
    if df_T['GROWTH'].notna().sum() > 1:
        growth_mean = df_T['GROWTH'].mean()
        growth_std = df_T['GROWTH'].std()
        if growth_std > 0:
            df_T['GROWTH'] = (df_T['GROWTH'] - growth_mean) / growth_std
    
    # Momentum, Volatility, Liquidity are already standardized
    
    # (h) CREATE SECTOR AND CONTINENT DUMMIES
    # Create dummy variables
    sector_dummies = pd.get_dummies(df_T['SECTOR'], prefix='SECTOR')
    continent_dummies = pd.get_dummies(df_T['CONTINENT'], prefix='CONTINENT')
    
    # Convert to float for sum-to-zero transform
    sector_dummies = sector_dummies.astype(float)
    continent_dummies = continent_dummies.astype(float)
    
    # (i) ENFORCE SUM-TO-ZERO ON DUMMIES
    for col in sector_dummies.columns:
        col_mean = sector_dummies[col].mean()
        sector_dummies[col] = sector_dummies[col] - col_mean
    
    for col in continent_dummies.columns:
        col_mean = continent_dummies[col].mean()
        continent_dummies[col] = continent_dummies[col] - col_mean
    
    # Combine all exposures
    style_factors = df_T[['FACTSET_ID', 'VALUE', 'PROFITABILITY', 'GROWTH', 
                          'MOMENTUM', 'VOLATILITY', 'LIQUIDITY']].copy()
    
    # Add DATE column
    style_factors['DATE'] = current_date
    
    # Combine with dummies
    exposures_df = pd.concat([
        style_factors.set_index('FACTSET_ID'),
        sector_dummies.set_index(style_factors.index),
        continent_dummies.set_index(style_factors.index)
    ], axis=1).reset_index()
    
    # Reorder columns: FACTSET_ID, DATE, then all factors
    factor_cols = [col for col in exposures_df.columns if col not in ['FACTSET_ID', 'DATE']]
    exposures_df = exposures_df[['FACTSET_ID', 'DATE'] + factor_cols]
    
    return exposures_df

