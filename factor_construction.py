"""
Factor construction module for MSCI-styled factor model
"""
import pandas as pd
import numpy as np
from typing import Dict, List
import config


class FactorConstructor:
    """Class to construct various factors for the factor model"""
    
    def __init__(self, prices_df: pd.DataFrame, fundamentals_df: pd.DataFrame):
        """
        Initialize with price and fundamental data
        
        Args:
            prices_df: DataFrame with price data (columns: ticker, date, close, volume, etc.)
            fundamentals_df: DataFrame with fundamental data (columns: ticker, date, pe_ratio, etc.)
        """
        self.prices_df = prices_df.copy()
        self.fundamentals_df = fundamentals_df.copy()
        self.factors = {}
        
        # Prepare data
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare and align data for factor construction"""
        # Convert date columns to datetime
        self.prices_df['date'] = pd.to_datetime(self.prices_df['date'])
        self.fundamentals_df['date'] = pd.to_datetime(self.fundamentals_df['date'])
        
        # Set date as index for easier manipulation
        self.prices_df = self.prices_df.set_index(['ticker', 'date']).sort_index()
        self.fundamentals_df = self.fundamentals_df.set_index(['ticker', 'date']).sort_index()
    
    def construct_all_factors(self) -> pd.DataFrame:
        """
        Construct all factors and return combined DataFrame
        
        Returns:
            DataFrame with all factors
        """
        # Construct individual factors
        momentum = self.construct_momentum_factor()
        value = self.construct_value_factor()
        quality = self.construct_quality_factor()
        size = self.construct_size_factor()
        volatility = self.construct_volatility_factor()
        reversal = self.construct_reversal_factor()
        
        # Combine all factors
        factors_list = [momentum, value, quality, size, volatility, reversal]
        factors_df = pd.concat(factors_list, axis=1)
        
        # Calculate returns for each stock
        returns = self._calculate_returns()
        factors_df = factors_df.join(returns)
        
        return factors_df.dropna()
    
    def construct_momentum_factor(self) -> pd.DataFrame:
        """
        Construct momentum factor (price momentum over lookback period)
        
        Returns:
            DataFrame with momentum factor
        """
        lookback = config.FACTOR_PARAMS['momentum_lookback']
        
        # Calculate returns over lookback period
        close_prices = self.prices_df['close'].unstack(level=0)
        momentum = close_prices.pct_change(lookback)
        
        # Rank and normalize
        momentum_ranked = momentum.rank(axis=1, pct=True)
        momentum_normalized = (momentum_ranked - 0.5) * 2  # Scale to [-1, 1]
        
        # Stack back
        momentum_factor = momentum_normalized.stack().to_frame('momentum')
        momentum_factor.index.names = ['date', 'ticker']
        momentum_factor = momentum_factor.swaplevel().sort_index()
        
        return momentum_factor
    
    def construct_value_factor(self) -> pd.DataFrame:
        """
        Construct value factor based on valuation metrics
        
        Returns:
            DataFrame with value factor
        """
        value_metrics = config.FACTOR_PARAMS['value_metrics']
        value_scores = []
        
        for metric in value_metrics:
            if metric in self.fundamentals_df.columns:
                metric_data = self.fundamentals_df[metric].unstack(level=0)
                
                # For PE, PB, EV/EBITDA: lower is better (value)
                # Rank inversely and normalize
                if metric in ['pe_ratio', 'pb_ratio', 'ev_ebitda']:
                    ranked = (1 / metric_data).rank(axis=1, pct=True)
                else:
                    ranked = metric_data.rank(axis=1, pct=True)
                
                normalized = (ranked - 0.5) * 2
                value_scores.append(normalized)
        
        if value_scores:
            # Average across metrics
            value_combined = pd.concat(value_scores, axis=1).mean(axis=1)
            value_factor = value_combined.stack().to_frame('value')
            value_factor.index.names = ['date', 'ticker']
            value_factor = value_factor.swaplevel().sort_index()
        else:
            # Return zeros if no value metrics available
            dates = self.prices_df.index.get_level_values('date').unique()
            tickers = self.prices_df.index.get_level_values('ticker').unique()
            value_factor = pd.DataFrame(
                index=pd.MultiIndex.from_product([tickers, dates], names=['ticker', 'date']),
                columns=['value'],
                data=0.0
            )
        
        return value_factor
    
    def construct_quality_factor(self) -> pd.DataFrame:
        """
        Construct quality factor based on profitability and leverage metrics
        
        Returns:
            DataFrame with quality factor
        """
        quality_metrics = config.FACTOR_PARAMS['quality_metrics']
        quality_scores = []
        
        for metric in quality_metrics:
            if metric in self.fundamentals_df.columns:
                metric_data = self.fundamentals_df[metric].unstack(level=0)
                
                # For ROE, ROA: higher is better
                # For debt_to_equity: lower is better
                if metric == 'debt_to_equity':
                    ranked = (1 / (1 + metric_data)).rank(axis=1, pct=True)
                else:
                    ranked = metric_data.rank(axis=1, pct=True)
                
                normalized = (ranked - 0.5) * 2
                quality_scores.append(normalized)
        
        if quality_scores:
            quality_combined = pd.concat(quality_scores, axis=1).mean(axis=1)
            quality_factor = quality_combined.stack().to_frame('quality')
            quality_factor.index.names = ['date', 'ticker']
            quality_factor = quality_factor.swaplevel().sort_index()
        else:
            dates = self.prices_df.index.get_level_values('date').unique()
            tickers = self.prices_df.index.get_level_values('ticker').unique()
            quality_factor = pd.DataFrame(
                index=pd.MultiIndex.from_product([tickers, dates], names=['ticker', 'date']),
                columns=['quality'],
                data=0.0
            )
        
        return quality_factor
    
    def construct_size_factor(self) -> pd.DataFrame:
        """
        Construct size factor based on market capitalization
        
        Returns:
            DataFrame with size factor
        """
        size_metric = config.FACTOR_PARAMS['size_metric']
        
        if size_metric in self.fundamentals_df.columns:
            market_cap = self.fundamentals_df[size_metric].unstack(level=0)
            
            # Log transform and rank (smaller is better for size factor)
            log_mcap = np.log1p(market_cap)
            ranked = (1 / log_mcap).rank(axis=1, pct=True)
            normalized = (ranked - 0.5) * 2
            
            size_factor = normalized.stack().to_frame('size')
            size_factor.index.names = ['date', 'ticker']
            size_factor = size_factor.swaplevel().sort_index()
        else:
            dates = self.prices_df.index.get_level_values('date').unique()
            tickers = self.prices_df.index.get_level_values('ticker').unique()
            size_factor = pd.DataFrame(
                index=pd.MultiIndex.from_product([tickers, dates], names=['ticker', 'date']),
                columns=['size'],
                data=0.0
            )
        
        return size_factor
    
    def construct_volatility_factor(self) -> pd.DataFrame:
        """
        Construct volatility factor (low volatility is better)
        
        Returns:
            DataFrame with volatility factor
        """
        lookback = config.FACTOR_PARAMS['volatility_lookback']
        
        close_prices = self.prices_df['close'].unstack(level=0)
        returns = close_prices.pct_change()
        
        # Calculate rolling volatility
        volatility = returns.rolling(window=lookback).std()
        
        # Rank inversely (low volatility is better)
        ranked = (1 / volatility).rank(axis=1, pct=True)
        normalized = (ranked - 0.5) * 2
        
        volatility_factor = normalized.stack().to_frame('volatility')
        volatility_factor.index.names = ['date', 'ticker']
        volatility_factor = volatility_factor.swaplevel().sort_index()
        
        return volatility_factor
    
    def construct_reversal_factor(self) -> pd.DataFrame:
        """
        Construct short-term reversal factor (mean reversion)
        
        Returns:
            DataFrame with reversal factor
        """
        close_prices = self.prices_df['close'].unstack(level=0)
        returns = close_prices.pct_change()
        
        # Short-term reversal (1-5 days)
        reversal_returns = -returns.rolling(window=5).mean()
        
        # Rank and normalize
        ranked = reversal_returns.rank(axis=1, pct=True)
        normalized = (ranked - 0.5) * 2
        
        reversal_factor = normalized.stack().to_frame('reversal')
        reversal_factor.index.names = ['date', 'ticker']
        reversal_factor = reversal_factor.swaplevel().sort_index()
        
        return reversal_factor
    
    def _calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns for each stock
        
        Returns:
            DataFrame with returns
        """
        close_prices = self.prices_df['close'].unstack(level=0)
        returns = close_prices.pct_change()
        returns_df = returns.stack().to_frame('returns')
        returns_df.index.names = ['date', 'ticker']
        returns_df = returns_df.swaplevel().sort_index()
        return returns_df



