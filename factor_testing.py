"""
Factor testing and validation module
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class FactorTester:
    """Class to test and validate factors"""
    
    def __init__(self, factors_df: pd.DataFrame, sp500_returns: pd.Series):
        """
        Initialize with factors and benchmark returns
        
        Args:
            factors_df: DataFrame with factors and stock returns
            sp500_returns: Series with S&P500 returns
        """
        self.factors_df = factors_df.copy()
        self.sp500_returns = sp500_returns.copy()
        self.test_results = {}
    
    def test_factor_ic(self, factor_name: str) -> Dict:
        """
        Test Information Coefficient (IC) of a factor
        
        Args:
            factor_name: Name of the factor to test
            
        Returns:
            Dictionary with IC statistics
        """
        if factor_name not in self.factors_df.columns:
            return {'error': f'Factor {factor_name} not found'}
        
        # Align dates
        dates = self.factors_df.index.get_level_values('date').unique()
        ic_values = []
        
        for date in dates:
            date_data = self.factors_df.loc[slice(None), date]
            
            if 'returns' in date_data.columns and len(date_data) > 1:
                factor_values = date_data[factor_name]
                forward_returns = date_data['returns']
                
                # Calculate correlation (IC)
                if len(factor_values.dropna()) > 1 and len(forward_returns.dropna()) > 1:
                    ic = factor_values.corr(forward_returns)
                    if not np.isnan(ic):
                        ic_values.append(ic)
        
        if ic_values:
            ic_series = pd.Series(ic_values)
            return {
                'mean_ic': ic_series.mean(),
                'std_ic': ic_series.std(),
                'ic_ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                'positive_ic_pct': (ic_series > 0).sum() / len(ic_series),
                'ic_values': ic_values
            }
        else:
            return {'error': 'Insufficient data for IC calculation'}
    
    def test_all_factors(self) -> pd.DataFrame:
        """
        Test all factors and return summary statistics
        
        Returns:
            DataFrame with test results for all factors
        """
        factor_names = [col for col in self.factors_df.columns if col != 'returns']
        results = []
        
        for factor_name in factor_names:
            ic_results = self.test_factor_ic(factor_name)
            if 'error' not in ic_results:
                results.append({
                    'factor': factor_name,
                    'mean_ic': ic_results['mean_ic'],
                    'ic_ir': ic_results['ic_ir'],
                    'positive_ic_pct': ic_results['positive_ic_pct']
                })
        
        return pd.DataFrame(results)
    
    def test_factor_returns(self, factor_name: str, quantiles: int = 5) -> Dict:
        """
        Test factor returns by creating quantile portfolios
        
        Args:
            factor_name: Name of the factor to test
            quantiles: Number of quantiles to create
            
        Returns:
            Dictionary with quantile portfolio returns
        """
        if factor_name not in self.factors_df.columns:
            return {'error': f'Factor {factor_name} not found'}
        
        dates = self.factors_df.index.get_level_values('date').unique()
        quantile_returns = {f'Q{i+1}': [] for i in range(quantiles)}
        
        for date in dates:
            date_data = self.factors_df.loc[slice(None), date].copy()
            
            if len(date_data) >= quantiles and 'returns' in date_data.columns:
                # Create quantiles
                date_data['quantile'] = pd.qcut(
                    date_data[factor_name], 
                    q=quantiles, 
                    labels=False, 
                    duplicates='drop'
                )
                
                # Calculate returns for each quantile
                for q in range(quantiles):
                    q_data = date_data[date_data['quantile'] == q]
                    if len(q_data) > 0:
                        q_return = q_data['returns'].mean()
                        quantile_returns[f'Q{q+1}'].append(q_return)
        
        # Calculate statistics
        results = {}
        for q_name, returns in quantile_returns.items():
            if returns:
                results[q_name] = {
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns),
                    'sharpe': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                }
        
        # Calculate spread (Q5 - Q1)
        if 'Q5' in results and 'Q1' in results:
            results['spread'] = {
                'mean_return': results['Q5']['mean_return'] - results['Q1']['mean_return'],
                'sharpe': (results['Q5']['mean_return'] - results['Q1']['mean_return']) / 
                         np.sqrt(results['Q5']['std_return']**2 + results['Q1']['std_return']**2)
            }
        
        return results
    
    def plot_factor_analysis(self, factor_name: str, save_path: str = None):
        """
        Create visualization for factor analysis
        
        Args:
            factor_name: Name of the factor to visualize
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # IC time series
        ic_results = self.test_factor_ic(factor_name)
        if 'ic_values' in ic_results:
            axes[0, 0].plot(ic_results['ic_values'])
            axes[0, 0].axhline(y=0, color='r', linestyle='--')
            axes[0, 0].set_title(f'{factor_name} - Information Coefficient')
            axes[0, 0].set_xlabel('Period')
            axes[0, 0].set_ylabel('IC')
        
        # Quantile returns
        quantile_results = self.test_factor_returns(factor_name)
        if 'spread' in quantile_results:
            quantile_names = [k for k in quantile_results.keys() if k.startswith('Q')]
            mean_returns = [quantile_results[q]['mean_return'] for q in quantile_names]
            axes[0, 1].bar(quantile_names, mean_returns)
            axes[0, 1].set_title(f'{factor_name} - Quantile Returns')
            axes[0, 1].set_xlabel('Quantile')
            axes[0, 1].set_ylabel('Mean Return')
        
        # Factor distribution
        factor_values = self.factors_df[factor_name].dropna()
        axes[1, 0].hist(factor_values, bins=50, edgecolor='black')
        axes[1, 0].set_title(f'{factor_name} - Distribution')
        axes[1, 0].set_xlabel('Factor Value')
        axes[1, 0].set_ylabel('Frequency')
        
        # Factor vs Returns scatter
        sample_data = self.factors_df[[factor_name, 'returns']].dropna().sample(min(1000, len(self.factors_df)))
        axes[1, 1].scatter(sample_data[factor_name], sample_data['returns'], alpha=0.3)
        axes[1, 1].set_title(f'{factor_name} vs Returns')
        axes[1, 1].set_xlabel('Factor Value')
        axes[1, 1].set_ylabel('Returns')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()



