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
    
    def __init__(self, factors_df: pd.DataFrame, sp500_returns: pd.Series):
        """
        Initialize with factors and benchmark returns
        
        Args:
            factors_df: DataFrame with factors and stock returns
            sp500_returns: Series with S&P500 returns
        """
        self.factors_df = factors_df.copy()
        self.sp500_returns = sp500_returns.copy()
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



