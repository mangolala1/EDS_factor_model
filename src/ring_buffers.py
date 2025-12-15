"""
Memory-efficient ring buffers for rolling calculations
Used for momentum, volatility, and liquidity calculations
"""
import numpy as np
from collections import deque
from typing import Dict, Optional
import pandas as pd


class RollingReturnBuffer:
    """
    Memory-efficient ring buffer for maintaining rolling returns
    Used for momentum and volatility calculations
    """
    
    def __init__(self, window_size: int = 252):
        """
        Initialize ring buffer
        
        Args:
            window_size: Maximum number of returns to store
        """
        self.window_size = window_size
        self.returns = deque(maxlen=window_size)
        self.dates = deque(maxlen=window_size)
    
    def add(self, date: pd.Timestamp, return_value: float):
        """Add a new return value"""
        self.returns.append(return_value)
        self.dates.append(date)
    
    def get_returns_array(self) -> np.ndarray:
        """Get returns as numpy array"""
        return np.array(self.returns)
    
    def get_dates_array(self) -> np.ndarray:
        """Get dates as numpy array"""
        return np.array(self.dates)
    
    def size(self) -> int:
        """Get current size"""
        return len(self.returns)
    
    def clear(self):
        """Clear the buffer"""
        self.returns.clear()
        self.dates.clear()


class MomentumCalculator:
    """
    Memory-efficient momentum calculation using ring buffer
    Calculates 12-1 month momentum (252 days excluding last 21 days)
    """
    
    def __init__(self, lookback: int = 252, exclude_days: int = 21):
        """
        Initialize momentum calculator
        
        Args:
            lookback: Total lookback period (default: 252 days)
            exclude_days: Days to exclude at the end (default: 21 days)
        """
        self.lookback = lookback
        self.exclude_days = exclude_days
        self.buffer = RollingReturnBuffer(window_size=lookback + exclude_days + 10)  # Extra buffer
    
    def calculate_momentum(self) -> Optional[float]:
        """
        Calculate momentum: (1+ret_12m).prod() - (1+ret_1m).prod()
        Returns None if insufficient data
        """
        if self.buffer.size() < self.lookback + self.exclude_days:
            return None
        
        returns = self.buffer.get_returns_array()
        
        # 12-month return (excluding last 21 days)
        ret_12m = returns[:-self.exclude_days]
        if len(ret_12m) < self.lookback - self.exclude_days:
            return None
        
        # Last 1-month return
        ret_1m = returns[-self.exclude_days:]
        
        # Calculate cumulative returns
        cum_ret_12m = np.prod(1 + ret_12m) - 1
        cum_ret_1m = np.prod(1 + ret_1m) - 1
        
        return cum_ret_12m - cum_ret_1m
    
    def add_return(self, date: pd.Timestamp, return_value: float):
        """Add a new return"""
        self.buffer.add(date, return_value)


class VolatilityCalculator:
    """
    Memory-efficient volatility calculation using ring buffer
    Maintains rolling sum and sum of squares for O(1) updates
    """
    
    def __init__(self, window_size: int = 60):
        """
        Initialize volatility calculator
        
        Args:
            window_size: Rolling window size (default: 60 days)
        """
        self.window_size = window_size
        self.buffer = RollingReturnBuffer(window_size=window_size)
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def calculate_volatility(self) -> Optional[float]:
        """
        Calculate rolling volatility (standard deviation)
        Returns None if insufficient data
        """
        if self.buffer.size() < self.window_size:
            return None
        
        returns = self.buffer.get_returns_array()
        
        # Recalculate sum and sum of squares from buffer
        # (in case we need to handle edge cases)
        n = len(returns)
        if n < 2:
            return None
        
        mean = np.mean(returns)
        variance = np.var(returns, ddof=1)  # Sample variance
        
        return np.sqrt(variance)
    
    def add_return(self, date: pd.Timestamp, return_value: float):
        """Add a new return and update running statistics"""
        if self.buffer.size() == self.window_size:
            # Remove oldest value
            old_return = self.buffer.returns[0]
            self._sum -= old_return
            self._sum_sq -= old_return ** 2
        
        # Add new value
        self.buffer.add(date, return_value)
        self._sum += return_value
        self._sum_sq += return_value ** 2
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self._sum = 0.0
        self._sum_sq = 0.0


class LiquidityCalculator:
    """
    Memory-efficient liquidity calculation
    Calculates log of 20-day average dollar volume
    """
    
    def __init__(self, window_size: int = 20):
        """
        Initialize liquidity calculator
        
        Args:
            window_size: Rolling window size (default: 20 days)
        """
        self.window_size = window_size
        self.dollar_volumes = deque(maxlen=window_size)
    
    def calculate_liquidity(self) -> Optional[float]:
        """
        Calculate liquidity: log(mean(dollar_volume))
        Returns None if insufficient data
        """
        if len(self.dollar_volumes) < self.window_size:
            return None
        
        avg_dollar_vol = np.mean(self.dollar_volumes)
        if avg_dollar_vol <= 0:
            return None
        
        return np.log(avg_dollar_vol)
    
    def add_dollar_volume(self, dollar_volume: float):
        """Add a new dollar volume"""
        if dollar_volume > 0:
            self.dollar_volumes.append(dollar_volume)
    
    def clear(self):
        """Clear the buffer"""
        self.dollar_volumes.clear()


class StockCalculators:
    """
    Container for all rolling calculators for a single stock
    """
    
    def __init__(self, factset_id: str):
        """
        Initialize calculators for a stock
        
        Args:
            factset_id: Stock identifier
        """
        self.factset_id = factset_id
        self.momentum = MomentumCalculator(lookback=252, exclude_days=21)
        self.volatility = VolatilityCalculator(window_size=60)
        self.liquidity = LiquidityCalculator(window_size=20)
    
    def update(self, date: pd.Timestamp, return_value: float, dollar_volume: float):
        """Update all calculators with new data"""
        self.momentum.add_return(date, return_value)
        self.volatility.add_return(date, return_value)
        self.liquidity.add_dollar_volume(dollar_volume)
    
    def get_momentum(self) -> Optional[float]:
        """Get current momentum value"""
        return self.momentum.calculate_momentum()
    
    def get_volatility(self) -> Optional[float]:
        """Get current volatility value"""
        return self.volatility.calculate_volatility()
    
    def get_liquidity(self) -> Optional[float]:
        """Get current liquidity value"""
        return self.liquidity.calculate_liquidity()


class StockCalculatorManager:
    """
    Manages rolling calculators for all stocks
    """
    
    def __init__(self):
        """Initialize manager"""
        self.calculators: Dict[str, StockCalculators] = {}
    
    def get_calculator(self, factset_id: str) -> StockCalculators:
        """Get or create calculator for a stock"""
        if factset_id not in self.calculators:
            self.calculators[factset_id] = StockCalculators(factset_id)
        return self.calculators[factset_id]
    
    def update_stock(self, factset_id: str, date: pd.Timestamp, 
                     return_value: float, dollar_volume: float):
        """Update calculator for a stock"""
        calc = self.get_calculator(factset_id)
        calc.update(date, return_value, dollar_volume)
    
    def get_momentum(self, factset_id: str) -> Optional[float]:
        """Get momentum for a stock"""
        if factset_id not in self.calculators:
            return None
        return self.calculators[factset_id].get_momentum()
    
    def get_volatility(self, factset_id: str) -> Optional[float]:
        """Get volatility for a stock"""
        if factset_id not in self.calculators:
            return None
        return self.calculators[factset_id].get_volatility()
    
    def get_liquidity(self, factset_id: str) -> Optional[float]:
        """Get liquidity for a stock"""
        if factset_id not in self.calculators:
            return None
        return self.calculators[factset_id].get_liquidity()
    
    def clear(self):
        """Clear all calculators"""
        self.calculators.clear()

