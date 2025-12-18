"""
Sequential snapshot approach for reading low-frequency data (fundamentals, EV, FX).
Reads data only ONCE, advancing pointer as trading dates progress.
This provides 10-100x speedup compared to date-filtered reads.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class SequentialSnapshot:
    """
    Sequential snapshot approach for reading low-frequency Parquet data.
    
    Instead of filtering by date each day (which triggers reading many row groups),
    this class reads the entire file sequentially once, maintaining a snapshot
    of the latest available data for each stock as dates progress.
    
    Key benefits:
    - Read data only ONCE (not per date)
    - O(1) lookup per stock
    - No date filtering overhead
    - Fixes "0 rows" issues from filter/type/row-group problems
    - 10-100x faster than daily filtering
    """
    
    def __init__(self, parquet_path: Path, id_col: str = "FACTSET_ID", 
                 date_col: str = "DATE", columns: Optional[List[str]] = None):
        """
        Initialize sequential snapshot reader.
        
        Args:
            parquet_path: Path to Parquet file
            id_col: Column name for stock identifier
            date_col: Column name for date
            columns: List of columns to read (None = all columns)
        """
        if not parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
        self.parquet_path = parquet_path
        self.id_col = id_col
        self.date_col = date_col
        self.columns = columns
        
        # State for sequential scanning
        self._scanner = None
        self._iter = None
        self.latest = {}  # id -> dict(row) - latest data per stock
        self._buffer = None  # Current batch being processed
        self._df = None  # Current batch as DataFrame
        self._pos = 0  # Position within current batch
        self._initialized = False
        
    def start(self):
        """Initialize sequential scanner (read only once, no date filters)"""
        if self._initialized:
            return  # Already started
        
        # Sequential scan without date filtering
        dataset = ds.dataset(self.parquet_path, format='parquet')
        self._scanner = dataset.scanner(columns=self.columns)
        self._iter = iter(self._scanner.to_batches())
        self._buffer = None
        self._df = None
        self._pos = 0
        self._initialized = True
        
    def advance_to(self, target_date: pd.Timestamp):
        """
        Advance snapshot to include all rows with DATE <= target_date.
        Called once per trading date in ascending order.
        
        Args:
            target_date: Target date (all rows with DATE <= this will be included)
        """
        if not self._initialized:
            self.start()
            
        # Ensure target_date is Timestamp
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        elif not isinstance(target_date, pd.Timestamp):
            target_date = pd.to_datetime(target_date)
            
        while True:
            # Load next batch if needed
            if self._buffer is None:
                try:
                    self._buffer = next(self._iter)
                    self._df = self._buffer.to_pandas()
                    self._pos = 0
                except StopIteration:
                    return  # Scan complete
            
            # Process rows in current batch with DATE <= target_date
            while self._pos < len(self._df):
                row_date = pd.to_datetime(self._df[self.date_col].iat[self._pos])
                
                if row_date > target_date:
                    # This batch contains future dates, stop here
                    # Will continue from this position on next trading date
                    return
                
                # Update latest data for this stock
                row = self._df.iloc[self._pos]
                stock_id = row[self.id_col]
                # Convert row to dict for efficient storage
                self.latest[stock_id] = row.to_dict()
                self._pos += 1
            
            # Current batch exhausted, load next batch
            self._buffer = None
            self._df = None
    
    def get_for_id(self, stock_id: str) -> Optional[Dict]:
        """
        Get latest data for a single stock ID.
        
        Args:
            stock_id: Stock identifier
            
        Returns:
            Dict with row data, or None if not found
        """
        return self.latest.get(stock_id)
    
    def get_for_ids(self, stock_ids: List[str]) -> List[Optional[Dict]]:
        """
        Get latest data for a list of stock IDs.
        
        Args:
            stock_ids: List of stock identifiers
            
        Returns:
            List of dicts (None if not found)
        """
        return [self.latest.get(sid) for sid in stock_ids]
    
    def get_latest_df(self, stock_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get latest data as DataFrame.
        
        Args:
            stock_ids: Optional list of stock IDs to filter (None = all stocks)
            
        Returns:
            DataFrame with latest data for each stock
        """
        if not self.latest:
            return pd.DataFrame()
        
        # Convert dict of dicts to DataFrame
        df = pd.DataFrame.from_dict(self.latest, orient='index')
        
        if stock_ids is not None:
            # Filter to requested stock IDs
            df = df[df[self.id_col].isin(stock_ids)]
        
        return df.reset_index(drop=True)
    
    def clear(self):
        """Clear snapshot state (useful for testing or restart)"""
        self.latest.clear()
        self._buffer = None
        self._df = None
        self._pos = 0
        self._initialized = False
        self._scanner = None
        self._iter = None


class FundamentalsSnapshot(SequentialSnapshot):
    """Specialized snapshot for fundamentals data"""
    
    def __init__(self, fundamentals_path: Path):
        columns = [
            'FACTSET_ID', 'DATE', 
            'EBITDA_LTM', 'SALES_LTM', 'COGS_LTM',
            'EPS_LTM', 'EPS_NTM', 'SALES_NTM'
        ]
        super().__init__(
            fundamentals_path,
            id_col='FACTSET_ID',
            date_col='DATE',
            columns=columns
        )


class EnterpriseValueSnapshot(SequentialSnapshot):
    """Specialized snapshot for enterprise value data"""
    
    def __init__(self, ev_path: Path):
        columns = ['FACTSET_ID', 'DATE', 'ENTERPRISE_VALUE']
        super().__init__(
            ev_path,
            id_col='FACTSET_ID',
            date_col='DATE',
            columns=columns
        )


class ExchangeRatesSnapshot(SequentialSnapshot):
    """Specialized snapshot for exchange rates data"""
    
    def __init__(self, fx_path: Path):
        columns = ['DATE', 'CURRENCY', 'EXCHANGE_RATE_TO_USD']
        super().__init__(
            fx_path,
            id_col='CURRENCY',  # Exchange rates are keyed by currency, not stock
            date_col='DATE',
            columns=columns
        )
    
    def get_for_currency(self, currency: str, target_date: pd.Timestamp) -> Optional[Dict]:
        """
        Get exchange rate for a currency on a specific date.
        Note: Exchange rates may update less frequently, so we get latest available.
        """
        # Advance to target date first
        self.advance_to(target_date)
        return self.get_for_id(currency)
    
    def get_for_currencies(self, currencies: List[str], target_date: pd.Timestamp) -> Dict[str, Optional[Dict]]:
        """
        Get exchange rates for multiple currencies on a specific date.
        
        Returns:
            Dict mapping currency -> exchange rate data (or None)
        """
        self.advance_to(target_date)
        return {curr: self.get_for_id(curr) for curr in currencies}

