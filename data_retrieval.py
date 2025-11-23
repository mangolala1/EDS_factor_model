"""
Data retrieval module for fetching data from Snowflake
"""
import snowflake.connector
import pandas as pd
import yfinance as yf
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import config


class SnowflakeDataRetriever:
    """Class to handle data retrieval from Snowflake"""
    
    def __init__(self, config_dict: Dict = None):
        """
        Initialize Snowflake connection
        
        Args:
            config_dict: Dictionary with Snowflake connection parameters
        """
        self.config = config_dict or config.SNOWFLAKE_CONFIG
        self.conn = None
        
    def connect(self):
        """Establish connection to Snowflake"""
        try:
            self.conn = snowflake.connector.connect(
                account=self.config['account'],
                user=self.config['user'],
                password=self.config['password'],
                warehouse=self.config['warehouse'],
                database=self.config['database'],
                schema=self.config['schema'],
                role=self.config.get('role')
            )
            print("Successfully connected to Snowflake")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Snowflake: {str(e)}")
    
    def disconnect(self):
        """Close Snowflake connection"""
        if self.conn:
            self.conn.close()
            print("Disconnected from Snowflake")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        if not self.conn:
            self.connect()
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            cursor.close()
            return df
        except Exception as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")
    
    def get_stock_data(self, 
                      tickers: List[str],
                      start_date: str,
                      end_date: str,
                      price_table: str = 'stock_prices',
                      fundamental_table: str = 'stock_fundamentals') -> Dict[str, pd.DataFrame]:
        """
        Retrieve stock price and fundamental data from Snowflake
        
        Args:
            tickers: List of stock tickers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            price_table: Name of price data table
            fundamental_table: Name of fundamental data table
            
        Returns:
            Dictionary with 'prices' and 'fundamentals' DataFrames
        """
        ticker_list = "', '".join(tickers)
        
        # Query for price data
        price_query = f"""
        SELECT 
            ticker,
            date,
            open,
            high,
            low,
            close,
            volume,
            adj_close
        FROM {price_table}
        WHERE ticker IN ('{ticker_list}')
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ticker, date
        """
        
        # Query for fundamental data
        fundamental_query = f"""
        SELECT 
            ticker,
            date,
            market_cap,
            pe_ratio,
            pb_ratio,
            ev_ebitda,
            roe,
            roa,
            debt_to_equity,
            revenue,
            earnings
        FROM {fundamental_table}
        WHERE ticker IN ('{ticker_list}')
        AND date BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY ticker, date
        """
        
        prices_df = self.execute_query(price_query)
        fundamentals_df = self.execute_query(fundamental_query)
        
        return {
            'prices': prices_df,
            'fundamentals': fundamentals_df
        }
    
    def get_sp500_returns(self, start_date: str, end_date: str) -> pd.Series:
        """
        Get S&P500 returns from Yahoo Finance
        
        Args:
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            Series with daily returns
        """
        sp500 = yf.download(config.SP500_TICKER, start=start_date, end=end_date)
        returns = sp500['Adj Close'].pct_change().dropna()
        returns.name = 'sp500_returns'
        return returns


def get_date_range(lookback_days: int = 252) -> tuple:
    """
    Get start and end dates for data retrieval
    
    Args:
        lookback_days: Number of days to look back
        
    Returns:
        Tuple of (start_date, end_date) as strings
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days * 2)  # Extra buffer
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')



