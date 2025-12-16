"""
Download additional historical data needed for specific risk calculations
This script downloads data BEFORE the main start_date to ensure we have enough
history for 60-day rolling windows.

Use this if you've already downloaded data starting from start_date but need
to backfill historical data for specific risk calculations.
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
try:
    from .data_retrieval import SnowflakeDataRetriever
except ImportError:
    from data_retrieval import SnowflakeDataRetriever


def download_lookback_data(
    start_date: str = '2020-01-01',
    data_dir: str = 'data',
    lookback_trading_days: int = 60
):
    """
    Download historical data needed for specific risk calculations
    
    Args:
        start_date: The main start date (we need data BEFORE this)
        data_dir: Directory where Parquet files are stored
        lookback_trading_days: Number of trading days needed before start_date (default: 60)
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠ Error: Data directory {data_dir} does not exist")
        return
    
    # Calculate lookback start date
    # 60 trading days ≈ 90 calendar days (accounting for weekends/holidays)
    # Add extra buffer: use 120 calendar days to ensure we get 60+ trading days
    lookback_start = (pd.to_datetime(start_date) - timedelta(days=120)).strftime('%Y-%m-%d')
    
    print("=" * 80)
    print("Downloading Lookback Data for Specific Risk Calculations")
    print("=" * 80)
    print(f"Main start date: {start_date}")
    print(f"Lookback start: {lookback_start} (need {lookback_trading_days} trading days before {start_date})")
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    retriever = SnowflakeDataRetriever()
    retriever.connect()
    
    # Check if existing files exist
    prices_path = data_path / 'prices.parquet'
    returns_path = data_path / 'returns.parquet'
    
    existing_prices = None
    existing_returns = None
    
    if prices_path.exists():
        print(f"\n[1] Loading existing prices from {prices_path}...")
        existing_prices = pd.read_parquet(prices_path)
        existing_prices['DATE'] = pd.to_datetime(existing_prices['DATE'])
        print(f"   ✓ Loaded {len(existing_prices):,} existing price rows")
        print(f"   Date range: {existing_prices['DATE'].min()} to {existing_prices['DATE'].max()}")
    
    if returns_path.exists():
        print(f"\n[2] Loading existing returns from {returns_path}...")
        existing_returns = pd.read_parquet(returns_path)
        if 'P_DATE' in existing_returns.columns:
            existing_returns['DATE'] = pd.to_datetime(existing_returns['P_DATE'])
        elif 'DATE' in existing_returns.columns:
            existing_returns['DATE'] = pd.to_datetime(existing_returns['DATE'])
        print(f"   ✓ Loaded {len(existing_returns):,} existing return rows")
        print(f"   Date range: {existing_returns['DATE'].min()} to {existing_returns['DATE'].max()}")
    
    # Download lookback prices
    print(f"\n[3] Downloading lookback prices ({lookback_start} to {start_date})...")
    lookback_prices = retriever.get_prices_data_by_date_range(lookback_start, start_date)
    
    if len(lookback_prices) > 0:
        lookback_prices['DATE'] = pd.to_datetime(lookback_prices['DATE'])
        print(f"   ✓ Downloaded {len(lookback_prices):,} lookback price rows")
        
        # Filter out dates that are >= start_date (we only want BEFORE start_date)
        lookback_prices = lookback_prices[lookback_prices['DATE'] < pd.to_datetime(start_date)]
        print(f"   ✓ Filtered to {len(lookback_prices):,} rows before {start_date}")
        
        # Merge with existing data
        if existing_prices is not None:
            # Remove any overlapping dates from existing data (keep lookback version)
            existing_prices = existing_prices[existing_prices['DATE'] >= pd.to_datetime(start_date)]
            combined_prices = pd.concat([lookback_prices, existing_prices], ignore_index=True)
            combined_prices = combined_prices.sort_values('DATE')
            print(f"   ✓ Combined: {len(combined_prices):,} total price rows")
        else:
            combined_prices = lookback_prices
        
        # Save combined prices
        combined_prices.to_parquet(prices_path, index=False, compression='snappy')
        print(f"   ✓ Saved to {prices_path}")
    else:
        print("   ⚠ No lookback price data found")
        combined_prices = existing_prices
    
    # Calculate and download lookback returns
    if combined_prices is not None and len(combined_prices) > 0:
        print(f"\n[4] Calculating lookback returns...")
        lookback_returns = retriever.calculate_returns_from_prices(lookback_prices if len(lookback_prices) > 0 else combined_prices)
        
        if len(lookback_returns) > 0:
            if 'P_DATE' in lookback_returns.columns:
                lookback_returns['DATE'] = pd.to_datetime(lookback_returns['P_DATE'])
            elif 'DATE' in lookback_returns.columns:
                lookback_returns['DATE'] = pd.to_datetime(lookback_returns['DATE'])
            
            # Filter to dates before start_date
            lookback_returns = lookback_returns[lookback_returns['DATE'] < pd.to_datetime(start_date)]
            print(f"   ✓ Calculated {len(lookback_returns):,} lookback return rows")
            
            # Merge with existing returns
            if existing_returns is not None:
                existing_returns = existing_returns[existing_returns['DATE'] >= pd.to_datetime(start_date)]
                combined_returns = pd.concat([lookback_returns, existing_returns], ignore_index=True)
                combined_returns = combined_returns.sort_values('DATE')
                print(f"   ✓ Combined: {len(combined_returns):,} total return rows")
            else:
                combined_returns = lookback_returns
            
            # Save combined returns
            combined_returns.to_parquet(returns_path, index=False, compression='snappy')
            print(f"   ✓ Saved to {returns_path}")
        else:
            print("   ⚠ No lookback return data calculated")
    
    retriever.disconnect()
    
    print("\n" + "=" * 80)
    print("Lookback Data Download Complete!")
    print("=" * 80)
    print(f"✓ Historical data now available from {lookback_start} onwards")
    print(f"✓ You can now calculate specific risk for dates starting from {start_date}")
    print("=" * 80)


if __name__ == "__main__":
    download_lookback_data(
        start_date='2020-01-01',
        data_dir='data',
        lookback_trading_days=60
    )

