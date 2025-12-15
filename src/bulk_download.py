"""
Bulk download all data from Snowflake upfront
Downloads everything needed, then we work entirely locally using Parquet files (fast!)
"""
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
from .data_retrieval import SnowflakeDataRetriever


def download_all_data(
    start_date: str = '2025-01-01',
    end_date: str = None,
    data_dir: str = 'data'
):
    """
    Download all data from Snowflake and save to local Parquet files (FAST!)
    
    Args:
        start_date: Start date for data
        end_date: End date (default: today)
        data_dir: Directory to save Parquet files
    """
    from data_retrieval import get_date_range
    
    if end_date is None:
        _, end_date = get_date_range(lookback_days=1)
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Bulk Download from Snowflake → Local Parquet Files")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    retriever = SnowflakeDataRetriever()
    retriever.connect()
    
    # Calculate lookback for historical buffers (400 days before start_date)
    lookback_start = (pd.to_datetime(start_date) - timedelta(days=400)).strftime('%Y-%m-%d')
    
    # Step 1: Download prices
    print("\n[1] Downloading prices...")
    with tqdm(total=1, desc="Querying prices", bar_format='{desc}: {elapsed}') as pbar:
        prices_df = retriever.get_prices_data_by_date_range(lookback_start, end_date)
        pbar.update(1)
    
    if len(prices_df) > 0:
        print(f"   ✓ Got {len(prices_df):,} price rows")
        print("   Saving to Parquet (fast local storage)...")
        prices_path = data_path / 'prices.parquet'
        prices_df.to_parquet(prices_path, index=False, compression='snappy')
        print(f"   ✓ Saved to {prices_path}")
    else:
        print("   ⚠ No price data found")
    
    # Step 2: Calculate and save returns
    print("\n[2] Calculating returns...")
    with tqdm(total=1, desc="Calculating returns", bar_format='{desc}: {elapsed}') as pbar:
        returns_df = retriever.calculate_returns_from_prices(prices_df)
        pbar.update(1)
    
    if len(returns_df) > 0:
        print(f"   ✓ Got {len(returns_df):,} return rows")
        print("   Saving to Parquet...")
        returns_path = data_path / 'returns.parquet'
        returns_df.to_parquet(returns_path, index=False, compression='snappy')
        print(f"   ✓ Saved to {returns_path}")
    else:
        print("   ⚠ No return data found")
    
    # Step 3: Download fundamentals (in chunks for efficiency)
    print("\n[3] Downloading fundamentals...")
    print("   (This may take a while - downloading in chunks...)")
    
    # Download in 1-year chunks
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    current_dt = start_dt
    
    fundamentals_chunks = []
    chunk_num = 0
    
    while current_dt <= end_dt:
        chunk_end = min(current_dt + timedelta(days=365), end_dt)
        chunk_num += 1
        
        print(f"\n   Chunk {chunk_num}: {current_dt.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
        with tqdm(total=1, desc="Querying fundamentals", bar_format='{desc}: {elapsed}') as pbar:
            fundamentals_chunk = retriever.get_fundamentals_data_by_date_range(
                start_date=current_dt.strftime('%Y-%m-%d'),
                end_date=chunk_end.strftime('%Y-%m-%d')
            )
            pbar.update(1)
        
        if len(fundamentals_chunk) > 0:
            print(f"      ✓ Got {len(fundamentals_chunk):,} rows")
            fundamentals_chunks.append(fundamentals_chunk)
        
        current_dt = chunk_end + timedelta(days=1)
    
    # Combine and save fundamentals
    if fundamentals_chunks:
        fundamentals_df = pd.concat(fundamentals_chunks, ignore_index=True)
        print(f"\n   ✓ Total fundamentals: {len(fundamentals_df):,} rows")
        print("   Saving to Parquet...")
        fundamentals_path = data_path / 'fundamentals.parquet'
        fundamentals_df.to_parquet(fundamentals_path, index=False, compression='snappy')
        print(f"   ✓ Saved to {fundamentals_path}")
    
    # Step 4: Get universe
    print("\n[4] Downloading universe metadata...")
    with tqdm(total=1, desc="Querying universe", bar_format='{desc}: {elapsed}') as pbar:
        universe_df = retriever.get_universe_data(bloomberg_tickers=None)
        pbar.update(1)
    
    print(f"   ✓ Got {len(universe_df):,} stocks")
    
    # Save universe to Parquet (and CSV for compatibility)
    universe_path = data_path / 'universe.parquet'
    universe_df.to_parquet(universe_path, index=False, compression='snappy')
    universe_df.to_csv(data_path / 'universe.csv', index=False)
    print(f"   ✓ Saved to {universe_path} and universe.csv")
    
    retriever.disconnect()
    
    print("\n" + "=" * 80)
    print("Bulk Download Complete!")
    print("=" * 80)
    print(f"✓ All data saved to {data_dir}/")
    print(f"  - prices.parquet")
    print(f"  - returns.parquet")
    print(f"  - fundamentals.parquet")
    print(f"  - universe.parquet & universe.csv")
    print("\nYou can now run the factor model workflow - it will work entirely locally!")
    print("No more Snowflake queries needed!")
    print("=" * 80)


if __name__ == "__main__":
    download_all_data(
        start_date='2020-01-01',
        end_date=None,
        data_dir='data'
    )

