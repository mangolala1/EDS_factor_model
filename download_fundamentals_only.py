"""
Download only fundamentals data from 2020-01-01 to now
"""
import sys
from pathlib import Path
import importlib.util

# Add parent directory to path for imports when running as script
_script_dir = Path(__file__).parent
_parent_dir = _script_dir.parent

# When running as script, set up package structure
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Set up src as a package
import types
if 'src' not in sys.modules:
    src_module = types.ModuleType('src')
    sys.modules['src'] = src_module

# Load config first (needed by data_retrieval)
if 'src.config' not in sys.modules:
    config_path = _script_dir / 'src' / 'config.py'
    if config_path.exists():
        spec = importlib.util.spec_from_file_location("src.config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        sys.modules['src.config'] = config_module
        spec.loader.exec_module(config_module)

# Load data_retrieval
if 'src.data_retrieval' not in sys.modules:
    data_retrieval_path = _script_dir / 'src' / 'data_retrieval.py'
    spec = importlib.util.spec_from_file_location("src.data_retrieval", data_retrieval_path)
    data_retrieval_module = importlib.util.module_from_spec(spec)
    sys.modules['src.data_retrieval'] = data_retrieval_module
    spec.loader.exec_module(data_retrieval_module)
    SnowflakeDataRetriever = data_retrieval_module.SnowflakeDataRetriever
    get_date_range = data_retrieval_module.get_date_range
else:
    from src.data_retrieval import SnowflakeDataRetriever
    from src.data_retrieval import get_date_range

import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


def download_fundamentals_only(
    start_date: str = '2020-01-01',
    end_date: str = None,
    data_dir: str = 'data'
):
    """
    Download only fundamentals data from Snowflake and save to local Parquet file.
    
    Args:
        start_date: Start date for fundamentals (default: '2020-01-01')
        end_date: End date (default: today)
        data_dir: Directory to save Parquet file
    """
    if end_date is None:
        _, end_date = get_date_range(lookback_days=1)
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Download Fundamentals Only from Snowflake")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    retriever = SnowflakeDataRetriever()
    retriever.connect()
    
    fundamentals_path = data_path / 'fundamentals.parquet'
    
    print(f"\n[Downloading fundamentals ({start_date} to {end_date})...")
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
        fundamentals_df['DATE'] = pd.to_datetime(fundamentals_df['DATE'])
        
        # Ensure we only have data from start_date onwards
        fundamentals_df = fundamentals_df[fundamentals_df['DATE'] >= start_dt].copy()
        
        # Sort and deduplicate
        fundamentals_df = fundamentals_df.sort_values('DATE')
        fundamentals_df = fundamentals_df.drop_duplicates(
            subset=['FACTSET_ID', 'DATE'], 
            keep='last'
        )
        
        print(f"\n   ✓ Total fundamentals: {len(fundamentals_df):,} rows")
        print(f"   ✓ Date range: {fundamentals_df['DATE'].min().date()} to {fundamentals_df['DATE'].max().date()}")
        print("   Saving to Parquet...")
        fundamentals_df.to_parquet(fundamentals_path, index=False, compression='snappy')
        print(f"   ✓ Saved to {fundamentals_path}")
    else:
        print("   ⚠ No fundamental data downloaded")
    
    retriever.disconnect()
    
    print("\n" + "=" * 80)
    print("Fundamentals Download Complete!")
    print("=" * 80)
    print(f"✓ Fundamentals saved to {data_dir}/fundamentals.parquet")
    print("=" * 80)


if __name__ == "__main__":
    import os
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    os.chdir(parent_dir)  # Change to project root
    
    download_fundamentals_only(
        start_date='2020-01-01',
        end_date=None,  # Will default to today
        data_dir='data'
    )

