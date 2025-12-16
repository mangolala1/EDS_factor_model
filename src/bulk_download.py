"""
Bulk download all data from Snowflake upfront
Downloads everything needed, then we work entirely locally using Parquet files (fast!)
"""
import sys
import os
from pathlib import Path
import importlib.util

# Add parent directory to path for imports when running as script
# This allows imports to work both as module and standalone script
_script_dir = Path(__file__).parent
_parent_dir = _script_dir.parent

# When running as script, set up package structure
if __name__ == "__main__":
    # Add parent to path
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    
    # Set up src as a package
    import types
    if 'src' not in sys.modules:
        src_module = types.ModuleType('src')
        sys.modules['src'] = src_module
    
    # Load config first (needed by data_retrieval)
    if 'src.config' not in sys.modules:
        config_path = _script_dir / 'config.py'
        if config_path.exists():
            spec = importlib.util.spec_from_file_location("src.config", config_path)
            config_module = importlib.util.module_from_spec(spec)
            sys.modules['src.config'] = config_module
            spec.loader.exec_module(config_module)
    
    # Load data_retrieval
    if 'src.data_retrieval' not in sys.modules:
        data_retrieval_path = _script_dir / 'data_retrieval.py'
        spec = importlib.util.spec_from_file_location("src.data_retrieval", data_retrieval_path)
        data_retrieval_module = importlib.util.module_from_spec(spec)
        sys.modules['src.data_retrieval'] = data_retrieval_module
        spec.loader.exec_module(data_retrieval_module)
        SnowflakeDataRetriever = data_retrieval_module.SnowflakeDataRetriever
        get_date_range = data_retrieval_module.get_date_range
    else:
        from src.data_retrieval import SnowflakeDataRetriever
        from src.data_retrieval import get_date_range
else:
    # Running as module - use relative imports
    from .data_retrieval import SnowflakeDataRetriever
    from .data_retrieval import get_date_range

import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm


def download_all_data(
    start_date: str = '2020-01-01',
    end_date: str = None,
    data_dir: str = 'data',
    lookback_start_date: str = '2019-01-01'
):
    """
    Download all data from Snowflake and save to local Parquet files (FAST!)
    Intelligently merges with existing data to avoid duplicates.
    
    Args:
        start_date: Start date for main modeling period (default: '2020-01-01')
        end_date: End date (default: today)
        data_dir: Directory to save Parquet files
        lookback_start_date: Start date for historical lookback data (default: '2019-01-01')
                          This ensures we have enough history for 60-day rolling windows
                          Downloads from this date to end_date
    """
    if end_date is None:
        _, end_date = get_date_range(lookback_days=1)
    
    # Create data directory
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("Bulk Download from Snowflake → Local Parquet Files")
    print("=" * 80)
    print(f"Main period: {start_date} to {end_date}")
    print(f"Lookback period: {lookback_start_date} to {start_date} (for rolling windows)")
    print(f"Data directory: {data_dir}")
    print("=" * 80)
    
    retriever = SnowflakeDataRetriever()
    retriever.connect()
    
    prices_path = data_path / 'prices.parquet'
    returns_path = data_path / 'returns.parquet'
    
    # Check existing data to determine what needs to be downloaded
    existing_prices = None
    existing_returns = None
    existing_prices_min_date = None
    existing_prices_max_date = None
    
    print("\n[Checking existing data]")
    
    if prices_path.exists():
        print("   Loading existing prices to check date range...")
        existing_prices = pd.read_parquet(prices_path)
        existing_prices['DATE'] = pd.to_datetime(existing_prices['DATE'])
        existing_prices_min_date = existing_prices['DATE'].min()
        existing_prices_max_date = existing_prices['DATE'].max()
        print(f"   ✓ Existing prices: {existing_prices_min_date.date()} to {existing_prices_max_date.date()}")
        print(f"   ✓ Existing prices: {len(existing_prices):,} rows")
    
    if returns_path.exists():
        existing_returns = pd.read_parquet(returns_path)
        if 'P_DATE' in existing_returns.columns:
            existing_returns['DATE'] = pd.to_datetime(existing_returns['P_DATE'])
        elif 'DATE' in existing_returns.columns:
            existing_returns['DATE'] = pd.to_datetime(existing_returns['DATE'])
        existing_returns_min_date = existing_returns['DATE'].min()
        existing_returns_max_date = existing_returns['DATE'].max()
        print(f"   ✓ Existing returns: {existing_returns_min_date.date()} to {existing_returns_max_date.date()}")
        print(f"   ✓ Existing returns: {len(existing_returns):,} rows")
    else:
        print("   ℹ No existing returns file found - will calculate from prices")
    
    # Determine what date range to download for prices/returns
    # We need data from lookback_start_date to end_date
    required_start = pd.to_datetime(lookback_start_date)
    required_end = pd.to_datetime(end_date)
    
    # Check if we need to download prices/returns
    prices_need_download = False
    returns_need_download = False
    prices_download_start = required_start
    prices_download_end = required_end
    returns_download_start = required_start
    returns_download_end = required_end
    
    if existing_prices is not None:
        # Check if we have gaps or missing data at the start
        if existing_prices_min_date > required_start:
            # Missing data before existing data
            prices_need_download = True
            prices_download_start = required_start
            prices_download_end = existing_prices_min_date - timedelta(days=1)
            print(f"\n   ⚠ Prices: Missing data from {required_start.date()} to {(existing_prices_min_date - timedelta(days=1)).date()}")
        elif existing_prices_max_date < required_end:
            # Missing data after existing data
            prices_need_download = True
            prices_download_start = existing_prices_max_date + timedelta(days=1)
            prices_download_end = required_end
            print(f"\n   ⚠ Prices: Missing data from {(existing_prices_max_date + timedelta(days=1)).date()} to {required_end.date()}")
        else:
            print(f"\n   ✓ Prices: All required data exists (from {required_start.date()} to {required_end.date()})")
    else:
        # No existing prices - download everything
        prices_need_download = True
        print(f"\n   ⚠ Prices: No existing data, downloading from {required_start.date()} to {required_end.date()}")
    
    if existing_returns is not None:
        # Check if we have gaps or missing data at the start
        if existing_returns_min_date > required_start:
            # Missing data before existing data
            returns_need_download = True
            returns_download_start = required_start
            returns_download_end = existing_returns_min_date - timedelta(days=1)
            print(f"   ⚠ Returns: Missing data from {required_start.date()} to {(existing_returns_min_date - timedelta(days=1)).date()}")
        elif existing_returns_max_date < required_end:
            # Missing data after existing data
            returns_need_download = True
            returns_download_start = existing_returns_max_date + timedelta(days=1)
            returns_download_end = required_end
            print(f"   ⚠ Returns: Missing data from {(existing_returns_max_date + timedelta(days=1)).date()} to {required_end.date()}")
        else:
            print(f"   ✓ Returns: All required data exists (from {required_start.date()} to {required_end.date()})")
    else:
        # No existing returns - will be calculated from prices
        returns_need_download = False  # Will be calculated from new prices
        print(f"   ℹ Returns: Will be calculated from prices")
    
    # If no downloads needed for prices/returns, skip
    if not prices_need_download and not returns_need_download:
        print(f"\n   ✓ All required prices/returns data already exists! Skipping prices/returns download.")
        new_prices_df = pd.DataFrame()
        new_returns_df = pd.DataFrame()
        combined_prices = existing_prices
        combined_returns = existing_returns
    else:
        # Download prices if needed
        if prices_need_download:
            # Step 1: Download prices
            print(f"\n[1] Downloading prices ({prices_download_start.date()} to {prices_download_end.date()})...")
            with tqdm(total=1, desc="Querying prices", bar_format='{desc}: {elapsed}') as pbar:
                new_prices_df = retriever.get_prices_data_by_date_range(
                    prices_download_start.strftime('%Y-%m-%d'),
                    prices_download_end.strftime('%Y-%m-%d')
                )
                pbar.update(1)
        else:
            new_prices_df = pd.DataFrame()
            print(f"\n[1] Skipping prices download (already exists)")
    
        if len(new_prices_df) > 0:
            new_prices_df['DATE'] = pd.to_datetime(new_prices_df['DATE'])
            print(f"   ✓ Downloaded {len(new_prices_df):,} new price rows")
            
            # Merge with existing data
            if existing_prices is not None:
                # Remove any overlapping dates from existing (keep new version)
                existing_prices = existing_prices[
                    (existing_prices['DATE'] < prices_download_start) | 
                    (existing_prices['DATE'] > prices_download_end)
                ]
                combined_prices = pd.concat([existing_prices, new_prices_df], ignore_index=True)
                combined_prices = combined_prices.sort_values('DATE')
                # Remove duplicates (keep last if any)
                combined_prices = combined_prices.drop_duplicates(
                    subset=['FACTSET_ID', 'DATE'], 
                    keep='last'
                )
                print(f"   ✓ Combined: {len(combined_prices):,} total price rows")
            else:
                combined_prices = new_prices_df
            
            # Save combined prices
            print("   Saving to Parquet (fast local storage)...")
            combined_prices.to_parquet(prices_path, index=False, compression='snappy')
            print(f"   ✓ Saved to {prices_path}")
        else:
            print("   ⚠ No new price data found")
            combined_prices = existing_prices
    
    # Step 2: Calculate and save returns
    print(f"\n[2] Calculating returns for new data...")
    if combined_prices is not None and len(combined_prices) > 0:
        # Only calculate returns for new prices
        if len(new_prices_df) > 0:
            new_returns_df = retriever.calculate_returns_from_prices(new_prices_df)
        else:
            new_returns_df = pd.DataFrame()
        
        if len(new_returns_df) > 0:
            if 'P_DATE' in new_returns_df.columns:
                new_returns_df['DATE'] = pd.to_datetime(new_returns_df['P_DATE'])
            elif 'DATE' in new_returns_df.columns:
                new_returns_df['DATE'] = pd.to_datetime(new_returns_df['DATE'])
            
            print(f"   ✓ Calculated {len(new_returns_df):,} new return rows")
            
            # Merge with existing returns
            if existing_returns is not None:
                # Remove overlapping dates (use prices download range if returns were calculated from prices)
                if prices_need_download:
                    overlap_start = prices_download_start
                    overlap_end = prices_download_end
                else:
                    overlap_start = returns_download_start
                    overlap_end = returns_download_end
                existing_returns = existing_returns[
                    (existing_returns['DATE'] < overlap_start) | 
                    (existing_returns['DATE'] > overlap_end)
                ]
                combined_returns = pd.concat([existing_returns, new_returns_df], ignore_index=True)
                combined_returns = combined_returns.sort_values('DATE')
                # Remove duplicates
                date_col = 'P_DATE' if 'P_DATE' in combined_returns.columns else 'DATE'
                id_col = 'FSYM_ID' if 'FSYM_ID' in combined_returns.columns else 'FACTSET_ID'
                combined_returns = combined_returns.drop_duplicates(
                    subset=[id_col, date_col], 
                    keep='last'
                )
                print(f"   ✓ Combined: {len(combined_returns):,} total return rows")
            else:
                combined_returns = new_returns_df
            
            # Save combined returns
            print("   Saving to Parquet...")
            combined_returns.to_parquet(returns_path, index=False, compression='snappy')
            print(f"   ✓ Saved to {returns_path}")
        else:
            print("   ⚠ No new return data calculated")
    
    # Step 3: Download fundamentals (in chunks for efficiency)
    fundamentals_path = data_path / 'fundamentals.parquet'
    existing_fundamentals = None
    
    if fundamentals_path.exists():
        print("\n[Checking existing fundamentals]")
        existing_fundamentals = pd.read_parquet(fundamentals_path)
        existing_fundamentals['DATE'] = pd.to_datetime(existing_fundamentals['DATE'])
        existing_fund_min = existing_fundamentals['DATE'].min()
        existing_fund_max = existing_fundamentals['DATE'].max()
        print(f"   ✓ Existing fundamentals: {existing_fund_min.date()} to {existing_fund_max.date()}")
        print(f"   ✓ Existing fundamentals: {len(existing_fundamentals):,} rows")
        
        # Determine what to download - check for gaps before and after
        fund_required_start = pd.to_datetime(lookback_start_date)
        fund_required_end = pd.to_datetime(end_date)
        fund_needs_download = False
        fund_download_ranges = []  # List of (start, end) tuples for gaps
        
        # Check for gap before existing data
        if existing_fund_min > fund_required_start:
            fund_needs_download = True
            fund_download_ranges.append((fund_required_start, existing_fund_min - timedelta(days=1)))
            print(f"   ⚠ Fundamentals: Missing data from {fund_required_start.date()} to {(existing_fund_min - timedelta(days=1)).date()}")
        
        # Check for gap after existing data
        if existing_fund_max < fund_required_end:
            fund_needs_download = True
            fund_download_ranges.append((existing_fund_max + timedelta(days=1), fund_required_end))
            print(f"   ⚠ Fundamentals: Missing data from {(existing_fund_max + timedelta(days=1)).date()} to {fund_required_end.date()}")
        
        if not fund_needs_download:
            print(f"   ✓ All required fundamentals already exist! Skipping download.")
        else:
            # Download all missing ranges
            print(f"   (This may take a while - downloading in chunks...)")
            
            # Download each missing range
            fundamentals_chunks = []
            for range_idx, (range_start, range_end) in enumerate(fund_download_ranges, 1):
                print(f"\n   Range {range_idx}/{len(fund_download_ranges)}: {range_start.date()} to {range_end.date()}")
                
                # Download in 1-year chunks for this range
                current_dt = range_start
                chunk_num = 0
                
                while current_dt <= range_end:
                    chunk_end = min(current_dt + timedelta(days=365), range_end)
                    chunk_num += 1
                    
                    print(f"      Chunk {chunk_num}: {current_dt.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                    with tqdm(total=1, desc="Querying fundamentals", bar_format='{desc}: {elapsed}') as pbar:
                        fundamentals_chunk = retriever.get_fundamentals_data_by_date_range(
                            start_date=current_dt.strftime('%Y-%m-%d'),
                            end_date=chunk_end.strftime('%Y-%m-%d')
                        )
                        pbar.update(1)
                    
                    if len(fundamentals_chunk) > 0:
                        print(f"         ✓ Got {len(fundamentals_chunk):,} rows")
                        fundamentals_chunks.append(fundamentals_chunk)
                    
                    current_dt = chunk_end + timedelta(days=1)
            
            # Combine with existing and save
            if fundamentals_chunks:
                new_fundamentals_df = pd.concat(fundamentals_chunks, ignore_index=True)
                new_fundamentals_df['DATE'] = pd.to_datetime(new_fundamentals_df['DATE'])
                
                if existing_fundamentals is not None:
                    # Remove overlapping dates from all downloaded ranges
                    overlap_mask = pd.Series(False, index=existing_fundamentals.index)
                    for range_start, range_end in fund_download_ranges:
                        range_mask = (existing_fundamentals['DATE'] >= range_start) & (existing_fundamentals['DATE'] <= range_end)
                        overlap_mask = overlap_mask | range_mask
                    existing_fundamentals = existing_fundamentals[~overlap_mask]
                    
                    combined_fundamentals = pd.concat([existing_fundamentals, new_fundamentals_df], ignore_index=True)
                    combined_fundamentals = combined_fundamentals.sort_values('DATE')
                    # Remove duplicates
                    combined_fundamentals = combined_fundamentals.drop_duplicates(
                        subset=['FACTSET_ID', 'DATE'], 
                        keep='last'
                    )
                    print(f"\n   ✓ Combined: {len(combined_fundamentals):,} total fundamental rows")
                else:
                    combined_fundamentals = new_fundamentals_df
                
                print("   Saving to Parquet...")
                combined_fundamentals.to_parquet(fundamentals_path, index=False, compression='snappy')
                print(f"   ✓ Saved to {fundamentals_path}")
            else:
                print("   ⚠ No new fundamental data downloaded")
    else:
        # No existing fundamentals - download everything
        print(f"\n[3] Downloading fundamentals ({lookback_start_date} to {end_date})...")
        print("   (This may take a while - downloading in chunks...)")
        
        # Download in 1-year chunks
        start_dt = pd.to_datetime(lookback_start_date)
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
    # When running as script, ensure we can import properly
    import os
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    os.chdir(parent_dir)  # Change to project root
    
    download_all_data(
        start_date='2020-01-01',
        end_date=None,  # Will default to today
        data_dir='data',
        lookback_start_date='2019-01-01'  # Download from 2019-01-01 to today
    )

