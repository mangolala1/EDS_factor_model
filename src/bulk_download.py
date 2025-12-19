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
    print(f"\n[2] Calculating returns from prices...")
    if combined_prices is not None and len(combined_prices) > 0:
        # If returns.parquet doesn't exist, calculate from ALL prices
        # Otherwise, only calculate returns for new prices
        if existing_returns is None or not returns_path.exists():
            print("   ℹ No existing returns file - calculating from all prices...")
            returns_to_calculate = combined_prices
        else:
            # Only calculate returns for new prices
            if len(new_prices_df) > 0:
                returns_to_calculate = new_prices_df
            else:
                returns_to_calculate = pd.DataFrame()
        
        if len(returns_to_calculate) > 0:
            new_returns_df = retriever.calculate_returns_from_prices(returns_to_calculate)
            
            if len(new_returns_df) > 0:
                if 'P_DATE' in new_returns_df.columns:
                    new_returns_df['DATE'] = pd.to_datetime(new_returns_df['P_DATE'])
                elif 'DATE' in new_returns_df.columns:
                    new_returns_df['DATE'] = pd.to_datetime(new_returns_df['DATE'])
                
                if existing_returns is None or not returns_path.exists():
                    print(f"   ✓ Calculated {len(new_returns_df):,} return rows from all prices")
                    combined_returns = new_returns_df
                else:
                    print(f"   ✓ Calculated {len(new_returns_df):,} new return rows")
                    
                    # Merge with existing returns
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
                
                # Save combined returns
                print("   Saving to Parquet...")
                combined_returns.to_parquet(returns_path, index=False, compression='snappy')
                print(f"   ✓ Saved to {returns_path}")
            else:
                print("   ⚠ No return data calculated")
        else:
            if existing_returns is not None:
                print("   ✓ Using existing returns (no new prices to calculate from)")
                combined_returns = existing_returns
            else:
                print("   ⚠ No prices available to calculate returns")
                combined_returns = None
    
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
    
    # Step 5: Download exchange rates
    exchange_rates_path = data_path / 'exchange_rates.parquet'
    existing_exchange_rates = None
    
    if exchange_rates_path.exists():
        print("\n[Checking existing exchange rates]")
        existing_exchange_rates = pd.read_parquet(exchange_rates_path)
        existing_exchange_rates['DATE'] = pd.to_datetime(existing_exchange_rates['DATE'])
        existing_exr_min = existing_exchange_rates['DATE'].min()
        existing_exr_max = existing_exchange_rates['DATE'].max()
        print(f"   ✓ Existing exchange rates: {existing_exr_min.date()} to {existing_exr_max.date()}")
        print(f"   ✓ Existing exchange rates: {len(existing_exchange_rates):,} rows")
        
        exr_required_start = pd.to_datetime(lookback_start_date)
        exr_required_end = pd.to_datetime(end_date)
        exr_needs_download = False
        exr_download_ranges = []
        
        if existing_exr_min > exr_required_start:
            exr_needs_download = True
            exr_download_ranges.append((exr_required_start, existing_exr_min - timedelta(days=1)))
            print(f"   ⚠ Exchange rates: Missing data from {exr_required_start.date()} to {(existing_exr_min - timedelta(days=1)).date()}")
        
        if existing_exr_max < exr_required_end:
            exr_needs_download = True
            exr_download_ranges.append((existing_exr_max + timedelta(days=1), exr_required_end))
            print(f"   ⚠ Exchange rates: Missing data from {(existing_exr_max + timedelta(days=1)).date()} to {exr_required_end.date()}")
        
        if not exr_needs_download:
            print(f"   ✓ All required exchange rates already exist! Skipping download.")
        else:
            print(f"\n[5] Downloading exchange rates...")
            exchange_rates_chunks = []
            for range_idx, (range_start, range_end) in enumerate(exr_download_ranges, 1):
                print(f"   Range {range_idx}/{len(exr_download_ranges)}: {range_start.date()} to {range_end.date()}")
                with tqdm(total=1, desc="Querying exchange rates", bar_format='{desc}: {elapsed}') as pbar:
                    exr_chunk = retriever.get_exchange_rates_by_date_range(
                        start_date=range_start.strftime('%Y-%m-%d'),
                        end_date=range_end.strftime('%Y-%m-%d')
                    )
                    pbar.update(1)
                
                if len(exr_chunk) > 0:
                    print(f"      ✓ Got {len(exr_chunk):,} rows")
                    exchange_rates_chunks.append(exr_chunk)
            
            if exchange_rates_chunks:
                new_exchange_rates_df = pd.concat(exchange_rates_chunks, ignore_index=True)
                new_exchange_rates_df['DATE'] = pd.to_datetime(new_exchange_rates_df['DATE'])
                
                # Remove overlapping dates from existing
                overlap_mask = pd.Series(False, index=existing_exchange_rates.index)
                for range_start, range_end in exr_download_ranges:
                    range_mask = (existing_exchange_rates['DATE'] >= range_start) & (existing_exchange_rates['DATE'] <= range_end)
                    overlap_mask = overlap_mask | range_mask
                existing_exchange_rates = existing_exchange_rates[~overlap_mask]
                
                combined_exchange_rates = pd.concat([existing_exchange_rates, new_exchange_rates_df], ignore_index=True)
                combined_exchange_rates = combined_exchange_rates.sort_values('DATE')
                combined_exchange_rates = combined_exchange_rates.drop_duplicates(
                    subset=['DATE', 'CURRENCY'], 
                    keep='last'
                )
                print(f"   ✓ Combined: {len(combined_exchange_rates):,} total exchange rate rows")
                combined_exchange_rates.to_parquet(exchange_rates_path, index=False, compression='snappy')
                print(f"   ✓ Saved to {exchange_rates_path}")
    else:
        # No existing exchange rates - download everything
        print(f"\n[5] Downloading exchange rates ({lookback_start_date} to {end_date})...")
        with tqdm(total=1, desc="Querying exchange rates", bar_format='{desc}: {elapsed}') as pbar:
            exchange_rates_df = retriever.get_exchange_rates_by_date_range(
                start_date=lookback_start_date,
                end_date=end_date
            )
            pbar.update(1)
        
        if len(exchange_rates_df) > 0:
            exchange_rates_df['DATE'] = pd.to_datetime(exchange_rates_df['DATE'])
            print(f"   ✓ Got {len(exchange_rates_df):,} exchange rate rows")
            print("   Saving to Parquet...")
            exchange_rates_df.to_parquet(exchange_rates_path, index=False, compression='snappy')
            print(f"   ✓ Saved to {exchange_rates_path}")
    
    # Step 6: Download market value (market cap) data
    market_value_path = data_path / 'market_value.parquet'
    existing_market_value = None
    
    if market_value_path.exists():
        print("\n[Checking existing market value]")
        existing_market_value = pd.read_parquet(market_value_path)
        existing_market_value['DATE'] = pd.to_datetime(existing_market_value['DATE'])
        existing_mv_min = existing_market_value['DATE'].min()
        existing_mv_max = existing_market_value['DATE'].max()
        print(f"   ✓ Existing market value: {existing_mv_min.date()} to {existing_mv_max.date()}")
        print(f"   ✓ Existing market value: {len(existing_market_value):,} rows")
        
        # Market value only needs data from 2020-01-01 onwards (no lookback)
        mv_required_start = pd.to_datetime('2020-01-01')
        mv_required_end = pd.to_datetime(end_date)
        mv_needs_download = False
        mv_download_ranges = []
        
        if existing_mv_min > mv_required_start:
            mv_needs_download = True
            mv_download_ranges.append((mv_required_start, existing_mv_min - timedelta(days=1)))
            print(f"   ⚠ Market value: Missing data from {mv_required_start.date()} to {(existing_mv_min - timedelta(days=1)).date()}")
        
        if existing_mv_max < mv_required_end:
            mv_needs_download = True
            mv_download_ranges.append((existing_mv_max + timedelta(days=1), mv_required_end))
            print(f"   ⚠ Market value: Missing data from {(existing_mv_max + timedelta(days=1)).date()} to {mv_required_end.date()}")
        
        if not mv_needs_download:
            print(f"   ✓ All required market value already exists! Skipping download.")
        else:
            print(f"\n[6] Downloading market value...")
            # Use incremental writing to avoid memory issues
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            temp_chunk_files = []
            chunk_idx = 0
            
            for range_idx, (range_start, range_end) in enumerate(mv_download_ranges, 1):
                print(f"   Range {range_idx}/{len(mv_download_ranges)}: {range_start.date()} to {range_end.date()}")
                # Download in 1-year chunks for this range
                current_dt = range_start
                chunk_num = 0
                
                while current_dt <= range_end:
                    chunk_end = min(current_dt + timedelta(days=365), range_end)
                    chunk_num += 1
                    
                    print(f"      Chunk {chunk_num}: {current_dt.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                    with tqdm(total=1, desc="Querying market value", bar_format='{desc}: {elapsed}') as pbar:
                        mv_chunk = retriever.get_market_value_data_by_date_range(
                            start_date=current_dt.strftime('%Y-%m-%d'),
                            end_date=chunk_end.strftime('%Y-%m-%d')
                        )
                        pbar.update(1)
                    
                    if len(mv_chunk) > 0:
                        mv_chunk['DATE'] = pd.to_datetime(mv_chunk['DATE'])
                        print(f"         ✓ Got {len(mv_chunk):,} rows")
                        # Save chunk to temporary file immediately (frees memory)
                        chunk_idx += 1
                        temp_file = data_path / f'market_value_temp_chunk_{chunk_idx}.parquet'
                        mv_chunk.to_parquet(temp_file, index=False, compression='snappy')
                        temp_chunk_files.append(temp_file)
                        del mv_chunk  # Free memory
                    
                    current_dt = chunk_end + timedelta(days=1)
            
            if temp_chunk_files:
                # Use streaming approach similar to fundamentals - write new chunks to temp file, then combine
                import pyarrow.parquet as pq
                import pyarrow as pa
                
                # Write new chunks to a temporary combined file (streaming write)
                new_market_value_path = data_path / 'market_value_new.parquet'
                
                # Read and write chunks one at a time to avoid loading all into memory
                first_chunk = True
                for temp_file in temp_chunk_files:
                    table = pq.read_table(temp_file)
                    if first_chunk:
                        pq.write_table(table, new_market_value_path, compression='snappy')
                        first_chunk = False
                    else:
                        # Append to existing file using PyArrow dataset
                        existing_table = pq.read_table(new_market_value_path)
                        combined_table = pa.concat_tables([existing_table, table])
                        pq.write_table(combined_table, new_market_value_path, compression='snappy')
                        del existing_table, combined_table
                    del table
                    # Add delay and error handling for file deletion
                    import time
                    time.sleep(0.1)
                    try:
                        temp_file.unlink()
                    except (PermissionError, FileNotFoundError):
                        pass  # File may be in use or already deleted
                
                # Now combine with existing data - process existing data in chunks
                print("   Combining with existing data...")
                
                # Filter existing data to remove overlapping dates
                overlap_mask = pd.Series(False, index=existing_market_value.index)
                for range_start, range_end in mv_download_ranges:
                    range_mask = (existing_market_value['DATE'] >= range_start) & (existing_market_value['DATE'] <= range_end)
                    overlap_mask = overlap_mask | range_mask
                existing_market_value_filtered = existing_market_value[~overlap_mask]
                del existing_market_value  # Free memory
                
                # Write filtered existing data to temp file
                existing_temp_path = data_path / 'market_value_existing_filtered.parquet'
                existing_market_value_filtered.to_parquet(existing_temp_path, index=False, compression='snappy')
                del existing_market_value_filtered  # Free memory
                
                # Combine the two files using PyArrow dataset (streaming)
                existing_table = pq.read_table(existing_temp_path)
                new_table = pq.read_table(new_market_value_path)
                combined_table = pa.concat_tables([existing_table, new_table])
                del existing_table, new_table
                
                # Write combined data (filter to 2020-01-01 onwards and remove NaN MARKETCAP)
                print(f"   ✓ Combined: {len(combined_table):,} total market value rows")
                print("   Filtering to dates >= 2020-01-01 and removing NaN MARKETCAP...")
                
                # Process in chunks to avoid memory issues
                cutoff_date = pd.to_datetime('2020-01-01')
                filtered_chunks = []
                batch_size = 10_000_000
                
                # Process combined_table in batches
                for i in range(0, len(combined_table), batch_size):
                    batch = combined_table.slice(i, min(batch_size, len(combined_table) - i))
                    batch_df = batch.to_pandas()
                    batch_df['DATE'] = pd.to_datetime(batch_df['DATE'])
                    
                    # Filter to 2020-01-01 onwards and remove NaN MARKETCAP
                    batch_filtered = batch_df[
                        (batch_df['DATE'] >= cutoff_date) & 
                        (batch_df['MARKETCAP'].notna())
                    ].copy()
                    
                    if len(batch_filtered) > 0:
                        filtered_chunks.append(batch_filtered)
                    del batch_df, batch_filtered
                
                del combined_table
                
                if len(filtered_chunks) > 0:
                    # Combine filtered chunks
                    filtered_df = pd.concat(filtered_chunks, ignore_index=True)
                    del filtered_chunks
                    
                    # Sort and deduplicate
                    filtered_df = filtered_df.sort_values('DATE')
                    filtered_df = filtered_df.drop_duplicates(
                        subset=['FACTSET_ID', 'DATE'], 
                        keep='last'
                    )
                    
                    print(f"   ✓ After filtering: {len(filtered_df):,} rows")
                    print("   Saving to Parquet...")
                    filtered_df.to_parquet(market_value_path, index=False, compression='snappy')
                    del filtered_df
                else:
                    print("   ⚠ No data remaining after filtering!")
                
                # Clean up temp files
                import time
                time.sleep(0.2)
                try:
                    new_market_value_path.unlink()
                except (PermissionError, FileNotFoundError):
                    pass
                try:
                    existing_temp_path.unlink()
                except (PermissionError, FileNotFoundError):
                    pass
                
                print(f"   ✓ Saved to {market_value_path}")
    else:
        # No existing market value - download from 2020-01-01 onwards (no lookback)
        print(f"\n[6] Downloading market value (2020-01-01 to {end_date})...")
        print("   (This may take a while - downloading in chunks...)")
        
        start_dt = pd.to_datetime('2020-01-01')
        end_dt = pd.to_datetime(end_date)
        current_dt = start_dt
        
        # Use incremental writing to avoid memory issues
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        temp_chunk_files = []
        chunk_num = 0
        total_rows = 0
        
        while current_dt <= end_dt:
            chunk_end = min(current_dt + timedelta(days=365), end_dt)
            chunk_num += 1
            
            print(f"\n   Chunk {chunk_num}: {current_dt.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            with tqdm(total=1, desc="Querying market value", bar_format='{desc}: {elapsed}') as pbar:
                mv_chunk = retriever.get_market_value_data_by_date_range(
                    start_date=current_dt.strftime('%Y-%m-%d'),
                    end_date=chunk_end.strftime('%Y-%m-%d')
                )
                pbar.update(1)
            
            if len(mv_chunk) > 0:
                mv_chunk['DATE'] = pd.to_datetime(mv_chunk['DATE'])
                print(f"      ✓ Got {len(mv_chunk):,} rows")
                total_rows += len(mv_chunk)
                
                # Save chunk to temporary file immediately (frees memory)
                temp_file = data_path / f'market_value_temp_chunk_{chunk_num}.parquet'
                mv_chunk.to_parquet(temp_file, index=False, compression='snappy')
                temp_chunk_files.append(temp_file)
                del mv_chunk  # Free memory
            
            current_dt = chunk_end + timedelta(days=1)
        
        # Combine temporary files efficiently using pyarrow (memory efficient)
        if temp_chunk_files:
            print(f"\n   ✓ Total market value: {total_rows:,} rows")
            print("   Combining chunks and saving to final Parquet file...")
            
            # Use pyarrow to combine all chunks at once (more memory efficient)
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            # Read all temp files as tables
            tables = []
            for temp_file in temp_chunk_files:
                table = pq.read_table(temp_file)
                tables.append(table)
                # Add delay and error handling for file deletion
                import time
                time.sleep(0.1)
                try:
                    temp_file.unlink()
                except (PermissionError, FileNotFoundError):
                    pass  # File may be in use or already deleted
            
            # Concatenate all tables and filter to 2020-01-01 onwards, remove NaN MARKETCAP
            if len(tables) > 0:
                combined_table = pa.concat_tables(tables)
                del tables
                
                print("   Filtering to dates >= 2020-01-01 and removing NaN MARKETCAP...")
                cutoff_date = pd.to_datetime('2020-01-01')
                filtered_chunks = []
                
                # Process in batches to avoid memory issues
                batch_size = 10_000_000
                for i in range(0, len(combined_table), batch_size):
                    batch = combined_table.slice(i, min(batch_size, len(combined_table) - i))
                    batch_df = batch.to_pandas()
                    batch_df['DATE'] = pd.to_datetime(batch_df['DATE'])
                    
                    # Filter to 2020-01-01 onwards and remove NaN MARKETCAP
                    batch_filtered = batch_df[
                        (batch_df['DATE'] >= cutoff_date) & 
                        (batch_df['MARKETCAP'].notna())
                    ].copy()
                    
                    if len(batch_filtered) > 0:
                        filtered_chunks.append(batch_filtered)
                    del batch_df, batch_filtered
                
                del combined_table
                
                if len(filtered_chunks) > 0:
                    # Combine filtered chunks
                    filtered_df = pd.concat(filtered_chunks, ignore_index=True)
                    del filtered_chunks
                    
                    # Sort and deduplicate
                    filtered_df = filtered_df.sort_values('DATE')
                    filtered_df = filtered_df.drop_duplicates(
                        subset=['FACTSET_ID', 'DATE'], 
                        keep='last'
                    )
                    
                    print(f"   ✓ After filtering: {len(filtered_df):,} rows")
                    filtered_df.to_parquet(market_value_path, index=False, compression='snappy')
                    del filtered_df
                else:
                    print("   ⚠ No data remaining after filtering!")
            
            print(f"   ✓ Saved to {market_value_path}")
    
    # Step 7: Download enterprise value data
    enterprise_value_path = data_path / 'enterprise_value.parquet'
    existing_enterprise_value = None
    
    if enterprise_value_path.exists():
        print("\n[Checking existing enterprise value]")
        existing_enterprise_value = pd.read_parquet(enterprise_value_path)
        existing_enterprise_value['DATE'] = pd.to_datetime(existing_enterprise_value['DATE'])
        existing_ev_min = existing_enterprise_value['DATE'].min()
        existing_ev_max = existing_enterprise_value['DATE'].max()
        print(f"   ✓ Existing enterprise value: {existing_ev_min.date()} to {existing_ev_max.date()}")
        print(f"   ✓ Existing enterprise value: {len(existing_enterprise_value):,} rows")
        
        # Enterprise value only needs data from 2020-01-01 onwards (no lookback)
        ev_required_start = pd.to_datetime('2020-01-01')
        ev_required_end = pd.to_datetime(end_date)
        ev_needs_download = False
        ev_download_ranges = []
        
        if existing_ev_min > ev_required_start:
            ev_needs_download = True
            ev_download_ranges.append((ev_required_start, existing_ev_min - timedelta(days=1)))
            print(f"   ⚠ Enterprise value: Missing data from {ev_required_start.date()} to {(existing_ev_min - timedelta(days=1)).date()}")
        
        if existing_ev_max < ev_required_end:
            ev_needs_download = True
            ev_download_ranges.append((existing_ev_max + timedelta(days=1), ev_required_end))
            print(f"   ⚠ Enterprise value: Missing data from {(existing_ev_max + timedelta(days=1)).date()} to {ev_required_end.date()}")
        
        if not ev_needs_download:
            print(f"   ✓ All required enterprise value already exists! Skipping download.")
        else:
            print(f"\n[7] Downloading enterprise value...")
            # Use incremental writing to avoid memory issues
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            temp_chunk_files = []
            chunk_idx = 0
            
            for range_idx, (range_start, range_end) in enumerate(ev_download_ranges, 1):
                print(f"   Range {range_idx}/{len(ev_download_ranges)}: {range_start.date()} to {range_end.date()}")
                # Download in 1-year chunks for this range
                current_dt = range_start
                chunk_num = 0
                
                while current_dt <= range_end:
                    chunk_end = min(current_dt + timedelta(days=365), range_end)
                    chunk_num += 1
                    
                    print(f"      Chunk {chunk_num}: {current_dt.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                    with tqdm(total=1, desc="Querying enterprise value", bar_format='{desc}: {elapsed}') as pbar:
                        ev_chunk = retriever.get_enterprise_value_data_by_date_range(
                            start_date=current_dt.strftime('%Y-%m-%d'),
                            end_date=chunk_end.strftime('%Y-%m-%d')
                        )
                        pbar.update(1)
                    
                    if len(ev_chunk) > 0:
                        ev_chunk['DATE'] = pd.to_datetime(ev_chunk['DATE'])
                        print(f"         ✓ Got {len(ev_chunk):,} rows")
                        # Save chunk to temporary file immediately (frees memory)
                        chunk_idx += 1
                        temp_file = data_path / f'enterprise_value_temp_chunk_{chunk_idx}.parquet'
                        ev_chunk.to_parquet(temp_file, index=False, compression='snappy')
                        temp_chunk_files.append(temp_file)
                        del ev_chunk  # Free memory
                    
                    current_dt = chunk_end + timedelta(days=1)
            
            if temp_chunk_files:
                # Combine new chunks efficiently using pyarrow ParquetDataset
                import pyarrow.parquet as pq
                import pyarrow as pa
                
                # Combine new chunks using pyarrow (more memory efficient)
                new_enterprise_value_path = data_path / 'enterprise_value_new.parquet'
                if len(temp_chunk_files) > 0:
                    # Read all temp files as a dataset and write to single file
                    tables = []
                    for temp_file in temp_chunk_files:
                        table = pq.read_table(temp_file)
                        tables.append(table)
                        temp_file.unlink()
                    
                    # Concatenate tables and write
                    combined_table = pa.concat_tables(tables)
                    pq.write_table(combined_table, new_enterprise_value_path, compression='snappy')
                    del tables, combined_table
                
                # Now combine with existing data using streaming approach
                # Filter existing data to remove overlapping dates
                overlap_mask = pd.Series(False, index=existing_enterprise_value.index)
                for range_start, range_end in ev_download_ranges:
                    range_mask = (existing_enterprise_value['DATE'] >= range_start) & (existing_enterprise_value['DATE'] <= range_end)
                    overlap_mask = overlap_mask | range_mask
                existing_enterprise_value_filtered = existing_enterprise_value[~overlap_mask]
                
                # Read new data
                new_enterprise_value_df = pd.read_parquet(new_enterprise_value_path)
                
                # Combine using pyarrow for memory efficiency
                existing_table = pa.Table.from_pandas(existing_enterprise_value_filtered)
                new_table = pa.Table.from_pandas(new_enterprise_value_df)
                combined_table = pa.concat_tables([existing_table, new_table])
                
                # Convert back to pandas for sorting and deduplication
                combined_enterprise_value = combined_table.to_pandas()
                combined_enterprise_value = combined_enterprise_value.sort_values('DATE')
                combined_enterprise_value = combined_enterprise_value.drop_duplicates(
                    subset=['FACTSET_ID', 'DATE'], 
                    keep='last'
                )
                
                print(f"   ✓ Combined: {len(combined_enterprise_value):,} total enterprise value rows")
                combined_enterprise_value.to_parquet(enterprise_value_path, index=False, compression='snappy')
                print(f"   ✓ Saved to {enterprise_value_path}")
                
                # Clean up
                new_enterprise_value_path.unlink()
                del existing_table, new_table, combined_table, combined_enterprise_value, new_enterprise_value_df, existing_enterprise_value_filtered
    else:
        # No existing enterprise value - download from 2020-01-01 onwards (no lookback)
        print(f"\n[7] Downloading enterprise value (2020-01-01 to {end_date})...")
        print("   (This may take a while - downloading in chunks...)")
        
        start_dt = pd.to_datetime('2020-01-01')
        end_dt = pd.to_datetime(end_date)
        current_dt = start_dt
        
        # Use incremental writing to avoid memory issues
        import pyarrow.parquet as pq
        import pyarrow as pa
        
        temp_chunk_files = []
        chunk_num = 0
        total_rows = 0
        
        while current_dt <= end_dt:
            chunk_end = min(current_dt + timedelta(days=365), end_dt)
            chunk_num += 1
            
            print(f"\n   Chunk {chunk_num}: {current_dt.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
            with tqdm(total=1, desc="Querying enterprise value", bar_format='{desc}: {elapsed}') as pbar:
                ev_chunk = retriever.get_enterprise_value_data_by_date_range(
                    start_date=current_dt.strftime('%Y-%m-%d'),
                    end_date=chunk_end.strftime('%Y-%m-%d')
                )
                pbar.update(1)
            
            if len(ev_chunk) > 0:
                ev_chunk['DATE'] = pd.to_datetime(ev_chunk['DATE'])
                print(f"      ✓ Got {len(ev_chunk):,} rows")
                total_rows += len(ev_chunk)
                
                # Save chunk to temporary file immediately (frees memory)
                temp_file = data_path / f'enterprise_value_temp_chunk_{chunk_num}.parquet'
                ev_chunk.to_parquet(temp_file, index=False, compression='snappy')
                temp_chunk_files.append(temp_file)
                del ev_chunk  # Free memory
            
            current_dt = chunk_end + timedelta(days=1)
        
        # Combine temporary files efficiently using pyarrow (memory efficient)
        if temp_chunk_files:
            print(f"\n   ✓ Total enterprise value: {total_rows:,} rows")
            print("   Combining chunks and saving to final Parquet file...")
            
            # Use pyarrow to combine all chunks at once (more memory efficient)
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            # Read all temp files as tables (ensure files are closed before unlinking)
            tables = []
            for temp_file in temp_chunk_files:
                table = pq.read_table(temp_file)
                tables.append(table)
                # Close the file explicitly and wait a moment before unlinking
                del table
                import time
                time.sleep(0.1)  # Brief pause to ensure file is released
                try:
                    temp_file.unlink()
                except PermissionError:
                    print(f"   ⚠ Warning: Could not delete {temp_file.name} (file may be in use, will retry later)")
            
            # Concatenate all tables and write to final file (filter to 2020-01-01 onwards)
            if len(tables) > 0:
                combined_table = pa.concat_tables(tables)
                del tables  # Free memory
                
                print("   Filtering to dates >= 2020-01-01...")
                cutoff_date = pd.to_datetime('2020-01-01')
                filtered_chunks = []
                
                # Process in batches to avoid memory issues
                batch_size = 10_000_000
                for i in range(0, len(combined_table), batch_size):
                    batch = combined_table.slice(i, min(batch_size, len(combined_table) - i))
                    batch_df = batch.to_pandas()
                    batch_df['DATE'] = pd.to_datetime(batch_df['DATE'])
                    
                    # Filter to 2020-01-01 onwards
                    batch_filtered = batch_df[batch_df['DATE'] >= cutoff_date].copy()
                    
                    if len(batch_filtered) > 0:
                        filtered_chunks.append(batch_filtered)
                    del batch_df, batch_filtered
                
                del combined_table
                
                if len(filtered_chunks) > 0:
                    # Combine filtered chunks
                    filtered_df = pd.concat(filtered_chunks, ignore_index=True)
                    del filtered_chunks
                    
                    # Sort and deduplicate
                    print("   Sorting and deduplicating...")
                    filtered_df = filtered_df.sort_values('DATE')
                    filtered_df = filtered_df.drop_duplicates(
                        subset=['FACTSET_ID', 'DATE'], 
                        keep='last'
                    )
                    
                    print(f"   ✓ After filtering: {len(filtered_df):,} rows")
                    print("   Writing to Parquet...")
                    filtered_df.to_parquet(enterprise_value_path, index=False, compression='snappy')
                    del filtered_df
                else:
                    print("   ⚠ No data remaining after filtering!")
            
            print(f"   ✓ Saved to {enterprise_value_path}")
    
    retriever.disconnect()
    
    print("\n" + "=" * 80)
    print("Bulk Download Complete!")
    print("=" * 80)
    print(f"✓ All data saved to {data_dir}/")
    print(f"  - prices.parquet")
    print(f"  - returns.parquet")
    print(f"  - fundamentals.parquet")
    print(f"  - universe.parquet & universe.csv")
    print(f"  - exchange_rates.parquet")
    print(f"  - market_value.parquet")
    print(f"  - enterprise_value.parquet")
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