"""
Process factor exposures by quarter in parallel
Much faster than processing dates sequentially
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import config
from ring_buffers import StockCalculatorManager
from continent_mapping import get_continent
import time


def get_quarter_dates(year: int, quarter: int) -> tuple:
    """Get start and end dates for a quarter"""
    if quarter == 1:
        start = f"{year}-01-01"
        end = f"{year}-03-31"
    elif quarter == 2:
        start = f"{year}-04-01"
        end = f"{year}-06-30"
    elif quarter == 3:
        start = f"{year}-07-01"
        end = f"{year}-09-30"
    else:  # Q4
        start = f"{year}-10-01"
        end = f"{year}-12-31"
    return start, end


def get_quarters_in_range(start_date: str, end_date: str) -> List[tuple]:
    """Get list of (year, quarter) tuples for date range"""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    quarters = []
    current = start_dt
    
    while current <= end_dt:
        year = current.year
        quarter = (current.month - 1) // 3 + 1
        quarters.append((year, quarter))
        
        # Move to next quarter
        if quarter == 4:
            current = pd.Timestamp(year=year + 1, month=1, day=1)
        else:
            current = pd.Timestamp(year=year, month=quarter * 3 + 1, day=1)
    
    return quarters


def compute_exposures_for_quarter(
    year: int,
    quarter: int,
    output_dir: Path,
    universe_df: pd.DataFrame,
    prices_path: Path,
    returns_path: Path,
    fundamentals_path: Path,
    start_date: str,
    end_date: str,
    neutralize: bool = False
) -> Dict:
    """
    Process all dates in a quarter
    
    Returns:
        Dict with 'processed', 'skipped', 'quarter', 'csv_path' keys
    """
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting {year}Q{quarter}...")
    
    quarter_start, quarter_end = get_quarter_dates(year, quarter)
    quarter_start_dt = pd.to_datetime(quarter_start)
    quarter_end_dt = pd.to_datetime(quarter_end)
    
    # CRITICAL FIX: Load data per-quarter from Parquet (chunked) instead of using pre-loaded DataFrames
    # Calculate lookback for this quarter (400 days before quarter start)
    lookback_start_dt = quarter_start_dt - timedelta(days=400)
    
    # Load prices for this quarter + lookback (chunked to avoid memory issues)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loading prices...")
    prices_start = time.time()
    prices_chunks = []
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(prices_path)
        batch_count = 0
        for batch in parquet_file.iter_batches(batch_size=500000):  # Smaller chunks
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Processed {batch_count} price batches...")
            chunk_df = batch.to_pandas()
            chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
            filtered = chunk_df[
                (chunk_df['DATE'] >= lookback_start_dt) & 
                (chunk_df['DATE'] <= quarter_end_dt)
            ]
            if len(filtered) > 0:
                prices_chunks.append(filtered)
        all_prices = pd.concat(prices_chunks, ignore_index=True) if prices_chunks else pd.DataFrame()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loaded {len(all_prices):,} price rows in {time.time() - prices_start:.1f}s")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading prices for {year}Q{quarter}: {e}")
        import traceback
        traceback.print_exc()
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    # Load returns for this quarter + lookback (chunked)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loading returns...")
    returns_start = time.time()
    returns_chunks = []
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(returns_path)
        batch_count = 0
        for batch in parquet_file.iter_batches(batch_size=500000):  # Smaller chunks
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Processed {batch_count} return batches...")
            chunk_df = batch.to_pandas()
            if 'FSYM_ID' in chunk_df.columns:
                chunk_df = chunk_df.rename(columns={'FSYM_ID': 'FACTSET_ID'})
            if 'P_DATE' in chunk_df.columns:
                chunk_df['P_DATE'] = pd.to_datetime(chunk_df['P_DATE'])
                chunk_df = chunk_df.rename(columns={'P_DATE': 'DATE'})
            elif 'DATE' in chunk_df.columns:
                chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
            filtered = chunk_df[
                (chunk_df['DATE'] >= lookback_start_dt) & 
                (chunk_df['DATE'] <= quarter_end_dt)
            ]
            if len(filtered) > 0:
                returns_chunks.append(filtered)
        all_returns = pd.concat(returns_chunks, ignore_index=True) if returns_chunks else pd.DataFrame()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loaded {len(all_returns):,} return rows in {time.time() - returns_start:.1f}s")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading returns for {year}Q{quarter}: {e}")
        import traceback
        traceback.print_exc()
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    # Now filter for just this quarter (smaller DataFrames)
    quarter_prices = all_prices[
        (all_prices['DATE'] >= quarter_start_dt) & 
        (all_prices['DATE'] <= quarter_end_dt)
    ].copy()
    quarter_returns = all_returns[
        (all_returns['DATE'] >= quarter_start_dt) & 
        (all_returns['DATE'] <= quarter_end_dt)
    ].copy()
    
    # Load fundamentals for this quarter only (chunked read to avoid memory issues)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loading fundamentals...")
    fund_start = time.time()
    quarter_fundamentals_chunks = []
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(fundamentals_path)
        batch_count = 0
        for batch in parquet_file.iter_batches(batch_size=500000):  # Smaller chunks
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Processed {batch_count} fundamental batches...")
            chunk_df = batch.to_pandas()
            chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
            filtered = chunk_df[
                (chunk_df['DATE'] >= quarter_start_dt) & 
                (chunk_df['DATE'] <= quarter_end_dt)
            ]
            if len(filtered) > 0:
                quarter_fundamentals_chunks.append(filtered)
        quarter_fundamentals = pd.concat(quarter_fundamentals_chunks, ignore_index=True) if quarter_fundamentals_chunks else pd.DataFrame()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loaded {len(quarter_fundamentals):,} fundamental rows in {time.time() - fund_start:.1f}s")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading fundamentals for {year}Q{quarter}: {e}")
        import traceback
        traceback.print_exc()
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    if len(quarter_prices) == 0 or len(quarter_fundamentals) == 0:
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}"}
    
    trading_dates = sorted(quarter_prices['DATE'].unique())
    
    if len(trading_dates) == 0:
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    # Filter dates to process range
    trading_dates = [d for d in trading_dates if pd.to_datetime(start_date) <= pd.to_datetime(d) <= pd.to_datetime(end_date)]
    
    if len(trading_dates) == 0:
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    # OPTIMIZATION: Index DataFrames by DATE for O(1) lookups
    quarter_prices_idx = quarter_prices.set_index('DATE', drop=False)
    quarter_returns_idx = quarter_returns.set_index('DATE', drop=False)
    quarter_fundamentals_idx = quarter_fundamentals.set_index('DATE', drop=False)
    
    # Initialize calculator manager for this quarter
    calculator_manager = StockCalculatorManager()
    
    # Populate historical buffers BEFORE processing dates
    # This ensures MOMENTUM, VOLATILITY, LIQUIDITY are available from the first date
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Populating historical buffers...")
    hist_start = time.time()
    
    # Get all historical dates (before the quarter start) for buffer population
    # Use all_prices/all_returns which include the full lookback period
    quarter_start = pd.to_datetime(start_date)
    historical_dates = sorted([d for d in all_prices['DATE'].unique() 
                              if pd.to_datetime(d) < quarter_start])
    
    # Process historical dates to populate buffers (only need to go back enough for calculations)
    # Momentum needs 252+21=273 days, so we need at least that many historical dates
    # Process in chronological order (oldest to newest) to build buffers correctly
    if len(historical_dates) > 0:
        # Only process the last ~300 days of history (enough for momentum calculation)
        # This is efficient - we don't need to process all 400 days, just enough for calculations
        historical_dates_to_process = historical_dates[-300:] if len(historical_dates) > 300 else historical_dates
        
        hist_processed = 0
        for hist_date in historical_dates_to_process:
            hist_returns = all_returns[all_returns['DATE'] == hist_date].copy()
            hist_prices = all_prices[all_prices['DATE'] == hist_date].copy()
            
            if len(hist_returns) > 0 and len(hist_prices) > 0:
                # Merge and update buffers
                hist_merged = hist_returns[['FACTSET_ID', 'ONE_DAY_PCT']].merge(
                    hist_prices[['FACTSET_ID', 'ADJUSTED_PRICE', 'ADJUSTED_VOLUME']],
                    on='FACTSET_ID',
                    how='inner'
                )
                hist_merged['DOLLAR_VOL'] = hist_merged['ADJUSTED_PRICE'] * hist_merged['ADJUSTED_VOLUME']
                
                # Update calculators with historical data
                for row in hist_merged.itertuples():
                    calculator_manager.update_stock(
                        row.FACTSET_ID,
                        pd.to_datetime(hist_date),
                        row.ONE_DAY_PCT,
                        row.DOLLAR_VOL
                    )
                hist_processed += 1
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Populated buffers with {hist_processed} historical dates in {time.time() - hist_start:.1f}s")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: No historical dates available (buffers will build during processing)")
    
    # OPTIMIZATION: Process all dates and batch write results
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Processing {len(trading_dates)} trading dates...")
    process_start = time.time()
    all_exposures = []
    processed = 0
    skipped = 0
    
    for i, date_T in enumerate(trading_dates):
        if i % 10 == 0 or i == len(trading_dates) - 1:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Processing date {i+1}/{len(trading_dates)} ({date_T})...")
        # OPTIMIZATION: Use .loc with datetime index (O(1) lookup, not O(n) scan)
        try:
            fundamentals_T = quarter_fundamentals_idx.loc[[date_T]].copy() if date_T in quarter_fundamentals_idx.index else pd.DataFrame()
            prices_T = quarter_prices_idx.loc[[date_T]].copy() if date_T in quarter_prices_idx.index else pd.DataFrame()
            returns_T = quarter_returns_idx.loc[[date_T]].copy() if date_T in quarter_returns_idx.index else pd.DataFrame()
        except (KeyError, TypeError):
            # Fallback if date not in index
            fundamentals_T = quarter_fundamentals[quarter_fundamentals['DATE'] == date_T].copy()
            prices_T = quarter_prices[quarter_prices['DATE'] == date_T].copy()
            returns_T = quarter_returns[quarter_returns['DATE'] == date_T].copy()
        
        if len(fundamentals_T) == 0 or len(prices_T) == 0 or len(returns_T) == 0:
            skipped += 1
            continue
        
        # OPTIMIZATION: Vectorized calculator updates (merge first, then iterate once)
        # Merge returns and prices first (no iterrows on returns_T!)
        returns_prices = returns_T[['FACTSET_ID', 'ONE_DAY_PCT']].merge(
            prices_T[['FACTSET_ID', 'ADJUSTED_PRICE', 'ADJUSTED_VOLUME']],
            on='FACTSET_ID',
            how='inner'
        )
        returns_prices['DOLLAR_VOL'] = returns_prices['ADJUSTED_PRICE'] * returns_prices['ADJUSTED_VOLUME']
        
        # Update calculators (use itertuples, faster than iterrows)
        for row in returns_prices.itertuples():
            calculator_manager.update_stock(
                row.FACTSET_ID,
                date_T,
                row.ONE_DAY_PCT,
                row.DOLLAR_VOL
            )
        
        # Merge data
        df_T = fundamentals_T.merge(
            universe_df[['FACTSET_ID', 'SECTOR', 'CONTINENT']],
            on='FACTSET_ID',
            how='inner'
        ).merge(
            prices_T[['FACTSET_ID', 'ADJUSTED_PRICE', 'ADJUSTED_VOLUME']],
            on='FACTSET_ID',
            how='inner'
        )
        
        if len(df_T) == 0:
            skipped += 1
            continue
        
        # Calculate characteristics (same as before)
        # Value
        df_T['EY_NTM'] = df_T['EPS_NTM'] / df_T['ADJUSTED_PRICE']
        df_T['SY_NTM'] = df_T['SALES_NTM'] / df_T['ADJUSTED_PRICE']
        df_T['EBITDA_Y_NTM'] = df_T['EBITDA_NTM'] / df_T['ADJUSTED_PRICE']
        df_T['EY_LTM'] = df_T['EPS_LTM'] / df_T['ADJUSTED_PRICE']
        df_T['SY_LTM'] = df_T['SALES_LTM'] / df_T['ADJUSTED_PRICE']
        
        df_T.loc[df_T['EPS_NTM'] <= 0, 'EY_NTM'] = np.nan
        df_T.loc[df_T['EBITDA_NTM'] <= 0, 'EBITDA_Y_NTM'] = np.nan
        df_T.loc[df_T['EPS_LTM'] <= 0, 'EY_LTM'] = np.nan
        
        # Profitability
        df_T['EBITDA_MARGIN'] = df_T['EBITDA_LTM'] / df_T['SALES_LTM']
        df_T['GROSS_MARGIN'] = 1 - (df_T['COGS_LTM'] / df_T['SALES_LTM'])
        
        df_T.loc[df_T['SALES_LTM'] <= 0, 'EBITDA_MARGIN'] = np.nan
        df_T.loc[df_T['SALES_LTM'] <= 0, 'GROSS_MARGIN'] = np.nan
        df_T.loc[df_T['COGS_LTM'] <= 0, 'GROSS_MARGIN'] = np.nan
        
        # Growth
        df_T['EPS_GROWTH'] = (df_T['EPS_NTM'] / df_T['EPS_LTM']) - 1
        df_T['SALES_GROWTH'] = (df_T['SALES_NTM'] / df_T['SALES_LTM']) - 1
        
        df_T.loc[df_T['EPS_LTM'] <= 0, 'EPS_GROWTH'] = np.nan
        df_T.loc[df_T['SALES_LTM'] <= 0, 'SALES_GROWTH'] = np.nan
        
        df_T = df_T.replace([np.inf, -np.inf], np.nan)
        
        # OPTIMIZATION: Batch get momentum/volatility/liquidity 
        # Use vectorized operations where possible
        factset_ids = df_T['FACTSET_ID'].values
        df_T['MOMENTUM'] = [calculator_manager.get_momentum(fid) for fid in factset_ids]
        df_T['VOLATILITY'] = [calculator_manager.get_volatility(fid) for fid in factset_ids]
        df_T['LIQUIDITY'] = [calculator_manager.get_liquidity(fid) for fid in factset_ids]
        
        # Winsorize
        winsorize_lower = config.FACTOR_PARAMS['winsorize_lower']
        winsorize_upper = config.FACTOR_PARAMS['winsorize_upper']
        
        value_chars = ['EY_NTM', 'SY_NTM', 'EBITDA_Y_NTM', 'EY_LTM', 'SY_LTM']
        profitability_chars = ['EBITDA_MARGIN', 'GROSS_MARGIN']
        growth_chars = ['EPS_GROWTH', 'SALES_GROWTH']
        all_chars = value_chars + profitability_chars + growth_chars + ['MOMENTUM', 'VOLATILITY', 'LIQUIDITY']
        
        for char in all_chars:
            if char not in df_T.columns:
                continue
            values = df_T[char].dropna()
            if len(values) > 0:
                lower_bound = values.quantile(winsorize_lower)
                upper_bound = values.quantile(winsorize_upper)
                df_T[char] = df_T[char].clip(lower=lower_bound, upper=upper_bound)
        
        # Standardize
        for char in all_chars:
            if char not in df_T.columns:
                continue
            values = df_T[char].dropna()
            if len(values) > 1:
                mean_val = values.mean()
                std_val = values.std()
                if std_val > 0:
                    df_T[char] = (df_T[char] - mean_val) / std_val
        
        # Combine into style factors
        df_T['VALUE'] = df_T[value_chars].mean(axis=1, skipna=True)
        if df_T['VALUE'].notna().sum() > 1:
            val_mean = df_T['VALUE'].mean()
            val_std = df_T['VALUE'].std()
            if val_std > 0:
                df_T['VALUE'] = (df_T['VALUE'] - val_mean) / val_std
        
        df_T['PROFITABILITY'] = df_T[profitability_chars].mean(axis=1, skipna=True)
        if df_T['PROFITABILITY'].notna().sum() > 1:
            prof_mean = df_T['PROFITABILITY'].mean()
            prof_std = df_T['PROFITABILITY'].std()
            if prof_std > 0:
                df_T['PROFITABILITY'] = (df_T['PROFITABILITY'] - prof_mean) / prof_std
        
        df_T['GROWTH'] = df_T[growth_chars].mean(axis=1, skipna=True)
        if df_T['GROWTH'].notna().sum() > 1:
            growth_mean = df_T['GROWTH'].mean()
            growth_std = df_T['GROWTH'].std()
            if growth_std > 0:
                df_T['GROWTH'] = (df_T['GROWTH'] - growth_mean) / growth_std
        
        # Create dummies
        sector_dummies = pd.get_dummies(df_T['SECTOR'], prefix='SECTOR')
        continent_dummies = pd.get_dummies(df_T['CONTINENT'], prefix='CONTINENT')
        
        sector_dummies.columns = sector_dummies.columns.str.replace(' ', '_')
        continent_dummies.columns = continent_dummies.columns.str.replace(' ', '_')
        
        sector_dummies = sector_dummies.astype(float)
        continent_dummies = continent_dummies.astype(float)
        
        # Sum-to-zero
        for col in sector_dummies.columns:
            col_mean = sector_dummies[col].mean()
            sector_dummies[col] = sector_dummies[col] - col_mean
        
        for col in continent_dummies.columns:
            col_mean = continent_dummies[col].mean()
            continent_dummies[col] = continent_dummies[col] - col_mean
        
        # Combine exposures
        # Ensure MOMENTUM, VOLATILITY, LIQUIDITY are included even if some values are NaN
        style_factor_cols = ['FACTSET_ID', 'VALUE', 'PROFITABILITY', 'GROWTH']
        for factor in ['MOMENTUM', 'VOLATILITY', 'LIQUIDITY']:
            if factor in df_T.columns:
                style_factor_cols.append(factor)
        
        style_factors = df_T[style_factor_cols].copy()
        style_factors = style_factors.set_index('FACTSET_ID')
        sector_dummies.index = df_T['FACTSET_ID'].values
        continent_dummies.index = df_T['FACTSET_ID'].values
        
        exposures_df = (
            style_factors
            .merge(sector_dummies, left_index=True, right_index=True, how='left')
            .merge(continent_dummies, left_index=True, right_index=True, how='left')
            .reset_index()
        )
        
        exposures_df.insert(1, 'DATE', date_T)
        
        # Collect for batch write (MAJOR SPEEDUP - write entire quarter at once!)
        all_exposures.append(exposures_df)
        processed += 1
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Date processing completed in {time.time() - process_start:.1f}s ({processed} processed, {skipped} skipped)")
    
    # Write quarter exposures to CSV file
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Writing CSV file...")
    write_start = time.time()
    csv_path = None
    if all_exposures:
        # Fix: Ensure all DataFrames have the same columns before concatenating
        # This prevents FutureWarning about all-NA columns
        if len(all_exposures) > 1:
            # Get union of all columns across all dates
            all_columns = set()
            for df in all_exposures:
                all_columns.update(df.columns)
            all_columns = sorted(list(all_columns))
            
            # Reindex each DataFrame to have all columns (fill missing with 0 for dummies, NaN for factors)
            aligned_exposures = []
            for df in all_exposures:
                # Create new DataFrame with all columns
                df_aligned = pd.DataFrame(index=df.index, columns=all_columns)
                # Copy existing columns
                for col in df.columns:
                    df_aligned[col] = df[col]
                # Fill missing dummy columns with 0 (they should be 0 if sector/continent not present)
                for col in all_columns:
                    if col not in df.columns:
                        if col.startswith('SECTOR_') or col.startswith('CONTINENT_'):
                            df_aligned[col] = 0.0
                        else:
                            df_aligned[col] = np.nan
                aligned_exposures.append(df_aligned)
            
            combined_exposures = pd.concat(aligned_exposures, ignore_index=True)
        else:
            combined_exposures = all_exposures[0]
        
        # Ensure DATE is formatted as string
        if 'DATE' in combined_exposures.columns:
            combined_exposures['DATE'] = pd.to_datetime(combined_exposures['DATE']).dt.strftime('%Y-%m-%d')
        
        # Write to CSV (one file per quarter for parallel processing)
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / f'exposures_{year}Q{quarter}.csv'
        combined_exposures.to_csv(csv_path, index=False)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Wrote {len(combined_exposures):,} rows to CSV in {time.time() - write_start:.1f}s")
    
    total_time = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: COMPLETE in {total_time:.1f}s total")
    
    return {'processed': processed, 'skipped': skipped, 'quarter': f"{year}Q{quarter}", 'csv_path': csv_path}


def process_quarters_parallel(
    start_date: str = '2025-01-01',
    end_date: str = None,
    output_dir: str = 'results',
    data_dir: str = 'data',
    max_workers: int = None,
    neutralize: bool = False
) -> Dict:
    """
    Process quarters in parallel for speed - works entirely with local Parquet files!
    
    Args:
        start_date: Start date
        end_date: End date
        output_dir: Directory to save CSV files
        data_dir: Directory with Parquet files (prices.parquet, returns.parquet, etc.)
        max_workers: Number of parallel workers (default: CPU cores)
        neutralize: Whether to neutralize factors
    """
    from data_retrieval import get_date_range
    from pathlib import Path
    import os
    
    if end_date is None:
        _, end_date = get_date_range(lookback_days=1)
    
    print("=" * 80)
    print("Quarter-Based Parallel Processing (Local Parquet Files)")
    print("=" * 80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(data_dir)
    
    # Check if data files exist
    prices_path = data_path / 'prices.parquet'
    returns_path = data_path / 'returns.parquet'
    fundamentals_path = data_path / 'fundamentals.parquet'
    universe_path = data_path / 'universe.parquet'
    
    if not prices_path.exists() or not returns_path.exists() or not fundamentals_path.exists():
        print(f"\n⚠ ERROR: Data files not found in {data_dir}/")
        print("   Please run 'python bulk_download.py' first to download data from Snowflake")
        return {'processed': 0, 'skipped': 0}
    
    # Don't create DB connection here - each worker thread creates its own (SQLite is not thread-safe!)
    
    # CRITICAL FIX: Don't load ALL data upfront - each quarter loads only what it needs
    # This avoids memory explosion when multiple workers try to copy 50M+ row DataFrames
    print("\n[1] Loading metadata (data loaded per-quarter to avoid memory issues)...")
    
    # Load only universe (small file) - can be shared across threads (read-only)
    if universe_path.exists():
        universe_df = pd.read_parquet(universe_path)
    else:
        csv_path = data_path / 'universe.csv'
        if csv_path.exists():
            universe_df = pd.read_csv(csv_path)
        else:
            print("   ⚠ universe file not found")
            return {'processed': 0, 'skipped': 0}
    
    if 'COUNTRY' in universe_df.columns:
        from continent_mapping import get_continent_developed
        universe_df['CONTINENT'] = universe_df['COUNTRY'].apply(get_continent_developed)
    
    print(f"   ✓ Universe: {len(universe_df):,} stocks")
    print(f"   ✓ Prices/Returns/Fundamentals: Loading per-quarter from Parquet (avoids memory issues)")
    
    # Get quarters to process
    quarters = get_quarters_in_range(start_date, end_date)
    print(f"\n[2] Processing {len(quarters)} quarters in parallel...")
    
    if max_workers is None:
        # REDUCE workers to avoid memory issues - each worker loads full DataFrames
        # With 19 workers and 50M row DataFrames, we'd need 19 * 500MB = 9.5GB+ just for data copies
        max_workers = min(4, max(1, os.cpu_count() - 1))  # Limit to 4 workers max
    
    print(f"   Using {max_workers} parallel workers (limited to avoid memory issues)")
    
    # Process quarters in parallel
    processed_total = 0
    skipped_total = 0
    
    quarter_csv_files = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_quarter = {
            executor.submit(
                compute_exposures_for_quarter,
                year, quarter, output_path, universe_df, prices_path, returns_path, fundamentals_path, start_date, end_date, neutralize
            ): (year, quarter)
            for year, quarter in quarters
        }
        
        for future in tqdm(as_completed(future_to_quarter), total=len(quarters), desc="Processing quarters"):
            year, quarter = future_to_quarter[future]
            try:
                result = future.result()
                processed_total += result['processed']
                skipped_total += result['skipped']
                if result.get('csv_path'):
                    quarter_csv_files.append(result['csv_path'])
            except Exception as e:
                print(f"\n⚠ Error processing {year}Q{quarter}: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Combine all quarter CSV files into one exposures.csv
    exposures_csv_path = output_path / 'exposures.csv'
    if quarter_csv_files:
        print(f"\n[3] Combining {len(quarter_csv_files)} quarter files into exposures.csv...")
        exposure_chunks = []
        for csv_file in quarter_csv_files:
            chunk = pd.read_csv(csv_file)
            exposure_chunks.append(chunk)
        
        combined_exposures = pd.concat(exposure_chunks, ignore_index=True)
        # Sort by DATE, FACTSET_ID
        combined_exposures['DATE'] = pd.to_datetime(combined_exposures['DATE'])
        combined_exposures = combined_exposures.sort_values(['DATE', 'FACTSET_ID'])
        
        # Convert to long format (MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE)
        from model_builder import FactorModelBuilder
        combined_exposures_long = FactorModelBuilder.convert_exposures_to_long_format(combined_exposures)
        
        # Sort by DATE, SECURITY_ID, FACTOR_NAME
        combined_exposures_long = combined_exposures_long.sort_values(['DATE', 'SECURITY_ID', 'FACTOR_NAME'])
        combined_exposures_long['DATE'] = pd.to_datetime(combined_exposures_long['DATE']).dt.strftime('%Y-%m-%d')
        combined_exposures_long.to_csv(exposures_csv_path, index=False)
        
        print(f"   ✓ Combined and converted {len(combined_exposures_long):,} exposure rows (long format) → {exposures_csv_path}")
        
        # Clean up temporary quarter files
        for csv_file in quarter_csv_files:
            csv_file.unlink()
    
    print(f"\n✓ Processed {processed_total} dates")
    print(f"  Skipped {skipped_total} dates")
    
    return {'processed': processed_total, 'skipped': skipped_total, 'exposures_csv': exposures_csv_path}


if __name__ == "__main__":
    process_quarters_parallel(
        start_date='2020-01-01',
        end_date=None,
        output_dir='results',
        data_dir='data',
        max_workers=None,
        neutralize=False
    )

