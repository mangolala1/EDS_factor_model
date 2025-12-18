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
from . import config
from .ring_buffers import StockCalculatorManager
from .continent_mapping import get_continent
from .sequential_snapshot import FundamentalsSnapshot, EnterpriseValueSnapshot, ExchangeRatesSnapshot
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
    neutralize: bool = False,
    market_value_path: Optional[Path] = None,
    exchange_rates_path: Optional[Path] = None,
    enterprise_value_path: Optional[Path] = None
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
            
            # Reset index to avoid duplicate label issues
            chunk_df = chunk_df.reset_index(drop=True)
            
            chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
            # Filter by date range (data only contains trading days)
            date_mask = (chunk_df['DATE'].values >= lookback_start_dt) & (chunk_df['DATE'].values <= quarter_end_dt)
            filtered = chunk_df.iloc[date_mask].copy()
            
            if len(filtered) > 0:
                prices_chunks.append(filtered)
        all_prices = pd.concat(prices_chunks, ignore_index=True) if prices_chunks else pd.DataFrame()
        # Remove duplicate columns after concatenation
        if len(all_prices) > 0:
            all_prices = all_prices.loc[:, ~all_prices.columns.duplicated()]
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
            
            # Reset index to avoid duplicate label issues - ensure simple integer index
            chunk_df = chunk_df.reset_index(drop=True)
            
            # Remove duplicate column names first
            chunk_df = chunk_df.loc[:, ~chunk_df.columns.duplicated()]
            
            if 'FSYM_ID' in chunk_df.columns:
                chunk_df = chunk_df.rename(columns={'FSYM_ID': 'FACTSET_ID'})
            
            # Handle DATE column - prefer P_DATE if both exist
            if 'P_DATE' in chunk_df.columns:
                chunk_df['P_DATE'] = pd.to_datetime(chunk_df['P_DATE'])
                # If DATE also exists, drop it first to avoid duplicates
                if 'DATE' in chunk_df.columns:
                    chunk_df = chunk_df.drop(columns=['DATE'])
                chunk_df = chunk_df.rename(columns={'P_DATE': 'DATE'})
            elif 'DATE' in chunk_df.columns:
                chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
            
            # Check if DATE column exists
            if 'DATE' not in chunk_df.columns:
                continue
            
            # Ensure DATE is a single column (not DataFrame)
            if isinstance(chunk_df['DATE'], pd.DataFrame):
                chunk_df['DATE'] = chunk_df['DATE'].iloc[:, 0]
            
            # Create boolean mask - ensure it's a 1D numpy array
            date_mask = (chunk_df['DATE'].values >= lookback_start_dt) & (chunk_df['DATE'].values <= quarter_end_dt)
            
            # Filter using boolean indexing with numpy array (avoids MultiIndex issues)
            filtered = chunk_df.iloc[date_mask].copy()
            
            if len(filtered) > 0:
                returns_chunks.append(filtered)
        all_returns = pd.concat(returns_chunks, ignore_index=True) if returns_chunks else pd.DataFrame()
        # Remove duplicate columns after concatenation
        if len(all_returns) > 0:
            all_returns = all_returns.loc[:, ~all_returns.columns.duplicated()]
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loaded {len(all_returns):,} return rows in {time.time() - returns_start:.1f}s")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading returns for {year}Q{quarter}: {e}")
        import traceback
        traceback.print_exc()
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    # Now filter for just this quarter (smaller DataFrames)
    # Data only contains trading days, so no holiday filtering needed
    # Use .iloc with boolean numpy array to avoid MultiIndex issues
    price_mask = (all_prices['DATE'].values >= quarter_start_dt) & (all_prices['DATE'].values <= quarter_end_dt)
    quarter_prices = all_prices.iloc[price_mask].copy()
    
    return_mask = (all_returns['DATE'].values >= quarter_start_dt) & (all_returns['DATE'].values <= quarter_end_dt)
    quarter_returns = all_returns.iloc[return_mask].copy()
    
    # OPTIMIZATION: Use sequential snapshot for fundamentals (10-100x faster than date-filtered reads)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Initializing fundamentals snapshot...")
    fund_start = time.time()
    fundamentals_snapshot = None
    try:
        fundamentals_snapshot = FundamentalsSnapshot(fundamentals_path)
        fundamentals_snapshot.start()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Fundamentals snapshot initialized in {time.time() - fund_start:.1f}s")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error initializing fundamentals snapshot for {year}Q{quarter}: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to old method if snapshot fails
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Falling back to date-filtered read...")
        fund_start = time.time()
        quarter_fundamentals_chunks = []
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(fundamentals_path)
            batch_count = 0
            for batch in parquet_file.iter_batches(batch_size=500000):
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Processed {batch_count} fundamental batches...")
                chunk_df = batch.to_pandas()
                chunk_df = chunk_df.reset_index(drop=True)
                chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
                date_mask = (chunk_df['DATE'].values >= quarter_start_dt) & (chunk_df['DATE'].values <= quarter_end_dt)
                filtered = chunk_df.iloc[date_mask].copy()
                if len(filtered) > 0:
                    quarter_fundamentals_chunks.append(filtered)
            quarter_fundamentals = pd.concat(quarter_fundamentals_chunks, ignore_index=True) if quarter_fundamentals_chunks else pd.DataFrame()
            if len(quarter_fundamentals) > 0:
                quarter_fundamentals = quarter_fundamentals.loc[:, ~quarter_fundamentals.columns.duplicated()]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loaded {len(quarter_fundamentals):,} fundamental rows in {time.time() - fund_start:.1f}s")
        except Exception as e2:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading fundamentals for {year}Q{quarter}: {e2}")
            return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    if len(quarter_prices) == 0:
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}"}
    
    # Load market value data for this quarter (if available)
    quarter_market_value = pd.DataFrame()
    if market_value_path and market_value_path.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loading market value...")
        mv_start = time.time()
        try:
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(market_value_path)
            mv_chunks = []
            for batch in parquet_file.iter_batches(batch_size=500000):
                chunk_df = batch.to_pandas()
                
                # Reset index to avoid duplicate label issues
                chunk_df = chunk_df.reset_index(drop=True)
                
                chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
                date_mask = (chunk_df['DATE'].values >= quarter_start_dt) & (chunk_df['DATE'].values <= quarter_end_dt)
                filtered = chunk_df.iloc[date_mask].copy()
                
                if len(filtered) > 0:
                    mv_chunks.append(filtered)
            quarter_market_value = pd.concat(mv_chunks, ignore_index=True) if mv_chunks else pd.DataFrame()
            # Remove duplicate columns after concatenation
            if len(quarter_market_value) > 0:
                quarter_market_value = quarter_market_value.loc[:, ~quarter_market_value.columns.duplicated()]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Loaded {len(quarter_market_value):,} market value rows in {time.time() - mv_start:.1f}s")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading market value for {year}Q{quarter}: {e}")
            quarter_market_value = pd.DataFrame()
    
    # OPTIMIZATION: Use sequential snapshot for exchange rates
    exchange_rates_snapshot = None
    if exchange_rates_path and exchange_rates_path.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Initializing exchange rates snapshot...")
        exr_start = time.time()
        try:
            exchange_rates_snapshot = ExchangeRatesSnapshot(exchange_rates_path)
            exchange_rates_snapshot.start()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Exchange rates snapshot initialized in {time.time() - exr_start:.1f}s")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error initializing exchange rates snapshot: {e}")
            # Fallback: load all exchange rates for quarter
            all_exchange_rates = pd.DataFrame()
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(exchange_rates_path)
                exr_chunks = []
                for batch in parquet_file.iter_batches(batch_size=500000):
                    chunk_df = batch.to_pandas()
                    chunk_df = chunk_df.reset_index(drop=True)
                    chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
                    date_mask = (chunk_df['DATE'].values >= lookback_start_dt) & (chunk_df['DATE'].values <= quarter_end_dt)
                    filtered = chunk_df.iloc[date_mask].copy()
                    if len(filtered) > 0:
                        exr_chunks.append(filtered)
                all_exchange_rates = pd.concat(exr_chunks, ignore_index=True) if exr_chunks else pd.DataFrame()
                if len(all_exchange_rates) > 0:
                    all_exchange_rates = all_exchange_rates.loc[:, ~all_exchange_rates.columns.duplicated()]
            except Exception as e2:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading exchange rates: {e2}")
                all_exchange_rates = pd.DataFrame()
    else:
        all_exchange_rates = pd.DataFrame()
    
    # OPTIMIZATION: Use sequential snapshot for enterprise value (10-100x faster)
    enterprise_value_snapshot = None
    if enterprise_value_path and enterprise_value_path.exists():
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Initializing enterprise value snapshot...")
        ev_start = time.time()
        try:
            enterprise_value_snapshot = EnterpriseValueSnapshot(enterprise_value_path)
            enterprise_value_snapshot.start()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {year}Q{quarter}: Enterprise value snapshot initialized in {time.time() - ev_start:.1f}s")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error initializing enterprise value snapshot: {e}")
            # Fallback: load all enterprise value for quarter
            quarter_enterprise_value = pd.DataFrame()
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(enterprise_value_path)
                ev_chunks = []
                for batch in parquet_file.iter_batches(batch_size=500000):
                    chunk_df = batch.to_pandas()
                    chunk_df = chunk_df.reset_index(drop=True)
                    chunk_df['DATE'] = pd.to_datetime(chunk_df['DATE'])
                    date_mask = (chunk_df['DATE'].values >= quarter_start_dt) & (chunk_df['DATE'].values <= quarter_end_dt)
                    filtered = chunk_df.iloc[date_mask].copy()
                    if len(filtered) > 0:
                        ev_chunks.append(filtered)
                quarter_enterprise_value = pd.concat(ev_chunks, ignore_index=True) if ev_chunks else pd.DataFrame()
                if len(quarter_enterprise_value) > 0:
                    quarter_enterprise_value = quarter_enterprise_value.loc[:, ~quarter_enterprise_value.columns.duplicated()]
            except Exception as e2:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error loading enterprise value: {e2}")
                quarter_enterprise_value = pd.DataFrame()
    else:
        quarter_enterprise_value = pd.DataFrame()
    
    trading_dates = sorted(quarter_prices['DATE'].unique())
    
    if len(trading_dates) == 0:
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    # Filter dates to process range
    trading_dates = [d for d in trading_dates if pd.to_datetime(start_date) <= pd.to_datetime(d) <= pd.to_datetime(end_date)]
    
    if len(trading_dates) == 0:
        return {'processed': 0, 'skipped': 0, 'quarter': f"{year}Q{quarter}", 'csv_path': None}
    
    # OPTIMIZATION: Index DataFrames by DATE for O(1) lookups (only for high-frequency data)
    # Remove duplicate columns first to avoid issues
    quarter_prices = quarter_prices.loc[:, ~quarter_prices.columns.duplicated()]
    quarter_returns = quarter_returns.loc[:, ~quarter_returns.columns.duplicated()]
    # Note: fundamentals now use snapshot, so we don't need to index it
    # But keep for fallback compatibility
    if fundamentals_snapshot is None:
        if 'quarter_fundamentals' in locals() and len(quarter_fundamentals) > 0:
            quarter_fundamentals = quarter_fundamentals.loc[:, ~quarter_fundamentals.columns.duplicated()]
    
    # Ensure DATE columns are 1D (not DataFrame)
    for df_name, df in [('prices', quarter_prices), ('returns', quarter_returns)]:
        if 'DATE' in df.columns and isinstance(df['DATE'], pd.DataFrame):
            df['DATE'] = df['DATE'].iloc[:, 0]
    if fundamentals_snapshot is None and 'quarter_fundamentals' in locals() and len(quarter_fundamentals) > 0:
        if 'DATE' in quarter_fundamentals.columns and isinstance(quarter_fundamentals['DATE'], pd.DataFrame):
            quarter_fundamentals['DATE'] = quarter_fundamentals['DATE'].iloc[:, 0]
    
    if len(quarter_market_value) > 0:
        quarter_market_value = quarter_market_value.loc[:, ~quarter_market_value.columns.duplicated()]
        if 'DATE' in quarter_market_value.columns and isinstance(quarter_market_value['DATE'], pd.DataFrame):
            quarter_market_value['DATE'] = quarter_market_value['DATE'].iloc[:, 0]
    
    if len(quarter_enterprise_value) > 0:
        quarter_enterprise_value = quarter_enterprise_value.loc[:, ~quarter_enterprise_value.columns.duplicated()]
        if 'DATE' in quarter_enterprise_value.columns and isinstance(quarter_enterprise_value['DATE'], pd.DataFrame):
            quarter_enterprise_value['DATE'] = quarter_enterprise_value['DATE'].iloc[:, 0]
    
    if len(all_exchange_rates) > 0:
        all_exchange_rates = all_exchange_rates.loc[:, ~all_exchange_rates.columns.duplicated()]
        if 'DATE' in all_exchange_rates.columns and isinstance(all_exchange_rates['DATE'], pd.DataFrame):
            all_exchange_rates['DATE'] = all_exchange_rates['DATE'].iloc[:, 0]
    
    quarter_prices_idx = quarter_prices.set_index('DATE', drop=False)
    quarter_returns_idx = quarter_returns.set_index('DATE', drop=False)
    # Only index fundamentals if not using snapshot (for fallback)
    if fundamentals_snapshot is None:
        quarter_fundamentals_idx = quarter_fundamentals.set_index('DATE', drop=False) if len(quarter_fundamentals) > 0 else None
    else:
        quarter_fundamentals_idx = None
    quarter_market_value_idx = quarter_market_value.set_index('DATE', drop=False) if len(quarter_market_value) > 0 else None
    quarter_enterprise_value_idx = quarter_enterprise_value.set_index('DATE', drop=False) if len(quarter_enterprise_value) > 0 else None
    all_exchange_rates_idx = all_exchange_rates.set_index('DATE', drop=False) if len(all_exchange_rates) > 0 else None
    
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
        
        # Advance snapshots to current date
        if fundamentals_snapshot is not None:
            fundamentals_snapshot.advance_to(date_T)
        if enterprise_value_snapshot is not None:
            enterprise_value_snapshot.advance_to(date_T)
        if exchange_rates_snapshot is not None:
            exchange_rates_snapshot.advance_to(date_T)
        
        # Get prices and returns for this date (high-frequency data, still use index lookup)
        try:
            prices_T = quarter_prices_idx.loc[[date_T]].copy() if date_T in quarter_prices_idx.index else pd.DataFrame()
            returns_T = quarter_returns_idx.loc[[date_T]].copy() if date_T in quarter_returns_idx.index else pd.DataFrame()
        except (KeyError, TypeError):
            # Fallback if date not in index
            price_mask = (quarter_prices['DATE'].values == date_T)
            prices_T = quarter_prices.iloc[price_mask].copy()
            return_mask = (quarter_returns['DATE'].values == date_T)
            returns_T = quarter_returns.iloc[return_mask].copy()
        
        if len(prices_T) == 0 or len(returns_T) == 0:
            skipped += 1
            continue
        
        # Get fundamentals for stocks on this date using snapshot
        if fundamentals_snapshot is not None:
            # Get latest fundamentals for all stocks with prices/returns
            stock_ids = list(set(prices_T['FACTSET_ID'].unique()) | set(returns_T['FACTSET_ID'].unique()))
            fundamentals_list = []
            for stock_id in stock_ids:
                fund = fundamentals_snapshot.get_for_id(stock_id)
                if fund:
                    fundamentals_list.append(fund)
            if fundamentals_list:
                fundamentals_T = pd.DataFrame(fundamentals_list)
            else:
                fundamentals_T = pd.DataFrame()
        else:
            # Fallback to old method
            try:
                fundamentals_T = quarter_fundamentals_idx.loc[[date_T]].copy() if date_T in quarter_fundamentals_idx.index else pd.DataFrame()
            except (KeyError, TypeError):
                fund_mask = (quarter_fundamentals['DATE'].values == date_T)
                fundamentals_T = quarter_fundamentals.iloc[fund_mask].copy()
        
        if len(fundamentals_T) == 0:
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
        
        # Merge market value data if available
        if quarter_market_value_idx is not None:
            try:
                market_value_T = quarter_market_value_idx.loc[[date_T]].copy() if date_T in quarter_market_value_idx.index else pd.DataFrame()
                if len(market_value_T) > 0:
                    df_T = df_T.merge(
                        market_value_T[['FACTSET_ID', 'MARKETCAP', 'CURRENCY']],
                        on='FACTSET_ID',
                        how='left'
                    )
            except (KeyError, TypeError):
                # Fallback if date not in index
                market_value_T = quarter_market_value[quarter_market_value['DATE'] == date_T].copy()
                if len(market_value_T) > 0:
                    df_T = df_T.merge(
                        market_value_T[['FACTSET_ID', 'MARKETCAP', 'CURRENCY']],
                        on='FACTSET_ID',
                        how='left'
                    )
        
        # Merge enterprise value data if available (using snapshot)
        if enterprise_value_snapshot is not None:
            # Get latest enterprise value for all stocks
            stock_ids = df_T['FACTSET_ID'].unique()
            ev_list = []
            for stock_id in stock_ids:
                ev = enterprise_value_snapshot.get_for_id(stock_id)
                if ev:
                    ev_list.append(ev)
            if ev_list:
                enterprise_value_T = pd.DataFrame(ev_list)
                df_T = df_T.merge(
                    enterprise_value_T[['FACTSET_ID', 'ENTERPRISE_VALUE']],
                    on='FACTSET_ID',
                    how='left'
                )
        elif quarter_enterprise_value_idx is not None:
            # Fallback to old method
            try:
                enterprise_value_T = quarter_enterprise_value_idx.loc[[date_T]].copy() if date_T in quarter_enterprise_value_idx.index else pd.DataFrame()
                if len(enterprise_value_T) > 0:
                    df_T = df_T.merge(
                        enterprise_value_T[['FACTSET_ID', 'ENTERPRISE_VALUE']],
                        on='FACTSET_ID',
                        how='left'
                    )
            except (KeyError, TypeError):
                enterprise_value_T = quarter_enterprise_value[quarter_enterprise_value['DATE'] == date_T].copy()
                if len(enterprise_value_T) > 0:
                    df_T = df_T.merge(
                        enterprise_value_T[['FACTSET_ID', 'ENTERPRISE_VALUE']],
                        on='FACTSET_ID',
                        how='left'
                    )
        
        if len(df_T) == 0:
            skipped += 1
            continue
        
        # Calculate SIZE characteristic (log of market cap in USD)
        if 'MARKETCAP' in df_T.columns:
            # Get exchange rates for this date (using snapshot if available)
            exchange_rates_T = pd.DataFrame()
            if exchange_rates_snapshot is not None:
                # Get unique currencies for stocks on this date
                currencies = df_T['CURRENCY'].unique() if 'CURRENCY' in df_T.columns else []
                if len(currencies) > 0:
                    # Get exchange rates for all currencies
                    fx_dict = exchange_rates_snapshot.get_for_currencies(list(currencies), date_T)
                    # Convert to DataFrame
                    fx_list = [fx_dict[curr] for curr in currencies if fx_dict[curr] is not None]
                    if fx_list:
                        exchange_rates_T = pd.DataFrame(fx_list)
            elif len(all_exchange_rates) > 0:
                # Fallback to old method
                if all_exchange_rates_idx is not None:
                    try:
                        exchange_rates_T = all_exchange_rates_idx.loc[[date_T]].copy() if date_T in all_exchange_rates_idx.index else pd.DataFrame()
                    except (KeyError, TypeError):
                        exchange_rates_T = all_exchange_rates[all_exchange_rates['DATE'] == date_T].copy()
                else:
                    exchange_rates_T = all_exchange_rates[all_exchange_rates['DATE'] == date_T].copy()
            
            # Convert market cap from local currency to USD
            if len(exchange_rates_T) > 0 and 'EXCHANGE_RATE_TO_USD' in exchange_rates_T.columns:
                # Merge exchange rates by currency
                df_T = df_T.merge(
                    exchange_rates_T[['CURRENCY', 'EXCHANGE_RATE_TO_USD']],
                    on='CURRENCY',
                    how='left'
                )
                # For USD currency, set exchange rate to 1.0 explicitly
                # For other currencies missing from exchange rates, fill with 1.0 (assume USD)
                df_T.loc[df_T['CURRENCY'] == 'USD', 'EXCHANGE_RATE_TO_USD'] = 1.0
                df_T['EXCHANGE_RATE_TO_USD'] = df_T['EXCHANGE_RATE_TO_USD'].fillna(1.0)
                # Convert to USD: multiply by exchange rate
                df_T['MARKETCAP_USD'] = df_T['MARKETCAP'] * df_T['EXCHANGE_RATE_TO_USD']
            else:
                # No exchange rates available - assume all are already in USD or set to NaN
                df_T['MARKETCAP_USD'] = df_T['MARKETCAP']
            
            # Calculate log of market cap in USD
            # Handle negative or zero values by setting to NaN
            df_T.loc[df_T['MARKETCAP_USD'] <= 0, 'MARKETCAP_USD'] = np.nan
            df_T['SIZE'] = np.log(df_T['MARKETCAP_USD'])
        
        # Calculate characteristics
        # Value: Calculate EBITDA_LTM/EV and SALES_LTM/EV
        if 'ENTERPRISE_VALUE' in df_T.columns:
            df_T['EBITDA_LTM_EV'] = df_T['EBITDA_LTM'] / df_T['ENTERPRISE_VALUE']
            df_T['SALES_LTM_EV'] = df_T['SALES_LTM'] / df_T['ENTERPRISE_VALUE']
            
            # Handle invalid values
            df_T.loc[df_T['ENTERPRISE_VALUE'] <= 0, 'EBITDA_LTM_EV'] = np.nan
            df_T.loc[df_T['ENTERPRISE_VALUE'] <= 0, 'SALES_LTM_EV'] = np.nan
            df_T.loc[df_T['EBITDA_LTM'] <= 0, 'EBITDA_LTM_EV'] = np.nan
        else:
            df_T['EBITDA_LTM_EV'] = np.nan
            df_T['SALES_LTM_EV'] = np.nan
        
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
        # Pre-fetch calculators dict to avoid repeated lookups in getter methods
        factset_ids = df_T['FACTSET_ID'].values
        calculators_dict = calculator_manager.calculators
        
        # Direct dict access + method call (faster than going through manager.get_* methods)
        momentum_vals = []
        volatility_vals = []
        liquidity_vals = []
        for fid in factset_ids:
            calc = calculators_dict.get(fid)
            if calc:
                momentum_vals.append(calc.get_momentum())
                volatility_vals.append(calc.get_volatility())
                liquidity_vals.append(calc.get_liquidity())
            else:
                momentum_vals.append(None)
                volatility_vals.append(None)
                liquidity_vals.append(None)
        
        df_T['MOMENTUM'] = momentum_vals
        df_T['VOLATILITY'] = volatility_vals
        df_T['LIQUIDITY'] = liquidity_vals
        
        # OPTIMIZATION: Vectorized winsorization and standardization
        # Use np.nanpercentile and vectorized operations across all characteristics at once
        winsorize_lower = config.FACTOR_PARAMS['winsorize_lower']
        winsorize_upper = config.FACTOR_PARAMS['winsorize_upper']
        
        # Characteristics that need general winsorization and z-scoring
        # Value, Profitability, and Growth are handled separately below
        all_chars = ['MOMENTUM', 'VOLATILITY', 'LIQUIDITY']
        
        # Add SIZE to characteristics if available
        if 'SIZE' in df_T.columns:
            all_chars.append('SIZE')
        
        # Filter to only characteristics that exist in df_T
        all_chars = [char for char in all_chars if char in df_T.columns]
        
        if len(all_chars) > 0:
            # Convert to NumPy array for faster operations
            # First ensure DataFrame columns are numeric (handles None values properly)
            char_df = df_T[all_chars].copy()
            for col in char_df.columns:
                # Convert to numeric, coercing errors (None, strings, etc.) to NaN
                char_df[col] = pd.to_numeric(char_df[col], errors='coerce')
            
            # Now convert to NumPy array - should be float64 with NaN for missing values
            char_data = char_df.values.astype(np.float64)  # (N stocks × K chars)
            
            # Compute percentiles across axis=0 (across stocks for each characteristic)
            # This is much faster than pandas quantile in a loop
            lower_bounds = np.nanpercentile(char_data, winsorize_lower * 100, axis=0)
            upper_bounds = np.nanpercentile(char_data, winsorize_upper * 100, axis=0)
            
            # Clip in one pass (vectorized) - broadcast lower/upper bounds to match char_data shape
            # np.clip broadcasts automatically: (N, K) clipped by (K,) bounds
            char_data_clipped = np.clip(char_data, lower_bounds, upper_bounds)
            
            # Standardize in one pass (vectorized) - compute mean/std across stocks (axis=0)
            char_means = np.nanmean(char_data_clipped, axis=0)  # (K,) means
            char_stds = np.nanstd(char_data_clipped, axis=0, ddof=0)  # (K,) stds
            char_stds = np.where(char_stds > 0, char_stds, 1.0)  # Avoid division by zero
            # Broadcast: (N, K) - (K,) / (K,) = (N, K)
            char_data_standardized = (char_data_clipped - char_means) / char_stds
            
            # Write back to DataFrame
            df_T[all_chars] = char_data_standardized
        
        # Combine into style factors
        # VALUE: Calculate EBITDA_LTM/EV and SALES_LTM/EV, z-score each, average, then z-score again
        value_signals = ['EBITDA_LTM_EV', 'SALES_LTM_EV']
        value_signals = [char for char in value_signals if char in df_T.columns]
        
        if len(value_signals) > 0:
            # Winsorize value signals
            for char in value_signals:
                # Ensure numeric type (handles None values)
                df_T[char] = pd.to_numeric(df_T[char], errors='coerce')
                if df_T[char].notna().sum() > 0:
                    char_values = df_T[char].values.astype(np.float64)
                    lower = np.nanpercentile(char_values, winsorize_lower * 100)
                    upper = np.nanpercentile(char_values, winsorize_upper * 100)
                    df_T[char] = np.clip(char_values, lower, upper)
            
            # Z-score each value signal separately
            value_z_scores_df = pd.DataFrame(index=df_T.index)
            for char in value_signals:
                if df_T[char].notna().sum() > 1:
                    mean_val = df_T[char].mean()
                    std_val = df_T[char].std()
                    if std_val > 0:
                        value_z_scores_df[char] = (df_T[char] - mean_val) / std_val
            
            # Average the z-scores
            if len(value_z_scores_df.columns) > 0:
                df_T['VALUE'] = value_z_scores_df.mean(axis=1, skipna=True)
                
                # Z-score the average again
                if df_T['VALUE'].notna().sum() > 1:
                    val_mean = df_T['VALUE'].mean()
                    val_std = df_T['VALUE'].std()
                    if val_std > 0:
                        df_T['VALUE'] = (df_T['VALUE'] - val_mean) / val_std
            else:
                df_T['VALUE'] = np.nan
        else:
            df_T['VALUE'] = np.nan
        
        # PROFITABILITY: Z-score each signal, average, then z-score again
        profitability_signals = ['EBITDA_MARGIN', 'GROSS_MARGIN']
        profitability_signals = [char for char in profitability_signals if char in df_T.columns]
        
        if len(profitability_signals) > 0:
            # Winsorize profitability signals
            for char in profitability_signals:
                # Ensure numeric type (handles None values)
                df_T[char] = pd.to_numeric(df_T[char], errors='coerce')
                if df_T[char].notna().sum() > 0:
                    char_values = df_T[char].values.astype(np.float64)
                    lower = np.nanpercentile(char_values, winsorize_lower * 100)
                    upper = np.nanpercentile(char_values, winsorize_upper * 100)
                    df_T[char] = np.clip(char_values, lower, upper)
            
            # Z-score each profitability signal separately
            profitability_z_scores_df = pd.DataFrame(index=df_T.index)
            for char in profitability_signals:
                if df_T[char].notna().sum() > 1:
                    mean_val = df_T[char].mean()
                    std_val = df_T[char].std()
                    if std_val > 0:
                        profitability_z_scores_df[char] = (df_T[char] - mean_val) / std_val
            
            # Average the z-scores
            if len(profitability_z_scores_df.columns) > 0:
                df_T['PROFITABILITY'] = profitability_z_scores_df.mean(axis=1, skipna=True)
                
                # Z-score the average again
                if df_T['PROFITABILITY'].notna().sum() > 1:
                    prof_mean = df_T['PROFITABILITY'].mean()
                    prof_std = df_T['PROFITABILITY'].std()
                    if prof_std > 0:
                        df_T['PROFITABILITY'] = (df_T['PROFITABILITY'] - prof_mean) / prof_std
            else:
                df_T['PROFITABILITY'] = np.nan
        else:
            df_T['PROFITABILITY'] = np.nan
        
        # GROWTH: Z-score each signal, average, then z-score again
        growth_signals = ['EPS_GROWTH', 'SALES_GROWTH']
        growth_signals = [char for char in growth_signals if char in df_T.columns]
        
        if len(growth_signals) > 0:
            # Winsorize growth signals
            for char in growth_signals:
                # Ensure numeric type (handles None values)
                df_T[char] = pd.to_numeric(df_T[char], errors='coerce')
                if df_T[char].notna().sum() > 0:
                    char_values = df_T[char].values.astype(np.float64)
                    lower = np.nanpercentile(char_values, winsorize_lower * 100)
                    upper = np.nanpercentile(char_values, winsorize_upper * 100)
                    df_T[char] = np.clip(char_values, lower, upper)
            
            # Z-score each growth signal separately
            growth_z_scores_df = pd.DataFrame(index=df_T.index)
            for char in growth_signals:
                if df_T[char].notna().sum() > 1:
                    mean_val = df_T[char].mean()
                    std_val = df_T[char].std()
                    if std_val > 0:
                        growth_z_scores_df[char] = (df_T[char] - mean_val) / std_val
            
            # Average the z-scores
            if len(growth_z_scores_df.columns) > 0:
                df_T['GROWTH'] = growth_z_scores_df.mean(axis=1, skipna=True)
                
                # Z-score the average again
                if df_T['GROWTH'].notna().sum() > 1:
                    growth_mean = df_T['GROWTH'].mean()
                    growth_std = df_T['GROWTH'].std()
                    if growth_std > 0:
                        df_T['GROWTH'] = (df_T['GROWTH'] - growth_mean) / growth_std
            else:
                df_T['GROWTH'] = np.nan
        else:
            df_T['GROWTH'] = np.nan
        
        # Create dummies
        sector_dummies = pd.get_dummies(df_T['SECTOR'], prefix='SECTOR')
        continent_dummies = pd.get_dummies(df_T['CONTINENT'], prefix='CONTINENT')
        
        sector_dummies.columns = sector_dummies.columns.str.replace(' ', '_')
        continent_dummies.columns = continent_dummies.columns.str.replace(' ', '_')
        
        sector_dummies = sector_dummies.astype(float)
        continent_dummies = continent_dummies.astype(float)
        
        # OPTIMIZATION: Vectorized sum-to-zero (subtract column means in one operation)
        if len(sector_dummies.columns) > 0:
            sector_dummies = sector_dummies - sector_dummies.mean(axis=0)
        if len(continent_dummies.columns) > 0:
            continent_dummies = continent_dummies - continent_dummies.mean(axis=0)
        
        # Combine exposures
        # Ensure MOMENTUM, VOLATILITY, LIQUIDITY, SIZE are included even if some values are NaN
        style_factor_cols = ['FACTSET_ID', 'VALUE', 'PROFITABILITY', 'GROWTH']
        for factor in ['MOMENTUM', 'VOLATILITY', 'LIQUIDITY', 'SIZE']:
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
    from .data_retrieval import get_date_range
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
    market_value_path = data_path / 'market_value.parquet'
    exchange_rates_path = data_path / 'exchange_rates.parquet'
    enterprise_value_path = data_path / 'enterprise_value.parquet'
    
    if not prices_path.exists() or not returns_path.exists() or not fundamentals_path.exists():
        print(f"\n⚠ ERROR: Data files not found in {data_dir}/")
        print("   Please run 'python bulk_download.py' first to download data from Snowflake")
        return {'processed': 0, 'skipped': 0}
    
    # Warn if market value or exchange rates are missing (but continue processing)
    if not market_value_path.exists():
        print(f"\n⚠ WARNING: market_value.parquet not found in {data_dir}/")
        print("   SIZE characteristic will not be calculated")
        print("   Run 'python bulk_download.py' to download market value data")
    if not exchange_rates_path.exists():
        print(f"\n⚠ WARNING: exchange_rates.parquet not found in {data_dir}/")
        print("   Market cap conversion to USD will not be performed")
        print("   Run 'python bulk_download.py' to download exchange rate data")
    if not enterprise_value_path.exists():
        print(f"\n⚠ WARNING: enterprise_value.parquet not found in {data_dir}/")
        print("   VALUE characteristic will not be calculated (requires EBITDA_LTM/EV and SALES_LTM/EV)")
        print("   Run 'python bulk_download.py' to download enterprise value data")
    
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
        from .continent_mapping import get_continent_developed
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
                year, quarter, output_path, universe_df, prices_path, returns_path, fundamentals_path, 
                start_date, end_date, neutralize, market_value_path if market_value_path.exists() else None,
                exchange_rates_path if exchange_rates_path.exists() else None,
                enterprise_value_path if enterprise_value_path.exists() else None
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
        from .model_builder import FactorModelBuilder
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

