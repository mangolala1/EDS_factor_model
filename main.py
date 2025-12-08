"""
Streamlined main workflow for EDS Factor Model
1. Download all data from Snowflake (one-time) → Parquet files
2. Process quarters in parallel → CSV files
3. Calculate factor returns, specific returns, risk, covariance → CSV files
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from quarter_processor import process_quarters_parallel
from model_builder import FactorModelBuilder
from data_retrieval import SnowflakeDataRetriever, get_date_range
import time


def main(
    start_date: str = '2020-01-01',
    end_date: str = None,
    neutralize: bool = False,
    output_dir: str = 'results',
    data_dir: str = 'data',
    skip_download: bool = False
):
    """
    Main workflow - streamlined and fast!
    Downloads data once, then works entirely with local Parquet files (no Snowflake queries!)
    All results saved to CSV files (no SQLite!)
    
    Args:
        start_date: Start date
        end_date: End date (default: today)
        neutralize: Whether to neutralize factors
        output_dir: Directory to save CSV result files
        data_dir: Directory with Parquet files (prices.parquet, returns.parquet, etc.)
        skip_download: If True, skip bulk download (assumes data already downloaded)
    """
    if end_date is None:
        _, end_date = get_date_range(lookback_days=1)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    main_start_time = time.time()
    print("=" * 80)
    print("EDS Factor Model - Streamlined Workflow (CSV Output)")
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting workflow...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Step 1: Bulk download (if needed)
    if not skip_download:
        print("\n[STAGE 1] Bulk Download from Snowflake → Local Parquet Files")
        print("=" * 80)
        from bulk_download import download_all_data
        download_all_data(start_date, end_date, data_dir)
    else:
        print("\n[STAGE 1] Skipping download (using local Parquet files)")
    
    # Step 2: Process quarters in parallel
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 2] Processing Quarters in Parallel (Local Parquet Files)")
    print("=" * 80)
    stage2_start = time.time()
    
    exposure_stats = process_quarters_parallel(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        data_dir=data_dir,
        max_workers=None,  # Auto-detect
        neutralize=neutralize
    )
    
    if exposure_stats['processed'] == 0:
        print("✗ ERROR: No dates were processed.")
        return None
    
    exposures_csv_path = exposure_stats.get('exposures_csv')
    if not exposures_csv_path or not Path(exposures_csv_path).exists():
        print("✗ ERROR: Exposures CSV file not found.")
        return None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 2 completed in {time.time() - stage2_start:.1f}s")
    
    # Step 3: Calculate factor returns
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 3] Calculating Factor Returns")
    print("=" * 80)
    stage3_start = time.time()
    
    try:
        print("Loading exposures from CSV...")
        with tqdm(total=1, desc="Loading exposures", bar_format='{desc}: {elapsed}') as pbar:
            exposures_df = pd.read_csv(exposures_csv_path)
            exposures_df['DATE'] = pd.to_datetime(exposures_df['DATE'])
            exposures_df = exposures_df.set_index(['FACTSET_ID', 'DATE'])
            pbar.update(1)
        
        print(f"   ✓ Loaded {len(exposures_df):,} exposure observations")
        
        # Load returns from Parquet (fast!)
        print("Loading returns from Parquet...")
        returns_path = Path(data_dir) / 'returns.parquet'
        if returns_path.exists():
            returns_df = pd.read_parquet(returns_path)
            if 'FSYM_ID' in returns_df.columns:
                returns_df = returns_df.rename(columns={'FSYM_ID': 'FACTSET_ID'})
            if 'P_DATE' in returns_df.columns:
                returns_df = returns_df.rename(columns={'P_DATE': 'DATE'})
            returns_df['DATE'] = pd.to_datetime(returns_df['DATE'])
            # Filter to date range and trading dates from exposures
            trading_dates = exposures_df.index.get_level_values('DATE').unique()
            returns_df = returns_df[returns_df['DATE'].isin(trading_dates)]
        else:
            print("   ⚠ returns.parquet not found - please run bulk_download.py first")
            returns_df = pd.DataFrame()
        
        print(f"   ✓ Loaded {len(returns_df):,} return observations")
        
        # Calculate factor returns
        print("Calculating factor returns...")
        with tqdm(total=1, desc="Factor returns", bar_format='{desc}: {elapsed}') as pbar:
            model_builder = FactorModelBuilder(exposures_df, pd.Series(dtype=float))
            factor_returns_df = model_builder.calculate_daily_factor_returns(
                exposures_df=exposures_df,
                returns_df=returns_df
            )
            pbar.update(1)
        
        if len(factor_returns_df) > 0:
            # Save to CSV (convert DATE to string for CSV)
            factor_returns_for_csv = factor_returns_df.copy()
            factor_returns_for_csv['DATE'] = pd.to_datetime(factor_returns_for_csv['DATE']).dt.strftime('%Y-%m-%d')
            factor_returns_csv = output_path / 'factor_returns.csv'
            factor_returns_for_csv.to_csv(factor_returns_csv, index=False)
            print(f"   ✓ Saved {len(factor_returns_df):,} factor return observations → {factor_returns_csv}")
            # Keep factor_returns_df with datetime for later calculations
        else:
            factor_returns_df = None
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        factor_returns_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 3 completed in {time.time() - stage3_start:.1f}s")
    
    # Step 4: Specific returns
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 4] Calculating Specific Returns")
    print("=" * 80)
    stage4_start = time.time()
    
    try:
        if factor_returns_df is not None:
            with tqdm(total=1, desc="Specific returns", bar_format='{desc}: {elapsed}') as pbar:
                specific_returns_df = model_builder.calculate_specific_returns(
                    exposures_df=exposures_df,
                    returns_df=returns_df,
                    factor_returns_df=factor_returns_df
                )
                pbar.update(1)
            if len(specific_returns_df) > 0:
                specific_returns_df['DATE'] = pd.to_datetime(specific_returns_df['DATE']).dt.strftime('%Y-%m-%d')
                specific_returns_csv = output_path / 'specific_returns.csv'
                specific_returns_df.to_csv(specific_returns_csv, index=False)
                print(f"   ✓ Saved {len(specific_returns_df):,} specific return observations → {specific_returns_csv}")
    except Exception as e:
        print(f"\n⚠ WARNING: {str(e)}")
        specific_returns_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 4 completed in {time.time() - stage4_start:.1f}s")
    
    # Step 5: Specific risk
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 5] Calculating Specific Risk (60-Day Window)")
    print("=" * 80)
    stage5_start = time.time()
    
    try:
        if factor_returns_df is not None:
            with tqdm(total=1, desc="Specific risk", bar_format='{desc}: {elapsed}') as pbar:
                specific_risk_df = model_builder.calculate_specific_risk(
                    exposures_df=exposures_df,
                    returns_df=returns_df,
                    factor_returns_df=factor_returns_df,
                    window=60
                )
                pbar.update(1)
            if len(specific_risk_df) > 0:
                specific_risk_df['DATE'] = pd.to_datetime(specific_risk_df['DATE']).dt.strftime('%Y-%m-%d')
                specific_risk_csv = output_path / 'specific_risk.csv'
                specific_risk_df.to_csv(specific_risk_csv, index=False)
                print(f"   ✓ Saved {len(specific_risk_df):,} specific risk observations → {specific_risk_csv}")
    except Exception as e:
        print(f"\n⚠ WARNING: {str(e)}")
        specific_risk_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 5 completed in {time.time() - stage5_start:.1f}s")
    
    # Step 6: Factor covariance
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 6] Calculating Factor Covariance (60-Day Window)")
    print("=" * 80)
    stage6_start = time.time()
    
    try:
        if factor_returns_df is not None:
            # factor_returns_df already has datetime DATE column
            with tqdm(total=1, desc="Covariance", bar_format='{desc}: {elapsed}') as pbar:
                covariance_df = model_builder.calculate_factor_covariance(
                    factor_returns_df=factor_returns_df,
                    lookback_window=60
                )
                pbar.update(1)
            if len(covariance_df) > 0:
                covariance_df['DATE'] = pd.to_datetime(covariance_df['DATE']).dt.strftime('%Y-%m-%d')
                covariance_csv = output_path / 'factor_covariance.csv'
                covariance_df.to_csv(covariance_csv, index=False)
                print(f"   ✓ Saved {len(covariance_df):,} covariance observations → {covariance_csv}")
    except Exception as e:
        print(f"\n⚠ WARNING: {str(e)}")
        covariance_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 6 completed in {time.time() - stage6_start:.1f}s")
    
    total_time = time.time() - main_start_time
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] WORKFLOW COMPLETE")
    print("=" * 80)
    print(f"✓ Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    print(f"✓ Processed {exposure_stats['processed']} dates")
    print(f"✓ All results saved to CSV files in: {output_dir}/")
    print("\nOutput CSV files:")
    print(f"  - {output_path / 'exposures.csv'}")
    if factor_returns_df is not None:
        print(f"  - {output_path / 'factor_returns.csv'}")
    if specific_returns_df is not None:
        print(f"  - {output_path / 'specific_returns.csv'}")
    if specific_risk_df is not None:
        print(f"  - {output_path / 'specific_risk.csv'}")
    if covariance_df is not None:
        print(f"  - {output_path / 'factor_covariance.csv'}")
    print("=" * 80)
    
    return {'output_dir': output_dir, 'processed': exposure_stats['processed']}


if __name__ == "__main__":
    # First run: download data, then process
    # Subsequent runs: skip download, just process
    results = main(
        start_date='2020-01-01',
        end_date=None,
        neutralize=False,
        output_dir='results',
        data_dir='data',
        skip_download=True  # Set to True if data already downloaded
    )

