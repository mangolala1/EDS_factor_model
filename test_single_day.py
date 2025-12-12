"""
Test script to run factor model workflow for a single day (2020-01-02)
This allows us to inspect the output tables before running the full pipeline
"""
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from quarter_processor import process_quarters_parallel
from model_builder import FactorModelBuilder
from data_retrieval import SnowflakeDataRetriever, get_date_range
import time

def test_single_day(
    test_date: str = '2020-01-02',  # Use 2020-01-02 (2020-01-01 is likely a holiday)
    output_dir: str = 'test_results',
    data_dir: str = 'data',
    skip_download: bool = True  # Assume data already downloaded
):
    """
    Test workflow for a single day
    
    Args:
        test_date: Single date to test (YYYY-MM-DD)
        output_dir: Directory to save test results
        data_dir: Directory with Parquet files
        skip_download: If True, skip bulk download
    """
    # Use a small date range around the test date (need some lookback for calculations)
    # Start a few days before to ensure we have data
    from datetime import datetime, timedelta
    test_dt = datetime.strptime(test_date, '%Y-%m-%d')
    start_date = (test_dt - timedelta(days=10)).strftime('%Y-%m-%d')
    end_date = test_date
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("EDS Factor Model - Single Day Test")
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Testing date: {test_date}")
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
    
    # Step 2: Process quarters (will only process Q1 2020 for this date)
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 2] Processing Quarters")
    print("=" * 80)
    stage2_start = time.time()
    
    exposure_stats = process_quarters_parallel(
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        data_dir=data_dir,
        max_workers=1,  # Single worker for test
        neutralize=False
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
            exposures_long_df = pd.read_csv(exposures_csv_path)
            exposures_long_df['DATE'] = pd.to_datetime(exposures_long_df['DATE'])
            
            # Filter to test date only
            exposures_long_df = exposures_long_df[exposures_long_df['DATE'] == test_date]
            print(f"   Filtered to {test_date}: {len(exposures_long_df):,} exposure rows")
            
            # Convert long format to wide format for regression calculations
            exposures_df = exposures_long_df.pivot_table(
                index=['SECURITY_ID', 'DATE'],
                columns='FACTOR_NAME',
                values='EXPOSURE',
                aggfunc='first'
            )
            exposures_df = exposures_df.rename_axis(None, axis=1)
            exposures_df = exposures_df.rename_axis(['FACTSET_ID', 'DATE'])
            pbar.update(1)
        
        print(f"   ✓ Loaded {len(exposures_long_df):,} exposure observations for {test_date}")
        
        # Load returns from Parquet
        print("Loading returns from Parquet...")
        returns_path = Path(data_dir) / 'returns.parquet'
        if returns_path.exists():
            returns_df = pd.read_parquet(returns_path)
            if 'FSYM_ID' in returns_df.columns:
                returns_df = returns_df.rename(columns={'FSYM_ID': 'FACTSET_ID'})
            if 'P_DATE' in returns_df.columns:
                returns_df = returns_df.rename(columns={'P_DATE': 'DATE'})
            returns_df['DATE'] = pd.to_datetime(returns_df['DATE'])
            # Filter to test date
            returns_df = returns_df[returns_df['DATE'] == test_date]
        else:
            print("   ⚠ returns.parquet not found - please run bulk_download.py first")
            returns_df = pd.DataFrame()
        
        print(f"   ✓ Loaded {len(returns_df):,} return observations for {test_date}")
        
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
            # Save to CSV
            factor_returns_for_csv = factor_returns_df.copy()
            factor_returns_for_csv['DATE'] = pd.to_datetime(factor_returns_for_csv['DATE']).dt.strftime('%Y-%m-%d')
            factor_returns_csv = output_path / 'factor_returns.csv'
            factor_returns_for_csv.to_csv(factor_returns_csv, index=False)
            print(f"   ✓ Saved {len(factor_returns_df):,} factor return observations → {factor_returns_csv}")
            
            # Display sample
            print("\n   Sample Factor Returns:")
            print(factor_returns_for_csv.head(20).to_string(index=False))
        else:
            factor_returns_df = None
            print("   ⚠ No factor returns calculated")
        
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
                
                # Display sample
                print("\n   Sample Specific Returns:")
                print(specific_returns_df.head(10).to_string(index=False))
        else:
            specific_returns_df = None
            print("   ⚠ Skipped (no factor returns)")
    except Exception as e:
        print(f"\n⚠ WARNING: {str(e)}")
        specific_returns_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 4 completed in {time.time() - stage4_start:.1f}s")
    
    # Step 5: Specific risk (skip for single day - needs rolling window)
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 5] Calculating Specific Risk")
    print("=" * 80)
    print("   Note: Specific risk requires 60 days of historical data (rolling window)")
    print("   For single day test, this will be skipped or show limited results")
    
    try:
        if factor_returns_df is not None and len(exposures_df) > 0:
            # Try to calculate with available data (may have limited results)
            specific_risk_df = model_builder.calculate_specific_risk(
                exposures_df=exposures_df,
                returns_df=returns_df,
                factor_returns_df=factor_returns_df,
                window=60
            )
            if len(specific_risk_df) > 0:
                specific_risk_df['DATE'] = pd.to_datetime(specific_risk_df['DATE']).dt.strftime('%Y-%m-%d')
                specific_risk_csv = output_path / 'specific_risk.csv'
                variance_csv = output_path / 'variance.csv'
                specific_risk_df.to_csv(specific_risk_csv, index=False)
                specific_risk_df.to_csv(variance_csv, index=False)
                print(f"   ✓ Saved {len(specific_risk_df):,} specific risk observations → {specific_risk_csv}")
                print(f"   ✓ Also saved as variance.csv → {variance_csv}")
            else:
                specific_risk_df = None
                print("   ⚠ No specific risk calculated (insufficient historical data for single day)")
        else:
            specific_risk_df = None
            print("   ⚠ Skipped (no factor returns or exposures)")
    except Exception as e:
        print(f"   ⚠ WARNING: {str(e)}")
        specific_risk_df = None
    
    # Step 6: Factor covariance (skip for single day - needs rolling window)
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 6] Calculating Factor Covariance (SKIPPED - needs rolling window)")
    print("=" * 80)
    covariance_df = None
    
    # Step 7: Generate factor names metadata table
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 7] Generating Factor Names Metadata Table")
    print("=" * 80)
    stage7_start = time.time()
    
    try:
        # Load exposures for factor name extraction
        exposures_for_names = pd.read_csv(exposures_csv_path)
        
        # Convert long format to wide format temporarily
        if 'FACTOR_NAME' in exposures_for_names.columns:
            exposures_wide_for_names = exposures_for_names.pivot_table(
                index=['SECURITY_ID', 'DATE'],
                columns='FACTOR_NAME',
                values='EXPOSURE',
                aggfunc='first'
            ).reset_index()
        else:
            exposures_wide_for_names = exposures_for_names
        
        # Generate factor names table
        with tqdm(total=1, desc="Factor names", bar_format='{desc}: {elapsed}') as pbar:
            factor_names_df = model_builder.create_factor_names_table(exposures_wide_for_names)
            pbar.update(1)
        
        if len(factor_names_df) > 0:
            factor_names_csv = output_path / 'factor_model_factor_names.csv'
            factor_names_df.to_csv(factor_names_csv, index=False)
            print(f"   ✓ Saved {len(factor_names_df):,} factor name entries → {factor_names_csv}")
            
            # Display full table
            print("\n   Factor Names Table:")
            print(factor_names_df.to_string(index=False))
        else:
            factor_names_df = None
    except Exception as e:
        print(f"\n⚠ WARNING: {str(e)}")
        import traceback
        traceback.print_exc()
        factor_names_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 7 completed in {time.time() - stage7_start:.1f}s")
    
    # Display exposure table sample
    print("\n" + "=" * 80)
    print("EXPOSURE TABLE SAMPLE (Long Format)")
    print("=" * 80)
    exposures_sample = exposures_long_df.head(20)
    print(f"Total exposure rows: {len(exposures_long_df):,}")
    print(f"Unique securities: {exposures_long_df['SECURITY_ID'].nunique():,}")
    print(f"Unique factors: {exposures_long_df['FACTOR_NAME'].nunique():,}")
    print(f"\nSample rows:")
    print(exposures_sample.to_string(index=False))
    
    total_time = time.time() - stage2_start
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] TEST COMPLETE")
    print("=" * 80)
    print(f"✓ Total time: {total_time:.1f} seconds")
    print(f"✓ Processed {exposure_stats['processed']} dates")
    print(f"✓ All results saved to CSV files in: {output_dir}/")
    print("\nOutput CSV files:")
    print(f"  - {output_path / 'exposures.csv'} (long format)")
    if factor_returns_df is not None:
        print(f"  - {output_path / 'factor_returns.csv'}")
    if specific_returns_df is not None:
        print(f"  - {output_path / 'specific_returns.csv'}")
    if factor_names_df is not None:
        print(f"  - {output_path / 'factor_model_factor_names.csv'}")
    print("=" * 80)
    
    return {
        'output_dir': output_dir,
        'processed': exposure_stats['processed'],
        'exposures_count': len(exposures_long_df),
        'factor_returns_count': len(factor_returns_df) if factor_returns_df is not None else 0,
        'specific_returns_count': len(specific_returns_df) if specific_returns_df is not None else 0
    }


if __name__ == "__main__":
    # Test on a single day
    # Note: 2020-01-01 is likely a holiday (New Year's Day), so using 2020-01-02
    # You can change test_date to any trading day you want to test
    results = test_single_day(
        test_date='2020-01-02',  # Change to any trading day (e.g., '2020-01-02', '2020-01-03')
        output_dir='test_results',
        data_dir='data',
        skip_download=True  # Set to False if you need to download data first
    )
    
    if results:
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"✓ Exposures: {results['exposures_count']:,} rows")
        print(f"✓ Factor Returns: {results['factor_returns_count']:,} rows")
        print(f"✓ Specific Returns: {results['specific_returns_count']:,} rows")
        print("\nYou can now inspect the CSV files in the 'test_results' directory")
        print("=" * 80)

