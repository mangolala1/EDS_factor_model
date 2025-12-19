"""
Test script to run factor model workflow for a single day (2020-01-02)
This allows us to inspect the output tables before running the full pipeline
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quarter_processor import process_quarters_parallel
from src.model_builder import FactorModelBuilder
from src.data_retrieval import SnowflakeDataRetriever, get_date_range

def test_single_day(
    test_date: str = '2020-01-02', 
    output_dir: str = 'test_results',
    data_dir: str = 'data',
    skip_download: bool = True 
):
    """
    Test workflow for a single day
    
    Args:
        test_date: Single date to test (YYYY-MM-DD)
        output_dir: Directory to save test results
        data_dir: Directory with Parquet files
        skip_download: If True, skip bulk download
    """
    # Use the test date as both start and end date
    start_date = test_date
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
        from src.bulk_download import download_all_data
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
    
    exposures_parquet_dir = exposure_stats.get('exposures_parquet_dir')
    if not exposures_parquet_dir or not Path(exposures_parquet_dir).exists():
        print("✗ ERROR: Exposures Parquet directory not found.")
        return None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 2 completed in {time.time() - stage2_start:.1f}s")
    
    # Step 3: Calculate factor returns
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 3] Calculating Factor Returns")
    print("=" * 80)
    stage3_start = time.time()
    
    # Initialize variables that may be used later
    exposures_df = None
    exposures_df_test_date = None
    returns_df = None
    model_builder = None
    
    try:
        print("Loading exposures from Parquet (date-partitioned, wide format)...")
        with tqdm(total=1, desc="Loading exposures", bar_format='{desc}: {elapsed}') as pbar:
            # Read all date partitions and combine
            exposure_chunks = []
            date_dirs = sorted([d for d in Path(exposures_parquet_dir).iterdir() if d.is_dir() and d.name.startswith('date=')])
            
            for date_dir in date_dirs:
                parquet_files = list(date_dir.glob('*.parquet'))
                for parquet_file in parquet_files:
                    chunk = pd.read_parquet(parquet_file)
                    exposure_chunks.append(chunk)
            
            if not exposure_chunks:
                raise ValueError("No exposure Parquet files found")
            
            # Combine all exposures
            exposures_df_all = pd.concat(exposure_chunks, ignore_index=True)
            
            # Ensure DATE is datetime
            exposures_df_all['DATE'] = pd.to_datetime(exposures_df_all['DATE'])
            
            # Set index to (FACTSET_ID, DATE) for regression calculations
            if 'FACTSET_ID' in exposures_df_all.columns:
                exposures_df_all = exposures_df_all.set_index(['FACTSET_ID', 'DATE'])
            elif 'SECURITY_ID' in exposures_df_all.columns:
                exposures_df_all = exposures_df_all.rename(columns={'SECURITY_ID': 'FACTSET_ID'})
                exposures_df_all = exposures_df_all.set_index(['FACTSET_ID', 'DATE'])
            else:
                raise ValueError("Exposures DataFrame must have FACTSET_ID or SECURITY_ID column")
            
            # Remove MODEL column if present
            if 'MODEL' in exposures_df_all.columns:
                exposures_df_all = exposures_df_all.drop(columns=['MODEL'])
            
            # Keep all exposures for factor returns calculation
            exposures_df = exposures_df_all
            
            # Also create a filtered version for the test date (for specific risk)
            test_date_dt = pd.to_datetime(test_date)
            exposures_df_test_date = exposures_df_all[exposures_df_all.index.get_level_values('DATE') == test_date_dt]
            
            pbar.update(1)
        
        print(f"   ✓ Loaded {len(exposures_df_all):,} exposure rows (wide format)")
        print(f"   ✓ Date range: {exposures_df_all.index.get_level_values('DATE').min()} to {exposures_df_all.index.get_level_values('DATE').max()}")
        print(f"   ✓ Exposures for test date ({test_date}): {len(exposures_df_test_date):,} rows")
        
        # Load returns from Parquet
        print("Loading returns from Parquet...")
        returns_path = Path(data_dir) / 'returns.parquet'
        if returns_path.exists():
            returns_df = pd.read_parquet(returns_path)
            
            # Remove duplicate columns first
            returns_df = returns_df.loc[:, ~returns_df.columns.duplicated()]
            
            if 'FSYM_ID' in returns_df.columns:
                returns_df = returns_df.rename(columns={'FSYM_ID': 'FACTSET_ID'})
            
            # Handle DATE column - prefer P_DATE if both exist
            if 'P_DATE' in returns_df.columns:
                # If DATE also exists, drop it first to avoid duplicates
                if 'DATE' in returns_df.columns:
                    returns_df = returns_df.drop(columns=['DATE'])
                returns_df = returns_df.rename(columns={'P_DATE': 'DATE'})
            elif 'DATE' not in returns_df.columns:
                # No DATE column at all - skip
                returns_df = pd.DataFrame()
            
            # Ensure DATE is a single column (not DataFrame) before converting
            if len(returns_df) > 0 and 'DATE' in returns_df.columns:
                if isinstance(returns_df['DATE'], pd.DataFrame):
                    returns_df['DATE'] = returns_df['DATE'].iloc[:, 0]
                returns_df['DATE'] = pd.to_datetime(returns_df['DATE'])
            
            # For specific risk calculation, we need ALL historical data up to test_date
            # Don't filter to just the processed dates - use all available data
            test_date_dt = pd.to_datetime(test_date)
            returns_df = returns_df[returns_df['DATE'] <= test_date_dt]
            
            print(f"   Loaded ALL returns up to {test_date}: {len(returns_df):,} rows")
        else:
            print("   ⚠ returns.parquet not found - please run bulk_download.py first")
            returns_df = pd.DataFrame()
        
        print(f"   ✓ Loaded {len(returns_df):,} return observations")
        if len(returns_df) > 0 and 'DATE' in returns_df.columns:
            print(f"   ✓ Date range: {returns_df['DATE'].min()} to {returns_df['DATE'].max()}")
            print(f"   ✓ Unique dates: {returns_df['DATE'].nunique()}")
        else:
            print("   ⚠ No returns data loaded")
        
        # Calculate factor returns
        # We need factor returns for ALL dates where we have both exposures and returns
        # This gives us the historical factor returns needed for the rolling window
        print("Calculating factor returns...")
        print(f"   Note: Calculating factor returns for all dates with both exposures and returns")
        
        # Get all dates where we have both exposures and returns
        exposure_dates = set(exposures_df.index.get_level_values('DATE').unique())
        return_dates = set(returns_df['DATE'].unique())
        common_dates = sorted(exposure_dates & return_dates)
        
        print(f"   Dates with exposures: {len(exposure_dates)}")
        print(f"   Dates with returns: {len(return_dates)}")
        print(f"   Common dates (will calculate factor returns): {len(common_dates)}")
        
        if len(common_dates) > 0:
            # Filter to common dates only
            exposures_for_factor_returns = exposures_df[exposures_df.index.get_level_values('DATE').isin(common_dates)]
            returns_for_factor_returns = returns_df[returns_df['DATE'].isin(common_dates)]
            
            with tqdm(total=1, desc="Factor returns", bar_format='{desc}: {elapsed}') as pbar:
                model_builder = FactorModelBuilder()
                factor_returns_df = model_builder.calculate_daily_factor_returns(
                    exposures_df=exposures_for_factor_returns,
                    returns_df=returns_for_factor_returns
                )
                pbar.update(1)
        else:
            print("   ⚠ No common dates between exposures and returns")
            factor_returns_df = None
        
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
        if factor_returns_df is not None and exposures_df is not None:
            # Use exposures for test date only
            test_date_dt = pd.to_datetime(test_date)
            if exposures_df_test_date is None:
                exposures_df_test_date = exposures_df[exposures_df.index.get_level_values('DATE') == test_date_dt]
            returns_df_test_date = returns_df[returns_df['DATE'] == test_date_dt] if returns_df is not None else pd.DataFrame()
            
            with tqdm(total=1, desc="Specific returns", bar_format='{desc}: {elapsed}') as pbar:
                specific_returns_df = model_builder.calculate_specific_returns(
                    exposures_df=exposures_df_test_date,
                    returns_df=returns_df_test_date,
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
    
    # Step 5: Specific risk
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 5] Calculating Specific Risk")
    print("=" * 80)
    
    specific_risk_df = None
    try:
        # Check prerequisites
        if factor_returns_df is None:
            print("   ⚠ Skipped (no factor returns)")
        elif exposures_df_test_date is None:
            print("   ⚠ Skipped (exposures_df_test_date is None)")
        elif len(exposures_df_test_date) == 0:
            print("   ⚠ Skipped (no exposures for test date)")
        elif returns_df is None or len(returns_df) == 0:
            print("   ⚠ Skipped (no returns data)")
        else:
            # Calculate specific risk (will return empty if insufficient historical data)
            specific_risk_df = model_builder.calculate_specific_risk(
                exposures_df=exposures_df_test_date,
                returns_df=returns_df,
                factor_returns_df=factor_returns_df,
                window=60
            )
            
            if len(specific_risk_df) > 0:
                specific_risk_df['DATE'] = pd.to_datetime(specific_risk_df['DATE']).dt.strftime('%Y-%m-%d')
                specific_risk_csv = output_path / 'specific_risk.csv'
                specific_risk_df.to_csv(specific_risk_csv, index=False)
                print(f"   ✓ Saved {len(specific_risk_df):,} specific risk observations → {specific_risk_csv}")
            else:
                specific_risk_df = None
                print("   ⚠ No specific risk calculated (insufficient historical data - need 60 days of factor returns)")
    except Exception as e:
        print(f"   ✗ ERROR in specific risk calculation: {str(e)}")
        import traceback
        traceback.print_exc()
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
        # Exposures are already in wide format, just reset index for factor names
        exposures_wide_for_names = exposures_df_all.reset_index()
        
        # Generate factor names table
        # Initialize model_builder if not already initialized
        if model_builder is None:
            model_builder = FactorModelBuilder()
        
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
    print("EXPOSURE TABLE SAMPLE (Wide Format)")
    print("=" * 80)
    exposures_sample = exposures_df_all.reset_index().head(20)
    print(f"Total exposure rows: {len(exposures_df_all):,}")
    print(f"Unique securities: {exposures_df_all.index.get_level_values('FACTSET_ID').nunique():,}")
    print(f"Unique dates: {exposures_df_all.index.get_level_values('DATE').nunique():,}")
    print(f"Factor columns: {len([c for c in exposures_df_all.columns if c not in ['FACTSET_ID', 'DATE']]):,}")
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
    print(f"  - {exposures_parquet_dir} (date-partitioned Parquet, wide format)")
    if factor_returns_df is not None:
        print(f"  - {output_path / 'factor_returns.csv'}")
    if specific_returns_df is not None:
        print(f"  - {output_path / 'specific_returns.csv'}")
    if specific_risk_df is not None:
        print(f"  - {output_path / 'specific_risk.csv'} (MODEL, DATE, SECURITY_ID, SPECIFIC_VAR)")
    if factor_names_df is not None:
        print(f"  - {output_path / 'factor_model_factor_names.csv'}")
    print("=" * 80)
    
    return {
        'output_dir': output_dir,
        'processed': exposure_stats['processed'],
        'exposures_count': len(exposures_df_all),
        'factor_returns_count': len(factor_returns_df) if factor_returns_df is not None else 0,
        'specific_returns_count': len(specific_returns_df) if specific_returns_df is not None else 0,
        'specific_risk_count': len(specific_risk_df) if specific_risk_df is not None else 0
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
        if results['specific_risk_count'] > 0:
            print(f"✓ Specific Risk: {results['specific_risk_count']:,} rows")
        else:
            print(f"⚠ Specific Risk: {results['specific_risk_count']:,} rows (not calculated)")
        print("\nYou can now inspect the CSV files in the 'test_results' directory")
        print("=" * 80)

