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
    # Use a date range that includes enough historical data for rolling window calculations
    # Specific risk needs 60 trading days, so we need ~90 calendar days to ensure we have enough
    from datetime import datetime, timedelta
    test_dt = datetime.strptime(test_date, '%Y-%m-%d')
    start_date = (test_dt - timedelta(days=90)).strftime('%Y-%m-%d')  # 90 days to ensure 60 trading days
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
    
    # Initialize variables that may be used later
    exposures_df = None
    exposures_df_test_date = None
    returns_df = None
    model_builder = None
    
    try:
        print("Loading exposures from CSV...")
        with tqdm(total=1, desc="Loading exposures", bar_format='{desc}: {elapsed}') as pbar:
            exposures_long_df = pd.read_csv(exposures_csv_path)
            exposures_long_df['DATE'] = pd.to_datetime(exposures_long_df['DATE'])
            
            # Load all exposures in the date range (needed for factor returns calculation)
            # But we'll filter to test date for specific risk calculation later
            print(f"   Loaded exposures for date range: {len(exposures_long_df):,} total exposure rows")
            print(f"   Date range: {exposures_long_df['DATE'].min()} to {exposures_long_df['DATE'].max()}")
            
            # Convert long format to wide format for regression calculations
            exposures_df_all = exposures_long_df.pivot_table(
                index=['SECURITY_ID', 'DATE'],
                columns='FACTOR_NAME',
                values='EXPOSURE',
                aggfunc='first'
            )
            exposures_df_all = exposures_df_all.rename_axis(None, axis=1)
            exposures_df_all = exposures_df_all.rename_axis(['FACTSET_ID', 'DATE'])
            
            # Keep all exposures for factor returns calculation
            exposures_df = exposures_df_all
            
            # Also create a filtered version for the test date (for specific risk)
            test_date_dt = pd.to_datetime(test_date)
            exposures_df_test_date = exposures_df_all[exposures_df_all.index.get_level_values('DATE') == test_date_dt]
            
            pbar.update(1)
        
        print(f"   ✓ Loaded {len(exposures_long_df):,} exposure observations")
        print(f"   ✓ Exposures for test date ({test_date}): {len(exposures_df_test_date):,} rows")
        
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
            
            # For specific risk calculation, we need ALL historical data up to test_date
            # Don't filter to just the processed dates - use all available data
            test_date_dt = pd.to_datetime(test_date)
            returns_df = returns_df[returns_df['DATE'] <= test_date_dt]
            
            print(f"   Loaded ALL returns up to {test_date}: {len(returns_df):,} rows")
        else:
            print("   ⚠ returns.parquet not found - please run bulk_download.py first")
            returns_df = pd.DataFrame()
        
        print(f"   ✓ Loaded {len(returns_df):,} return observations")
        print(f"   ✓ Date range: {returns_df['DATE'].min()} to {returns_df['DATE'].max()}")
        print(f"   ✓ Unique dates: {returns_df['DATE'].nunique()}")
        
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
                model_builder = FactorModelBuilder(exposures_for_factor_returns, pd.Series(dtype=float))
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
    
    # Step 5: Specific risk (skip for single day - needs rolling window)
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 5] Calculating Specific Risk")
    print("=" * 80)
    print("   Note: Specific risk requires 60 days of historical data (rolling window)")
    
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
            # Debug info
            print(f"   Checking prerequisites...")
            print(f"   - Exposures for test date: {len(exposures_df_test_date):,} rows")
            print(f"   - Returns data: {len(returns_df):,} rows, {returns_df['DATE'].nunique()} unique dates")
            print(f"   - Factor returns: {len(factor_returns_df):,} rows, {factor_returns_df['DATE'].nunique()} unique dates")
            print(f"   - Date range in returns: {returns_df['DATE'].min()} to {returns_df['DATE'].max()}")
            print(f"   - Date range in factor returns: {factor_returns_df['DATE'].min()} to {factor_returns_df['DATE'].max()}")
            
            # Check if we have enough historical data
            test_date_dt = pd.to_datetime(test_date)
            factor_returns_dates = pd.to_datetime(factor_returns_df['DATE'].unique())
            dates_before_test = factor_returns_dates[factor_returns_dates < test_date_dt]
            
            print(f"   - Factor return dates before test date: {len(dates_before_test)}")
            
            # If we don't have enough data, try to process more dates from parquet files
            if len(dates_before_test) < 60:
                print(f"   ⚠ Only {len(dates_before_test)} days of factor returns before test date")
                print(f"   ⚠ Need at least 60 days for rolling window calculation")
                print(f"   → Attempting to process additional historical dates from parquet files...")
                
                # Check if parquet files have more data
                needed_start_date = (test_date_dt - timedelta(days=90)).strftime('%Y-%m-%d')
                returns_in_range = returns_df[returns_df['DATE'] >= needed_start_date]
                return_dates_available = sorted(returns_in_range['DATE'].unique())
                
                print(f"   - Returns available in parquet from {needed_start_date}: {len(return_dates_available)} dates")
                
                if len(return_dates_available) >= 60:
                    # We have returns data, but need to process exposures for those dates
                    print(f"   → Processing exposures for {len(return_dates_available)} additional dates...")
                    
                    try:
                        # Process additional dates using quarter_processor
                        additional_exposure_stats = process_quarters_parallel(
                            start_date=needed_start_date,
                            end_date=test_date,
                            output_dir=output_dir,
                            data_dir=data_dir,
                            max_workers=1,
                            neutralize=False
                        )
                        
                        if additional_exposure_stats['processed'] > 0:
                            # Reload exposures CSV to get the new data
                            print(f"   → Reloading exposures with additional {additional_exposure_stats['processed']} dates...")
                            exposures_long_df_updated = pd.read_csv(exposures_csv_path)
                            exposures_long_df_updated['DATE'] = pd.to_datetime(exposures_long_df_updated['DATE'])
                            
                            # Convert to wide format
                            exposures_df_updated = exposures_long_df_updated.pivot_table(
                                index=['SECURITY_ID', 'DATE'],
                                columns='FACTOR_NAME',
                                values='EXPOSURE',
                                aggfunc='first'
                            )
                            exposures_df_updated = exposures_df_updated.rename_axis(None, axis=1)
                            exposures_df_updated = exposures_df_updated.rename_axis(['FACTSET_ID', 'DATE'])
                            
                            # Recalculate factor returns with all dates
                            exposure_dates_updated = set(exposures_df_updated.index.get_level_values('DATE').unique())
                            return_dates_set = set(returns_df['DATE'].unique())
                            common_dates_updated = sorted(exposure_dates_updated & return_dates_set)
                            
                            if len(common_dates_updated) >= 60:
                                print(f"   → Recalculating factor returns for {len(common_dates_updated)} dates...")
                                exposures_for_factor_returns_updated = exposures_df_updated[
                                    exposures_df_updated.index.get_level_values('DATE').isin(common_dates_updated)
                                ]
                                returns_for_factor_returns_updated = returns_df[returns_df['DATE'].isin(common_dates_updated)]
                                
                                model_builder = FactorModelBuilder(exposures_for_factor_returns_updated, pd.Series(dtype=float))
                                factor_returns_df = model_builder.calculate_daily_factor_returns(
                                    exposures_df=exposures_for_factor_returns_updated,
                                    returns_df=returns_for_factor_returns_updated
                                )
                                
                                # Update exposures_df_test_date
                                exposures_df_test_date = exposures_df_updated[
                                    exposures_df_updated.index.get_level_values('DATE') == test_date_dt
                                ]
                                
                                # Update main exposures_df for consistency
                                exposures_df = exposures_df_updated
                                
                                # Check again
                                factor_returns_dates = pd.to_datetime(factor_returns_df['DATE'].unique())
                                dates_before_test = factor_returns_dates[factor_returns_dates < test_date_dt]
                                print(f"   ✓ Now have {len(dates_before_test)} days of factor returns before test date")
                            else:
                                print(f"   ⚠ Still only {len(common_dates_updated)} common dates after processing")
                        else:
                            print(f"   ⚠ Could not process additional dates (may need to download more data)")
                    except Exception as e:
                        print(f"   ⚠ Error processing additional dates: {str(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"   ⚠ Not enough returns data in parquet files")
                    print(f"   ⚠ Need to download more data from Snowflake (run bulk_download.py)")
            
            # Final check before calculating specific risk
            # Re-check dates_before_test in case we updated factor_returns_df
            factor_returns_dates_final = pd.to_datetime(factor_returns_df['DATE'].unique())
            dates_before_test_final = factor_returns_dates_final[factor_returns_dates_final < test_date_dt]
            
            if test_date_dt not in factor_returns_dates_final:
                print(f"   ⚠ WARNING: Test date {test_date} not found in factor returns")
                print(f"   ⚠ Cannot calculate specific risk for this date")
                specific_risk_df = None
            elif len(dates_before_test_final) < 60:
                print(f"   ⚠ Still insufficient data: {len(dates_before_test_final)} days (need 60)")
                print(f"   ⚠ Cannot calculate specific risk - need more historical data")
                print(f"   ⚠ Try: Download more data or use a later test date")
                specific_risk_df = None
            else:
                # Use exposures for test date only, but returns and factor_returns need full history
                print(f"   ✓ Sufficient data available ({len(dates_before_test_final)} days), calculating specific risk...")
                # model_builder is already available (either original or updated)
                specific_risk_df = model_builder.calculate_specific_risk(
                    exposures_df=exposures_df_test_date,
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
                print("   ⚠ No specific risk calculated (returned empty DataFrame)")
                print("   ⚠ This usually means insufficient historical data (need 60 trading days)")
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
    if specific_risk_df is not None:
        print(f"  - {output_path / 'specific_risk.csv'} (MODEL, DATE, SECURITY_ID, SPECIFIC_VAR)")
    if factor_names_df is not None:
        print(f"  - {output_path / 'factor_model_factor_names.csv'}")
    print("=" * 80)
    
    return {
        'output_dir': output_dir,
        'processed': exposure_stats['processed'],
        'exposures_count': len(exposures_long_df),
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

