"""
Test script to run factor model workflow for Q1 2020 (2020-01-01 to 2020-03-31)
This generates all output tables for the quarter
"""
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.quarter_processor import process_quarters_parallel
from src.model_builder import FactorModelBuilder
from src.data_retrieval import SnowflakeDataRetriever, get_date_range


def test_quarter(
    start_date: str = '2020-01-01',
    end_date: str = '2020-03-31',
    output_dir: str = 'test_results_q1_2020',
    data_dir: str = 'data',
    skip_download: bool = True  # Assume data already downloaded
):
    """
    Test workflow for Q1 2020
    
    Args:
        start_date: Start date (default: 2020-01-01)
        end_date: End date (default: 2020-03-31)
        output_dir: Directory to save test results
        data_dir: Directory with Parquet files
        skip_download: If True, skip bulk download
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("EDS Factor Model - Q1 2020 Test")
    print("=" * 80)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing Q1 2020")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    
    # Step 1: Bulk download (if needed)
    if not skip_download:
        print("\n[STAGE 1] Bulk Download from Snowflake → Local Parquet Files")
        print("=" * 80)
        from src.bulk_download import download_all_data
        # Need lookback for rolling calculations, so start earlier
        lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=90)).strftime('%Y-%m-%d')
        download_all_data(lookback_start, end_date, data_dir)
    else:
        print("\n[STAGE 1] Skipping download (using local Parquet files)")
    
    # Step 2: Process quarters
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
            
            print(f"   ✓ Loaded {len(exposures_long_df):,} exposure observations")
            print(f"   ✓ Date range: {exposures_long_df['DATE'].min()} to {exposures_long_df['DATE'].max()}")
            print(f"   ✓ Unique dates: {exposures_long_df['DATE'].nunique()}")
            print(f"   ✓ Unique securities: {exposures_long_df['SECURITY_ID'].nunique()}")
            
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
            
            # Filter to date range and trading dates from exposures
            trading_dates = exposures_df.index.get_level_values('DATE').unique()
            returns_df = returns_df[returns_df['DATE'].isin(trading_dates)]
            print(f"   ✓ Unique securities in returns: {returns_df['FACTSET_ID'].nunique():,}")
            print(f"   ✓ Unique dates in returns: {returns_df['DATE'].nunique():,}")
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
            # Save to CSV
            factor_returns_for_csv = factor_returns_df.copy()
            factor_returns_for_csv['DATE'] = pd.to_datetime(factor_returns_for_csv['DATE']).dt.strftime('%Y-%m-%d')
            factor_returns_csv = output_path / 'factor_returns.csv'
            factor_returns_for_csv.to_csv(factor_returns_csv, index=False)
            print(f"   ✓ Saved {len(factor_returns_df):,} factor return observations → {factor_returns_csv}")
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
        else:
            specific_returns_df = None
            print("   ⚠ Skipped (no factor returns)")
    except Exception as e:
        print(f"\n⚠ WARNING: {str(e)}")
        import traceback
        traceback.print_exc()
        specific_returns_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 4 completed in {time.time() - stage4_start:.1f}s")
    
    # Step 5: Specific risk
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 5] Calculating Specific Risk (60-Day Window)")
    print("=" * 80)
    stage5_start = time.time()
    
    try:
        if factor_returns_df is not None and len(exposures_df) > 0:
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
                # Save as both specific_risk.csv and variance.csv (workflow specification)
                specific_risk_csv = output_path / 'specific_risk.csv'
                variance_csv = output_path / 'variance.csv'
                specific_risk_df.to_csv(specific_risk_csv, index=False)
                specific_risk_df.to_csv(variance_csv, index=False)
                print(f"   ✓ Saved {len(specific_risk_df):,} specific risk observations → {specific_risk_csv}")
                print(f"   ✓ Also saved as variance.csv → {variance_csv}")
            else:
                specific_risk_df = None
                print("   ⚠ No specific risk calculated (insufficient historical data)")
        else:
            specific_risk_df = None
            print("   ⚠ Skipped (no factor returns or exposures)")
    except Exception as e:
        print(f"\n⚠ WARNING in specific risk calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        specific_risk_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 5 completed in {time.time() - stage5_start:.1f}s")
    
    # Step 6: Factor covariance
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 6] Calculating Factor Covariance (60-Day Window)")
    print("=" * 80)
    stage6_start = time.time()
    
    try:
        if factor_returns_df is not None:
            with tqdm(total=1, desc="Factor covariance", bar_format='{desc}: {elapsed}') as pbar:
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
            else:
                covariance_df = None
                print("   ⚠ No covariance calculated (insufficient historical data)")
        else:
            covariance_df = None
            print("   ⚠ Skipped (no factor returns)")
    except Exception as e:
        print(f"\n⚠ WARNING in covariance calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        covariance_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 6 completed in {time.time() - stage6_start:.1f}s")
    
    # Step 7: Generate factor names metadata table
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [STAGE 7] Generating Factor Names Metadata Table")
    print("=" * 80)
    stage7_start = time.time()
    
    try:
        # Convert exposures to wide format for factor names
        exposures_wide = exposures_long_df.pivot_table(
            index=['SECURITY_ID', 'DATE'],
            columns='FACTOR_NAME',
            values='EXPOSURE',
            aggfunc='first'
        ).reset_index()
        
        # Generate factor names table
        with tqdm(total=1, desc="Factor names", bar_format='{desc}: {elapsed}') as pbar:
            factor_names_df = model_builder.create_factor_names_table(exposures_wide)
            pbar.update(1)
        
        if len(factor_names_df) > 0:
            factor_names_csv = output_path / 'factor_model_factor_names.csv'
            factor_names_df.to_csv(factor_names_csv, index=False)
            print(f"   ✓ Saved {len(factor_names_df):,} factor name entries → {factor_names_csv}")
        else:
            factor_names_df = None
    except Exception as e:
        print(f"\n⚠ WARNING: {str(e)}")
        import traceback
        traceback.print_exc()
        factor_names_df = None
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Stage 7 completed in {time.time() - stage7_start:.1f}s")
    
    # Summary
    total_time = time.time() - stage2_start
    print("\n" + "=" * 80)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] TEST COMPLETE")
    print("=" * 80)
    print(f"✓ Total time: {total_time:.1f} seconds")
    print(f"✓ Processed {exposure_stats['processed']} dates")
    print(f"✓ All results saved to CSV files in: {output_dir}/")
    print("\nOutput CSV files:")
    print(f"  - {output_path / 'exposures.csv'} (long format: MODEL, DATE, SECURITY_ID, FACTOR_NAME, EXPOSURE)")
    if factor_returns_df is not None:
        print(f"  - {output_path / 'factor_returns.csv'} (MODEL, DATE, FACTOR_NAME, RETURN)")
    if specific_returns_df is not None:
        print(f"  - {output_path / 'specific_returns.csv'} (MODEL, DATE, SECURITY_ID, SPECIFIC_RETURN)")
    if specific_risk_df is not None:
        print(f"  - {output_path / 'specific_risk.csv'} (MODEL, DATE, SECURITY_ID, SPECIFIC_VAR)")
    if covariance_df is not None:
        print(f"  - {output_path / 'factor_covariance.csv'} (MODEL, DATE, FACTOR_NAME_1, FACTOR_NAME_2, COVARIANCE)")
    if factor_names_df is not None:
        print(f"  - {output_path / 'factor_model_factor_names.csv'} (MODEL, FACTOR_DISPLAY_NAME, FACTOR_GROUP)")
    print("=" * 80)
    
    return {
        'output_dir': output_dir,
        'processed': exposure_stats['processed'],
        'exposures_count': len(exposures_long_df),
        'factor_returns_count': len(factor_returns_df) if factor_returns_df is not None else 0,
        'specific_returns_count': len(specific_returns_df) if specific_returns_df is not None else 0,
        'specific_risk_count': len(specific_risk_df) if specific_risk_df is not None else 0,
        'covariance_count': len(covariance_df) if covariance_df is not None else 0
    }


if __name__ == "__main__":
    # Test Q1 2020
    results = test_quarter(
        start_date='2020-01-01',
        end_date='2020-03-31',
        output_dir='test_results_q1_2020',
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
        if results['covariance_count'] > 0:
            print(f"✓ Factor Covariance: {results['covariance_count']:,} rows")
        else:
            print(f"⚠ Factor Covariance: {results['covariance_count']:,} rows (not calculated)")
        print("\nYou can now inspect the CSV files in the 'test_results_q1_2020' directory")
        print("=" * 80)

