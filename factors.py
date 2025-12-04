"""
Main workflow script for EDS Factor Model - MSCI Equity Risk Factor Model
Implements the complete workflow: Data Retrieval -> Factor Construction -> Factor Returns
Processes by date to handle full universe efficiently
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path
import config
from data_retrieval import SnowflakeDataRetriever, get_date_range
from factor_construction import FactorConstructor
from model_builder import FactorModelBuilder


def main(
    start_date: str = '2020-01-01',
    end_date: str = None,
    neutralize: bool = False,
    min_stocks_per_date: int = 50
):
    """
    Main workflow for building MSCI-styled factor model
    Processes by date to handle full universe efficiently
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format (default: '2020-01-01')
        end_date: End date in 'YYYY-MM-DD' format (default: today)
        neutralize: Whether to neutralize factors by industry/continent
        min_stocks_per_date: Minimum number of stocks required per date for factor calculation
    """
    print("=" * 80)
    print("EDS Factor Model - MSCI Equity Risk Factor Model")
    print("=" * 80)
    print(f"Workflow: Process by Date -> Construct Exposures -> Calculate Factor Returns")
    print("=" * 80)
    
    # Step 1: Connect to Snowflake and get trading dates
    print("\n[STEP 1] Connecting to Snowflake and retrieving trading dates...")
    retriever = SnowflakeDataRetriever()
    
    try:
        retriever.connect()
        
        # Determine date range
        if end_date is None:
            _, end_date_calc = get_date_range(lookback_days=1)
            end_date = end_date_calc
        
        # Enforce minimum date of 2015-01-01 for data retrieval
        min_data_date = '2015-01-01'
        effective_start_date = max(start_date, min_data_date) if start_date < min_data_date else start_date
        
        print(f"Date range: {start_date} to {end_date}")
        print(f"Data retrieval from: {effective_start_date} (for historical lookback)")
        
        # Get list of trading dates (non-holiday dates)
        print("\n[Step 1.1] Getting list of trading dates...")
        trading_dates = retriever.get_trading_dates(start_date, end_date)
        
        if len(trading_dates) == 0:
            print("✗ ERROR: No trading dates found in the specified range.")
            retriever.disconnect()
            return None
        
        print(f"✓ Found {len(trading_dates)} trading dates")
        print(f"  First date: {trading_dates[0]}")
        print(f"  Last date: {trading_dates[-1]}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to connect to Snowflake: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # STAGE A: Bulk load all raw data once
    print("\n" + "=" * 80)
    print("[STAGE A] Bulk Loading Raw Data")
    print("=" * 80)
    
    try:
        raw_data = retriever.load_raw_data_for_model(start_date, end_date)
        universe_df = raw_data['universe']
        fundamentals_all = raw_data['fundamentals']
        prices_all = raw_data['prices']
        returns_all = raw_data['returns']
        
        print(f"\n✓ Bulk data loading complete!")
        print(f"  Universe: {len(universe_df)} stocks")
        print(f"  Fundamentals: {len(fundamentals_all):,} rows")
        print(f"  Prices: {len(prices_all):,} rows")
        print(f"  Returns: {len(returns_all):,} rows")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to bulk load data: {str(e)}")
        import traceback
        traceback.print_exc()
        retriever.disconnect()
        return None
    
    # STAGE B: Process each date using preloaded data
    print("\n" + "=" * 80)
    print("[STAGE B] Processing Dates (Slicing from Preloaded Data)")
    print("=" * 80)
    
    # Create output directory for daily exposures
    exposures_dir = Path('outputs/exposures')
    exposures_dir.mkdir(parents=True, exist_ok=True)
    
    processed_dates = 0
    skipped_dates = 0
    
    print(f"\nProcessing {len(trading_dates)} dates sequentially...")
    
    for i, date_str in enumerate(trading_dates):
        if (i + 1) % 50 == 0:
            print(f"  Processing date {i+1}/{len(trading_dates)}: {date_str}...")
        
        try:
            date_dt = pd.to_datetime(date_str)
            
            # Check if output file already exists
            output_file = exposures_dir / f"exposures_{date_str}.parquet"
            if output_file.exists():
                continue  # Skip already processed dates
            
            # Slice data for this date (no SQL queries - just DataFrame filtering)
            fundamentals_date = fundamentals_all[fundamentals_all['DATE'] == date_dt].copy()
            prices_date = prices_all[prices_all['DATE'] == date_dt].copy()
            
            # Filter out holidays
            if 'IS_HOLIDAY' in prices_date.columns:
                if prices_date['IS_HOLIDAY'].all():
                    skipped_dates += 1
                    continue
                prices_date = prices_date[prices_date['IS_HOLIDAY'] == False]
            
            if len(fundamentals_date) == 0 or len(prices_date) == 0:
                skipped_dates += 1
                continue
            
            # Slice returns for lookback window (needed for momentum, volatility, liquidity)
            max_lookback = timedelta(days=365)  # Safe buffer
            lookback_start = date_dt - max_lookback
            
            returns_window = returns_all[
                (pd.to_datetime(returns_all['P_DATE']) >= lookback_start) &
                (pd.to_datetime(returns_all['P_DATE']) <= date_dt)
            ].copy()
            
            prices_window = prices_all[
                (prices_all['DATE'] >= lookback_start) &
                (prices_all['DATE'] <= date_dt)
            ].copy()
            
            # Initialize factor constructor with sliced data
            factor_constructor = FactorConstructor(
                fundamentals_df=fundamentals_all[fundamentals_all['DATE'] <= date_dt].copy(),
                prices_df=prices_window,
                returns_df=returns_window,
                universe_df=universe_df
            )
            
            # Construct factors (will process all dates in slice, then we filter)
            all_exposures = factor_constructor.construct_all_factors(neutralize=neutralize)
            
            # Filter to only the target date
            if 'DATE' in all_exposures.index.names:
                exposures_date = all_exposures[all_exposures.index.get_level_values('DATE') == date_dt]
            else:
                exposures_reset = all_exposures.reset_index()
                if 'DATE' in exposures_reset.columns:
                    exposures_reset['DATE'] = pd.to_datetime(exposures_reset['DATE'])
                    exposures_date = exposures_reset[exposures_reset['DATE'] == date_dt]
                    if len(exposures_date) > 0:
                        exposures_date = exposures_date.set_index(['FACTSET_ID', 'DATE'])
                    else:
                        exposures_date = pd.DataFrame()
                else:
                    exposures_date = pd.DataFrame()
            
            if len(exposures_date) == 0:
                skipped_dates += 1
                continue
            
            # Check minimum stocks requirement
            if len(exposures_date) < min_stocks_per_date:
                skipped_dates += 1
                continue
            
            # Save to parquet file
            exposures_date.reset_index().to_parquet(output_file, index=False)
            processed_dates += 1
            
        except Exception as e:
            print(f"  ⚠ Warning: Failed to process date {date_str}: {str(e)}")
            skipped_dates += 1
            continue
    
    print(f"\n✓ Processed {processed_dates} dates successfully")
    if skipped_dates > 0:
        print(f"  Skipped {skipped_dates} dates (holidays, insufficient data, or already processed)")
    
    if processed_dates == 0:
        print("✗ ERROR: No dates were processed successfully.")
        retriever.disconnect()
        return None
    
    # Step 3: Concatenate all daily parquet files
    print("\n" + "=" * 80)
    print("[STEP 3] Concatenating Daily Exposures")
    print("=" * 80)
    
    try:
        all_files = sorted(glob.glob(str(exposures_dir / "exposures_*.parquet")))
        
        if len(all_files) == 0:
            print("✗ ERROR: No exposure files found.")
            retriever.disconnect()
            return None
        
        print(f"  Found {len(all_files)} exposure files")
        print("  Concatenating...")
        
        exposures_list = []
        for file in all_files:
            try:
                df = pd.read_parquet(file)
                exposures_list.append(df)
            except Exception as e:
                print(f"  ⚠ Warning: Failed to read {file}: {str(e)}")
                continue
        
        if len(exposures_list) == 0:
            print("✗ ERROR: No valid exposure files could be read.")
            retriever.disconnect()
            return None
        
        exposures_df = pd.concat(exposures_list, ignore_index=True)
        
        # Set index
        if 'FACTSET_ID' in exposures_df.columns and 'DATE' in exposures_df.columns:
            exposures_df['DATE'] = pd.to_datetime(exposures_df['DATE'])
            exposures_df = exposures_df.set_index(['FACTSET_ID', 'DATE'])
        
        # Save combined file
        combined_file = Path('outputs') / f"all_exposures_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
        exposures_df.reset_index().to_parquet(combined_file, index=False)
        print(f"✓ Combined exposures saved to: {combined_file}")
        print(f"  Total observations: {len(exposures_df):,}")
        print(f"  Unique dates: {exposures_df.index.get_level_values('DATE').nunique()}")
        print(f"  Unique stocks: {exposures_df.index.get_level_values('FACTSET_ID').nunique()}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to concatenate exposures: {str(e)}")
        import traceback
        traceback.print_exc()
        retriever.disconnect()
        return None
    
    # Step 4: Prepare returns data for factor return calculation (use preloaded data)
    print("\n" + "=" * 80)
    print("[STEP 4] Preparing Returns Data for Factor Return Calculation")
    print("=" * 80)
    
    try:
        # Use preloaded returns data
        returns_prep = returns_all.copy()
        
        # Rename columns to match expected format
        returns_prep = returns_prep.rename(columns={'FSYM_ID': 'FACTSET_ID', 'P_DATE': 'DATE'})
        returns_prep['DATE'] = pd.to_datetime(returns_prep['DATE'])
        
        # Filter returns to trading dates only
        trading_dates_dt = pd.to_datetime(trading_dates)
        returns_prep = returns_prep[returns_prep['DATE'].isin(trading_dates_dt)]
        
        print(f"✓ Returns data prepared: {len(returns_prep):,} observations")
        print(f"  Unique dates: {returns_prep['DATE'].nunique()}")
        print(f"  Unique stocks: {returns_prep['FACTSET_ID'].nunique()}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to prepare returns data: {str(e)}")
        import traceback
        traceback.print_exc()
        retriever.disconnect()
        return None
    
    # Step 5: Calculate daily factor returns via OLS
    print("\n" + "=" * 80)
    print("[STEP 5] Calculating Daily Factor Returns (OLS Regression)")
    print("=" * 80)
    
    try:
        # Initialize model builder
        model_builder = FactorModelBuilder(exposures_df, pd.Series(dtype=float))
        
        print("Calculating factor returns via cross-sectional OLS regression...")
        factor_returns_df = model_builder.calculate_daily_factor_returns(
            exposures_df=exposures_df,
            returns_df=returns_prep
        )
        
        if len(factor_returns_df) == 0:
            print("✗ ERROR: No factor returns calculated. Check data alignment.")
            retriever.disconnect()
            return None
        
        print(f"\n✓ Factor returns calculated successfully!")
        print(f"  - Total factor return observations: {len(factor_returns_df):,}")
        print(f"  - Unique dates: {factor_returns_df['DATE'].nunique()}")
        print(f"  - Factors: {factor_returns_df['FACTOR_NAME'].unique().tolist()}")
        
        # Save factor returns
        factor_returns_file = Path('outputs') / f"factor_returns_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
        factor_returns_df.to_parquet(factor_returns_file, index=False)
        print(f"✓ Factor returns saved to: {factor_returns_file}")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to calculate factor returns: {str(e)}")
        import traceback
        traceback.print_exc()
        retriever.disconnect()
        return None
    
    # Step 6: Calculate specific returns
    print("\n" + "=" * 80)
    print("[STEP 6] Calculating Specific Returns")
    print("=" * 80)
    
    try:
        print("Calculating specific (idiosyncratic) returns as regression residuals...")
        specific_returns_df = model_builder.calculate_specific_returns(
            exposures_df=exposures_df,
            returns_df=returns_prep,
            factor_returns_df=factor_returns_df
        )
        
        print(f"\n✓ Specific returns calculated successfully!")
        print(f"  - Total specific return observations: {len(specific_returns_df):,}")
        
        # Save specific returns
        specific_returns_file = Path('outputs') / f"specific_returns_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
        specific_returns_df.to_parquet(specific_returns_file, index=False)
        print(f"✓ Specific returns saved to: {specific_returns_file}")
        
    except Exception as e:
        print(f"\n⚠ WARNING: Failed to calculate specific returns: {str(e)}")
        print("Continuing without specific returns...")
        specific_returns_df = None
    
    # Step 7: Calculate specific risk (rolling 60-day window)
    print("\n" + "=" * 80)
    print("[STEP 7] Calculating Specific Risk (Rolling 60-Day Window)")
    print("=" * 80)
    
    try:
        print("Calculating specific risk with rolling 60-day window...")
        specific_risk_df = model_builder.calculate_specific_risk(
            exposures_df=exposures_df,
            returns_df=returns_prep,
            factor_returns_df=factor_returns_df,
            window=60
        )
        
        if len(specific_risk_df) > 0:
            print(f"\n✓ Specific risk calculated successfully!")
            print(f"  - Total specific risk observations: {len(specific_risk_df):,}")
            
            # Save specific risk
            specific_risk_file = Path('outputs') / f"specific_risk_{start_date.replace('-', '')}_{end_date.replace('-', '')}.parquet"
            specific_risk_df.to_parquet(specific_risk_file, index=False)
            print(f"✓ Specific risk saved to: {specific_risk_file}")
        else:
            print("⚠ WARNING: No specific risk calculated")
            specific_risk_df = None
            
    except Exception as e:
        print(f"\n⚠ WARNING: Failed to calculate specific risk: {str(e)}")
        print("Continuing without specific risk...")
        import traceback
        traceback.print_exc()
        specific_risk_df = None
    
    retriever.disconnect()
    
    # Final Summary
    print("\n" + "=" * 80)
    print("WORKFLOW COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  ✓ Processed {processed_dates} dates")
    print(f"  ✓ Factor exposures: {len(exposures_df):,} observations")
    print(f"  ✓ Factor returns: {len(factor_returns_df):,} observations")
    if specific_returns_df is not None:
        print(f"  ✓ Specific returns: {len(specific_returns_df):,} observations")
    if specific_risk_df is not None:
        print(f"  ✓ Specific risk: {len(specific_risk_df):,} observations")
    print("\nOutput files saved to: outputs/")
    print("=" * 80)
    
    # Return results dictionary
    results = {
        'exposures': exposures_df,
        'factor_returns': factor_returns_df,
        'specific_returns': specific_returns_df,
        'specific_risk': specific_risk_df
    }
    
    return results


if __name__ == "__main__":
    # Run main workflow
    # Process from 2020-01-01 to today
    results = main(
        start_date='2020-01-01',
        end_date=None,  # Will default to today
        neutralize=False,
        min_stocks_per_date=50
    )
    
    if results:
        print("\n✓ Main workflow completed successfully!")
    else:
        print("\n✗ Main workflow failed. Please check errors above.")
