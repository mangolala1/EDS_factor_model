"""
Main workflow script for EDS Factor Model
"""
import pandas as pd
import numpy as np
from datetime import datetime
import config
from data_retrieval import SnowflakeDataRetriever, get_date_range
from factor_construction import FactorConstructor
from factor_testing import FactorTester
from model_builder import FactorModelBuilder


def main(
    price_table: str = None,
    fundamental_table: str = None,
    ticker_query: str = None,
    max_tickers: int = 500
):
    """
    Main workflow for building MSCI-styled factor model
    
    Args:
        price_table: Name of price data table in Snowflake (default: 'stock_prices')
        fundamental_table: Name of fundamental data table (default: 'stock_fundamentals')
        ticker_query: Custom SQL query to get tickers (optional)
        max_tickers: Maximum number of tickers to use
    """
    print("=" * 60)
    print("EDS Factor Model - MSCI-Styled Factor Model Builder")
    print("=" * 60)
    
    # Configuration
    price_table = price_table or 'stock_prices'
    fundamental_table = fundamental_table or 'stock_fundamentals'
    
    # Step 1: Get data from Snowflake
    print("\n[Step 1] Retrieving data from Snowflake...")
    retriever = SnowflakeDataRetriever()
    
    try:
        retriever.connect()
        
        # Get date range
        start_date, end_date = get_date_range(lookback_days=config.LOOKBACK_PERIOD)
        print(f"Date range: {start_date} to {end_date}")
        
        # Get ticker list
        print("\n[Step 1.1] Retrieving ticker list...")
        if ticker_query:
            tickers_df = retriever.execute_query(ticker_query)
        else:
            ticker_query_default = f"SELECT DISTINCT ticker FROM {price_table} ORDER BY ticker LIMIT {max_tickers}"
            tickers_df = retriever.execute_query(ticker_query_default)
        
        if 'ticker' in tickers_df.columns:
            tickers = tickers_df['ticker'].tolist()
        else:
            # Try first column if 'ticker' not found
            tickers = tickers_df.iloc[:, 0].tolist()
        
        print(f"Found {len(tickers)} tickers")
        
        if len(tickers) == 0:
            print("ERROR: No tickers found. Please check your table names and query.")
            retriever.disconnect()
            return
        
        # Get stock data
        print(f"\n[Step 1.2] Retrieving stock data for {len(tickers)} tickers...")
        print("This may take a few minutes...")
        data = retriever.get_stock_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            price_table=price_table,
            fundamental_table=fundamental_table
        )
        prices_df = data['prices']
        fundamentals_df = data['fundamentals']
        
        print(f"Retrieved {len(prices_df)} price observations")
        print(f"Retrieved {len(fundamentals_df)} fundamental observations")
        
        # Get S&P500 returns
        print("\n[Step 1.3] Retrieving S&P500 returns...")
        sp500_returns = retriever.get_sp500_returns(start_date, end_date)
        print(f"Retrieved {len(sp500_returns)} S&P500 return observations")
        
        retriever.disconnect()
        
    except Exception as e:
        print(f"\nERROR: Failed to retrieve data: {str(e)}")
        print("\nPlease ensure:")
        print("1. Snowflake credentials are set in .env file or environment variables")
        print("2. Table names match your Snowflake schema")
        print("3. You have appropriate permissions")
        print(f"4. Tables '{price_table}' and '{fundamental_table}' exist")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Construct factors
    print("\n[Step 2] Constructing factors...")
    try:
        factor_constructor = FactorConstructor(prices_df, fundamentals_df)
        factors_df = factor_constructor.construct_all_factors()
        print(f"✓ Constructed {len([c for c in factors_df.columns if c != 'returns'])} factors")
        print(f"✓ Total observations: {len(factors_df)}")
        print(f"  Factors: {', '.join([c for c in factors_df.columns if c != 'returns'])}")
    except Exception as e:
        print(f"ERROR: Failed to construct factors: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 3: Test factors
    print("\n[Step 3] Testing factors...")
    try:
        factor_tester = FactorTester(factors_df, sp500_returns)
        test_results = factor_tester.test_all_factors()
        print("\nFactor Test Results (Information Coefficient):")
        print(test_results.to_string(index=False))
        
        # Show best factors
        if len(test_results) > 0:
            best_factor = test_results.loc[test_results['mean_ic'].abs().idxmax()]
            print(f"\nBest Factor: {best_factor['factor']} (IC: {best_factor['mean_ic']:.4f})")
    except Exception as e:
        print(f"WARNING: Factor testing failed: {str(e)}")
        print("Continuing with model building...")
    
    # Step 4: Build factor model
    print("\n[Step 4] Building factor model...")
    print("Target correlation: {:.1%}".format(config.TARGET_CORRELATION))
    try:
        model_builder = FactorModelBuilder(factors_df, sp500_returns)
        model_results = model_builder.build_optimized_model()
        
        if 'error' in model_results:
            print(f"ERROR: {model_results['error']}")
            return
        
        correlation = model_results.get('correlation', 0)
        target_achieved = correlation >= config.TARGET_CORRELATION
        
        print(f"\n{'='*60}")
        print("MODEL RESULTS")
        print(f"{'='*60}")
        print(f"Model Correlation: {correlation:.4f} ({correlation:.1%})")
        print(f"Target Correlation: {config.TARGET_CORRELATION:.4f} ({config.TARGET_CORRELATION:.1%})")
        print(f"Target Achieved: {'✓ YES' if target_achieved else '✗ NO'}")
        
        if 'r2_score' in model_results:
            print(f"R² Score: {model_results['r2_score']:.4f}")
        
        print(f"\nFactor Weights:")
        for factor, weight in model_results.get('factor_weights', {}).items():
            print(f"  {factor:15s}: {weight:8.4f}")
        
        if 'intercept' in model_results:
            print(f"  {'intercept':15s}: {model_results['intercept']:8.4f}")
        
        # Compare with S&P500
        print(f"\n{'='*60}")
        print("S&P500 COMPARISON")
        print(f"{'='*60}")
        comparison = model_builder.compare_with_sp500()
        if 'error' not in comparison:
            print(f"Correlation with S&P500: {comparison['correlation']:.4f} ({comparison['correlation']:.1%})")
            print(f"R² Score: {comparison['r2_score']:.4f}")
            print(f"Mean Squared Error: {comparison['mse']:.6f}")
            print(f"Mean Absolute Error: {comparison['mae']:.6f}")
            print(f"Target Achieved: {'✓ YES' if comparison['target_achieved'] else '✗ NO'}")
        else:
            print(f"WARNING: {comparison.get('error', 'Could not compare with S&P500')}")
        
        print(f"\n{'='*60}")
        if target_achieved:
            print("SUCCESS: Model achieved target correlation!")
        else:
            print("NOTE: Model did not achieve target correlation.")
            print("Consider:")
            print("  - Increasing lookback period")
            print("  - Adding more factors")
            print("  - Adjusting regularization parameters")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"ERROR: Failed to build model: {str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

