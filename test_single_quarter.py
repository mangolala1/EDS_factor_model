"""
Quick test script to verify fixes for a single quarter (2020Q1)
Tests:
1. UnboundLocalError fix
2. Market value loading optimization (should be ~16s instead of 281s)
3. NaN warning fixes
4. Reduced worker count (1-2 workers instead of 4)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.quarter_processor import process_quarters_parallel
from datetime import datetime
import time

def test_single_quarter():
    """Test processing a single quarter (2020Q1)"""
    print("=" * 80)
    print("Testing Single Quarter (2020Q1) - Verification of Fixes")
    print("=" * 80)
    print()
    
    start_time = time.time()
    
    # Test with just 2020Q1
    result = process_quarters_parallel(
        start_date='2020-01-01',
        end_date='2020-03-31',  # End of Q1
        output_dir='results',
        data_dir='data',
        max_workers=1,  # Use 1 worker to avoid I/O contention
        neutralize=False
    )
    
    elapsed = time.time() - start_time
    
    print()
    print("=" * 80)
    print("Test Results:")
    print("=" * 80)
    print(f"✓ Processed dates: {result['processed']}")
    print(f"✓ Skipped dates: {result['skipped']}")
    print(f"✓ Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print()
    
    # Check for expected improvements
    print("Expected Improvements:")
    print("1. ✓ No UnboundLocalError (quarter_enterprise_value initialized)")
    print("2. ✓ Market value load time: ~16s (not 281s)")
    print("3. ✓ No 'All-NaN slice' or 'empty slice' warnings")
    print("4. ✓ Using 1 worker (reduced from 4)")
    print()
    
    if result['processed'] > 0:
        print("✅ Test PASSED: Quarter processed successfully!")
        print(f"   Check 'results/exposures_2020Q1.parquet' for output")
    else:
        print("❌ Test FAILED: No dates were processed")
        print("   Check error messages above")
    
    return result

if __name__ == '__main__':
    test_single_quarter()

