# Workflow Explanation - EDS Factor Model

## Overview: 3-Level Chunking Strategy

The workflow uses **3 levels of chunking** to handle large datasets efficiently:

### Level 1: **Chunk by QUARTERS** (Parallel Processing Unit)
- **Why**: Split 2020-2025 into 24 quarters
- **Parallelization**: Process 4 quarters simultaneously (4 workers)
- **Benefit**: Natural time boundaries, manageable memory per worker

```
2020Q1, 2020Q2, 2020Q3, 2020Q4, 2021Q1, ... → 24 quarters total
    ↓        ↓        ↓        ↓
Worker1  Worker2  Worker3  Worker4  (4 parallel workers)
```

### Level 2: **Chunk by ROWS** (Parquet File Reading)
- **Why**: Parquet files are HUGE (50M+ rows). Can't load all at once!
- **How**: Read Parquet files in **500,000 row batches**
- **Benefit**: Memory-efficient, processes files incrementally

```
prices.parquet (50M rows)
    ↓
Read 500k rows → Filter by date → Keep relevant
Read 500k rows → Filter by date → Keep relevant
...
Repeat until file done → Concatenate filtered chunks
```

### Level 3: **Chunk by DATES** (Date-by-Date Processing)
- **Why**: Process each trading day sequentially within a quarter
- **How**: Loop through ~65 trading dates per quarter
- **Benefit**: Ring buffers build up naturally, manageable per-date operations

```
2020Q1: [Date1, Date2, Date3, ..., Date65]
         ↓      ↓      ↓
      Process Process Process
```

---

## Detailed Workflow

### STEP 1: Identify Quarters to Process
```
start_date = '2020-01-01'
end_date = '2025-12-31'

→ Split into 24 quarters:
   [(2020, 1), (2020, 2), (2020, 3), (2020, 4),
    (2021, 1), (2021, 2), ..., (2025, 4)]
```

### STEP 2: For Each Quarter (4 workers in parallel)

#### 2A. Load Prices (Chunked from Parquet)
```
For 2020Q1:
- Quarter range: 2020-01-01 to 2020-03-31
- Need lookback: 2018-11-26 to 2020-03-31 (400 days before)
- Read prices.parquet in 500k batches
- Filter each batch to date range
- Combine filtered chunks → ~13M rows
```

#### 2B. Load Returns (Chunked from Parquet)
```
- Same process as prices
- Read returns.parquet in 500k batches
- Filter to same date range
- Combine → ~13M rows
```

#### 2C. Load Fundamentals (Chunked from Parquet)
```
- Read fundamentals.parquet in 500k batches
- Filter to quarter only (2020-01-01 to 2020-03-31)
- No lookback needed for fundamentals
- Combine → ~5M rows
```

#### 2D. Process Each Trading Date in Quarter
```
For each date in 2020Q1 (65 dates):
  1. Get data for that date:
     - fundamentals_T (all stocks on that date)
     - prices_T (all stocks on that date)
     - returns_T (all stocks on that date)
  
  2. Update ring buffers:
     - Add today's return/price to each stock's buffer
  
  3. Calculate characteristics:
     - Value factors (EY_NTM, SY_NTM, etc.)
     - Profitability (EBITDA_MARGIN, GROSS_MARGIN)
     - Growth (EPS_GROWTH, SALES_GROWTH)
     - Momentum (from ring buffer - last 252 days)
     - Volatility (from ring buffer - last 60 days)
     - Liquidity (from ring buffer - last 60 days)
  
  4. Winsorize and standardize
  
  5. Create dummies (sector, continent)
  
  6. Save exposures for this date
```

#### 2E. Write Quarter Results to CSV
```
- Combine all 65 dates → exposures_2020Q1.csv
- One CSV file per quarter
```

### STEP 3: Combine All Quarter CSVs
```
exposures_2020Q1.csv
exposures_2020Q2.csv
...
exposures_2025Q4.csv
    ↓
Combine into: exposures.csv (all dates, sorted)
```

---

## Why This Chunking Strategy?

### Quarter-Level Chunking:
- ✅ **Natural boundaries**: Quarters align with business cycles
- ✅ **Parallelizable**: Can run 4 quarters simultaneously
- ✅ **Memory manageable**: Each worker holds ~13M rows (not 300M)
- ✅ **Fault tolerant**: If one quarter fails, others continue

### Row-Level Chunking (Parquet):
- ✅ **Memory efficient**: Only loads 500k rows at a time
- ✅ **Handles huge files**: Works with 50M+ row files
- ✅ **Fast filtering**: Filter each chunk before concatenating

### Date-Level Chunking:
- ✅ **Incremental buffers**: Ring buffers build up naturally
- ✅ **Manageable operations**: Process ~20k stocks per date
- ✅ **Batch writes**: Collect all dates, write once per quarter

---

## Memory Usage Per Worker

For 2020Q1 (one worker):
- **Prices**: ~13M rows × ~200 bytes = ~2.6 GB
- **Returns**: ~13M rows × ~100 bytes = ~1.3 GB
- **Fundamentals**: ~5M rows × ~150 bytes = ~0.75 GB
- **Universe**: ~75k rows × ~500 bytes = ~37 MB
- **Total per worker**: ~5 GB

With 4 workers: 4 × 5 GB = **~20 GB total** (but staggered, not all at once)

---

## Performance Timeline

**Per Quarter** (2020Q1 example):
1. Load prices: ~50 seconds (130 batches × 0.4s)
2. Load returns: ~43 seconds (130 batches × 0.33s)
3. Load fundamentals: ~73 seconds (230 batches × 0.32s)
4. Process 65 dates: ~2-3 minutes (each date ~2-3 seconds)
5. Write CSV: ~10 seconds
**Total per quarter**: ~4-5 minutes

**All 24 quarters** (4 workers in parallel):
- Sequential would be: 24 × 5 min = 120 minutes
- With 4 workers: ~30 minutes (24 ÷ 4 × 5 min)

---

## Current Status from Your Terminal

You're seeing:
- ✅ **2020Q1-Q4 starting in parallel** - Good!
- ✅ **Loading data in batches** - Working as designed
- ✅ **Processing dates** - Moving forward
- ⏱️ **~20 seconds per 10 dates** - Normal speed

Each date takes ~2 seconds because:
- Merging fundamentals, prices, returns
- Calculating 9 characteristics
- Winsorizing and standardizing
- Creating dummies (sector/continent)
- Updating ring buffers

This is expected! The process is working correctly.

