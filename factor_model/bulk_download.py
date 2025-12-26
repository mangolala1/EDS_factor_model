"""Bulk download from Snowflake into local Parquet files."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from .data_retrieval import SnowflakeDataRetriever

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _as_dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def download_all_data(
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    data_dir: str | Path = "data",
    chunk_days: int = 90,
    lookback_days: int = 400,
) -> None:
    """Download prices/fundamentals/universe and save to ./data/*.parquet."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    end_dt = _as_dt(end_date) if end_date else datetime.today()
    start_dt = _as_dt(start_date)

    query_start_dt = start_dt - timedelta(days=lookback_days)

    retriever = SnowflakeDataRetriever()

    def _pbar(total, desc):
        if tqdm is None:
            return None
        return tqdm(total=total, desc=desc, bar_format="{desc}: {elapsed}")

    print("=" * 80)
    print("Bulk download (Snowflake → local Parquet)")
    print(f"Target range: {start_dt.date()} → {end_dt.date()}")
    print(f"Query start (with lookback): {query_start_dt.date()} (lookback_days={lookback_days})")
    print(f"Output dir: {data_path.resolve()}")
    print("=" * 80)

    print("\n[1] Downloading prices (chunked)...")
    prices_chunks = []
    current = query_start_dt
    n_chunks = ((end_dt - current).days // chunk_days) + 1
    pbar = _pbar(n_chunks, "Querying prices")

    while current <= end_dt:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_dt)
        df = retriever.get_prices(
            start_date=current.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
        )
        if not df.empty:
            prices_chunks.append(df)
        if pbar:
            pbar.update(1)
        current = chunk_end + timedelta(days=1)

    if pbar:
        pbar.close()

    prices_df = pd.concat(prices_chunks, ignore_index=True) if prices_chunks else pd.DataFrame(
        columns=["DATE","FACTSET_ID","ADJUSTED_PRICE","ADJUSTED_VOLUME"]
    )
    prices_df = prices_df.sort_values(["FACTSET_ID","DATE"]).reset_index(drop=True)

    prices_path = data_path / "prices.parquet"
    prices_df.to_parquet(prices_path, index=False, compression="snappy")
    print(f"   ✓ Saved prices: {len(prices_df):,} rows -> {prices_path}")

    print("\n[2] Calculating returns from prices...")
    returns_df = retriever.calculate_returns_from_prices(prices_df)
    returns_path = data_path / "returns.parquet"
    returns_df.to_parquet(returns_path, index=False, compression="snappy")
    print(f"   ✓ Saved returns: {len(returns_df):,} rows -> {returns_path}")

    print("\n[3] Downloading fundamentals (chunked)...")
    fundamentals_chunks = []
    current = query_start_dt
    n_chunks = ((end_dt - current).days // chunk_days) + 1
    pbar = _pbar(n_chunks, "Querying fundamentals")

    while current <= end_dt:
        chunk_end = min(current + timedelta(days=chunk_days - 1), end_dt)
        df = retriever.get_fundamentals(
            start_date=current.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
        )
        if not df.empty:
            fundamentals_chunks.append(df)
        if pbar:
            pbar.update(1)
        current = chunk_end + timedelta(days=1)

    if pbar:
        pbar.close()

    fundamentals_df = pd.concat(fundamentals_chunks, ignore_index=True) if fundamentals_chunks else pd.DataFrame()
    if not fundamentals_df.empty:
        fundamentals_df = fundamentals_df.sort_values(["FACTSET_ID","DATE"]).reset_index(drop=True)
    fundamentals_path = data_path / "fundamentals.parquet"
    fundamentals_df.to_parquet(fundamentals_path, index=False, compression="snappy")
    print(f"   ✓ Saved fundamentals: {len(fundamentals_df):,} rows -> {fundamentals_path}")

    print("\n[4] Downloading universe...")
    universe_df = retriever.get_universe()
    universe_path = data_path / "universe.parquet"
    universe_df.to_parquet(universe_path, index=False, compression="snappy")
    universe_csv = data_path / "universe.csv"
    universe_df.to_csv(universe_csv, index=False)
    print(f"   ✓ Saved universe: {len(universe_df):,} rows -> {universe_path} (+ CSV)")

    print("\nDone.")
    print("=" * 80)


if __name__ == "__main__":  # pragma: no cover
    download_all_data(start_date="2020-01-01", end_date=None, data_dir="data")
