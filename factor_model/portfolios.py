from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
import pandas as pd

from .config import ModelConfig

log = logging.getLogger(__name__)


def write_style_factor_mimicking_portfolios_parquet(
    factor_exposures_base: pd.DataFrame,
    out_path: Path,
    chunk_start: pd.Timestamp,
    weight_dtype: str = "float32",
) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    model_id = ModelConfig().model_id

    df = factor_exposures_base.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    style_cols = [c for c in ["VALUE","PROFITABILITY","GROWTH","MOMENTUM","VOLATILITY","LIQUIDITY"] if c in df.columns]
    dummy_cols = [c for c in df.columns if c.startswith("SECTOR_") or c.startswith("CONTINENT_")]
    all_cols_no_intercept = style_cols + dummy_cols

    if not style_cols:
        raise ValueError("No style factor columns found to build factor-mimicking portfolios.")

    X = df[all_cols_no_intercept].astype("float64").copy()
    X["INTERCEPT"] = 1.0
    factor_names = list(X.columns)

    ids = df["FACTSET_ID"].to_numpy()
    dates = df["DATE"].to_numpy()
    X_arr = X.to_numpy(dtype="float64")

    order = np.argsort(dates)
    dates = dates[order]
    ids = ids[order]
    X_arr = X_arr[order, :]

    uniq_dates, start_idx = np.unique(dates, return_index=True)
    end_idx = np.r_[start_idx[1:], len(dates)]

    style_idx = [factor_names.index(c) for c in style_cols]

    writer = None
    rows_written = 0

    for d, a, b in zip(uniq_dates, start_idx, end_idx):
        d_ts = pd.Timestamp(d)
        if d_ts < chunk_start:
            continue

        X_t = X_arr[a:b, :]
        ids_t = ids[a:b]

        valid = np.isfinite(X_t).all(axis=1)
        if valid.sum() < X_t.shape[1]:
            continue

        Xv = X_t[valid]
        idv = ids_t[valid]

        Ginv = np.linalg.pinv(Xv.T @ Xv)
        A = Xv @ Ginv

        n = idv.shape[0]
        k = len(style_idx)

        out_df = pd.DataFrame({
            "MODEL": model_id,
            "DATE": np.repeat(d_ts, n * k),
            "FACTOR_NAME": np.repeat(np.array(style_cols, dtype=object), n),
            "SECURITY_ID": np.tile(idv, k),
            "WEIGHT": np.concatenate([A[:, j] for j in style_idx]).astype(weight_dtype, copy=False),
        })

        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
        writer.write_table(table)
        rows_written += table.num_rows

    if writer is not None:
        writer.close()

    log.info(f"[FACTOR_PORTFOLIO] wrote ~{rows_written:,} rows to {out_path.name}")
    return out_path
