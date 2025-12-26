from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


def _pick_id_col(cols):
    for c in ["SECURITY_ID","FACTSET_ID","ID","FSYM_ID","TICKER","RIC","BBGID"]:
        if c in cols:
            return c
    return None


def combine_quarterly_parquets_fast(
    out_dir: Path,
    client: str,
    model: str,
    stem: str,
    out_name: str | None = None,
    do_postprocess: bool = False,
    batch_size: int = 1_000_000,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    pattern = f"{client}_{model}_{stem}_*.parquet"
    files = sorted(out_dir.glob(pattern))
    if not files:
        log.info(f"[SKIP] No files matched {pattern} in {out_dir}")
        return None

    out_path = out_dir / (out_name or f"{client}_{model}_{stem}.parquet")
    if out_path.exists():
        out_path.unlink()

    writer = None
    rows_written = 0

    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
            writer.write_table(table)
            rows_written += table.num_rows

    if writer is not None:
        writer.close()

    log.info(f"[COMBINE] {stem}: wrote ~{rows_written:,} rows -> {out_path.name}")

    if do_postprocess:
        df = pd.read_parquet(out_path)
        cols = list(df.columns)
        id_col = _pick_id_col(cols)

        if "DATE" in cols and id_col:
            keys = ["DATE", id_col]
            df = df.drop_duplicates(subset=keys, keep="last").sort_values(keys)
            df.to_parquet(out_path, index=False, compression="snappy")
            log.info(f"[POST] sorted/deduped on {keys}")

    return out_path
