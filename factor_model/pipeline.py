from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from dateutil.relativedelta import relativedelta

from .config import ModelConfig, PathsConfig, default_paths
from .features_raw import build_factor_raw_base
from .exposures import build_factor_exposures_base
from .regression import run_factor_model
from .tables import build_returns_table, build_beta_wide, compute_periodic_factor_returns, build_peer_group_data, build_factor_metadata
from .portfolios import write_style_factor_mimicking_portfolios_parquet
from .combine import combine_quarterly_parquets_fast

log = logging.getLogger(__name__)


def _infer_overall_end_from_prices(paths: PathsConfig) -> pd.Timestamp:
    prices = pd.read_parquet(paths.prices_file, columns=["DATE"])
    return pd.to_datetime(prices["DATE"]).max()


def process_chunk(chunk_start_str: str, chunk_end_str: str, out_dir: Path, paths: PathsConfig | None = None, model_cfg: ModelConfig | None = None, specific_var_method: str = "total_minus_explained"):
    paths = paths or default_paths()
    model_cfg = model_cfg or ModelConfig()

    chunk_start = pd.Timestamp(chunk_start_str)
    chunk_end = pd.Timestamp(chunk_end_str)
    window_start = chunk_start - pd.Timedelta(days=model_cfg.lookback_days)

    log.info(f"=== Processing chunk {chunk_start.date()} → {chunk_end.date()} (window from {window_start.date()}) ===")

    factor_raw = build_factor_raw_base(window_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), paths=paths)
    factor_exposures = build_factor_exposures_base(factor_raw, winsor_low=model_cfg.winsor_low, winsor_high=model_cfg.winsor_high, drop_one_dummy_per_group=True)

    exposure_long, fr_long, specific_returns, covariance_long, factor_names_df, specific_variance = run_factor_model(
        factor_exposures,
        start_date=window_start.strftime("%Y-%m-%d"),
        end_date=chunk_end.strftime("%Y-%m-%d"),
        cov_window=model_cfg.cov_window,
        var_window=model_cfg.var_window,
        specific_var_method=specific_var_method,
    )

    returns_table = build_returns_table(factor_raw)
    beta_wide = build_beta_wide(factor_exposures)
    periodic_factor_returns = compute_periodic_factor_returns(fr_long, freqs=("M","Q"))
    peer_group_data = build_peer_group_data(factor_exposures)
    factor_metadata = build_factor_metadata(factor_exposures)

    out_dir.mkdir(parents=True, exist_ok=True)

    factor_portfolio_path = out_dir / f"{model_cfg.client}_{model_cfg.model}_FACTOR_PORTFOLIO_{chunk_start.date()}_{chunk_end.date()}.parquet"
    write_style_factor_mimicking_portfolios_parquet(factor_exposures, out_path=factor_portfolio_path, chunk_start=chunk_start)

    # trim overlap
    exposure_long = exposure_long[exposure_long["DATE"] >= chunk_start]
    fr_long = fr_long[fr_long["DATE"] >= chunk_start]
    specific_returns = specific_returns[specific_returns["DATE"] >= chunk_start]
    covariance_long = covariance_long[covariance_long["DATE"] >= chunk_start]
    specific_variance = specific_variance[specific_variance["DATE"] >= chunk_start]

    returns_table = returns_table[returns_table["DATE"] >= chunk_start]
    beta_wide = beta_wide[beta_wide["DATE"] >= chunk_start]
    peer_group_data = peer_group_data[peer_group_data["DATE"] >= chunk_start]
    periodic_factor_returns = periodic_factor_returns[periodic_factor_returns["PERIOD_END_DATE"] >= chunk_start]

    suffix = f"{chunk_start.date()}_{chunk_end.date()}"

    exposure_long.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_EXPOSURE_{suffix}.parquet", index=False)
    fr_long.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_FACTOR_RETURNS_{suffix}.parquet", index=False)
    specific_returns.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_SPECIFIC_RETURNS_{suffix}.parquet", index=False)
    covariance_long.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_COVARIANCE_{suffix}.parquet", index=False)
    specific_variance.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_SPECIFIC_VARIANCE_{suffix}.parquet", index=False)
    factor_names_df.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_FACTOR_NAMES_{suffix}.parquet", index=False)

    returns_table.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_RETURNS_{suffix}.parquet", index=False)
    periodic_factor_returns.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_PERIODIC_FACTOR_RETURNS_{suffix}.parquet", index=False)
    peer_group_data.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_PEER_GROUP_DATA_{suffix}.parquet", index=False)
    factor_metadata.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_FACTOR_METADATA_{suffix}.parquet", index=False)
    beta_wide.to_parquet(out_dir / f"{model_cfg.client}_{model_cfg.model}_BETA_{suffix}.parquet", index=False)

    log.info(f"Chunk {chunk_start.date()} → {chunk_end.date()} done.")


def run_quarterly(out_dir: Path | None = None, overall_start: str = "2023-01-01", overall_end: str | None = None, paths: PathsConfig | None = None, model_cfg: ModelConfig | None = None, specific_var_method: str = "total_minus_explained"):
    paths = paths or default_paths()
    model_cfg = model_cfg or ModelConfig()

    out_dir = Path(out_dir) if out_dir is not None else (paths.outputs_dir / "NEW_FY2023_25")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ts = pd.Timestamp(overall_start)
    end_ts = pd.Timestamp(overall_end) if overall_end else _infer_overall_end_from_prices(paths)

    chunk_start = start_ts
    while chunk_start <= end_ts:
        chunk_end = (chunk_start + relativedelta(months=3) - pd.Timedelta(days=1))
        if chunk_end > end_ts:
            chunk_end = end_ts
        process_chunk(chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"), out_dir, paths=paths, model_cfg=model_cfg, specific_var_method=specific_var_method)
        chunk_start = chunk_start + relativedelta(months=3)

    combine_all_outputs(out_dir, model_cfg=model_cfg)
    return out_dir


def combine_all_outputs(out_dir: Path, model_cfg: ModelConfig | None = None):
    model_cfg = model_cfg or ModelConfig()
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "EXPOSURE", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "FACTOR_RETURNS", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "SPECIFIC_RETURNS", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "COVARIANCE", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "SPECIFIC_VARIANCE", do_postprocess=False)

    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "RETURNS", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "PERIODIC_FACTOR_RETURNS", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "PEER_GROUP_DATA", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "BETA", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "FACTOR_NAMES", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "FACTOR_METADATA", do_postprocess=False)
    combine_quarterly_parquets_fast(out_dir, model_cfg.client, model_cfg.model, "FACTOR_PORTFOLIO", do_postprocess=False)
