from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from .config import PathsConfig, default_paths
from .continent_mapping import map_countries_to_continent_developed

log = logging.getLogger(__name__)

_PRICES_ALL: pd.DataFrame | None = None
_FUND_ALL: pd.DataFrame | None = None
_UNIVERSE_ALL: pd.DataFrame | None = None
_CACHE_MIN_DATE: pd.Timestamp | None = None


def safe_div(a, b):
    a = a.astype("float64")
    b = b.astype("float64")
    out = np.full_like(a, np.nan, dtype="float64")
    mask = np.isfinite(a) & np.isfinite(b) & (b != 0)
    out[mask] = (a[mask] / b[mask]).astype("float64")
    return out


def fillna_unknown_cat(s: pd.Series) -> pd.Series:
    if pd.api.types.is_categorical_dtype(s):
        if "Unknown" not in s.cat.categories:
            s = s.cat.add_categories(["Unknown"])
        return s.fillna("Unknown")
    return s.fillna("Unknown")


def _ensure_inputs_loaded(min_date: pd.Timestamp, paths: PathsConfig) -> None:
    global _PRICES_ALL, _FUND_ALL, _UNIVERSE_ALL, _CACHE_MIN_DATE

    if _CACHE_MIN_DATE is None or min_date < _CACHE_MIN_DATE or _PRICES_ALL is None:
        price_cols = ["DATE", "FACTSET_ID", "ADJUSTED_PRICE", "ADJUSTED_VOLUME"]
        fundamentals_cols = [
            "DATE", "FACTSET_ID",
            "SALES_LTM", "SALES_NTM",
            "EBITDA_LTM", "EBITDA_NTM",
            "EPS_LTM", "EPS_NTM",
            "COGS_LTM", "COGS_NTM",
        ]
        universe_cols = ["FACTSET_ID", "SECTOR", "COUNTRY"]

        _CACHE_MIN_DATE = min_date

        _PRICES_ALL = pd.read_parquet(paths.prices_file, columns=price_cols, filters=[("DATE", ">=", _CACHE_MIN_DATE)])
        _FUND_ALL = pd.read_parquet(paths.fundamentals_file, columns=fundamentals_cols, filters=[("DATE", ">=", _CACHE_MIN_DATE)])
        _UNIVERSE_ALL = pd.read_parquet(paths.universe_file, columns=universe_cols)

        _PRICES_ALL["DATE"] = pd.to_datetime(_PRICES_ALL["DATE"])
        _FUND_ALL["DATE"] = pd.to_datetime(_FUND_ALL["DATE"])
        _PRICES_ALL = _PRICES_ALL.sort_values(["FACTSET_ID", "DATE"]).reset_index(drop=True)

        _UNIVERSE_ALL["SECTOR"] = _UNIVERSE_ALL["SECTOR"].astype("category")
        _UNIVERSE_ALL["COUNTRY"] = _UNIVERSE_ALL["COUNTRY"].astype("category")


def build_factor_raw_base(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    paths: PathsConfig | None = None,
) -> pd.DataFrame:
    paths = paths or default_paths()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) if end_date is not None else None

    _ensure_inputs_loaded(start_ts, paths)

    assert _PRICES_ALL is not None and _FUND_ALL is not None and _UNIVERSE_ALL is not None
    prices = _PRICES_ALL
    fundamentals = _FUND_ALL
    universe = _UNIVERSE_ALL

    if end_ts is not None:
        prices = prices[(prices["DATE"] >= start_ts) & (prices["DATE"] <= end_ts)].copy()
        fundamentals = fundamentals[(fundamentals["DATE"] >= start_ts) & (fundamentals["DATE"] <= end_ts)].copy()
    else:
        prices = prices[prices["DATE"] >= start_ts].copy()
        fundamentals = fundamentals[fundamentals["DATE"] >= start_ts].copy()

    prices = prices.sort_values(["FACTSET_ID", "DATE"]).reset_index(drop=True)
    g_prices = prices.groupby("FACTSET_ID", sort=False)
    prices["RETURN"] = g_prices["ADJUSTED_PRICE"].pct_change()
    prices["DOLLAR_VOL"] = prices["ADJUSTED_PRICE"].astype("float64") * prices["ADJUSTED_VOLUME"].astype("float64")

    joined = prices.merge(universe, on="FACTSET_ID", how="inner")
    joined = joined.merge(fundamentals, on=["DATE", "FACTSET_ID"], how="left")

    joined["SECTOR"] = fillna_unknown_cat(joined["SECTOR"].astype("category"))
    joined["COUNTRY"] = fillna_unknown_cat(joined["COUNTRY"].astype("category"))

    joined["CONTINENT"] = map_countries_to_continent_developed(joined["COUNTRY"])
    joined["CONTINENT"] = fillna_unknown_cat(joined["CONTINENT"].astype("category"))

    joined = joined.sort_values(["FACTSET_ID", "DATE"]).reset_index(drop=True)
    joined["PRICE"] = joined["ADJUSTED_PRICE"].astype("float64")

    # Value yields
    joined["EY_NTM"]       = safe_div(joined["EPS_NTM"],    joined["PRICE"])
    joined["SY_NTM"]       = safe_div(joined["SALES_NTM"],  joined["PRICE"])
    joined["EBITDA_Y_NTM"] = safe_div(joined["EBITDA_NTM"], joined["PRICE"])
    joined["EY_LTM"]       = safe_div(joined["EPS_LTM"],    joined["PRICE"])
    joined["SY_LTM"]       = safe_div(joined["SALES_LTM"],  joined["PRICE"])

    joined.loc[joined["EPS_NTM"] <= 0, "EY_NTM"] = np.nan
    joined.loc[joined["EBITDA_NTM"] <= 0, "EBITDA_Y_NTM"] = np.nan
    joined.loc[joined["EPS_LTM"] <= 0, "EY_LTM"] = np.nan

    # Profitability
    joined["EBITDA_MARGIN"] = safe_div(joined["EBITDA_LTM"], joined["SALES_LTM"])
    joined["GROSS_MARGIN"]  = 1.0 - safe_div(joined["COGS_LTM"], joined["SALES_LTM"])
    joined.loc[joined["SALES_LTM"] <= 0, ["EBITDA_MARGIN","GROSS_MARGIN"]] = np.nan
    joined.loc[joined["COGS_LTM"] <= 0, "GROSS_MARGIN"] = np.nan

    # Growth
    joined["EPS_GROWTH"]   = safe_div(joined["EPS_NTM"], joined["EPS_LTM"]) - 1.0
    joined["SALES_GROWTH"] = safe_div(joined["SALES_NTM"], joined["SALES_LTM"]) - 1.0
    joined.loc[joined["EPS_LTM"] <= 0, "EPS_GROWTH"] = np.nan
    joined.loc[joined["SALES_LTM"] <= 0, "SALES_GROWTH"] = np.nan

    # Momentum / Vol / Liquidity
    r = joined["RETURN"].astype("float64")
    joined["LOG1P_RET"] = np.log1p(r.where(np.isfinite(r), np.nan))
    g = joined.groupby("FACTSET_ID", sort=False)

    joined["SUM_LOG_252"] = g["LOG1P_RET"].rolling(252, min_periods=252).sum().reset_index(level=0, drop=True)
    joined["SUM_LOG_21"]  = g["LOG1P_RET"].rolling(21,  min_periods=21 ).sum().reset_index(level=0, drop=True)
    joined["CUMRET_252"] = np.expm1(joined["SUM_LOG_252"])
    joined["CUMRET_21"]  = np.expm1(joined["SUM_LOG_21"])
    joined["MOMENTUM"]   = joined["CUMRET_252"] - joined["CUMRET_21"]

    joined["VOLATILITY"] = g["RETURN"].rolling(60, min_periods=60).std(ddof=1).reset_index(level=0, drop=True)

    joined["AVG_DVOL_20"] = g["DOLLAR_VOL"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    joined["LIQUIDITY"] = np.where(joined["AVG_DVOL_20"] > 0, np.log(joined["AVG_DVOL_20"]), np.nan)

    joined.drop(columns=["LOG1P_RET","SUM_LOG_252","SUM_LOG_21","CUMRET_252","CUMRET_21","AVG_DVOL_20"], inplace=True)

    return joined
