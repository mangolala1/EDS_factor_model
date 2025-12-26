from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig
from .features_raw import fillna_unknown_cat


def build_returns_table(factor_raw_base: pd.DataFrame) -> pd.DataFrame:
    model_id = ModelConfig().model_id
    out = factor_raw_base[["DATE","FACTSET_ID","RETURN"]].copy()
    out.rename(columns={"FACTSET_ID":"SECURITY_ID"}, inplace=True)
    out["MODEL"] = model_id
    return out[["MODEL","DATE","SECURITY_ID","RETURN"]]


def build_beta_wide(factor_exposures_base: pd.DataFrame) -> pd.DataFrame:
    model_id = ModelConfig().model_id
    df = factor_exposures_base.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.rename(columns={"FACTSET_ID":"SECURITY_ID"}, inplace=True)

    style_cols = [c for c in ["VALUE","PROFITABILITY","GROWTH","MOMENTUM","VOLATILITY","LIQUIDITY"] if c in df.columns]
    dummy_cols = [c for c in df.columns if c.startswith("SECTOR_") or c.startswith("CONTINENT_")]
    cols = ["DATE","SECURITY_ID"] + style_cols + dummy_cols

    out = df[cols].copy()
    out["INTERCEPT"] = 1.0
    out["MODEL"] = model_id
    return out[["MODEL","DATE","SECURITY_ID","INTERCEPT"] + style_cols + dummy_cols]


def compute_periodic_factor_returns(fr_long: pd.DataFrame, freqs: tuple[str, ...] = ("M","Q")) -> pd.DataFrame:
    if fr_long.empty:
        return pd.DataFrame(columns=["MODEL","PERIOD_END_DATE","FREQUENCY","FACTOR_NAME","PERIODIC_RETURN"])

    df = fr_long.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    out_parts = []
    for freq in freqs:
        period = df["DATE"].dt.to_period(freq)
        period_end = period.dt.to_timestamp(how="end").dt.normalize()
        tmp = df.assign(PERIOD_END_DATE=period_end, FREQUENCY=freq)

        grp = tmp.groupby(["MODEL","FREQUENCY","PERIOD_END_DATE","FACTOR_NAME"], sort=True)["FACTOR_RETURN"]
        periodic = grp.apply(lambda x: float(np.prod(1.0 + x.dropna().to_numpy(dtype="float64")) - 1.0)).reset_index()
        periodic.rename(columns={"FACTOR_RETURN":"PERIODIC_RETURN"}, inplace=True)
        out_parts.append(periodic)

    return pd.concat(out_parts, ignore_index=True)


def build_peer_group_data(factor_exposures_base: pd.DataFrame) -> pd.DataFrame:
    model_id = ModelConfig().model_id
    df = factor_exposures_base[["DATE","FACTSET_ID","RETURN","SECTOR","CONTINENT","COUNTRY"]].copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df.rename(columns={"FACTSET_ID":"SECURITY_ID"}, inplace=True)

    df["SECTOR"] = fillna_unknown_cat(df["SECTOR"])
    df["CONTINENT"] = fillna_unknown_cat(df["CONTINENT"])
    df["COUNTRY"] = fillna_unknown_cat(df["COUNTRY"])

    df["PEER_GROUP_SECTOR"] = df["SECTOR"].astype(str)
    df["PEER_GROUP_REGION_SECTOR"] = df["CONTINENT"].astype(str) + " | " + df["SECTOR"].astype(str)

    sec_stats = (
        df.groupby(["DATE","PEER_GROUP_SECTOR"], sort=False)["RETURN"]
          .agg(PEER_N="count", PEER_MEAN_RETURN="mean", PEER_STD_RETURN="std")
          .reset_index()
    )
    rs_stats = (
        df.groupby(["DATE","PEER_GROUP_REGION_SECTOR"], sort=False)["RETURN"]
          .agg(PEER_RS_N="count", PEER_RS_MEAN_RETURN="mean", PEER_RS_STD_RETURN="std")
          .reset_index()
    )

    out = df.merge(sec_stats, on=["DATE","PEER_GROUP_SECTOR"], how="left").merge(rs_stats, on=["DATE","PEER_GROUP_REGION_SECTOR"], how="left")
    out["MODEL"] = model_id
    return out[[
        "MODEL","DATE","SECURITY_ID","COUNTRY","CONTINENT","SECTOR",
        "PEER_GROUP_SECTOR","PEER_N","PEER_MEAN_RETURN","PEER_STD_RETURN",
        "PEER_GROUP_REGION_SECTOR","PEER_RS_N","PEER_RS_MEAN_RETURN","PEER_RS_STD_RETURN",
    ]]


def build_factor_metadata(factor_exposures_base: pd.DataFrame) -> pd.DataFrame:
    model_id = ModelConfig().model_id
    df = factor_exposures_base
    style_cols = [c for c in ["VALUE","PROFITABILITY","GROWTH","MOMENTUM","VOLATILITY","LIQUIDITY"] if c in df.columns]
    sector_cols = [c for c in df.columns if c.startswith("SECTOR_")]
    geo_cols    = [c for c in df.columns if c.startswith("CONTINENT_")]

    rows = []
    style_desc = {
        "VALUE": "Composite value signal (standardized cross-sectionally each day).",
        "PROFITABILITY": "Composite profitability signal (standardized cross-sectionally each day).",
        "GROWTH": "Composite growth signal (standardized cross-sectionally each day).",
        "MOMENTUM": "12-1 momentum (252d cumret minus 21d cumret), standardized cross-sectionally each day.",
        "VOLATILITY": "Rolling 60d return volatility, standardized cross-sectionally each day.",
        "LIQUIDITY": "Log 20d average dollar volume, standardized cross-sectionally each day.",
    }
    for c in style_cols:
        rows.append({
            "MODEL": model_id,
            "FACTOR_NAME": c,
            "FACTOR_DISPLAY_NAME": c,
            "FACTOR_GROUP": "Style Factors",
            "FACTOR_TYPE": "STYLE",
            "IS_STYLE": True,
            "IS_DUMMY": False,
            "DESCRIPTION": style_desc.get(c, ""),
        })

    for c in sector_cols:
        rows.append({
            "MODEL": model_id,
            "FACTOR_NAME": c,
            "FACTOR_DISPLAY_NAME": c.replace("SECTOR_", ""),
            "FACTOR_GROUP": "Sector Factors",
            "FACTOR_TYPE": "DUMMY",
            "IS_STYLE": False,
            "IS_DUMMY": True,
            "DESCRIPTION": "Sector dummy (mean-centered per day). One dummy dropped for identifiability.",
        })

    for c in geo_cols:
        rows.append({
            "MODEL": model_id,
            "FACTOR_NAME": c,
            "FACTOR_DISPLAY_NAME": c.replace("CONTINENT_", ""),
            "FACTOR_GROUP": "Geographic Factors",
            "FACTOR_TYPE": "DUMMY",
            "IS_STYLE": False,
            "IS_DUMMY": True,
            "DESCRIPTION": "Continent/region dummy (mean-centered per day). One dummy dropped for identifiability.",
        })

    rows.append({
        "MODEL": model_id,
        "FACTOR_NAME": "INTERCEPT",
        "FACTOR_DISPLAY_NAME": "INTERCEPT",
        "FACTOR_GROUP": "Market Factor",
        "FACTOR_TYPE": "MARKET",
        "IS_STYLE": False,
        "IS_DUMMY": False,
        "DESCRIPTION": "Global market intercept (cross-sectional mean return each day under centered exposures).",
    })

    meta = pd.DataFrame(rows)
    return meta[["MODEL","FACTOR_NAME","FACTOR_DISPLAY_NAME","FACTOR_GROUP","FACTOR_TYPE","IS_STYLE","IS_DUMMY","DESCRIPTION"]].drop_duplicates()
