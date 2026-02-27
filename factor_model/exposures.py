from __future__ import annotations

import numpy as np
import pandas as pd

from .features_raw import fillna_unknown_cat


def build_factor_exposures_base(
    factor_raw_base: pd.DataFrame,
    winsor_low: float = 0.01,
    winsor_high: float = 0.99,
    drop_one_dummy_per_group: bool = True,
    fill_missing_cats: bool = True,
) -> pd.DataFrame:
    df = factor_raw_base.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    value_chars = ["EY_NTM", "SY_NTM", "EBITDA_Y_NTM", "EY_LTM", "SY_LTM"]
    profitability_chars = ["EBITDA_MARGIN", "GROSS_MARGIN"]
    growth_chars = ["EPS_GROWTH", "SALES_GROWTH"]
    extra_chars = [c for c in ["MOMENTUM","VOLATILITY","LIQUIDITY"] if c in df.columns]
    all_chars = value_chars + profitability_chars + growth_chars + extra_chars
    all_chars = [c for c in all_chars if c in df.columns]

    if fill_missing_cats:
        for col in ["SECTOR", "COUNTRY", "CONTINENT"]:
            if col in df.columns:
                df[col] = fillna_unknown_cat(df[col].astype("category"))

    if all_chars:
        qs = df.groupby("DATE")[all_chars].quantile([winsor_low, winsor_high])
        lo = qs.xs(winsor_low, level=1).reset_index().rename(columns={c: f"{c}_LO" for c in all_chars})
        hi = qs.xs(winsor_high, level=1).reset_index().rename(columns={c: f"{c}_HI" for c in all_chars})
        df = df.merge(lo, on="DATE", how="left").merge(hi, on="DATE", how="left")

        for c in all_chars:
            df[c] = df[c].clip(lower=df[f"{c}_LO"], upper=df[f"{c}_HI"])
            df.drop(columns=[f"{c}_LO", f"{c}_HI"], inplace=True)

        means = df.groupby("DATE")[all_chars].transform("mean")
        stds  = df.groupby("DATE")[all_chars].transform("std").replace(0.0, np.nan)
        df[all_chars] = (df[all_chars] - means) / stds

    # Composite style factors
    vc = [c for c in value_chars if c in df.columns]
    df["VALUE"] = df[vc].mean(axis=1, skipna=True) if vc else np.nan

    pc = [c for c in profitability_chars if c in df.columns]
    df["PROFITABILITY"] = df[pc].mean(axis=1, skipna=True) if pc else np.nan

    gc = [c for c in growth_chars if c in df.columns]
    df["GROWTH"] = df[gc].mean(axis=1, skipna=True) if gc else np.nan

    comp_cols = [c for c in ["VALUE","PROFITABILITY","GROWTH"] + extra_chars if c in df.columns]
    if comp_cols:
        comp_means = df.groupby("DATE")[comp_cols].transform("mean")
        comp_stds  = df.groupby("DATE")[comp_cols].transform("std").replace(0.0, np.nan)
        df[comp_cols] = (df[comp_cols] - comp_means) / comp_stds

    # Dummies (mean-centered per day)
    sector_d = pd.get_dummies(df["SECTOR"].astype("category"), prefix="SECTOR", dtype="float32") if "SECTOR" in df.columns else pd.DataFrame(index=df.index)
    cont_d = pd.get_dummies(df["CONTINENT"].astype("category"), prefix="CONTINENT", dtype="float32") if "CONTINENT" in df.columns else pd.DataFrame(index=df.index)

    if not sector_d.empty:
        sector_d = sector_d - sector_d.groupby(df["DATE"]).transform("mean")
    if not cont_d.empty:
        cont_d = cont_d - cont_d.groupby(df["DATE"]).transform("mean")

    if drop_one_dummy_per_group:
        if sector_d.shape[1] > 0:
            sector_d = sector_d.iloc[:, :-1]
        if cont_d.shape[1] > 0:
            cont_d = cont_d.iloc[:, :-1]

    df = pd.concat([df, sector_d, cont_d], axis=1)

    style_cols = [c for c in ["VALUE","PROFITABILITY","GROWTH","MOMENTUM","VOLATILITY","LIQUIDITY"] if c in df.columns]
    dummy_cols = list(sector_d.columns) + list(cont_d.columns)
    keep = ["DATE","FACTSET_ID","RETURN","SECTOR","COUNTRY","CONTINENT"] + style_cols + dummy_cols
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()
