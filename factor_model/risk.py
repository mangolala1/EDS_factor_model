from __future__ import annotations

from typing import Dict, List
import numpy as np
import pandas as pd


def compute_factor_covariance(fr_long: pd.DataFrame, cov_window: int, model_id: str, factor_order: List[str] | None = None):
    if fr_long.empty:
        return pd.DataFrame(columns=["MODEL","DATE","FACTOR1","FACTOR2","COVARIANCE"]), {}

    fr_wide = fr_long.pivot(index="DATE", columns="FACTOR_NAME", values="FACTOR_RETURN").sort_index()
    if factor_order is not None:
        fr_wide = fr_wide.reindex(columns=factor_order)

    sigma_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    cov_dfs = []
    for i, d in enumerate(fr_wide.index):
        if i + 1 < cov_window:
            continue
        window_df = fr_wide.iloc[i + 1 - cov_window : i + 1]
        sigma = window_df.cov(ddof=1)
        sigma_by_date[pd.Timestamp(d)] = sigma

        cov_long = sigma.rename_axis(index="FACTOR1", columns="FACTOR2").stack(dropna=False).rename("COVARIANCE").reset_index()
        cov_long["DATE"] = pd.Timestamp(d)
        cov_long["MODEL"] = model_id
        cov_dfs.append(cov_long[["MODEL","DATE","FACTOR1","FACTOR2","COVARIANCE"]])

    covariance_long = pd.concat(cov_dfs, ignore_index=True) if cov_dfs else pd.DataFrame(
        columns=["MODEL","DATE","FACTOR1","FACTOR2","COVARIANCE"]
    )
    return covariance_long, sigma_by_date


def compute_specific_variance_residual_rolling(specific_returns: pd.DataFrame, var_window: int, spec_floor: float, model_id: str):
    if specific_returns.empty:
        return pd.DataFrame(columns=["MODEL","DATE","SECURITY_ID","SPECIFIC_VAR","SPECIFIC_VOL"])

    sr = specific_returns.copy().sort_values(["SECURITY_ID","DATE"]).reset_index(drop=True)
    g = sr.groupby("SECURITY_ID", sort=False)
    sr["SPECIFIC_VAR"] = g["SPECIFIC_RETURN"].rolling(var_window, min_periods=var_window).var(ddof=1).reset_index(level=0, drop=True)
    sr["SPECIFIC_VAR"] = sr["SPECIFIC_VAR"].clip(lower=spec_floor)
    sr["SPECIFIC_VOL"] = np.sqrt(sr["SPECIFIC_VAR"])
    out = sr.dropna(subset=["SPECIFIC_VAR"])[["DATE","SECURITY_ID","SPECIFIC_VAR","SPECIFIC_VOL"]].copy()
    out["MODEL"] = model_id
    return out[["MODEL","DATE","SECURITY_ID","SPECIFIC_VAR","SPECIFIC_VOL"]]


def compute_specific_variance_total_minus_explained(exposures_df: pd.DataFrame, sigma_by_date: Dict[pd.Timestamp, pd.DataFrame], factor_names: list[str], var_window: int, spec_floor: float, model_id: str):
    if not sigma_by_date:
        return pd.DataFrame(columns=["MODEL","DATE","SECURITY_ID","SPECIFIC_VAR","SPECIFIC_VOL"])

    df = exposures_df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["FACTSET_ID","DATE"]).reset_index(drop=True)

    g = df.groupby("FACTSET_ID", sort=False)
    df["TOTAL_VAR"] = g["RETURN"].rolling(var_window, min_periods=var_window).var(ddof=1).reset_index(level=0, drop=True)

    explained_records = []
    for d, sub in df.groupby("DATE", sort=True):
        sigma = sigma_by_date.get(pd.Timestamp(d))
        if sigma is None:
            continue
        sigma = sigma.reindex(index=factor_names, columns=factor_names)
        B = sub[factor_names].to_numpy(dtype="float64")
        S = sigma.to_numpy(dtype="float64")
        BS = B @ S
        exp_var = np.einsum("ij,ij->i", BS, B)
        explained_records.append(pd.DataFrame({"DATE": pd.Timestamp(d), "FACTSET_ID": sub["FACTSET_ID"].to_numpy(), "EXPLAINED_VAR": exp_var}))

    explained_df = pd.concat(explained_records, ignore_index=True) if explained_records else pd.DataFrame(columns=["DATE","FACTSET_ID","EXPLAINED_VAR"])
    risk_df = df.merge(explained_df, on=["DATE","FACTSET_ID"], how="left")
    risk_df["SPECIFIC_VAR"] = (risk_df["TOTAL_VAR"] - risk_df["EXPLAINED_VAR"]).clip(lower=spec_floor)
    risk_df["SPECIFIC_VOL"] = np.sqrt(risk_df["SPECIFIC_VAR"])

    out = risk_df.dropna(subset=["SPECIFIC_VAR"])[["DATE","FACTSET_ID","SPECIFIC_VAR","SPECIFIC_VOL"]].copy()
    out["MODEL"] = model_id
    out.rename(columns={"FACTSET_ID":"SECURITY_ID"}, inplace=True)
    return out[["MODEL","DATE","SECURITY_ID","SPECIFIC_VAR","SPECIFIC_VOL"]]
