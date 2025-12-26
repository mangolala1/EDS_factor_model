from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ModelConfig
from .risk import compute_factor_covariance, compute_specific_variance_residual_rolling, compute_specific_variance_total_minus_explained


def run_factor_model(
    factor_exposures_base: pd.DataFrame,
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    cov_window: int = 60,
    var_window: int = 60,
    spec_floor: float = 1e-10,
    specific_var_method: str = "total_minus_explained",
):
    model_id = ModelConfig().model_id

    df = factor_exposures_base.copy()
    df["DATE"] = pd.to_datetime(df["DATE"])

    start_ts = pd.Timestamp(start_date)
    mask = df["DATE"] >= start_ts
    if end_date is not None:
        mask &= df["DATE"] <= pd.Timestamp(end_date)
    df = df.loc[mask].copy()

    style_cols = [c for c in ["VALUE","PROFITABILITY","GROWTH","MOMENTUM","VOLATILITY","LIQUIDITY"] if c in df.columns]
    dummy_cols = [c for c in df.columns if c.startswith("SECTOR_") or c.startswith("CONTINENT_")]

    factor_cols_no_intercept = style_cols + dummy_cols
    factor_names = factor_cols_no_intercept + ["INTERCEPT"]

    df = df.sort_values(["DATE","FACTSET_ID"]).reset_index(drop=True)

    df_X = df[factor_cols_no_intercept].copy() if factor_cols_no_intercept else pd.DataFrame(index=df.index)
    for c in factor_cols_no_intercept:
        df_X[c] = df_X[c].astype("float64")
    df_X["INTERCEPT"] = 1.0

    y = df["RETURN"].astype("float64").to_numpy()
    X_arr = df_X[factor_names].to_numpy(dtype="float64")
    dates = df["DATE"].to_numpy()
    ids = df["FACTSET_ID"].to_numpy()

    unique_dates, first_idx = np.unique(dates, return_index=True)
    last_idx = np.append(first_idx[1:], len(dates))

    factor_returns_records = []
    residual_dfs = []

    for d, a, b in zip(unique_dates, first_idx, last_idx):
        Xv = X_arr[a:b]
        yv = y[a:b]

        valid = np.isfinite(yv) & np.isfinite(Xv).all(axis=1)
        if valid.sum() < len(factor_names):
            continue

        Xv = Xv[valid]
        yv = yv[valid]

        XtX = Xv.T @ Xv
        Xty = Xv.T @ yv
        try:
            fr_t = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            fr_t = np.linalg.lstsq(Xv, yv, rcond=None)[0]

        for fname, fval in zip(factor_names, fr_t):
            factor_returns_records.append({"MODEL": model_id, "DATE": pd.Timestamp(d), "FACTOR_NAME": fname, "FACTOR_RETURN": float(fval)})

        resid = yv - (Xv @ fr_t)
        residual_dfs.append(pd.DataFrame({"MODEL": model_id, "DATE": pd.Timestamp(d), "SECURITY_ID": ids[a:b][valid], "SPECIFIC_RETURN": resid}))

    fr_long = pd.DataFrame(factor_returns_records)
    specific_returns = pd.concat(residual_dfs, ignore_index=True) if residual_dfs else pd.DataFrame(columns=["MODEL","DATE","SECURITY_ID","SPECIFIC_RETURN"])

    expo = df[["DATE","FACTSET_ID"]].copy()
    for c in factor_cols_no_intercept:
        expo[c] = df[c].astype("float64")
    expo["INTERCEPT"] = 1.0
    exposure_long = expo.set_index(["DATE","FACTSET_ID"]).stack(dropna=False).rename("EXPOSURE").reset_index().rename(
        columns={"FACTSET_ID":"SECURITY_ID","level_2":"FACTOR_NAME"}
    )
    exposure_long["MODEL"] = model_id
    exposure_long = exposure_long[["MODEL","DATE","SECURITY_ID","FACTOR_NAME","EXPOSURE"]]

    meta_records = []
    for col in style_cols:
        meta_records.append({"MODEL": model_id, "FACTOR_DISPLAY_NAME": col, "FACTOR_GROUP": "Style Factors"})
    for col in [c for c in dummy_cols if c.startswith("SECTOR_")]:
        meta_records.append({"MODEL": model_id, "FACTOR_DISPLAY_NAME": col, "FACTOR_GROUP": "Sector Factors"})
    for col in [c for c in dummy_cols if c.startswith("CONTINENT_")]:
        meta_records.append({"MODEL": model_id, "FACTOR_DISPLAY_NAME": col, "FACTOR_GROUP": "Geographic Factors"})
    meta_records.append({"MODEL": model_id, "FACTOR_DISPLAY_NAME": "INTERCEPT", "FACTOR_GROUP": "Market Factor"})
    factor_names_df = pd.DataFrame(meta_records).drop_duplicates()

    covariance_long, sigma_by_date = compute_factor_covariance(fr_long, cov_window=cov_window, model_id=model_id, factor_order=factor_names)

    if specific_var_method == "residual_rolling":
        specific_variance = compute_specific_variance_residual_rolling(specific_returns, var_window=var_window, spec_floor=spec_floor, model_id=model_id)
    elif specific_var_method == "total_minus_explained":
        df_for_risk = df[["DATE","FACTSET_ID","RETURN"] + factor_names].copy()
        specific_variance = compute_specific_variance_total_minus_explained(df_for_risk, sigma_by_date=sigma_by_date, factor_names=factor_names, var_window=var_window, spec_floor=spec_floor, model_id=model_id)
    else:
        raise ValueError("specific_var_method must be 'residual_rolling' or 'total_minus_explained'")

    return exposure_long, fr_long, specific_returns, covariance_long, factor_names_df, specific_variance
