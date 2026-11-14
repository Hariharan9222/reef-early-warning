"""
src/features/sst_features.py
Extracts satellite SST-derived thermal stress features:
- 7/14/30-day rolling mean SSTA
- DHW (Degree Heating Weeks)
- max SST over 30 days
- SST rate of change (e.g., 7-day difference)
- Days above bleaching threshold (SST > MMM + 1.0 deg C) over rolling 30 and 90 days
"""

import pandas as pd
import numpy as np
from src.utils import get_logger

logger = get_logger("sst_features")

def compute_sst_features(df_sst: pd.DataFrame, mmm_offset: float = 1.0) -> pd.DataFrame:
    """
    Computes time-series rolling features on daily SST data per site.
    Input df_sst must contain: ['date', 'site_id', 'sst', 'ssta', 'dhw']
    """
    df = df_sst.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["site_id", "date"]).reset_index(drop=True)

    # Estimate site-specific MMM (climatological warmest monthly mean)
    # Using 95th percentile of historical baseline as MMM approximation
    site_mmm = df.groupby("site_id")["sst"].transform(lambda s: s.quantile(0.95))
    df["site_mmm"] = site_mmm
    df["bleaching_threshold"] = df["site_mmm"] + mmm_offset
    df["above_threshold"] = (df["sst"] > df["bleaching_threshold"]).astype(int)

    feature_dfs = []
    for site_id, site_data in df.groupby("site_id"):
        site_data = site_data.set_index("date").sort_index()

        # Rolling SSTA
        site_data["ssta_roll_7d"] = site_data["ssta"].rolling("7D", min_periods=1).mean()
        site_data["ssta_roll_14d"] = site_data["ssta"].rolling("14D", min_periods=1).mean()
        site_data["ssta_roll_30d"] = site_data["ssta"].rolling("30D", min_periods=1).mean()

        # Max SST over 30 days
        site_data["sst_max_30d"] = site_data["sst"].rolling("30D", min_periods=1).max()

        # SST rate of change (7-day delta)
        site_data["sst_rate_of_change_7d"] = site_data["sst"] - site_data["sst"].shift(7)
        site_data["sst_rate_of_change_7d"] = site_data["sst_rate_of_change_7d"].bfill().fillna(0.0)

        # Days above bleaching threshold in last 30 and 90 days
        site_data["days_above_threshold_30d"] = site_data["above_threshold"].rolling("30D", min_periods=1).sum()
        site_data["days_above_threshold_90d"] = site_data["above_threshold"].rolling("90D", min_periods=1).sum()

        site_data = site_data.reset_index()
        feature_dfs.append(site_data)

    result_df = pd.concat(feature_dfs, ignore_index=True)
    logger.info(f"Computed SST features for {len(result_df)} rows across {result_df['site_id'].nunique()} sites.")
    return result_df

if __name__ == "__main__":
    import os
    from src.utils import load_config
    cfg = load_config("config.yaml")
    sst_p = os.path.join(cfg["paths"]["processed_dir"], "sst_processed.parquet")
    if os.path.exists(sst_p):
        df_raw = pd.read_parquet(sst_p)
        res = compute_sst_features(df_raw, mmm_offset=cfg["sst"]["mmm_climatology_offset"])
        print(f"Computed SST features: {res.shape}")
    else:
        logger.warning(f"File {sst_p} not found. Run ingestion first.")

