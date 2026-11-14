"""
src/features/build_features.py
Integrates Feature Engineering (Phase 3) and Data Alignment (Phase 4):
1. Resamples and joins SST, Citizen Reports, and Acoustics into weekly site-grid bins (0.25 deg)
2. Applies masked-modality approach for missing acoustic surveys (indicator + regional imputation)
3. Computes 14-day and 28-day forward-looking bleaching severity targets
4. Performs strict time-based split: Train (<= 2021), Val (2022), Test (2023-2024)
"""

import os
import numpy as np
import pandas as pd
from src.utils import get_logger, load_config, set_seed
from src.features.sst_features import compute_sst_features
from src.features.acoustic_features import process_all_acoustics
from src.features.citizen_features import compute_citizen_features_for_sites

logger = get_logger("build_features")

def align_and_build_features(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    proc_dir = cfg["paths"]["processed_dir"]
    sst_parquet = os.path.join(proc_dir, "sst_processed.parquet")
    citizen_parquet = os.path.join(proc_dir, "citizen_processed.parquet")
    acoustics_meta_parquet = os.path.join(proc_dir, "acoustics_metadata.parquet")

    logger.info("Step 1: Extracting rolling SST features...")
    df_sst = pd.read_parquet(sst_parquet)
    df_sst_feats = compute_sst_features(df_sst, mmm_offset=cfg["sst"]["mmm_climatology_offset"])

    logger.info("Step 2: Resampling SST features to weekly bins (W-MON)...")
    # Resample daily SST to weekly (W-MON) per site
    df_sst_feats["date"] = pd.to_datetime(df_sst_feats["date"])
    
    weekly_records = []
    for site_id, site_group in df_sst_feats.groupby("site_id"):
        site_group = site_group.set_index("date").sort_index()
        weekly = site_group.resample("W-MON").last().reset_index()
        weekly["site_id"] = site_id
        # Forward fill any coordinates or region names
        weekly["region"] = site_group["region"].iloc[0]
        weekly["lat"] = site_group["lat"].iloc[0]
        weekly["lon"] = site_group["lon"].iloc[0]
        weekly_records.append(weekly)

    df_weekly = pd.concat(weekly_records, ignore_index=True)
    logger.info(f"Generated {len(df_weekly)} weekly site-grid observation points.")

    logger.info("Step 3: Extracting citizen diver features...")
    df_citizen = pd.read_parquet(citizen_parquet)
    df_cit_feats = compute_citizen_features_for_sites(
        df_sites=df_weekly[["date", "site_id", "lat", "lon"]],
        df_citizen=df_citizen,
        search_radius_km=cfg["citizen"]["search_radius_km"],
        window_days=cfg["citizen"]["window_days"]
    )
    df_merged = pd.merge(df_weekly, df_cit_feats, on=["site_id", "date"], how="left")

    logger.info("Step 4: Extracting acoustic soundscape features...")
    acoustic_feats_parquet = os.path.join(proc_dir, "acoustics_features.parquet")
    if not os.path.exists(acoustic_feats_parquet):
        acoustic_feats_parquet = process_all_acoustics(acoustics_meta_parquet, cfg, proc_dir)
    df_acoustics = pd.read_parquet(acoustic_feats_parquet)
    df_acoustics["date"] = pd.to_datetime(df_acoustics["date"])

    logger.info("Step 5: Aligning acoustic features with masked-modality strategy...")
    # Find closest acoustic survey within +/- 21 days for the same site
    acoustic_cols = [c for c in df_acoustics.columns if c not in ["clip_id", "site_id", "date"]]
    
    # Initialize acoustic columns as NaN
    for col in acoustic_cols:
        df_merged[col] = np.nan
    df_merged["acoustic_is_missing"] = 1

    for site_id, ac_group in df_acoustics.groupby("site_id"):
        site_mask = df_merged["site_id"] == site_id
        site_indices = df_merged[site_mask].index
        site_dates = df_merged.loc[site_indices, "date"]

        for _, ac_row in ac_group.iterrows():
            ac_date = ac_row["date"]
            # Time delta in days
            deltas = np.abs((site_dates - ac_date).dt.days)
            near_mask = deltas <= 21
            match_indices = site_indices[near_mask]
            if len(match_indices) > 0:
                for col in acoustic_cols:
                    df_merged.loc[match_indices, col] = ac_row[col]
                df_merged.loc[match_indices, "acoustic_is_missing"] = 0

    # Impute missing acoustic features with regional means + missing indicator
    for col in acoustic_cols:
        regional_mean = df_merged.groupby("region")[col].transform("mean")
        global_mean = df_merged[col].mean() if not np.isnan(df_merged[col].mean()) else 0.0
        df_merged[col] = df_merged[col].fillna(regional_mean).fillna(global_mean)

    logger.info("Step 6: Creating 14-day and 28-day forward-looking targets...")
    # Target definition at t + lead_days:
    # 0 = None, 1 = Mild, 2 = Severe
    # Corresponds to future DHW >= 8 (Severe) or 4 <= DHW < 8 (Mild)
    df_merged = df_merged.sort_values(by=["site_id", "date"]).reset_index(drop=True)

    df_merged["target_t14"] = np.nan
    df_merged["target_t28"] = np.nan

    for site_id, group in df_merged.groupby("site_id"):
        # Shift 2 weeks (2 rows of weekly data) for t+14
        future_dhw_14 = group["dhw"].shift(-2)
        future_diver_14 = group["diver_weighted_severity_30d"].shift(-2)

        sev_14 = np.zeros(len(group), dtype=int)
        # Condition for Mild: DHW >= 4 or diver severity >= 1.0
        sev_14[(future_dhw_14 >= 4.0) | (future_diver_14 >= 0.8)] = 1
        # Condition for Severe: DHW >= 8 or diver severity >= 1.6
        sev_14[(future_dhw_14 >= 8.0) | (future_diver_14 >= 1.6)] = 2
        df_merged.loc[group.index, "target_t14"] = sev_14

        # Shift 4 weeks (4 rows of weekly data) for t+28
        future_dhw_28 = group["dhw"].shift(-4)
        future_diver_28 = group["diver_weighted_severity_30d"].shift(-4)

        sev_28 = np.zeros(len(group), dtype=int)
        sev_28[(future_dhw_28 >= 4.0) | (future_diver_28 >= 0.8)] = 1
        sev_28[(future_dhw_28 >= 8.0) | (future_diver_28 >= 1.6)] = 2
        df_merged.loc[group.index, "target_t28"] = sev_28

    # Drop the trailing rows where future target is out of bounds
    df_final = df_merged.dropna(subset=["target_t14", "target_t28"]).reset_index(drop=True)
    df_final["target_t14"] = df_final["target_t14"].astype(int)
    df_final["target_t28"] = df_final["target_t28"].astype(int)

    logger.info("Step 7: Performing strict time-based split...")
    # Train: <= 2021, Val: 2022, Test: 2023-2024
    years = df_final["date"].dt.year
    train_mask = years <= cfg["dates"]["train_end_year"]
    val_mask = years == cfg["dates"]["val_year"]
    test_mask = (years >= cfg["dates"]["test_start_year"]) & (years <= cfg["dates"]["test_end_year"])

    df_train = df_final[train_mask].reset_index(drop=True)
    df_val = df_final[val_mask].reset_index(drop=True)
    df_test = df_final[test_mask].reset_index(drop=True)

    # Save to disk
    full_path = os.path.join(proc_dir, "full_aligned_dataset.parquet")
    train_path = os.path.join(proc_dir, "train_data.parquet")
    val_path = os.path.join(proc_dir, "val_data.parquet")
    test_path = os.path.join(proc_dir, "test_data.parquet")

    df_final.to_parquet(full_path, index=False)
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    logger.info(f"Dataset successfully built and aligned:")
    logger.info(f"  - Full: {len(df_final)} rows, {df_final.shape[1]} columns")
    logger.info(f"  - Train (<= {cfg['dates']['train_end_year']}): {len(df_train)} rows")
    logger.info(f"  - Val ({cfg['dates']['val_year']}): {len(df_val)} rows")
    logger.info(f"  - Test ({cfg['dates']['test_start_year']}-{cfg['dates']['test_end_year']}): {len(df_test)} rows")
    logger.info(f"  - Target distribution (t+14 Train): {dict(df_train['target_t14'].value_counts())}")
    logger.info(f"  - Target distribution (t+14 Test):  {dict(df_test['target_t14'].value_counts())}")

    return {
        "full": full_path,
        "train": train_path,
        "val": val_path,
        "test": test_path
    }

if __name__ == "__main__":
    align_and_build_features()

