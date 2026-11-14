"""
src/ingest/ingest_citizen.py
Ingests and cleans citizen diver bleaching reports (Reef Check, CoralWatch, etc.).
Standardizes bleaching severity to:
0 = none (< 10% bleached)
1 = mild (10 - 50% bleached)
2 = severe (> 50% bleached)
Saves raw and cleaned data/processed/citizen_processed.parquet.
"""

import os
import pandas as pd
from src.utils import get_logger, load_config
from src.synthetic import generate_synthetic_citizen_reports

logger = get_logger("ingest_citizen")

def load_or_fetch_citizen_reports(config: dict, output_dir: str = "data/raw") -> str:
    """
    Checks for external citizen science CSV; if absent or fails, falls back to synthetic generator.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, "citizen_diver_reports_raw.csv")

    # If external raw file not pre-existing, trigger fallback
    if not os.path.exists(raw_file):
        logger.info("External citizen science download not present. Initiating documented synthetic generator.")
        raw_file = generate_synthetic_citizen_reports(config, output_dir)
    else:
        logger.info(f"Found existing raw citizen reports at {raw_file}")

    return raw_file

def standardize_and_clean_citizen_reports(raw_file: str, config: dict, output_dir: str = "data/processed") -> str:
    """
    Cleans citizen diver reports:
    - Standardizes date format
    - Standardizes labels: 0=none, 1=mild, 2=severe
    - Validates coordinates and depths
    - Ensures observer confidence weighting
    """
    os.makedirs(output_dir, exist_ok=True)
    out_parquet = os.path.join(output_dir, "citizen_processed.parquet")

    df = pd.read_csv(raw_file)
    df["date"] = pd.to_datetime(df["date"])

    # Standardize bleaching category if needed
    if "bleaching_category" not in df.columns and "bleaching_percent" in df.columns:
        df["bleaching_category"] = pd.cut(
            df["bleaching_percent"],
            bins=[-float("inf"), 10.0, 50.0, float("inf")],
            labels=[0, 1, 2]
        ).astype(int)

    # Observer confidence default
    if "observer_confidence" not in df.columns:
        df["observer_confidence"] = 0.8

    # Filter invalid GPS
    df = df[(df["lat"] >= -90) & (df["lat"] <= 90) & (df["lon"] >= -180) & (df["lon"] <= 180)]
    df["depth"] = df["depth"].clip(lower=0, upper=100)

    # Sort
    df = df.sort_values(by="date").reset_index(drop=True)

    df.to_parquet(out_parquet, index=False)
    logger.info(f"Processed citizen reports: {len(df)} reports saved to {out_parquet}")
    return out_parquet

def run_ingest_citizen(config_path: str = "config.yaml") -> str:
    cfg = load_config(config_path)
    raw_path = load_or_fetch_citizen_reports(cfg, cfg["paths"]["raw_dir"])
    processed_path = standardize_and_clean_citizen_reports(raw_path, cfg, cfg["paths"]["processed_dir"])
    return processed_path

if __name__ == "__main__":
    run_ingest_citizen()

