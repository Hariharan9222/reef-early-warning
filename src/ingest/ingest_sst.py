"""
src/ingest/ingest_sst.py
Downloads or falls back to synthetic data for Satellite SST, SSTA, and DHW.
Cleans and standardizes to data/processed/sst_processed.parquet.
"""

import os
import requests
import pandas as pd
from src.utils import get_logger, load_config
from src.synthetic import generate_synthetic_sst

logger = get_logger("ingest_sst")

def download_noaa_erddap_sst(config: dict, output_dir: str = "data/raw") -> str:
    """
    Attempts to download NOAA Coral Reef Watch / OISST data via ERDDAP.
    On network failure or timeout, gracefully falls back to synthetic generator.
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_file = os.path.join(output_dir, "satellite_sst_raw.csv")

    base_url = config.get("sst", {}).get("erddap_base_url")
    download_success = False

    if base_url:
        try:
            logger.info(f"Attempting live download from NOAA ERDDAP: {base_url}")
            # Request a small sample to test endpoint connectivity
            params = {
                "time>=": f"{config['dates']['start_year']}-01-01T00:00:00Z",
                "time<=": f"{config['dates']['start_year']}-01-05T00:00:00Z"
            }
            resp = requests.get(base_url, params=params, timeout=10)
            if resp.status_code == 200 and len(resp.content) > 100:
                with open(raw_file, "wb") as f:
                    f.write(resp.content)
                logger.info(f"Successfully downloaded NOAA ERDDAP data to {raw_file}")
                download_success = True
            else:
                logger.warning(f"NOAA ERDDAP returned status code {resp.status_code}. Initiating synthetic fallback.")
        except Exception as e:
            logger.warning(f"Live ERDDAP SST download failed ({e}). Auto-falling back to documented synthetic generator.")

    if not download_success:
        raw_file = generate_synthetic_sst(config, output_dir)
        # Record fallback flag in metadata/README requirement
        logger.info(f"Using synthetic SST data at {raw_file}")

    return raw_file

def process_sst(raw_file: str, config: dict, output_dir: str = "data/processed") -> str:
    """
    Processes, cleans, and standardizes SST, SSTA, and DHW data.
    Saves to parquet.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_parquet = os.path.join(output_dir, "sst_processed.parquet")

    df = pd.read_csv(raw_file)
    df["date"] = pd.to_datetime(df["date"])
    
    # Sort and remove duplicates
    df = df.sort_values(by=["site_id", "date"]).drop_duplicates(subset=["site_id", "date"]).reset_index(drop=True)

    # Validate essential columns
    expected_cols = ["date", "site_id", "region", "lat", "lon", "sst", "ssta", "dhw"]
    for col in expected_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in SST data")

    # Interpolate minor missing data per site
    df["sst"] = df.groupby("site_id")["sst"].transform(lambda s: s.interpolate(method="linear").bfill().ffill())
    df["ssta"] = df.groupby("site_id")["ssta"].transform(lambda s: s.interpolate(method="linear").bfill().ffill())
    df["dhw"] = df.groupby("site_id")["dhw"].transform(lambda s: s.interpolate(method="linear").bfill().ffill())

    df.to_parquet(out_parquet, index=False)
    logger.info(f"Processed SST data: {len(df)} records saved to {out_parquet}")
    return out_parquet

def run_ingest_sst(config_path: str = "config.yaml") -> str:
    cfg = load_config(config_path)
    raw_path = download_noaa_erddap_sst(cfg, cfg["paths"]["raw_dir"])
    processed_path = process_sst(raw_path, cfg, cfg["paths"]["processed_dir"])
    return processed_path

if __name__ == "__main__":
    run_ingest_sst()

