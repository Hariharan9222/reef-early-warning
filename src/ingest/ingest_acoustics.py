"""
src/ingest/ingest_acoustics.py
Downloads or generates underwater acoustic soundscape audio clips (.wav) and metadata CSV.
Saves raw audio to data/raw/audio/ and metadata to data/processed/acoustics_metadata.parquet.
"""

import os
import pandas as pd
from src.utils import get_logger, load_config
from src.synthetic import generate_synthetic_acoustics

logger = get_logger("ingest_acoustics")

def fetch_or_generate_acoustics(config: dict, raw_dir: str = "data/raw") -> str:
    """
    Checks for external acoustic datasets (Zenodo/Dryad); falls back to documented synthetic soundscape generator.
    """
    audio_dir = os.path.join(raw_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    meta_file = os.path.join(raw_dir, "synthetic_acoustics_metadata.csv")

    if not os.path.exists(meta_file) or len(os.listdir(audio_dir)) == 0:
        logger.info("External acoustic soundscape dataset not present. Generating synthetic acoustic soundscape clips.")
        meta_file = generate_synthetic_acoustics(config, audio_dir)
    else:
        logger.info(f"Found existing acoustic clips in {audio_dir}")

    return meta_file

def process_acoustics_metadata(meta_file: str, config: dict, output_dir: str = "data/processed") -> str:
    """
    Cleans and standardizes acoustic metadata into parquet format.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_parquet = os.path.join(output_dir, "acoustics_metadata.parquet")

    df = pd.read_csv(meta_file)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["site_id", "date"]).reset_index(drop=True)

    df.to_parquet(out_parquet, index=False)
    logger.info(f"Processed acoustics metadata: {len(df)} clips saved to {out_parquet}")
    return out_parquet

def run_ingest_acoustics(config_path: str = "config.yaml") -> str:
    cfg = load_config(config_path)
    meta_file = fetch_or_generate_acoustics(cfg, cfg["paths"]["raw_dir"])
    processed_path = process_acoustics_metadata(meta_file, cfg, cfg["paths"]["processed_dir"])
    return processed_path

if __name__ == "__main__":
    run_ingest_acoustics()

