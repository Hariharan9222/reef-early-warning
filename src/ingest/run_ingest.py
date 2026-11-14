"""
src/ingest/run_ingest.py
Orchestrates Phase 2: Data Ingestion for Satellite SST, Citizen Reports, and Underwater Acoustics.
Saves raw files to data/raw and standardized parquet files to data/processed.
"""

from src.utils import get_logger, load_config
from src.ingest.ingest_sst import run_ingest_sst
from src.ingest.ingest_citizen import run_ingest_citizen
from src.ingest.ingest_acoustics import run_ingest_acoustics

logger = get_logger("run_ingest")

def run_all_ingest(config_path: str = "config.yaml"):
    logger.info("Starting Phase 2: Data Ingestion across all modalities...")
    
    sst_parquet = run_ingest_sst(config_path)
    logger.info(f"SST ingestion completed: {sst_parquet}")

    citizen_parquet = run_ingest_citizen(config_path)
    logger.info(f"Citizen reports ingestion completed: {citizen_parquet}")

    acoustics_parquet = run_ingest_acoustics(config_path)
    logger.info(f"Acoustics ingestion completed: {acoustics_parquet}")

    logger.info("Phase 2: All modalities ingested and standardized to parquet.")
    return {
        "sst": sst_parquet,
        "citizen": citizen_parquet,
        "acoustics": acoustics_parquet
    }

if __name__ == "__main__":
    run_all_ingest()

