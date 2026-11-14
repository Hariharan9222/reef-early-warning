"""
src/run_all.py
Master execution pipeline for CoralGuard:
Executes Phase 2 (Ingestion) -> Phase 3 & 4 (Features & Alignment) -> Phase 5 (Modeling) -> Phase 6 & 7 (Evaluation)
"""

import sys
import time
from src.utils import get_logger, load_config, set_seed
from src.ingest.run_ingest import run_all_ingest
from src.features.build_features import align_and_build_features
from src.models.train_all import train_all_models
from src.models.evaluate import run_evaluation

logger = get_logger("run_all")

def run_pipeline():
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("   CORALGUARD: MULTIMODAL EARLY WARNING OF CORAL REEF BLEACHING PIPELINE")
    logger.info("================================================================================")

    cfg = load_config("config.yaml")
    set_seed(cfg.get("seed", 42))

    # Phase 2: Ingestion
    logger.info(">>> STEP 1/4: Ingestion across SST, Citizen Science, and Acoustics...")
    run_all_ingest("config.yaml")

    # Phase 3 & 4: Features & Alignment
    logger.info(">>> STEP 2/4: Feature Engineering and Masked-Modality Time-Based Alignment...")
    align_and_build_features("config.yaml")

    # Phase 5: Modeling & Tuning
    logger.info(">>> STEP 3/4: Training Baselines, Ablations, Boosted Trees, and Late-Fusion NN...")
    train_all_models("config.yaml")

    # Phase 6 & 7: Evaluation & Early Warning Benchmarking
    logger.info(">>> STEP 4/4: Generating Test Metrics, SHAP, Calibration, and FAR Benchmark...")
    run_evaluation("config.yaml")

    elapsed = time.time() - start_time
    logger.info("================================================================================")
    logger.info(f"   PIPELINE EXECUTION COMPLETE IN {elapsed:.1f} SECONDS!")
    logger.info("   Launch dashboard with: streamlit run app/streamlit_app.py")
    logger.info("================================================================================")

if __name__ == "__main__":
    run_pipeline()

