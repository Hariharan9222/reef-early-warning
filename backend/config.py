"""
backend/config.py
Centralized configuration for CoralGuard FastAPI Backend.
Reads environment variables from .env with production defaults.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load .env file
load_dotenv()

class Settings:
    PROJECT_NAME: str = "CoralGuard Backend API"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Database Configuration (PostgreSQL+PostGIS with SQLite local fallback)
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./data/coralguard.db"
    )

    # Redis Cache Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_DEFAULT_TTL: int = int(os.getenv("CACHE_DEFAULT_TTL", 3600))  # 1 hour

    # NOAA ERDDAP Sea Surface Temperature Source
    NOAA_ERDDAP_BASE_URL: str = os.getenv(
        "NOAA_ERDDAP_BASE_URL",
        "https://coastwatch.pfeg.noaa.gov/erddap/griddap/NOAA_DHW.csv"
    )
    ERDDAP_TIMEOUT_SEC: int = int(os.getenv("ERDDAP_TIMEOUT_SEC", 15))
    ERDDAP_MAX_RETRIES: int = int(os.getenv("ERDDAP_MAX_RETRIES", 3))

    # Ingestion Scheduler
    SCHEDULER_ENABLED: bool = os.getenv("SCHEDULER_ENABLED", "True").lower() in ("true", "1", "yes")
    INGESTION_CRON_HOUR: int = int(os.getenv("INGESTION_CRON_HOUR", 2))  # 02:00 UTC daily

    # Machine Learning Models
    MODEL_T14_PATH: str = os.getenv("MODEL_T14_PATH", "models/xgb_multimodal_t14.joblib")
    MODEL_T28_PATH: str = os.getenv("MODEL_T28_PATH", "models/xgb_multimodal_t28.joblib")

    # Operational Alert Thresholds
    WATCH_THRESHOLD: float = float(os.getenv("WATCH_THRESHOLD", 0.25))
    WARNING_THRESHOLD: float = float(os.getenv("WARNING_THRESHOLD", 0.50))
    ALERT_THRESHOLD: float = float(os.getenv("ALERT_THRESHOLD", 0.75))

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "*"
    ]

settings = Settings()
