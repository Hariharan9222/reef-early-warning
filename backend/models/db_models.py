"""
backend/models/db_models.py
SQLAlchemy ORM models for CoralGuard PostgreSQL+PostGIS / SQLite tables.
"""

from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, Boolean, DateTime, Date, ForeignKey, Text, Index
)
from sqlalchemy.orm import relationship
from backend.database import Base

class ReefSite(Base):
    __tablename__ = "sites"

    id = Column(String(100), primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    region = Column(String(100), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    depth_meters = Column(Float, default=10.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    sst_records = relationship("SSTDaily", back_populates="site", cascade="all, delete-orphan")
    acoustics = relationship("AcousticRecording", back_populates="site", cascade="all, delete-orphan")
    citizen_reports = relationship("CitizenReport", back_populates="site")
    predictions = relationship("BleachingRisk", back_populates="site", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_site_coords", "latitude", "longitude"),
    )

class SSTDaily(Base):
    __tablename__ = "sst_daily"

    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(100), ForeignKey("sites.id"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    sst = Column(Float, nullable=True)                  # Sea Surface Temp (°C)
    ssta = Column(Float, nullable=True)                 # SST Anomaly (°C)
    dhw = Column(Float, nullable=True)                  # Degree Heating Weeks (°C-weeks)
    bleaching_alert_level = Column(Integer, default=0) # 0: None, 1: Watch, 2: Warning, 3: Alert 1, 4: Alert 2
    status = Column(String(50), default="verified")    # "verified", "no_data", "source_offline"
    fetched_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("ReefSite", back_populates="sst_records")

    __table_args__ = (
        Index("idx_site_date", "site_id", "date", unique=True),
    )

class AcousticRecording(Base):
    __tablename__ = "acoustic_recordings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(100), ForeignKey("sites.id"), nullable=False, index=True)
    recorded_at = Column(DateTime, nullable=False, index=True)
    file_path = Column(String(500), nullable=True)
    duration_sec = Column(Float, default=5.0)
    sample_rate = Column(Integer, default=22050)
    hydrophone_type = Column(String(100), default="SoundTrap 300HF")

    # Ecoacoustic & Bioacoustic Indices
    aci = Column(Float, nullable=True)                 # Acoustic Complexity Index
    adi = Column(Float, nullable=True)                 # Acoustic Diversity Index
    spectral_entropy = Column(Float, nullable=True)    # Spectral Entropy
    bioacoustic_index = Column(Float, nullable=True)   # Bioacoustic Index (2-8 kHz)
    ndsi = Column(Float, nullable=True)                # Normalized Difference Soundscape Index
    shrimp_band_energy_ratio = Column(Float, nullable=True) # 2-20 kHz
    fish_chorus_energy_ratio = Column(Float, nullable=True) # 50-1000 Hz
    spectral_centroid_mean = Column(Float, nullable=True)
    mfccs_json = Column(Text, nullable=True)           # JSON array of 13 MFCC means
    status = Column(String(50), default="verified")    # "verified", "no_data", "sensor_fault"
    created_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("ReefSite", back_populates="acoustics")

class CitizenReport(Base):
    __tablename__ = "citizen_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(100), ForeignKey("sites.id"), nullable=True, index=True)
    observed_at = Column(Date, nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    observer_id = Column(String(100), nullable=False)
    observer_type = Column(String(100), default="recreational_diver")  # "dive_master", "marine_biologist", "recreational_diver", "resident"
    survey_program = Column(String(100), default="Direct Submission") # "Reef Check", "CoralWatch", "Reef Life Survey"
    depth_meters = Column(Float, default=8.0)
    bleaching_percent = Column(Float, nullable=False)                  # 0 to 100
    has_photo = Column(Boolean, default=False)
    photo_url = Column(String(500), nullable=True)
    notes = Column(Text, nullable=True)
    trust_score = Column(Float, default=0.5)                          # 0.0 to 1.0 automated trust score
    status = Column(String(50), default="verified")                   # "verified", "flagged", "pending"
    submitted_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("ReefSite", back_populates="citizen_reports")

    __table_args__ = (
        Index("idx_citizen_coords", "latitude", "longitude"),
    )

class WeeklyFeature(Base):
    __tablename__ = "weekly_features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(100), ForeignKey("sites.id"), nullable=False, index=True)
    week_start_date = Column(Date, nullable=False, index=True)

    # Aggregated SST features
    sst = Column(Float, nullable=True)
    ssta = Column(Float, nullable=True)
    dhw = Column(Float, nullable=True)
    ssta_roll_7d = Column(Float, nullable=True)
    ssta_roll_14d = Column(Float, nullable=True)
    ssta_roll_30d = Column(Float, nullable=True)
    sst_max_30d = Column(Float, nullable=True)

    # Aggregated Acoustic features
    acoustic_is_missing = Column(Integer, default=1)
    shrimp_band_energy_ratio = Column(Float, nullable=True)
    fish_chorus_energy_ratio = Column(Float, nullable=True)
    aci = Column(Float, nullable=True)
    adi = Column(Float, nullable=True)
    bioacoustic_index = Column(Float, nullable=True)
    ndsi = Column(Float, nullable=True)
    spectral_entropy = Column(Float, nullable=True)

    # Aggregated Citizen features
    diver_reports_count_50km_30d = Column(Float, default=0.0)
    diver_weighted_severity_30d = Column(Float, default=0.0)
    diver_mean_confidence_30d = Column(Float, default=0.0)
    diver_has_reports = Column(Integer, default=0)

    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_site_week", "site_id", "week_start_date", unique=True),
    )

class BleachingRisk(Base):
    __tablename__ = "bleaching_predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    site_id = Column(String(100), ForeignKey("sites.id"), nullable=False, index=True)
    forecast_date = Column(Date, nullable=False, index=True)
    lead_days = Column(Integer, nullable=False)        # 14 or 28 days
    p_none = Column(Float, nullable=False)
    p_mild = Column(Float, nullable=False)
    p_severe = Column(Float, nullable=False)
    p_bleach = Column(Float, nullable=False)
    alert_tier = Column(String(50), nullable=False)    # Normal, Watch, Warning, Alert
    model_version = Column(String(50), default="xgb_v2")
    created_at = Column(DateTime, default=datetime.utcnow)

    site = relationship("ReefSite", back_populates="predictions")

