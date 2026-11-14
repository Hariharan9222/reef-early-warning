"""
backend/models/schemas.py
Pydantic v2 schemas for request validation and response serialization.
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

class ReefSiteBase(BaseModel):
    id: str
    name: str
    region: str
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    depth_meters: float = Field(10.0, ge=0.0, le=100.0)

class ReefSiteOut(ReefSiteBase):
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class SSTDataPoint(BaseModel):
    date: date
    sst: Optional[float] = None
    ssta: Optional[float] = None
    dhw: Optional[float] = None
    bleaching_alert_level: int = 0
    status: str = "verified"

class SSTResponse(BaseModel):
    site_id: str
    region: str
    latest: Optional[SSTDataPoint] = None
    history: List[SSTDataPoint] = []
    status: str = "available"  # "available" or "no_data"

class AcousticsDataPoint(BaseModel):
    recorded_at: datetime
    aci: Optional[float] = None
    adi: Optional[float] = None
    spectral_entropy: Optional[float] = None
    bioacoustic_index: Optional[float] = None
    ndsi: Optional[float] = None
    shrimp_band_energy_ratio: Optional[float] = None
    fish_chorus_energy_ratio: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None
    status: str = "verified"

class AcousticsResponse(BaseModel):
    site_id: str
    region: str
    latest: Optional[AcousticsDataPoint] = None
    history: List[AcousticsDataPoint] = []
    status: str = "available"  # "available" or "no_data"

class CitizenReportCreate(BaseModel):
    site_id: Optional[str] = None
    observed_at: date
    latitude: float = Field(..., ge=-90.0, le=90.0)
    longitude: float = Field(..., ge=-180.0, le=180.0)
    observer_id: str = Field(..., min_length=2, max_length=100)
    observer_type: str = Field("recreational_diver", description="dive_master, marine_biologist, recreational_diver, resident")
    survey_program: str = Field("Direct Submission", description="Reef Check, CoralWatch, Reef Life Survey, Direct Submission")
    depth_meters: float = Field(8.0, ge=0.0, le=60.0)
    bleaching_percent: float = Field(..., ge=0.0, le=100.0)
    has_photo: bool = False
    photo_url: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("observer_type")
    @classmethod
    def validate_observer_type(cls, v: str) -> str:
        valid = {"dive_master", "marine_biologist", "recreational_diver", "resident", "volunteer"}
        if v.lower() not in valid:
            raise ValueError(f"observer_type must be one of {valid}")
        return v.lower()

class CitizenReportOut(CitizenReportCreate):
    id: int
    trust_score: float
    status: str
    submitted_at: datetime

    class Config:
        from_attributes = True

class RiskProbabilities(BaseModel):
    no_bleaching: float
    mild_bleaching: float
    severe_bleaching: float
    total_bleaching_risk: float

class SiteRiskResponse(BaseModel):
    site_id: str
    region: str
    forecast_date: date
    lead_days: int
    probabilities: RiskProbabilities
    alert_tier: str
    model_version: str
    sst_summary: Optional[Dict[str, Any]] = None
    soundscape_summary: Optional[Dict[str, Any]] = None
    diver_summary: Optional[Dict[str, Any]] = None

class ActiveAlertItem(BaseModel):
    site_id: str
    site_name: str
    region: str
    latitude: float
    longitude: float
    lead_days: int
    alert_tier: str
    total_bleaching_risk: float
    severe_risk: float
    sst: Optional[float] = None
    dhw: Optional[float] = None

class ActiveAlertsResponse(BaseModel):
    lead_days: int
    reference_date: date
    active_alerts_count: int
    total_monitored_sites: int
    alerts: List[ActiveAlertItem]

class ModelMetricsResponse(BaseModel):
    evaluated_at: str
    test_period: str
    lead_14d: Optional[Dict[str, Any]] = None
    lead_28d: Optional[Dict[str, Any]] = None
    ablation_summary: List[Dict[str, Any]]
    false_alarm_rate_benchmark: Dict[str, Any]
    regional_breakdown: Optional[Dict[str, Any]] = None
