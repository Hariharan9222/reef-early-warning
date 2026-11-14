"""
backend/api/routes.py
FastAPI router implementing all required endpoints:
- GET  /sites
- GET  /risk/{site_id}
- GET  /sst/{site_id}
- GET  /acoustics/{site_id}
- GET  /alerts
- POST /reports
- GET  /metrics
"""

import os
import json
import logging
from datetime import date, timedelta
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models.db_models import ReefSite, SSTDaily, AcousticRecording, CitizenReport
from backend.models.schemas import (
    ReefSiteOut, SSTResponse, SSTDataPoint,
    AcousticsResponse, AcousticsDataPoint,
    CitizenReportCreate, CitizenReportOut,
    SiteRiskResponse, ActiveAlertsResponse, ActiveAlertItem,
    ModelMetricsResponse
)
from backend.features.feature_pipeline import FeatureEngineeringPipeline
from backend.models.ml_model import predictor
from backend.ingestion.citizen_ingest import CitizenIngestionEngine
from backend.cache import cache

logger = logging.getLogger("coralguard.api")
router = APIRouter()

# ----------------- 1. GET /sites -----------------
@router.get("/sites", response_model=List[ReefSiteOut], summary="List all monitored coral reef sites")
def list_sites(db: Session = Depends(get_db)):
    """Returns catalog of monitored reef sites with GPS coordinates and depth."""
    cache_key = "api:sites"
    cached = cache.get_json(cache_key)
    if cached:
        return cached

    sites = db.query(ReefSite).all()
    results = [ReefSiteOut.model_validate(s).model_dump(mode="json") for s in sites]
    cache.set_json(cache_key, results, ttl=300)
    return results

# ----------------- 2. GET /risk/{site_id} -----------------
@router.get("/risk/{site_id}", response_model=SiteRiskResponse, summary="Get forecasted bleaching risk for a site")
def get_site_risk(
    site_id: str,
    lead: int = Query(14, description="Forecast horizon in days (14 or 28)"),
    db: Session = Depends(get_db)
):
    """
    Computes real-time multi-modal risk forecast for a specific reef site.
    Returns predicted probabilities for No Bleaching, Mild Stress, and Severe Bleaching.
    """
    if lead not in (14, 28):
        lead = 14

    site = db.query(ReefSite).filter_by(id=site_id).first()
    if not site:
        raise HTTPException(status_code=404, detail=f"Reef site '{site_id}' not found.")

    try:
        features = FeatureEngineeringPipeline.compute_site_weekly_features(db, site_id)
    except Exception as e:
        logger.error(f"Feature computation error for {site_id}: {e}")
        raise HTTPException(status_code=500, detail="Feature computation failed.")

    pred = predictor.predict_risk(features, lead_days=lead)

    return SiteRiskResponse(
        site_id=site.id,
        region=site.region,
        forecast_date=features["date"],
        lead_days=lead,
        probabilities=pred["probabilities"],
        alert_tier=pred["alert_tier"],
        model_version=pred["model_version"],
        sst_summary={
            "sst": features["sst"],
            "ssta": features["ssta"],
            "dhw": features["dhw"],
            "ssta_roll_7d": features["ssta_roll_7d"],
            "status": features["sst_status"]
        },
        soundscape_summary={
            "acoustic_is_missing": bool(features["acoustic_is_missing"]),
            "shrimp_band_energy_ratio": features["shrimp_band_energy_ratio"],
            "fish_chorus_energy_ratio": features["fish_chorus_energy_ratio"],
            "aci": features["aci"],
            "spectral_entropy": features["spectral_entropy"],
            "status": features["soundscape_status"]
        },
        diver_summary={
            "diver_has_reports": bool(features["diver_has_reports"]),
            "diver_reports_count_50km_30d": int(features["diver_reports_count_50km_30d"]),
            "diver_weighted_severity_30d": features["diver_weighted_severity_30d"],
            "status": features["diver_status"]
        }
    )

# ----------------- 3. GET /sst/{site_id} -----------------
@router.get("/sst/{site_id}", response_model=SSTResponse, summary="Get satellite SST observations for a site")
def get_site_sst(site_id: str, days: int = Query(30, ge=1, le=90), db: Session = Depends(get_db)):
    """Returns trailing daily SST observations from NOAA ERDDAP or clear 'no data' status if unavailable."""
    site = db.query(ReefSite).filter_by(id=site_id).first()
    if not site:
        raise HTTPException(status_code=404, detail=f"Reef site '{site_id}' not found.")

    records = (
        db.query(SSTDaily)
        .filter(SSTDaily.site_id == site_id)
        .order_by(SSTDaily.date.desc())
        .limit(days)
        .all()
    )
    records.reverse()

    if not records:
        return SSTResponse(
            site_id=site.id,
            region=site.region,
            latest=None,
            history=[],
            status="no_data"
        )

    history_points = [
        SSTDataPoint(
            date=r.date, sst=r.sst, ssta=r.ssta, dhw=r.dhw,
            bleaching_alert_level=r.bleaching_alert_level, status=r.status
        )
        for r in records
    ]

    return SSTResponse(
        site_id=site.id,
        region=site.region,
        latest=history_points[-1],
        history=history_points,
        status="available"
    )

# ----------------- 4. GET /acoustics/{site_id} -----------------
@router.get("/acoustics/{site_id}", response_model=AcousticsResponse, summary="Get acoustic recordings & soundscape features")
def get_site_acoustics(site_id: str, limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)):
    """Returns acoustic soundscape indices for a site, or 'no data' if hydrophone is unavailable."""
    site = db.query(ReefSite).filter_by(id=site_id).first()
    if not site:
        raise HTTPException(status_code=404, detail=f"Reef site '{site_id}' not found.")

    records = (
        db.query(AcousticRecording)
        .filter_by(site_id=site_id)
        .order_by(AcousticRecording.recorded_at.desc())
        .limit(limit)
        .all()
    )

    if not records:
        return AcousticsResponse(
            site_id=site.id,
            region=site.region,
            latest=None,
            history=[],
            status="no_data"
        )

    points = [
        AcousticsDataPoint(
            recorded_at=r.recorded_at,
            aci=r.aci,
            adi=r.adi,
            spectral_entropy=r.spectral_entropy,
            bioacoustic_index=r.bioacoustic_index,
            ndsi=r.ndsi,
            shrimp_band_energy_ratio=r.shrimp_band_energy_ratio,
            fish_chorus_energy_ratio=r.fish_chorus_energy_ratio,
            spectral_centroid_mean=r.spectral_centroid_mean,
            status=r.status
        )
        for r in records
    ]

    return AcousticsResponse(
        site_id=site.id,
        region=site.region,
        latest=points[0] if points else None,
        history=points,
        status="available"
    )

# ----------------- 5. GET /alerts -----------------
@router.get("/alerts", response_model=ActiveAlertsResponse, summary="List all active bleaching warnings and alerts")
def get_active_alerts(
    lead: int = Query(14, description="Lead horizon (14 or 28 days)"),
    db: Session = Depends(get_db)
):
    """Scans all sites and returns those triggering Watch, Warning, or Alert thresholds."""
    if lead not in (14, 28):
        lead = 14

    sites = db.query(ReefSite).all()
    alerts = []
    ref_date = date.today()

    for site in sites:
        features = FeatureEngineeringPipeline.compute_site_weekly_features(db, site.id)
        pred = predictor.predict_risk(features, lead_days=lead)

        if "NORMAL" not in pred["alert_tier"]:
            alerts.append(
                ActiveAlertItem(
                    site_id=site.id,
                    site_name=site.name,
                    region=site.region,
                    latitude=site.latitude,
                    longitude=site.longitude,
                    lead_days=lead,
                    alert_tier=pred["alert_tier"],
                    total_bleaching_risk=pred["probabilities"]["total_bleaching_risk"],
                    severe_risk=pred["probabilities"]["severe_bleaching"],
                    sst=features["sst"],
                    dhw=features["dhw"]
                )
            )

    return ActiveAlertsResponse(
        lead_days=lead,
        reference_date=ref_date,
        active_alerts_count=len(alerts),
        total_monitored_sites=len(sites),
        alerts=alerts
    )

# ----------------- 6. POST /reports -----------------
@router.post("/reports", response_model=CitizenReportOut, status_code=status.HTTP_201_CREATED, summary="Submit diver survey report")
def submit_diver_report(
    report_in: CitizenReportCreate,
    db: Session = Depends(get_db)
):
    """
    Submits a field diver survey report with automated validation and trust score calculation.
    Computes trust score based on observer role, survey completeness, depth plausibility, and photo proof.
    """
    known_sites = db.query(ReefSite).all()
    processed_dict, matched_site = CitizenIngestionEngine.process_submission(report_in, known_sites)

    report_db = CitizenReport(**processed_dict)
    db.add(report_db)
    db.commit()
    db.refresh(report_db)

    # Invalidate cache
    cache.delete("api:sites")

    logger.info(f"Accepted citizen report {report_db.id} from {report_db.observer_id} (trust score: {report_db.trust_score:.2f})")
    return report_db

# ----------------- 7. GET /metrics -----------------
@router.get("/metrics", response_model=ModelMetricsResponse, summary="Get model evaluation benchmark metrics")
def get_model_metrics():
    """Exports PR-AUC, recall, lead-time accuracy, confusion matrix, and FAR benchmarks."""
    from backend.evaluation.evaluator import get_evaluation_metrics
    return get_evaluation_metrics()
