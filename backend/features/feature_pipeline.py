"""
backend/features/feature_pipeline.py
Weekly per-site feature engineering pipeline joining SST, acoustic, and citizen diver modalities.
Computes multi-modal feature vectors without hardcoded or random mock values.
"""

import logging
from datetime import date, timedelta
from typing import Dict, Any, Optional
import numpy as np
from sqlalchemy.orm import Session
from backend.models.db_models import ReefSite, SSTDaily, AcousticRecording, CitizenReport, WeeklyFeature
from backend.ingestion.citizen_ingest import haversine_distance_km

logger = logging.getLogger("coralguard.features")

class FeatureEngineeringPipeline:
    """Aggregates multi-modal data into weekly feature representations per site."""

    @staticmethod
    def compute_site_weekly_features(
        db: Session,
        site_id: str,
        target_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """
        Extracts rolling 7/14/30d SST, acoustic soundscape, and citizen diver features
        for a given site on a reference date (or latest available date).
        """
        site = db.query(ReefSite).filter_by(id=site_id).first()
        if not site:
            raise KeyError(f"Reef site '{site_id}' not found")

        if target_date is None:
            latest_sst = db.query(SSTDaily).filter_by(site_id=site_id).order_by(SSTDaily.date.desc()).first()
            target_date = latest_sst.date if latest_sst else date.today()

        window_start = target_date - timedelta(days=30)

        # 1. SST Dynamics (Trailing 30 Days)
        sst_records = (
            db.query(SSTDaily)
            .filter(
                SSTDaily.site_id == site_id,
                SSTDaily.date <= target_date,
                SSTDaily.date >= window_start
            )
            .order_by(SSTDaily.date.asc())
            .all()
        )

        sst_features = {
            "sst": None, "ssta": None, "dhw": None,
            "ssta_roll_7d": None, "ssta_roll_14d": None, "ssta_roll_30d": None,
            "sst_max_30d": None,
            "sst_status": "available" if sst_records else "no_data"
        }

        if sst_records:
            latest_rec = sst_records[-1]
            sst_features["sst"] = latest_rec.sst
            sst_features["ssta"] = latest_rec.ssta
            sst_features["dhw"] = latest_rec.dhw

            # Rolling calculations
            ssta_vals = [r.ssta for r in sst_records if r.ssta is not None]
            sst_vals = [r.sst for r in sst_records if r.sst is not None]

            if ssta_vals:
                sst_features["ssta_roll_30d"] = float(np.mean(ssta_vals))
                sst_features["ssta_roll_14d"] = float(np.mean(ssta_vals[-14:]))
                sst_features["ssta_roll_7d"] = float(np.mean(ssta_vals[-7:]))
            if sst_vals:
                sst_features["sst_max_30d"] = float(np.max(sst_vals))

        # 2. Acoustic Soundscape Features (Trailing 30 Days)
        ac_records = (
            db.query(AcousticRecording)
            .filter(
                AcousticRecording.site_id == site_id,
                AcousticRecording.recorded_at <= target_date,
                AcousticRecording.recorded_at >= window_start,
                AcousticRecording.status == "verified"
            )
            .all()
        )

        ac_features = {
            "acoustic_is_missing": 1,
            "shrimp_band_energy_ratio": 0.0,
            "fish_chorus_energy_ratio": 0.0,
            "aci": 0.0,
            "adi": 0.0,
            "bioacoustic_index": 0.0,
            "ndsi": 0.0,
            "spectral_entropy": 0.0,
            "soundscape_status": "no_data"
        }

        if ac_records:
            ac_features["acoustic_is_missing"] = 0
            ac_features["soundscape_status"] = "verified"
            shrimp_vals = [r.shrimp_band_energy_ratio for r in ac_records if r.shrimp_band_energy_ratio is not None]
            fish_vals = [r.fish_chorus_energy_ratio for r in ac_records if r.fish_chorus_energy_ratio is not None]
            aci_vals = [r.aci for r in ac_records if r.aci is not None]
            entropy_vals = [r.spectral_entropy for r in ac_records if r.spectral_entropy is not None]
            bi_vals = [r.bioacoustic_index for r in ac_records if r.bioacoustic_index is not None]
            ndsi_vals = [r.ndsi for r in ac_records if r.ndsi is not None]

            if shrimp_vals:
                ac_features["shrimp_band_energy_ratio"] = float(np.mean(shrimp_vals))
            if fish_vals:
                ac_features["fish_chorus_energy_ratio"] = float(np.mean(fish_vals))
            if aci_vals:
                ac_features["aci"] = float(np.mean(aci_vals))
            if entropy_vals:
                ac_features["spectral_entropy"] = float(np.mean(entropy_vals))
            if bi_vals:
                ac_features["bioacoustic_index"] = float(np.mean(bi_vals))
            if ndsi_vals:
                ac_features["ndsi"] = float(np.mean(ndsi_vals))

        # 3. Citizen Diver Reports (Within 50km and 30 Days)
        all_reports = (
            db.query(CitizenReport)
            .filter(
                CitizenReport.observed_at <= target_date,
                CitizenReport.observed_at >= window_start,
                CitizenReport.status == "verified"
            )
            .all()
        )

        nearby_reports = []
        for rep in all_reports:
            dist = haversine_distance_km(site.latitude, site.longitude, rep.latitude, rep.longitude)
            if dist <= 50.0:
                nearby_reports.append(rep)

        cit_features = {
            "diver_has_reports": 1 if nearby_reports else 0,
            "diver_reports_count_50km_30d": float(len(nearby_reports)),
            "diver_weighted_severity_30d": 0.0,
            "diver_mean_confidence_30d": 0.0,
            "diver_status": "available" if nearby_reports else "no_data"
        }

        if nearby_reports:
            weights = np.array([r.trust_score for r in nearby_reports])
            # Normalize bleaching percent (0-100) to 3-class scale (0: None, 1: Mild, 2: Severe)
            sevs = np.array([r.bleaching_percent / 50.0 for r in nearby_reports])
            cit_features["diver_weighted_severity_30d"] = float(np.sum(sevs * weights) / (np.sum(weights) + 1e-8))
            cit_features["diver_mean_confidence_30d"] = float(np.mean(weights))

        # Combined multi-modal weekly vector
        combined = {
            "site_id": site_id,
            "site_name": site.name,
            "region": site.region,
            "latitude": site.latitude,
            "longitude": site.longitude,
            "date": target_date,
            **sst_features,
            **ac_features,
            **cit_features
        }
        return combined

