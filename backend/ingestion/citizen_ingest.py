"""
backend/ingestion/citizen_ingest.py
Ingestion and trust score computation engine for Citizen Diver Science.
Ingests surveys from Reef Check, CoralWatch, and Reef Life Survey,
and processes live diver submissions via POST /reports.
"""

import math
import logging
from typing import Dict, Any, Optional, Tuple
from backend.models.schemas import CitizenReportCreate

logger = logging.getLogger("coralguard.citizen")

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two GPS coordinates in kilometers."""
    R = 6371.0  # Earth's radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def compute_trust_score(report: CitizenReportCreate) -> float:
    """
    Computes an empirical observer trust score in range [0.10, 1.00].
    Factors in observer credentials, photographic verification, depth plausibility, and documentation quality.
    """
    score = 0.50  # Baseline neutral trust

    # 1. Observer Credential Weight
    role_weights = {
        "marine_biologist": 0.25,
        "dive_master": 0.20,
        "recreational_diver": 0.05,
        "volunteer": 0.02,
        "resident": 0.00
    }
    score += role_weights.get(report.observer_type.lower(), 0.0)

    # 2. Photographic Proof
    if report.has_photo:
        score += 0.15
    else:
        score -= 0.05

    # 3. Survey Program Provenance
    program_weights = {
        "Reef Check": 0.10,
        "Reef Life Survey": 0.10,
        "CoralWatch": 0.08,
        "Direct Submission": 0.00
    }
    score += program_weights.get(report.survey_program, 0.0)

    # 4. Depth Plausibility
    # Coral reefs rarely thrive deeper than 40m due to photosynthetic light attenuation
    if 2.0 <= report.depth_meters <= 35.0:
        score += 0.05
    elif report.depth_meters > 45.0 or report.depth_meters < 0.0:
        score -= 0.20

    # 5. Descriptive Documentation
    if report.notes and len(report.notes.strip()) >= 20:
        score += 0.05

    return round(float(min(1.0, max(0.10, score))), 2)

class CitizenIngestionEngine:
    """Validates and processes live and historical diver reports."""

    @staticmethod
    def process_submission(
        report_data: CitizenReportCreate,
        known_sites: list
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Validates a citizen diver report, computes trust score,
        and matches it to the nearest reef site within 50 km.
        """
        trust = compute_trust_score(report_data)

        # Match to nearest reef site
        matched_site_id = report_data.site_id
        min_dist = float("inf")

        if not matched_site_id and known_sites:
            for site in known_sites:
                dist = haversine_distance_km(
                    report_data.latitude, report_data.longitude,
                    site.latitude, site.longitude
                )
                if dist < min_dist:
                    min_dist = dist
                    matched_site_id = site.id

            # If nearest reef is farther than 100km, leave as regional open observation
            if min_dist > 100.0:
                matched_site_id = None

        status = "verified" if trust >= 0.40 else "flagged"

        processed = {
            "site_id": matched_site_id,
            "observed_at": report_data.observed_at,
            "latitude": report_data.latitude,
            "longitude": report_data.longitude,
            "observer_id": report_data.observer_id,
            "observer_type": report_data.observer_type,
            "survey_program": report_data.survey_program,
            "depth_meters": report_data.depth_meters,
            "bleaching_percent": report_data.bleaching_percent,
            "has_photo": report_data.has_photo,
            "photo_url": report_data.photo_url,
            "notes": report_data.notes,
            "trust_score": trust,
            "status": status
        }
        return processed, matched_site_id

