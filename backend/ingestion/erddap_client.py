"""
backend/ingestion/erddap_client.py
Production client for NOAA Coral Reef Watch 5km Daily via CoastWatch ERDDAP.
Includes automated retries, caching, and strict "no data" error reporting.
"""

import time
import logging
import requests
from datetime import date, timedelta
from typing import Optional, Dict, Any
from backend.config import settings
from backend.cache import cache

logger = logging.getLogger("coralguard.erddap")

class NOAAERDDAPClient:
    """Client for querying NOAA Coral Reef Watch Daily 5km ERDDAP server."""

    def __init__(
        self,
        base_url: str = settings.NOAA_ERDDAP_BASE_URL,
        timeout: int = settings.ERDDAP_TIMEOUT_SEC,
        max_retries: int = settings.ERDDAP_MAX_RETRIES
    ):
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()

    def fetch_daily_sst_point(
        self,
        lat: float,
        lon: float,
        obs_date: date
    ) -> Dict[str, Any]:
        """
        Fetches daily SST, SSTA, and DHW for a specific geographic coordinate from NOAA ERDDAP.
        Returns dictionary with thermal metrics or status="no_data" if unavailable.
        """
        cache_key = f"erddap:{lat:.2f}:{lon:.2f}:{obs_date.isoformat()}"
        cached = cache.get_json(cache_key)
        if cached:
            logger.debug(f"Cache hit for ERDDAP query: {cache_key}")
            return cached

        date_str = obs_date.strftime("%Y-%m-%dT12:00:00Z")
        # Format ERDDAP griddap query
        # NOAA_DHW.csv?sea_surface_temperature,sea_surface_temperature_anomaly,degree_heating_week
        query_url = (
            f"{self.base_url}?CRW_SST,CRW_SSTANOMALY,CRW_DHW,CRW_BAA_max_7d"
            f"[({date_str}):1:({date_str})]"
            f"[({lat}):1:({lat})]"
            f"[({lon}):1:({lon})]"
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Fetching NOAA ERDDAP (Attempt {attempt}/{self.max_retries}): {lat:.2f}, {lon:.2f} on {obs_date}")
                resp = self.session.get(query_url, timeout=self.timeout)

                if resp.status_code == 200:
                    lines = resp.text.strip().split("\n")
                    # ERDDAP CSV returns 2 header lines: variable names and units
                    if len(lines) >= 3:
                        data_parts = lines[2].split(",")
                        sst_val = float(data_parts[3]) if data_parts[3] != "NaN" else None
                        ssta_val = float(data_parts[4]) if data_parts[4] != "NaN" else None
                        dhw_val = float(data_parts[5]) if data_parts[5] != "NaN" else None
                        baa_val = int(data_parts[6]) if len(data_parts) > 6 and data_parts[6] != "NaN" else 0

                        result = {
                            "date": obs_date.isoformat(),
                            "latitude": lat,
                            "longitude": lon,
                            "sst": sst_val,
                            "ssta": ssta_val,
                            "dhw": dhw_val,
                            "bleaching_alert_level": baa_val,
                            "status": "verified" if sst_val is not None else "no_data",
                            "source": "NOAA_CoralReefWatch_ERDDAP"
                        }
                        # Cache verified response
                        cache.set_json(cache_key, result, ttl=86400)
                        return result

                elif resp.status_code == 404:
                    logger.warning(f"NOAA ERDDAP coordinate/date out of bounds: {lat}, {lon} ({resp.status_code})")
                    return {
                        "date": obs_date.isoformat(),
                        "latitude": lat,
                        "longitude": lon,
                        "sst": None, "ssta": None, "dhw": None, "bleaching_alert_level": 0,
                        "status": "no_data",
                        "error": "coordinate_or_date_out_of_bounds"
                    }

            except (requests.RequestException, Exception) as e:
                logger.warning(f"NOAA ERDDAP attempt {attempt} failed ({e})")
                if attempt < self.max_retries:
                    time.sleep(1.5 * attempt)

        # All retries failed - return strict "source_offline" status instead of mock fabrication
        logger.error(f"NOAA ERDDAP unreachable for {lat}, {lon} on {obs_date} after {self.max_retries} retries.")
        return {
            "date": obs_date.isoformat(),
            "latitude": lat,
            "longitude": lon,
            "sst": None,
            "ssta": None,
            "dhw": None,
            "bleaching_alert_level": 0,
            "status": "source_offline",
            "error": "NOAA CoastWatch ERDDAP service unavailable"
        }

