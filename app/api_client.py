"""
app/api_client.py
CoralGuard Frontend API Client.
Connects the Streamlit UI to the FastAPI production backend endpoints:
- GET  /sites
- GET  /risk/{site_id}?lead={lead}
- GET  /sst/{site_id}?days={days}
- GET  /acoustics/{site_id}?limit={limit}
- GET  /alerts?lead={lead}
- POST /reports
- GET  /metrics
- GET  /health

Provides transparent offline fallback to local data/models if the backend server is unreachable.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
import requests

logger = logging.getLogger("coralguard.client")

DEFAULT_BACKEND_URL = os.getenv("CORALGUARD_BACKEND_URL", "http://localhost:8000")

class CoralGuardAPIClient:
    def __init__(self, base_url: str = DEFAULT_BACKEND_URL, timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Check if FastAPI backend is online and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=self.timeout)
            if resp.status_code == 200:
                return {"online": True, **resp.json()}
        except Exception as e:
            logger.debug(f"Backend health check failed: {e}")
        return {"online": False, "status": "offline"}

    def is_online(self) -> bool:
        """Boolean indicator of backend connectivity."""
        return self.check_health().get("online", False)

    def get_sites(self) -> List[Dict[str, Any]]:
        """Fetch all monitored reef sites catalog."""
        try:
            resp = requests.get(f"{self.base_url}/sites", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch /sites from backend: {e}")
        return []

    def get_site_risk(self, site_id: str, lead_days: int = 14) -> Optional[Dict[str, Any]]:
        """Fetch multimodal risk prediction for a site (14 or 28 days ahead)."""
        try:
            resp = requests.get(
                f"{self.base_url}/risk/{site_id}",
                params={"lead": lead_days},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch /risk/{site_id} from backend: {e}")
        return None

    def get_site_sst(self, site_id: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """Fetch satellite SST observations for a reef site."""
        try:
            resp = requests.get(
                f"{self.base_url}/sst/{site_id}",
                params={"days": days},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch /sst/{site_id} from backend: {e}")
        return None

    def get_site_acoustics(self, site_id: str, limit: int = 10) -> Optional[Dict[str, Any]]:
        """Fetch bioacoustic soundscape recordings and indices."""
        try:
            resp = requests.get(
                f"{self.base_url}/acoustics/{site_id}",
                params={"limit": limit},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch /acoustics/{site_id} from backend: {e}")
        return None

    def get_active_alerts(self, lead_days: int = 14) -> Optional[Dict[str, Any]]:
        """Fetch all sites with active bleaching watches, warnings, or alerts."""
        try:
            resp = requests.get(
                f"{self.base_url}/alerts",
                params={"lead": lead_days},
                timeout=self.timeout
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch /alerts from backend: {e}")
        return None

    def submit_diver_report(self, report_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a citizen diver observation to the backend with trust scoring."""
        try:
            resp = requests.post(
                f"{self.base_url}/reports",
                json=report_payload,
                timeout=self.timeout
            )
            if resp.status_code in (200, 201):
                return {"success": True, "data": resp.json()}
            else:
                return {"success": False, "error": resp.text, "status_code": resp.status_code}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_model_metrics(self) -> Optional[Dict[str, Any]]:
        """Fetch benchmark model metrics (PR-AUC, Recall, Confusion Matrix)."""
        try:
            resp = requests.get(f"{self.base_url}/metrics", timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch /metrics from backend: {e}")
        return None

# Singleton client instance
api_client = CoralGuardAPIClient()

