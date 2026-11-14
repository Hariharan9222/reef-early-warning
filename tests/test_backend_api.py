"""
tests/test_backend_api.py
Comprehensive unit and integration tests for FastAPI backend endpoints:
- GET  /health
- GET  /sites
- GET  /risk/{site_id}
- GET  /sst/{site_id}
- GET  /acoustics/{site_id}
- GET  /alerts
- POST /reports
- GET  /metrics
"""

import unittest
from datetime import date
from fastapi.testclient import TestClient
from backend.api.main import app

class TestBackendAPIRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_01_health_check(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["database"], "connected")

    def test_02_get_sites(self):
        resp = self.client.get("/sites")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 15)
        site_ids = [s["id"] for s in data]
        self.assertIn("gbr_lizard_island", site_ids)
        self.assertIn("carib_florida_keys", site_ids)
        self.assertIn("ct_raja_ampat", site_ids)

    def test_03_get_site_risk(self):
        resp = self.client.get("/risk/gbr_lizard_island?lead=14")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["site_id"], "gbr_lizard_island")
        self.assertIn("probabilities", data)
        self.assertIn("total_bleaching_risk", data["probabilities"])
        self.assertIn("severe_bleaching", data["probabilities"])
        self.assertIn("alert_tier", data)
        self.assertIn("sst_summary", data)
        self.assertIn("soundscape_summary", data)
        self.assertIn("diver_summary", data)

    def test_04_get_site_risk_not_found(self):
        resp = self.client.get("/risk/nonexistent_reef_site_999")
        self.assertEqual(resp.status_code, 404)

    def test_05_get_site_sst(self):
        resp = self.client.get("/sst/gbr_lizard_island?days=30")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["site_id"], "gbr_lizard_island")
        self.assertEqual(data["status"], "available")
        self.assertIsNotNone(data["latest"])
        self.assertGreaterEqual(len(data["history"]), 1)

    def test_06_get_site_sst_not_found(self):
        resp = self.client.get("/sst/nonexistent_reef_site_999")
        self.assertEqual(resp.status_code, 404)

    def test_07_get_site_acoustics(self):
        resp = self.client.get("/acoustics/gbr_lizard_island?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["site_id"], "gbr_lizard_island")
        self.assertEqual(data["status"], "available")
        self.assertIsNotNone(data["latest"])
        self.assertIn("aci", data["latest"])
        self.assertIn("spectral_entropy", data["latest"])

    def test_08_get_active_alerts(self):
        resp = self.client.get("/alerts?lead=14")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["lead_days"], 14)
        self.assertIn("active_alerts_count", data)
        self.assertIn("total_monitored_sites", data)
        self.assertGreaterEqual(data["total_monitored_sites"], 15)
        self.assertIsInstance(data["alerts"], list)

    def test_09_post_citizen_report_valid(self):
        payload = {
            "latitude": -14.670,
            "longitude": 145.465,
            "observed_at": str(date.today()),
            "observer_id": "diver_unit_test",
            "observer_type": "marine_biologist",
            "survey_program": "Reef Check",
            "depth_meters": 11.5,
            "bleaching_percent": 28.0,
            "has_photo": True,
            "photo_url": "https://example.org/photo.jpg",
            "notes": "Verified bleaching patch on northern crest."
        }
        resp = self.client.post("/reports", json=payload)
        self.assertEqual(resp.status_code, 201)
        data = resp.json()
        self.assertIn("id", data)
        self.assertEqual(data["site_id"], "gbr_lizard_island")
        self.assertGreaterEqual(data["trust_score"], 0.70)
        self.assertEqual(data["status"], "verified")

    def test_10_post_citizen_report_invalid(self):
        # Invalid payload: depth > 60m should fail validation
        invalid_payload = {
            "latitude": -14.670,
            "longitude": 145.465,
            "observed_at": str(date.today()),
            "observer_id": "diver_unit_test",
            "bleaching_percent": 150.0,  # Invalid: > 100%
            "depth_meters": 85.0        # Invalid: > 60m
        }
        resp = self.client.post("/reports", json=invalid_payload)
        self.assertEqual(resp.status_code, 422)

    def test_11_get_metrics(self):
        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("lead_14d", data)
        self.assertIn("lead_28d", data)
        self.assertIn("pr_auc", data["lead_14d"])
        self.assertIn("recall_severe", data["lead_14d"])
        self.assertIn("lead_time_accuracy", data["lead_14d"])
        self.assertIn("confusion_matrix", data["lead_14d"])

if __name__ == "__main__":
    unittest.main()

