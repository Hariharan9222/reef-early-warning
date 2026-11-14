"""
tests/test_server.py
Integration unit tests for the CoralGuard REST API (src/server.py).
Tests all API endpoints: /health, /sites, /alerts, /predict (GET & POST), /metrics.
"""

import unittest
from src.server import CoralGuardAPI

class TestCoralGuardAPI(unittest.TestCase):
    def test_health_endpoint(self):
        status, data = CoralGuardAPI.dispatch("GET", "/health", {})
        self.assertEqual(status, 200)
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("regions_monitored", data)

    def test_sites_endpoint(self):
        status, data = CoralGuardAPI.dispatch("GET", "/sites", {})
        self.assertEqual(status, 200)
        self.assertGreater(data.get("count", 0), 0)
        self.assertIsInstance(data.get("sites"), list)

    def test_alerts_endpoint(self):
        status, data = CoralGuardAPI.dispatch("GET", "/alerts", {"lead": ["14"]})
        self.assertEqual(status, 200)
        self.assertEqual(data.get("lead_days"), 14)
        self.assertIn("active_alerts_count", data)

    def test_predict_get_endpoint(self):
        status, data = CoralGuardAPI.dispatch("GET", "/predict", {"lead": ["14"]})
        self.assertEqual(status, 200)
        self.assertIn("results", data)
        self.assertIsInstance(data["results"], list)
        first = data["results"][0]
        self.assertIn("probabilities", first)
        self.assertIn("alert_tier", first)

    def test_predict_get_single_site(self):
        # Fetch available site
        _, sites_data = CoralGuardAPI.dispatch("GET", "/sites", {})
        sample_site = sites_data["sites"][0]["site_id"]

        status, data = CoralGuardAPI.dispatch("GET", "/predict", {"site": [sample_site], "lead": ["14"]})
        self.assertEqual(status, 200)
        res = data.get("results")
        self.assertEqual(res["site_id"], sample_site)
        self.assertIn("probabilities", res)

    def test_predict_post_endpoint(self):
        sample_payload = {
            "lead": 14,
            "features": {
                "sst": 30.5,
                "ssta": 1.8,
                "dhw": 7.5,
                "snapping_shrimp_band_ratio": 0.15
            }
        }
        status, data = CoralGuardAPI.dispatch("POST", "/predict", {}, sample_payload)
        self.assertEqual(status, 200)
        self.assertIn("probabilities", data)
        probs = data["probabilities"]
        self.assertIn("severe_bleaching", probs)
        self.assertAlmostEqual(probs["no_bleaching"] + probs["mild_bleaching"] + probs["severe_bleaching"], 1.0, places=3)

    def test_metrics_endpoint(self):
        status, data = CoralGuardAPI.dispatch("GET", "/metrics", {})
        self.assertEqual(status, 200)
        self.assertIn("far_comparison", data)

    def test_not_found_endpoint(self):
        status, data = CoralGuardAPI.dispatch("GET", "/non_existent", {})
        self.assertEqual(status, 404)
        self.assertIn("error", data)

if __name__ == "__main__":
    unittest.main()
