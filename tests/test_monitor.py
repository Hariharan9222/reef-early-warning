"""
tests/test_monitor.py
Unit tests for the CoralGuard automated monitoring and alert dispatch daemon (src/monitor.py).
"""

import os
import json
import unittest
from src.monitor import CoralGuardMonitor

class TestCoralGuardMonitor(unittest.TestCase):
    def setUp(self):
        self.test_log = "reports/test_alerts_log.jsonl"
        self.monitor = CoralGuardMonitor(log_file=self.test_log)

    def tearDown(self):
        if os.path.exists(self.test_log):
            os.remove(self.test_log)

    def test_check_alerts_format(self):
        alerts = self.monitor.check_alerts(lead_days=14)
        self.assertIsInstance(alerts, list)
        if alerts:
            first = alerts[0]
            self.assertIn("site_id", first)
            self.assertIn("alert_tier", first)
            self.assertIn("p_bleach", first)
            self.assertIn("p_severe", first)

    def test_run_check_cycle_logging(self):
        res = self.monitor.run_check_cycle()
        self.assertEqual(res.get("status"), "scan_complete")
        self.assertIn("alerts_14d_count", res)
        self.assertIn("alerts_28d_count", res)
        self.assertTrue(os.path.exists(self.test_log), "Alert log file must be created")

        with open(self.test_log, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        logged = json.loads(lines[0])
        self.assertEqual(logged.get("status"), "scan_complete")

if __name__ == "__main__":
    unittest.main()

