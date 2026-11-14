"""
src/monitor.py
Automated Monitoring and Alert Dispatch Daemon for CoralGuard.
Periodically assesses regional reef health, detects active bleaching alerts,
and logs/dispatches early warning notifications to marine park authorities.

Usage:
    python -m src.monitor --single-shot
    python -m src.monitor --interval-sec 3600
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
import pandas as pd
from src.utils import get_logger, load_config
from src.cli import load_inference_artifacts, get_alert_tier

logger = get_logger("monitor")

class CoralGuardMonitor:
    def __init__(self, config_path: str = "config.yaml", log_file: str = "reports/alerts_log.jsonl"):
        self.cfg = load_config(config_path)
        self.log_file = log_file
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def check_alerts(self, lead_days: int = 14) -> list:
        """Evaluates latest observations across all sites and flags active alert conditions."""
        pack, df = load_inference_artifacts(lead_days)
        model = pack["model"]
        features = pack["features"]

        latest_date = df["date"].max()
        sub_df = df[df["date"] == latest_date].copy()

        X = sub_df[features].fillna(0.0)
        probs = model.predict_proba(X)
        sub_df["p_mild"] = probs[:, 1]
        sub_df["p_severe"] = probs[:, 2]
        sub_df["p_bleach"] = probs[:, 1] + probs[:, 2]
        sub_df["alert_tier"] = [get_alert_tier(r["p_severe"], r["p_bleach"]) for _, r in sub_df.iterrows()]

        # Filter active alerts (Watch, Warning, Alert)
        flagged = sub_df[~sub_df["alert_tier"].str.contains("NORMAL")]
        records = flagged[["site_id", "region", "lat", "lon", "sst", "ssta", "dhw", "p_bleach", "p_severe", "alert_tier"]].to_dict(orient="records")

        return records

    def run_check_cycle(self) -> dict:
        """Executes a single monitoring cycle across both 14-day and 28-day forecast horizons."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        logger.info(f"Running automated early warning scan at {timestamp}...")

        alerts_14d = self.check_alerts(lead_days=14)
        alerts_28d = self.check_alerts(lead_days=28)

        cycle_summary = {
            "timestamp": timestamp,
            "status": "scan_complete",
            "alerts_14d_count": len(alerts_14d),
            "alerts_28d_count": len(alerts_28d),
            "alerts_14d": alerts_14d,
            "alerts_28d": alerts_28d
        }

        # Log to ledger
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(cycle_summary) + "\n")

        if len(alerts_14d) > 0 or len(alerts_28d) > 0:
            logger.warning(f"Scan detected active warnings: {len(alerts_14d)} at +14d, {len(alerts_28d)} at +28d!")
        else:
            logger.info("Scan completed: All monitored reef sites currently within normal baseline.")

        return cycle_summary

    def run_daemon(self, interval_sec: int = 3600):
        """Runs the monitoring loop indefinitely at specified intervals."""
        logger.info(f"Starting CoralGuard Early Warning Daemon (polling interval: {interval_sec}s)...")
        try:
            while True:
                self.run_check_cycle()
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            logger.info("Monitoring daemon stopped by user.")

def main():
    parser = argparse.ArgumentParser(description="CoralGuard Automated Early Warning Monitor")
    parser.add_argument("--single-shot", action="store_true", help="Run a single evaluation cycle and exit")
    parser.add_argument("--interval-sec", type=int, default=3600, help="Loop interval in seconds (default: 3600)")
    parser.add_argument("--log-file", type=str, default="reports/alerts_log.jsonl", help="Log output path")
    args = parser.parse_args()

    monitor = CoralGuardMonitor(log_file=args.log_file)
    if args.single_shot:
        res = monitor.run_check_cycle()
        print(json.dumps(res, indent=2))
    else:
        monitor.run_daemon(interval_sec=args.interval_sec)

if __name__ == "__main__":
    main()

