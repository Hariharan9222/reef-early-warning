"""
src/server.py
Lightweight standard-library REST API for CoralGuard.
Provides programmatic HTTP endpoints for external integration, webhooks, and IoT hydrophone feeds.

Endpoints:
    GET  /health
    GET  /sites
    GET  /alerts?lead=14
    GET  /predict?site=<site_id>&lead=14
    GET  /metrics
    POST /predict
"""

import os
import sys
import json
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np
import pandas as pd
from src.utils import get_logger, load_config
from src.cli import load_inference_artifacts, get_alert_tier

logger = get_logger("server")

class CoralGuardAPI:
    """Core REST API business logic decoupled from HTTP transport."""

    @staticmethod
    def get_health() -> dict:
        return {
            "status": "healthy",
            "service": "CoralGuard Multimodal Early Warning API",
            "version": "1.0.0",
            "regions_monitored": ["Great Barrier Reef", "Caribbean", "Coral Triangle"]
        }

    @staticmethod
    def get_sites() -> dict:
        _, df = load_inference_artifacts(14)
        latest_date = df["date"].max()
        sites_df = df[df["date"] == latest_date][["site_id", "region", "lat", "lon"]].drop_duplicates()
        records = sites_df.to_dict(orient="records")
        return {
            "count": len(records),
            "sites": records
        }

    @staticmethod
    def get_alerts(lead: int = 14) -> dict:
        if lead not in [14, 28]:
            lead = 14
        pack, df = load_inference_artifacts(lead)
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

        alert_sites = sub_df[~sub_df["alert_tier"].str.contains("NORMAL")]
        records = alert_sites[["site_id", "region", "lat", "lon", "sst", "dhw", "p_bleach", "p_severe", "alert_tier"]].to_dict(orient="records")

        return {
            "lead_days": lead,
            "reference_date": latest_date.strftime("%Y-%m-%d"),
            "total_monitored": len(sub_df),
            "active_alerts_count": len(records),
            "alerts": records
        }

    @staticmethod
    def predict_site(site_id: str = None, lead: int = 14) -> dict:
        if lead not in [14, 28]:
            lead = 14

        pack, df = load_inference_artifacts(lead)
        model = pack["model"]
        features = pack["features"]

        latest_date = df["date"].max()
        sub_df = df[df["date"] == latest_date].copy()

        if site_id:
            sub_df = sub_df[sub_df["site_id"] == site_id]
            if sub_df.empty:
                raise KeyError(f"Site '{site_id}' not found")

        X = sub_df[features].fillna(0.0)
        probs = model.predict_proba(X)

        results = []
        for idx, (_, row) in enumerate(sub_df.iterrows()):
            p_none = float(probs[idx, 0])
            p_mild = float(probs[idx, 1])
            p_severe = float(probs[idx, 2])
            p_bleach = p_mild + p_severe
            tier = get_alert_tier(p_severe, p_bleach)
            results.append({
                "site_id": row["site_id"],
                "region": row["region"],
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "sst": float(row["sst"]),
                "dhw": float(row["dhw"]),
                "probabilities": {
                    "no_bleaching": round(p_none, 4),
                    "mild_bleaching": round(p_mild, 4),
                    "severe_bleaching": round(p_severe, 4),
                    "total_bleaching_risk": round(p_bleach, 4)
                },
                "alert_tier": tier
            })

        return {
            "lead_days": lead,
            "reference_date": latest_date.strftime("%Y-%m-%d"),
            "results": results[0] if site_id else results
        }

    @staticmethod
    def predict_custom(data: dict) -> dict:
        lead = int(data.get("lead", 14))
        pack, _ = load_inference_artifacts(lead)
        model = pack["model"]
        features = pack["features"]

        input_dict = {f: 0.0 for f in features}
        for k, v in data.get("features", {}).items():
            if k in input_dict:
                input_dict[k] = float(v)

        X_in = pd.DataFrame([input_dict])[features]
        probs = model.predict_proba(X_in)[0]

        p_none, p_mild, p_severe = float(probs[0]), float(probs[1]), float(probs[2])
        p_bleach = p_mild + p_severe
        tier = get_alert_tier(p_severe, p_bleach)

        return {
            "lead_days": lead,
            "probabilities": {
                "no_bleaching": round(p_none, 4),
                "mild_bleaching": round(p_mild, 4),
                "severe_bleaching": round(p_severe, 4),
                "total_bleaching_risk": round(p_bleach, 4)
            },
            "alert_tier": tier
        }

    @staticmethod
    def get_metrics() -> dict:
        far_path = "reports/early_warning_far_comparison.json"
        reg_path = "reports/regional_seasonal_breakdown.json"
        table_path = "reports/ablation_metrics_table.csv"

        payload = {}
        if os.path.exists(far_path):
            with open(far_path, "r") as f:
                payload["far_comparison"] = json.load(f)
        if os.path.exists(reg_path):
            with open(reg_path, "r") as f:
                payload["regional_breakdown"] = json.load(f)
        if os.path.exists(table_path):
            df_tab = pd.read_csv(table_path)
            payload["ablation_summary"] = df_tab.to_dict(orient="records")
        return payload

    @classmethod
    def dispatch(cls, method: str, path: str, query: dict, body: dict = None) -> tuple:
        """Dispatches an incoming request to appropriate API function."""
        try:
            if method == "GET":
                if path == "/health":
                    return 200, cls.get_health()
                elif path == "/sites":
                    return 200, cls.get_sites()
                elif path == "/alerts":
                    lead = int(query.get("lead", [14])[0])
                    return 200, cls.get_alerts(lead=lead)
                elif path == "/predict":
                    site_id = query.get("site", [None])[0]
                    lead = int(query.get("lead", [14])[0])
                    return 200, cls.predict_site(site_id=site_id, lead=lead)
                elif path == "/metrics":
                    return 200, cls.get_metrics()
                else:
                    return 404, {"error": f"Endpoint '{path}' not found"}
            elif method == "POST":
                if path == "/predict":
                    return 200, cls.predict_custom(body or {})
                else:
                    return 404, {"error": f"Endpoint '{path}' not found"}
            else:
                return 405, {"error": f"Method '{method}' not allowed"}
        except KeyError as e:
            return 404, {"error": str(e)}
        except Exception as e:
            return 500, {"error": str(e)}


class CoralGuardRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler implementing CoralGuard REST API endpoints."""

    def _set_headers(self, status: int = 200, content_type: str = "application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(204)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        status, payload = CoralGuardAPI.dispatch("GET", parsed.path, query)
        self._set_headers(status)
        self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        content_length = int(self.headers.get("Content-Length", 0))
        body = {}
        if content_length > 0:
            raw_body = self.rfile.read(content_length)
            try:
                body = json.loads(raw_body.decode("utf-8"))
            except Exception:
                body = {}
        status, payload = CoralGuardAPI.dispatch("POST", parsed.path, query, body)
        self._set_headers(status)
        self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

    def log_message(self, format, *args):
        return

def run_server(port: int = 8000):
    server_address = ("", port)
    httpd = HTTPServer(server_address, CoralGuardRequestHandler)
    logger.info(f"Starting CoralGuard REST API server on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Stopping CoralGuard REST API server...")
        httpd.server_close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run CoralGuard REST API Server")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    args = parser.parse_args()
    run_server(args.port)
