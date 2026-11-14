"""
src/tracking.py
Experiment Tracking Abstraction:
Gracefully integrates MLflow tracking if installed; otherwise falls back to a
structured local JSONL experiment ledger in reports/experiments/runs.jsonl.
"""

import os
import json
import time
from datetime import datetime
from src.utils import get_logger

logger = get_logger("tracking")

class ExperimentTracker:
    def __init__(self, experiment_name: str = "coral_guard_bleaching", log_dir: str = "reports/experiments"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.runs_file = os.path.join(log_dir, "runs.jsonl")
        
        self.has_mlflow = False
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_experiment(experiment_name)
            self.has_mlflow = True
            logger.info(f"MLflow experiment tracker initialized: {experiment_name}")
        except (ImportError, Exception):
            logger.info("MLflow not installed/active. Utilizing local structured experiment ledger.")

        self.active_run = None

    def start_run(self, run_name: str):
        self.active_run = {
            "run_id": f"run_{int(time.time() * 1000)}",
            "run_name": run_name,
            "experiment": self.experiment_name,
            "start_time": datetime.utcnow().isoformat() + "Z",
            "params": {},
            "metrics": {},
            "artifacts": []
        }
        if self.has_mlflow:
            self.mlflow.start_run(run_name=run_name)
        return self

    def log_params(self, params: dict):
        if self.active_run:
            self.active_run["params"].update(params)
        if self.has_mlflow:
            for k, v in params.items():
                self.mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict):
        if self.active_run:
            self.active_run["metrics"].update(metrics)
        if self.has_mlflow:
            for k, v in metrics.items():
                self.mlflow.log_metric(k, float(v))

    def log_artifact(self, file_path: str):
        if self.active_run and os.path.exists(file_path):
            self.active_run["artifacts"].append(file_path)
        if self.has_mlflow and os.path.exists(file_path):
            self.mlflow.log_artifact(file_path)

    def end_run(self):
        if self.active_run:
            self.active_run["end_time"] = datetime.utcnow().isoformat() + "Z"
            with open(self.runs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.active_run) + "\n")
            logger.info(f"Logged experiment run '{self.active_run['run_name']}' to {self.runs_file}")
            self.active_run = None
        if self.has_mlflow:
            self.mlflow.end_run()

    def get_runs_dataframe(self):
        """Loads logged runs as a pandas DataFrame."""
        import pandas as pd
        if not os.path.exists(self.runs_file):
            return pd.DataFrame()
        records = []
        with open(self.runs_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return pd.json_normalize(records)

