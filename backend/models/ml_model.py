"""
backend/models/ml_model.py
XGBoost inference and management engine predicting bleaching risk 1-4 weeks ahead.
Loads serialized production model artifacts with zero mock values.
"""

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from backend.config import settings

logger = logging.getLogger("coralguard.ml")

def get_operational_alert_tier(p_severe: float, p_bleach: float) -> str:
    """Assigns standard operational alert tier according to probability thresholds."""
    if p_severe >= 0.35 or p_bleach >= settings.ALERT_THRESHOLD:
        return "🚨 BLEACHING ALERT (Severe Risk)"
    elif p_bleach >= settings.WARNING_THRESHOLD:
        return "⚠️ BLEACHING WARNING (Moderate Risk)"
    elif p_bleach >= settings.WATCH_THRESHOLD:
        return "👀 BLEACHING WATCH (Early Advisory)"
    else:
        return "✅ NORMAL (No Thermal Stress)"

class BleachingPredictor:
    """Loads trained XGBoost artifacts and computes multi-horizon risk forecasts."""

    def __init__(self, model_t14_path: str = settings.MODEL_T14_PATH, model_t28_path: str = settings.MODEL_T28_PATH):
        self.model_t14_path = model_t14_path
        self.model_t28_path = model_t28_path
        self.pack_t14 = None
        self.pack_t28 = None
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.model_t14_path):
            try:
                self.pack_t14 = joblib.load(self.model_t14_path)
                logger.info(f"Loaded t+14d XGBoost model from {self.model_t14_path}")
            except Exception as e:
                logger.error(f"Failed to load t+14d model: {e}")

        if os.path.exists(self.model_t28_path):
            try:
                self.pack_t28 = joblib.load(self.model_t28_path)
                logger.info(f"Loaded t+28d XGBoost model from {self.model_t28_path}")
            except Exception as e:
                logger.error(f"Failed to load t+28d model: {e}")

    def predict_risk(
        self,
        features_dict: Dict[str, Any],
        lead_days: int = 14
    ) -> Dict[str, Any]:
        """
        Runs inference on multi-modal feature dictionary for a given lead horizon (14 or 28 days).
        Returns class probabilities and operational alert tier.
        """
        pack = self.pack_t14 if lead_days <= 14 else self.pack_t28
        if not pack:
            raise RuntimeError(f"Trained model artifact for lead_days={lead_days} is not loaded.")

        model = pack["model"]
        feature_names = pack["features"]

        # Build feature vector matching model schema
        input_data = {}
        for feat in feature_names:
            val = features_dict.get(feat, 0.0)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                val = 0.0
            input_data[feat] = float(val)

        df_input = pd.DataFrame([input_data])[feature_names]
        probs = model.predict_proba(df_input)[0]

        p_none = float(probs[0])
        p_mild = float(probs[1])
        p_severe = float(probs[2])
        p_bleach = p_mild + p_severe

        alert_tier = get_operational_alert_tier(p_severe, p_bleach)

        return {
            "lead_days": lead_days,
            "probabilities": {
                "no_bleaching": round(p_none, 4),
                "mild_bleaching": round(p_mild, 4),
                "severe_bleaching": round(p_severe, 4),
                "total_bleaching_risk": round(p_bleach, 4)
            },
            "alert_tier": alert_tier,
            "model_version": f"xgboost_multimodal_v2_t{lead_days}d"
        }

predictor = BleachingPredictor()

