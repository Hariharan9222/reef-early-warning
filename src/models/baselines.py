"""
src/models/baselines.py
Implements baseline models:
1. NOAA DHW-threshold rule:
   - DHW < 4.0: None (0)
   - 4.0 <= DHW < 8.0: Mild (1)
   - DHW >= 8.0: Severe (2)
2. Logistic Regression (SST only)
3. XGBoost (SST only)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from src.utils import get_logger, set_seed

logger = get_logger("baselines")

class NOAADHWThresholdRule:
    """NOAA Coral Reef Watch standard DHW threshold decision rule."""
    def __init__(self, watch_thresh: float = 4.0, severe_thresh: float = 8.0):
        self.watch_thresh = watch_thresh
        self.severe_thresh = severe_thresh

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        dhw = df["dhw"].values
        preds = np.zeros(len(dhw), dtype=int)
        preds[(dhw >= self.watch_thresh) & (dhw < self.severe_thresh)] = 1
        preds[dhw >= self.severe_thresh] = 2
        return preds

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        dhw = df["dhw"].values
        probs = np.zeros((len(dhw), 3))
        # Smooth proxy probabilities based on DHW
        p_severe = 1.0 / (1.0 + np.exp(-(dhw - self.severe_thresh)))
        p_mild = np.exp(-((dhw - (self.watch_thresh + self.severe_thresh) / 2) ** 2) / 4.0)
        p_none = np.maximum(0.0, 1.0 - (p_severe + p_mild))
        
        raw = np.column_stack([p_none, p_mild, p_severe])
        sums = raw.sum(axis=1, keepdims=True) + 1e-8
        return raw / sums

def train_logistic_regression_sst(df_train: pd.DataFrame, sst_feature_cols: list, target_col: str = "target_t14", seed: int = 42):
    """Logistic Regression baseline using SST features only."""
    set_seed(seed)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[sst_feature_cols].fillna(0.0))
    y_train = df_train[target_col].values

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=seed
    )
    clf.fit(X_train, y_train)
    return clf, scaler

def train_xgboost_sst(df_train: pd.DataFrame, sst_feature_cols: list, target_col: str = "target_t14", seed: int = 42):
    """XGBoost baseline using SST features only."""
    set_seed(seed)
    X_train = df_train[sst_feature_cols].fillna(0.0).values
    y_train = df_train[target_col].values

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        objective="multi:softprob",
        num_class=3,
        random_state=seed,
        eval_metric="mlogloss"
    )
    model.fit(X_train, y_train)
    return model

