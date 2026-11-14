"""
src/models/tree_models.py
Gradient Boosted Tree models (XGBoost & LightGBM):
- Feature subset selections for ablation studies
- Class imbalance handling via compute_sample_weight
- Optuna hyperparameter optimization (30 trials)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import optuna
from src.utils import get_logger, load_config, set_seed

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = get_logger("tree_models")

def get_feature_subsets(df: pd.DataFrame):
    """Identifies and partitions available columns into distinct modality feature subsets."""
    sst_cols = [
        "sst", "ssta", "dhw", "ssta_roll_7d", "ssta_roll_14d", "ssta_roll_30d",
        "sst_max_30d", "sst_rate_of_change_7d", "days_above_threshold_30d", "days_above_threshold_90d"
    ]
    sst_cols = [c for c in sst_cols if c in df.columns]

    citizen_cols = [
        "diver_reports_count_50km_30d", "diver_report_density", "diver_weighted_severity_30d",
        "diver_mean_confidence_30d", "diver_spatial_neighbor_severity_knn", "diver_has_reports"
    ]
    citizen_cols = [c for c in citizen_cols if c in df.columns]

    acoustic_cols = [
        c for c in df.columns if c.startswith("acoustic_") or c in [
            "spectral_centroid_mean", "spectral_centroid_std", "aci", "adi",
            "bioacoustic_index", "ndsi",
            "shrimp_band_energy_ratio", "fish_chorus_energy_ratio"
        ] or c.startswith("mfcc_")
    ]

    modalities = {
        "sst_only": sst_cols,
        "sst_citizen": sst_cols + citizen_cols,
        "sst_acoustic": sst_cols + acoustic_cols,
        "multimodal": sst_cols + citizen_cols + acoustic_cols
    }
    return modalities

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    params: dict = None,
    seed: int = 42
):
    """Trains an XGBoost multi-class classifier with balanced sample weights."""
    set_seed(seed)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    default_params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softprob",
        "num_class": 3,
        "random_state": seed,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "n_jobs": 2
    }
    if params:
        default_params.update(params)

    model = xgb.XGBClassifier(**default_params)
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

    return model

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    params: dict = None,
    seed: int = 42
):
    """Trains a LightGBM multi-class classifier with balanced class weights."""
    set_seed(seed)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    default_params = {
        "n_estimators": 150,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multiclass",
        "num_class": 3,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": 2
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
        )
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)

    return model

def tune_xgboost_optuna(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_trials: int = 30,
    seed: int = 42
) -> dict:
    """Uses Optuna to optimize XGBoost hyperparameters maximizing Validation Macro F1."""
    from sklearn.metrics import f1_score
    set_seed(seed)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 80, 250),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0.0, 2.0),
            "objective": "multi:softprob",
            "num_class": 3,
            "random_state": seed,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": 2
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
        val_preds = clf.predict(X_val)
        # Maximize macro F1 score
        return f1_score(y_val, val_preds, average="macro")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    logger.info(f"Optuna tuning completed ({n_trials} trials). Best Macro F1: {study.best_value:.4f}")
    return study.best_params

