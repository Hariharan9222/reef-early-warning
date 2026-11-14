"""
src/models/train_all.py
Orchestrates Phase 5 Modeling:
- Trains baselines (NOAA DHW rule, Logistic Regression, XGBoost SST only)
- Trains ablation models (SST only, SST+citizen, SST+acoustic, all 3 modalities)
- Runs Optuna tuning (30 trials) for XGBoost
- Trains PyTorch Multimodal Late-Fusion model with Focal Loss
- Trains t+14 and t+28 variants
- Saves all trained models to models/
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd
from src.utils import get_logger, load_config, set_seed
from src.tracking import ExperimentTracker
from src.models.baselines import NOAADHWThresholdRule, train_logistic_regression_sst
from src.models.tree_models import get_feature_subsets, train_xgboost, train_lightgbm, tune_xgboost_optuna
from src.models.multimodal_nn import train_multimodal_nn

logger = get_logger("train_all")

def train_all_models(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    models_dir = cfg["paths"]["models_dir"]
    proc_dir = cfg["paths"]["processed_dir"]
    os.makedirs(models_dir, exist_ok=True)

    logger.info("Loading processed train, val, and test datasets...")
    df_train = pd.read_parquet(os.path.join(proc_dir, "train_data.parquet"))
    df_val = pd.read_parquet(os.path.join(proc_dir, "val_data.parquet"))
    df_test = pd.read_parquet(os.path.join(proc_dir, "test_data.parquet"))

    modality_features = get_feature_subsets(df_train)
    sst_cols = modality_features["sst_only"]
    cit_cols = [c for c in modality_features["sst_citizen"] if c not in sst_cols]
    ac_cols = [c for c in modality_features["multimodal"] if c not in sst_cols and c not in cit_cols]

    logger.info(f"Feature counts: SST={len(sst_cols)}, Citizen={len(cit_cols)}, Acoustic={len(ac_cols)}, Total={len(modality_features['multimodal'])}")

    tracker = ExperimentTracker(experiment_name="coral_guard_multimodal")
    trained_artifacts = {}

    for lead_time, target_col in [(14, "target_t14"), (28, "target_t28")]:
        logger.info(f"=== Training Models for Target: {target_col} (Lead time: {lead_time} days) ===")
        y_train = df_train[target_col].values
        y_val = df_val[target_col].values
        y_test = df_test[target_col].values

        # 1. Baseline: NOAA DHW Rule
        noaa_rule = NOAADHWThresholdRule()
        trained_artifacts[f"noaa_rule_t{lead_time}"] = noaa_rule

        # 2. Baseline: Logistic Regression (SST only)
        lr_model, lr_scaler = train_logistic_regression_sst(df_train, sst_cols, target_col=target_col, seed=cfg["seed"])
        joblib.dump({"model": lr_model, "scaler": lr_scaler, "features": sst_cols}, os.path.join(models_dir, f"lr_sst_t{lead_time}.joblib"))
        trained_artifacts[f"lr_sst_t{lead_time}"] = (lr_model, lr_scaler)

        # 3. Ablation Runs using XGBoost
        for abl_name, feat_cols in modality_features.items():
            logger.info(f"Training XGBoost ablation: {abl_name} (features: {len(feat_cols)})...")
            X_tr = df_train[feat_cols].fillna(0.0)
            X_va = df_val[feat_cols].fillna(0.0)

            xgb_abl = train_xgboost(X_tr, y_train, X_va, y_val, params=cfg["modeling"]["xgboost"], seed=cfg["seed"])
            joblib.dump({"model": xgb_abl, "features": feat_cols}, os.path.join(models_dir, f"xgb_{abl_name}_t{lead_time}.joblib"))
            trained_artifacts[f"xgb_{abl_name}_t{lead_time}"] = xgb_abl

        # 4. LightGBM Multimodal
        logger.info("Training LightGBM on multimodal features...")
        X_tr_multi = df_train[modality_features["multimodal"]].fillna(0.0)
        X_va_multi = df_val[modality_features["multimodal"]].fillna(0.0)
        lgb_model = train_lightgbm(X_tr_multi, y_train, X_va_multi, y_val, params=cfg["modeling"]["lightgbm"], seed=cfg["seed"])
        joblib.dump({"model": lgb_model, "features": modality_features["multimodal"]}, os.path.join(models_dir, f"lgb_multimodal_t{lead_time}.joblib"))
        trained_artifacts[f"lgb_multimodal_t{lead_time}"] = lgb_model

        # 5. Optuna Tuning on Multimodal XGBoost (30 trials)
        logger.info(f"Running Optuna tuning (30 trials) for XGBoost (lead time {lead_time}d)...")
        best_params = tune_xgboost_optuna(
            X_train=X_tr_multi,
            y_train=y_train,
            X_val=X_va_multi,
            y_val=y_val,
            n_trials=cfg["modeling"]["optuna"]["n_trials"],
            seed=cfg["seed"]
        )
        xgb_tuned = train_xgboost(X_tr_multi, y_train, X_va_multi, y_val, params=best_params, seed=cfg["seed"])
        joblib.dump(
            {"model": xgb_tuned, "features": modality_features["multimodal"], "best_params": best_params},
            os.path.join(models_dir, f"xgb_optuna_tuned_t{lead_time}.joblib")
        )
        trained_artifacts[f"xgb_optuna_tuned_t{lead_time}"] = xgb_tuned

        # 6. PyTorch Multimodal Late-Fusion Neural Network
        logger.info("Training PyTorch Late-Fusion Neural Network with Focal Loss...")
        nn_model, nn_meta = train_multimodal_nn(
            df_train=df_train,
            df_val=df_val,
            sst_cols=sst_cols,
            cit_cols=cit_cols,
            ac_cols=ac_cols,
            target_col=target_col,
            epochs=cfg["modeling"]["neural_network"]["epochs"],
            batch_size=cfg["modeling"]["neural_network"]["batch_size"],
            lr=cfg["modeling"]["neural_network"]["learning_rate"],
            seed=cfg["seed"]
        )
        torch.save({
            "model_state": nn_model.state_dict(),
            "meta": nn_meta
        }, os.path.join(models_dir, f"multimodal_late_fusion_t{lead_time}.pt"))
        trained_artifacts[f"nn_multimodal_t{lead_time}"] = (nn_model, nn_meta)

        # Log training run metadata
        tracker.start_run(f"train_cycle_t{lead_time}d")
        tracker.log_params({"lead_time": lead_time, "target_col": target_col, "n_features": len(modality_features['multimodal'])})
        tracker.log_params(best_params)
        tracker.log_artifact(os.path.join(models_dir, f"xgb_optuna_tuned_t{lead_time}.joblib"))
        tracker.end_run()

    logger.info("Phase 5: All models and ablations trained and saved successfully to models/")
    return trained_artifacts

if __name__ == "__main__":
    train_all_models()

