"""
src/models/evaluate.py
Executes Phase 6 (Evaluation) and Phase 7 (Early Warning Logic):
1. Evaluates all models on strict test set (2023-2024):
   - Macro F1, PR-AUC, ROC-AUC (Severe vs rest), Severe Recall, Brier Score
2. Generates:
   - Ablation table across modalities and baselines
   - Confusion matrices (png & json)
   - Probability calibration curves
   - SHAP feature importance summary plot
   - Error analysis across regions and seasons
   - Early warning logic (Watch, Warning, Alert) and False Alarm Rate (FAR) comparison vs NOAA DHW baseline
3. Saves all outputs to reports/
"""

import os
import json
import joblib
import warnings

os.environ["MPLCONFIGDIR"] = os.path.abspath(".cache/matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

from src.utils import get_logger, load_config, set_seed
from src.models.baselines import NOAADHWThresholdRule
from src.models.multimodal_nn import MultimodalLateFusionNet, MultimodalDataset
from torch.utils.data import DataLoader

logger = get_logger("evaluate")

def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int = 3) -> float:
    """Calculates multi-class Brier score."""
    y_onehot = np.eye(num_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray = None) -> dict:
    """Computes comprehensive suite of evaluation metrics."""
    if y_pred is None:
        y_pred = np.argmax(y_prob, axis=1)

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    prec, rec, f1s, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    severe_recall = float(rec[2])
    mild_recall = float(rec[1])

    # PR-AUC for severe class (class 2)
    y_true_sev = (y_true == 2).astype(int)
    p_sev = y_prob[:, 2]
    pr_auc_sev = float(average_precision_score(y_true_sev, p_sev))

    # ROC-AUC severe vs rest
    try:
        roc_auc_sev = float(roc_auc_score(y_true_sev, p_sev))
    except Exception:
        roc_auc_sev = 0.5

    brier = multiclass_brier_score(y_true, y_prob)

    return {
        "macro_f1": round(macro_f1, 4),
        "severe_recall": round(severe_recall, 4),
        "mild_recall": round(mild_recall, 4),
        "pr_auc_severe": round(pr_auc_sev, 4),
        "roc_auc_severe": round(roc_auc_sev, 4),
        "brier_score": round(brier, 4)
    }

def run_evaluation(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    proc_dir = cfg["paths"]["processed_dir"]
    models_dir = cfg["paths"]["models_dir"]
    rep_dir = cfg["paths"]["reports_dir"]
    os.makedirs(rep_dir, exist_ok=True)

    df_test = pd.read_parquet(os.path.join(proc_dir, "test_data.parquet"))
    logger.info(f"Loaded test dataset: {len(df_test)} samples ({cfg['dates']['test_start_year']}-{cfg['dates']['test_end_year']}).")

    results_table = []
    test_predictions_store = {}

    for lead_time in [14, 28]:
        target_col = f"target_t{lead_time}"
        y_test = df_test[target_col].values

        # 1. NOAA DHW Rule
        noaa_rule = NOAADHWThresholdRule()
        p_noaa = noaa_rule.predict_proba(df_test)
        pred_noaa = noaa_rule.predict(df_test)
        m_noaa = evaluate_predictions(y_test, p_noaa, pred_noaa)
        m_noaa.update({"model": "NOAA DHW Threshold Rule (Baseline)", "lead_time": f"t+{lead_time}d", "modality": "DHW rule"})
        results_table.append(m_noaa)
        test_predictions_store[f"noaa_t{lead_time}"] = (y_test, p_noaa, pred_noaa)

        # 2. Logistic Regression (SST)
        lr_pack = joblib.load(os.path.join(models_dir, f"lr_sst_t{lead_time}.joblib"))
        X_lr = lr_pack["scaler"].transform(df_test[lr_pack["features"]].fillna(0.0))
        p_lr = lr_pack["model"].predict_proba(X_lr)
        m_lr = evaluate_predictions(y_test, p_lr)
        m_lr.update({"model": "Logistic Regression (SST only)", "lead_time": f"t+{lead_time}d", "modality": "SST"})
        results_table.append(m_lr)

        # 3. XGBoost Ablations
        for abl_key, abl_label, mod_label in [
            ("sst_only", "XGBoost (SST only)", "SST"),
            ("sst_citizen", "XGBoost (SST + Citizen)", "SST + Citizen"),
            ("sst_acoustic", "XGBoost (SST + Acoustic)", "SST + Acoustic"),
            ("multimodal", "XGBoost (Multimodal: All 3)", "SST + Citizen + Acoustic")
        ]:
            pack = joblib.load(os.path.join(models_dir, f"xgb_{abl_key}_t{lead_time}.joblib"))
            X_abl = df_test[pack["features"]].fillna(0.0)
            p_abl = pack["model"].predict_proba(X_abl)
            m_abl = evaluate_predictions(y_test, p_abl)
            m_abl.update({"model": abl_label, "lead_time": f"t+{lead_time}d", "modality": mod_label})
            results_table.append(m_abl)
            if abl_key == "multimodal":
                test_predictions_store[f"xgb_multi_t{lead_time}"] = (y_test, p_abl, np.argmax(p_abl, axis=1))

        # 4. LightGBM Multimodal
        lgb_pack = joblib.load(os.path.join(models_dir, f"lgb_multimodal_t{lead_time}.joblib"))
        X_lgb = df_test[lgb_pack["features"]].fillna(0.0)
        p_lgb = lgb_pack["model"].predict_proba(X_lgb)
        m_lgb = evaluate_predictions(y_test, p_lgb)
        m_lgb.update({"model": "LightGBM (Multimodal)", "lead_time": f"t+{lead_time}d", "modality": "SST + Citizen + Acoustic"})
        results_table.append(m_lgb)

        # 5. Optuna Tuned XGBoost
        tuned_pack = joblib.load(os.path.join(models_dir, f"xgb_optuna_tuned_t{lead_time}.joblib"))
        X_tuned = df_test[tuned_pack["features"]].fillna(0.0)
        p_tuned = tuned_pack["model"].predict_proba(X_tuned)
        m_tuned = evaluate_predictions(y_test, p_tuned)
        m_tuned.update({"model": "XGBoost (Optuna Tuned Multimodal)", "lead_time": f"t+{lead_time}d", "modality": "SST + Citizen + Acoustic"})
        results_table.append(m_tuned)
        test_predictions_store[f"best_xgb_t{lead_time}"] = (y_test, p_tuned, np.argmax(p_tuned, axis=1), tuned_pack["model"], X_tuned)

        # 6. PyTorch Late-Fusion Neural Network
        nn_ckpt = torch.load(os.path.join(models_dir, f"multimodal_late_fusion_t{lead_time}.pt"), weights_only=False)
        meta = nn_ckpt["meta"]
        X_test_sst = meta["scaler_sst"].transform(df_test[meta["sst_cols"]].fillna(0.0))
        X_test_cit = meta["scaler_cit"].transform(df_test[meta["cit_cols"]].fillna(0.0))
        X_test_ac = meta["scaler_ac"].transform(df_test[meta["ac_cols"]].fillna(0.0))

        net = MultimodalLateFusionNet(len(meta["sst_cols"]), len(meta["cit_cols"]), len(meta["ac_cols"]), num_classes=3)
        net.load_state_dict(nn_ckpt["model_state"])
        net.eval()
        with torch.no_grad():
            t_sst = torch.tensor(X_test_sst, dtype=torch.float32)
            t_cit = torch.tensor(X_test_cit, dtype=torch.float32)
            t_ac = torch.tensor(X_test_ac, dtype=torch.float32)
            p_nn = net.predict_proba(t_sst, t_cit, t_ac)

        m_nn = evaluate_predictions(y_test, p_nn)
        m_nn.update({"model": "Multimodal Late-Fusion Neural Net (Focal Loss)", "lead_time": f"t+{lead_time}d", "modality": "SST + Citizen + Acoustic"})
        results_table.append(m_nn)
        test_predictions_store[f"nn_multi_t{lead_time}"] = (y_test, p_nn, np.argmax(p_nn, axis=1))

    # Save comprehensive ablation table
    df_results = pd.DataFrame(results_table)
    df_results = df_results[["lead_time", "model", "modality", "macro_f1", "severe_recall", "pr_auc_severe", "roc_auc_severe", "brier_score"]]
    df_results.to_csv(os.path.join(rep_dir, "ablation_metrics_table.csv"), index=False)
    logger.info("Saved ablation table to reports/ablation_metrics_table.csv")

    # Generate Figures
    # 1. Confusion Matrix Plot (Best Model at t+14)
    y_t14, p_t14, pred_t14, best_model_t14, X_t14 = test_predictions_store["best_xgb_t14"]
    cm = confusion_matrix(y_t14, pred_t14)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlOrRd", cbar=True,
                xticklabels=["None", "Mild", "Severe"],
                yticklabels=["None", "Mild", "Severe"], ax=ax)
    ax.set_title("Test Set Normalized Confusion Matrix (t+14d Best Multimodal Model)")
    ax.set_xlabel("Predicted Bleaching Severity")
    ax.set_ylabel("True Bleaching Severity")
    plt.tight_layout()
    fig.savefig(os.path.join(rep_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)

    # 2. Probability Calibration Curve (Severe Class)
    fig, ax = plt.subplots(figsize=(8, 6))
    for mod_key, label_name, color in [
        ("noaa_t14", "NOAA DHW Rule", "#7f7f7f"),
        ("xgb_multi_t14", "XGBoost Multimodal", "#1f77b4"),
        ("nn_multi_t14", "Late-Fusion NN", "#2ca02c")
    ]:
        y_true, p_pred, _ = test_predictions_store[mod_key][:3]
        prob_true, prob_pred = calibration_curve((y_true == 2).astype(int), p_pred[:, 2], n_bins=8, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o", label=label_name, color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability (Severe Class)")
    ax.set_ylabel("Fraction of Positives (Severe Class)")
    ax.set_title("Reliability Calibration Curve (14-Day Forecast)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(rep_dir, "calibration_curve.png"), dpi=300)
    plt.close(fig)

    # 3. SHAP Summary Plot
    logger.info("Computing exact Tree SHAP summary values for multimodal tree model...")
    try:
        import shap
        import xgboost as xgb
        sample_size = min(300, len(X_t14))
        X_sample = X_t14.sample(sample_size, random_state=42)
        dmat = xgb.DMatrix(X_sample)
        # Native exact Tree SHAP computation: shape is (N, num_classes, num_features + 1)
        contribs = best_model_t14.get_booster().predict(dmat, pred_contribs=True)
        # Class 2 is Severe Bleaching, exclude the last bias term column
        sv_severe = contribs[:, 2, :-1]

        fig = plt.figure(figsize=(10, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.summary_plot(sv_severe, X_sample, show=False, max_display=12)
        plt.title("SHAP Feature Importance for Predicting Severe Bleaching (t+14d)", fontsize=13, pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(rep_dir, "shap_summary.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved SHAP beeswarm summary plot to reports/shap_summary.png")
    except Exception as e:
        logger.warning(f"SHAP plot generation exception ({e}). Fallback to native feature importance.")
        fig, ax = plt.subplots(figsize=(10, 8))
        importances = pd.Series(best_model_t14.feature_importances_, index=X_t14.columns).nlargest(15)
        importances.plot(kind="barh", ax=ax, color="#1f77b4")
        ax.set_title("Top 15 Feature Importances (Multimodal XGBoost t+14d)")
        ax.set_xlabel("Gain Importance")
        plt.tight_layout()
        fig.savefig(os.path.join(rep_dir, "shap_summary.png"), dpi=300)
        plt.close(fig)

    # 4. Error Analysis by Region and Season
    logger.info("Performing Error Analysis by Region and Season...")
    df_eval = df_test.copy()
    df_eval["pred_sev"] = pred_t14
    df_eval["is_correct"] = (df_eval["target_t14"] == df_eval["pred_sev"]).astype(int)
    df_eval["is_severe_missed"] = ((df_eval["target_t14"] == 2) & (df_eval["pred_sev"] != 2)).astype(int)

    # Regional accuracy & severe recall
    regional_metrics = []
    for region, reg_group in df_eval.groupby("region"):
        reg_f1 = f1_score(reg_group["target_t14"], reg_group["pred_sev"], average="macro")
        _, rec, _, _ = precision_recall_fscore_support(reg_group["target_t14"], reg_group["pred_sev"], labels=[0, 1, 2], zero_division=0)
        regional_metrics.append({
            "region": region,
            "samples": len(reg_group),
            "macro_f1": round(reg_f1, 4),
            "severe_recall": round(rec[2], 4),
            "overall_accuracy": round(reg_group["is_correct"].mean(), 4)
        })
    df_regional = pd.DataFrame(regional_metrics)
    df_regional.to_csv(os.path.join(rep_dir, "error_analysis_region.csv"), index=False)

    # Seasonal error analysis
    df_eval["month"] = df_eval["date"].dt.month
    df_eval["season"] = df_eval["month"].apply(lambda m: "Summer Peak" if m in [1, 2, 3, 7, 8, 9] else "Cooler/Transition")
    seasonal_metrics = []
    for season, s_group in df_eval.groupby("season"):
        s_f1 = f1_score(s_group["target_t14"], s_group["pred_sev"], average="macro")
        _, rec, _, _ = precision_recall_fscore_support(s_group["target_t14"], s_group["pred_sev"], labels=[0, 1, 2], zero_division=0)
        seasonal_metrics.append({
            "season": season,
            "samples": len(s_group),
            "macro_f1": round(s_f1, 4),
            "severe_recall": round(rec[2], 4),
            "accuracy": round(s_group["is_correct"].mean(), 4)
        })
    df_seasonal = pd.DataFrame(seasonal_metrics)
    df_seasonal.to_csv(os.path.join(rep_dir, "error_analysis_season.csv"), index=False)

    # 5. Phase 7: Early Warning Logic & False Alarm Rate (FAR) vs NOAA DHW Baseline
    logger.info("Executing Phase 7: Early Warning Alert conversion & FAR benchmarking...")
    # Operational alert levels:
    # 0: Normal (No stress)
    # 1: Bleaching Watch
    # 2: Bleaching Warning
    # 3: Bleaching Alert Level (Severe Risk)
    
    # Multimodal alert rule:
    # Alert if p_severe >= 0.35 OR (p_mild + p_severe) >= 0.75
    # Warning if (p_mild + p_severe) >= 0.50
    # Watch if (p_mild + p_severe) >= 0.25
    def compute_alert_level(probs: np.ndarray) -> np.ndarray:
        p_bleach = probs[:, 1] + probs[:, 2]
        p_sev = probs[:, 2]
        alerts = np.zeros(len(probs), dtype=int)
        alerts[p_bleach >= 0.25] = 1 # Watch
        alerts[p_bleach >= 0.50] = 2 # Warning
        alerts[(p_sev >= 0.35) | (p_bleach >= 0.75)] = 3 # Alert
        return alerts

    alerts_multimodal = compute_alert_level(p_t14)
    # NOAA baseline alerts:
    # DHW >= 8 -> Alert (3), 4 <= DHW < 8 -> Warning (2), Hotspot > 0 -> Watch (1)
    dhw = df_test["dhw"].values
    alerts_noaa = np.zeros(len(dhw), dtype=int)
    alerts_noaa[df_test["ssta"] > 0] = 1
    alerts_noaa[dhw >= 4.0] = 2
    alerts_noaa[dhw >= 8.0] = 3

    # Ground truth binary severe bleaching (class 2)
    y_binary_severe = (y_t14 == 2).astype(int)

    # Alert triggers for severe: alert level 3
    pred_alert_multi = (alerts_multimodal == 3).astype(int)
    pred_alert_noaa = (alerts_noaa == 3).astype(int)

    # False Alarm Rate (FAR) = FP / (FP + TN)
    def calc_far_and_recall(y_true, y_pred_alert):
        tp = np.sum((y_true == 1) & (y_pred_alert == 1))
        fp = np.sum((y_true == 0) & (y_pred_alert == 1))
        fn = np.sum((y_true == 1) & (y_pred_alert == 0))
        tn = np.sum((y_true == 0) & (y_pred_alert == 0))
        far = fp / (fp + tn + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        return round(float(far), 4), round(float(recall), 4), round(float(precision), 4)

    multi_far, multi_rec, multi_prec = calc_far_and_recall(y_binary_severe, pred_alert_multi)
    noaa_far, noaa_rec, noaa_prec = calc_far_and_recall(y_binary_severe, pred_alert_noaa)

    far_comparison = {
        "lead_time": "14 days",
        "noaa_dhw_baseline": {
            "false_alarm_rate": noaa_far,
            "severe_recall": noaa_rec,
            "precision": noaa_prec
        },
        "multimodal_early_warning": {
            "false_alarm_rate": multi_far,
            "severe_recall": multi_rec,
            "precision": multi_prec,
            "far_reduction_percentage": round(((noaa_far - multi_far) / (noaa_far + 1e-8)) * 100.0, 2)
        }
    }

    with open(os.path.join(rep_dir, "early_warning_far_comparison.json"), "w") as f:
        json.dump(far_comparison, f, indent=2)

    logger.info(f"Phase 6 & 7 completed successfully! Saved all evaluation outputs to {rep_dir}")
    return df_results, far_comparison

if __name__ == "__main__":
    run_evaluation()
