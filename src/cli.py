"""
src/cli.py
Command-Line Interface (CLI) for CoralGuard:
Allows marine scientists and reef park managers to run queries, predictions,
and early warning risk assessments directly from the terminal.

Usage:
    python -m src.cli predict --lead 14 --site great_barrier_reef_-18.00_146.00
    python -m src.cli alerts --lead 14
    python -m src.cli evaluate
    python -m src.cli info
"""

import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from src.utils import get_logger, load_config
from src.models.evaluate import run_evaluation
from src.models.tree_models import get_feature_subsets

logger = get_logger("cli")

def load_inference_artifacts(lead_days: int = 14):
    """Loads model and latest test data snapshot."""
    model_path = f"models/xgb_multimodal_t{lead_days}.joblib"
    data_path = "data/processed/test_data.parquet"

    if not os.path.exists(model_path) or not os.path.exists(data_path):
        raise FileNotFoundError("Models or test data not found. Please run 'python -m src.run_all' first.")

    pack = joblib.load(model_path)
    df = pd.read_parquet(data_path)
    return pack, df

def get_alert_tier(p_severe: float, p_bleach: float) -> str:
    """Classifies risk probabilities into standard operational alert tiers."""
    if p_severe >= 0.35 or p_bleach >= 0.75:
        return "🚨 BLEACHING ALERT (Severe Risk)"
    elif p_bleach >= 0.50:
        return "⚠️ BLEACHING WARNING (Moderate Risk)"
    elif p_bleach >= 0.25:
        return "👀 BLEACHING WATCH (Early Advisory)"
    else:
        return "✅ NORMAL (No Thermal Stress)"

def cmd_predict(args):
    pack, df = load_inference_artifacts(args.lead)
    model = pack["model"]
    features = pack["features"]

    # Filter by site if provided, else take latest snapshot of all sites
    latest_date = df["date"].max()
    sub_df = df[df["date"] == latest_date].copy()

    if args.site:
        sub_df = sub_df[sub_df["site_id"] == args.site]
        if sub_df.empty:
            print(f"Error: Site '{args.site}' not found. Available sites include:")
            print(df["site_id"].unique()[:5])
            return

    X = sub_df[features].fillna(0.0)
    probs = model.predict_proba(X)
    sub_df["p_none"] = probs[:, 0]
    sub_df["p_mild"] = probs[:, 1]
    sub_df["p_severe"] = probs[:, 2]
    sub_df["p_bleach"] = probs[:, 1] + probs[:, 2]
    sub_df["alert_tier"] = [get_alert_tier(r["p_severe"], r["p_bleach"]) for _, r in sub_df.iterrows()]

    print("\n================================================================================")
    print(f"   CORALGUARD INFERENCE REPORT: +{args.lead} DAYS FORECAST HORIZON")
    print(f"   Reference Date: {latest_date.strftime('%Y-%m-%d')}")
    print("================================================================================")

    for _, row in sub_df.iterrows():
        print(f"\nReef Site: {row['site_id']} ({row['region']})")
        print(f"  Coordinates:       {row['lat']:.2f}°, {row['lon']:.2f}°")
        print(f"  Thermal State:     SST = {row['sst']:.1f}°C | SSTA = {row['ssta']:+.2f}°C | DHW = {row['dhw']:.1f}°C-weeks")
        print(f"  Diver Reports:     {int(row['diver_reports_count_50km_30d'])} in past 30 days (avg conf: {row['diver_mean_confidence_30d']:.2f})")
        print(f"  Acoustic Survey:   {'Live Hydrophone' if row['acoustic_is_missing']==0 else 'Regional Imputation (No Hydrophone)'}")
        print(f"  Bleaching Risk:    {row['p_bleach']*100:.1f}%")
        print(f"  P(Severe Class):   {row['p_severe']*100:.1f}%")
        print(f"  Operational Tier:  {row['alert_tier']}")
    print("\n================================================================================\n")

def cmd_alerts(args):
    pack, df = load_inference_artifacts(args.lead)
    model = pack["model"]
    features = pack["features"]

    latest_date = df["date"].max()
    sub_df = df[df["date"] == latest_date].copy()

    X = sub_df[features].fillna(0.0)
    probs = model.predict_proba(X)
    sub_df["p_severe"] = probs[:, 2]
    sub_df["p_bleach"] = probs[:, 1] + probs[:, 2]
    sub_df["alert_tier"] = [get_alert_tier(r["p_severe"], r["p_bleach"]) for _, r in sub_df.iterrows()]

    # Filter out normal sites
    alert_sites = sub_df[~sub_df["alert_tier"].str.contains("NORMAL")].sort_values("p_bleach", ascending=False)

    print("\n================================================================================")
    print(f"   CORALGUARD ACTIVE BLEACHING ALERTS (+{args.lead} DAYS AHEAD)")
    print(f"   Date: {latest_date.strftime('%Y-%m-%d')} | Active Alerts: {len(alert_sites)} / {len(sub_df)}")
    print("================================================================================")

    if alert_sites.empty:
        print("No active warnings or alerts. All monitored reef sites are currently unstressed.")
    else:
        table = alert_sites[["site_id", "region", "p_bleach", "p_severe", "sst", "dhw", "alert_tier"]].copy()
        table["p_bleach"] = (table["p_bleach"] * 100).round(1).astype(str) + "%"
        table["p_severe"] = (table["p_severe"] * 100).round(1).astype(str) + "%"
        print(table.to_string(index=False))
    print("\n================================================================================\n")

def cmd_audio(args):
    from src.features.acoustic_features import extract_single_audio_features
    if not os.path.exists(args.file):
        print(f"Error: Audio file not found at '{args.file}'")
        return
    print(f"\nAnalyzing soundscape clip: {args.file} ...")
    feats = extract_single_audio_features(args.file)
    shrimp = feats["shrimp_band_energy_ratio"]
    fish = feats["fish_chorus_energy_ratio"]
    aci = feats["aci"]
    bi = feats["bioacoustic_index"]
    ndsi = feats["ndsi"]

    if shrimp >= 0.20 and ndsi > 0.2:
        verdict = "HEALTHY REEF (High biophonic activity, vibrant snapping shrimp crackle)"
    elif shrimp >= 0.10:
        verdict = "STRESSED REEF (Moderate biophonic attenuation, potential early thermal impact)"
    else:
        verdict = "DEGRADED REEF (Severe acoustic silencing, biophony collapse)"

    print("=" * 80)
    print("   CORALGUARD ACOUSTIC SOUNDSCAPE HEALTH REPORT")
    print("=" * 80)
    print(f"File Path:                {args.file}")
    print(f"Acoustic Complexity (ACI):{aci:.4f}")
    print(f"Bioacoustic Index (BI):   {bi:.4f}")
    print(f"Soundscape Index (NDSI):  {ndsi:+.4f}")
    print(f"Snapping Shrimp Ratio:    {shrimp*100:.1f}% (2 kHz - 20 kHz)")
    print(f"Fish Chorus Band Ratio:   {fish*100:.1f}% (50 Hz - 1 kHz)")
    print(f"Spectral Centroid Mean:   {feats['spectral_centroid_mean']:.1f} Hz")
    print("-" * 80)
    print(f"Ecological Verdict:       {verdict}")
    print("=" * 80 + "\n")

def cmd_evaluate(args):
    print("Running evaluation suite...")
    results_df, far_dict = run_evaluation("config.yaml")
    print("\n=== Test Holdout Benchmark Results ===")
    print(results_df.to_string(index=False))
    print("\n=== False Alarm Rate (FAR) Benchmark vs NOAA Baseline ===")
    print(json.dumps(far_dict, indent=2))

def cmd_export(args):
    from src.export_report import generate_html_report
    out = generate_html_report(args.output)
    print(f"\nSuccessfully exported standalone executive report to: {out}\n")

def cmd_monitor(args):
    from src.monitor import CoralGuardMonitor
    monitor = CoralGuardMonitor()
    if args.single_shot:
        res = monitor.run_check_cycle()
        print(json.dumps(res, indent=2))
    else:
        monitor.run_daemon(interval_sec=args.interval_sec)

def cmd_info(args):
    cfg = load_config("config.yaml")
    print("\n================================================================================")
    print("   CORALGUARD SYSTEM CONFIGURATION & DIRECTORY AUDIT")
    print("================================================================================")
    print(f"Random Seed:       {cfg.get('seed')}")
    print(f"Target Regions:    {list(cfg.get('regions', {}).keys())}")
    print(f"Date Span:         {cfg['dates']['start_year']} to {cfg['dates']['end_year']}")
    print(f"Temporal Split:    Train <= {cfg['dates']['train_end_year']} | Val = {cfg['dates']['val_year']} | Test = {cfg['dates']['test_start_year']}-{cfg['dates']['test_end_year']}")
    print(f"Models Directory:  {cfg['paths']['models_dir']}")
    print(f"Reports Directory: {cfg['paths']['reports_dir']}")
    print("================================================================================\n")

def main():
    parser = argparse.ArgumentParser(description="CoralGuard: Multimodal Early Warning CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict bleaching risk for sites")
    p_pred.add_argument("--lead", type=int, default=14, choices=[14, 28], help="Lead time in days")
    p_pred.add_argument("--site", type=str, default=None, help="Optional site ID filter")

    # alerts
    p_alerts = subparsers.add_parser("alerts", help="List all sites currently under Watch, Warning, or Alert")
    p_alerts.add_argument("--lead", type=int, default=14, choices=[14, 28], help="Lead time in days")

    # evaluate
    subparsers.add_parser("evaluate", help="Re-evaluate all models on test set")

    # audio
    p_audio = subparsers.add_parser("audio", help="Analyze underwater audio soundscape file")
    p_audio.add_argument("--file", type=str, required=True, help="Path to WAV audio file")

    # export
    p_exp = subparsers.add_parser("export", help="Export standalone HTML executive report")
    p_exp.add_argument("--output", type=str, default="reports/executive_report.html", help="Output HTML path")

    # monitor
    p_mon = subparsers.add_parser("monitor", help="Run automated early warning monitoring daemon")
    p_mon.add_argument("--single-shot", action="store_true", help="Run a single scan cycle and exit")
    p_mon.add_argument("--interval-sec", type=int, default=3600, help="Loop interval in seconds")

    # info
    subparsers.add_parser("info", help="Display system configuration")

    args = parser.parse_args()
    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "alerts":
        cmd_alerts(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "audio":
        cmd_audio(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

