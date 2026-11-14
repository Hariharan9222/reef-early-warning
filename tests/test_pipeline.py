"""
tests/test_pipeline.py
Unit and integration test suite for CoralGuard:
1. Config validation
2. Synthetic data generation & ingestion integrity
3. Feature engineering & bioacoustic metric sanity
4. Strict temporal data leakage verification
5. Model inference & probability calibration bounds
6. Operational early warning threshold logic
"""

import os
import unittest
import numpy as np
import pandas as pd
import torch
import joblib

from src.utils import load_config, set_seed
from src.synthetic import generate_synthetic_sst, generate_synthetic_citizen_reports, generate_synthetic_acoustics
from src.features.sst_features import compute_sst_features
from src.features.acoustic_features import compute_aci, compute_adi, compute_bioacoustic_index, compute_ndsi
from src.features.citizen_features import haversine_np
from src.models.baselines import NOAADHWThresholdRule
from src.models.multimodal_nn import MultimodalLateFusionNet
from src.tracking import ExperimentTracker

class TestCoralGuardPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("config.yaml")
        set_seed(cls.config.get("seed", 42))

    def test_config_structure(self):
        """Verify essential configuration fields exist."""
        self.assertIn("regions", self.config)
        self.assertIn("dates", self.config)
        self.assertIn("sst", self.config)
        self.assertIn("acoustics", self.config)
        self.assertIn("citizen", self.config)
        self.assertIn("modeling", self.config)
        self.assertEqual(self.config["seed"], 42)

    def test_haversine_distance(self):
        """Verify haversine distance calculation is accurate."""
        # Distance between (0, 0) and (0, 1 degree) is approx 111.19 km
        d = haversine_np(0.0, 0.0, np.array([0.0]), np.array([1.0]))
        self.assertAlmostEqual(d[0], 111.19, delta=1.5)
        # Distance to self is 0
        d_self = haversine_np(-18.5, 147.2, np.array([-18.5]), np.array([147.2]))
        self.assertAlmostEqual(d_self[0], 0.0, delta=0.01)

    def test_acoustic_indices(self):
        """Verify ACI, ADI, Bioacoustic Index, and NDSI mathematical properties."""
        dummy_spec = np.random.uniform(0.1, 1.0, (64, 100))
        aci_val = compute_aci(dummy_spec)
        self.assertIsInstance(aci_val, float)
        self.assertGreater(aci_val, 0.0)

        adi_val = compute_adi(dummy_spec, num_bands=8)
        self.assertIsInstance(adi_val, float)
        self.assertGreater(adi_val, 0.0)

        dummy_freqs = np.linspace(0, 11025, 64)
        bi_val = compute_bioacoustic_index(dummy_spec, dummy_freqs)
        self.assertIsInstance(bi_val, float)
        self.assertGreaterEqual(bi_val, 0.0)

        ndsi_val = compute_ndsi(dummy_spec, dummy_freqs)
        self.assertIsInstance(ndsi_val, float)
        self.assertGreaterEqual(ndsi_val, -1.0)
        self.assertLessEqual(ndsi_val, 1.0)

    def test_sst_feature_engineering(self):
        """Verify rolling SST features are computed without NaN corruption."""
        sample_df = pd.DataFrame({
            "site_id": ["site_A"] * 60,
            "date": pd.date_range("2020-01-01", periods=60, freq="D"),
            "region": ["GBR"] * 60,
            "lat": [-18.0] * 60,
            "lon": [147.0] * 60,
            "sst": [28.0 + np.sin(i / 10.0) for i in range(60)],
            "ssta": [0.5 * np.cos(i / 10.0) for i in range(60)],
            "dhw": [float(i / 10.0) for i in range(60)]
        })
        feats = compute_sst_features(sample_df, mmm_offset=1.0)
        self.assertIn("ssta_roll_7d", feats.columns)
        self.assertIn("ssta_roll_14d", feats.columns)
        self.assertIn("ssta_roll_30d", feats.columns)
        self.assertIn("sst_max_30d", feats.columns)
        self.assertIn("days_above_threshold_30d", feats.columns)
        self.assertFalse(feats["ssta_roll_7d"].isnull().any())

    def test_no_data_leakage_in_splits(self):
        """Strict verification: Train <= 2021, Val == 2022, Test >= 2023. No temporal overlap."""
        train_path = "data/processed/train_data.parquet"
        val_path = "data/processed/val_data.parquet"
        test_path = "data/processed/test_data.parquet"

        self.assertTrue(os.path.exists(train_path), "Train parquet must exist")
        self.assertTrue(os.path.exists(val_path), "Val parquet must exist")
        self.assertTrue(os.path.exists(test_path), "Test parquet must exist")

        df_train = pd.read_parquet(train_path)
        df_val = pd.read_parquet(val_path)
        df_test = pd.read_parquet(test_path)

        max_train_date = df_train["date"].max()
        min_val_date = df_val["date"].min()
        max_val_date = df_val["date"].max()
        min_test_date = df_test["date"].min()

        self.assertLess(max_train_date, min_val_date, "Train dates must precede Val dates (no leakage)")
        self.assertLess(max_val_date, min_test_date, "Val dates must precede Test dates (no leakage)")
        self.assertLessEqual(max_train_date.year, 2021)
        self.assertEqual(min_val_date.year, 2022)
        self.assertGreaterEqual(min_test_date.year, 2023)

    def test_model_inference_and_probabilities(self):
        """Verify saved models load and produce valid probability distributions summing to 1.0."""
        model_path = "models/xgb_multimodal_t14.joblib"
        self.assertTrue(os.path.exists(model_path), "Saved model must exist")
        pack = joblib.load(model_path)
        model = pack["model"]
        features = pack["features"]

        test_df = pd.read_parquet("data/processed/test_data.parquet")
        X = test_df[features].iloc[:20].fillna(0.0)
        probs = model.predict_proba(X)

        self.assertEqual(probs.shape, (20, 3))
        self.assertTrue(np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be in [0, 1]")
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg="Probabilities must sum to 1.0")

    def test_neural_network_architecture(self):
        """Verify PyTorch late-fusion model forward pass."""
        net = MultimodalLateFusionNet(sst_dim=10, cit_dim=6, ac_dim=49, num_classes=3)
        dummy_sst = torch.randn(8, 10)
        dummy_cit = torch.randn(8, 6)
        dummy_ac = torch.randn(8, 49)

        logits = net(dummy_sst, dummy_cit, dummy_ac)
        self.assertEqual(logits.shape, (8, 3))
        probs = net.predict_proba(dummy_sst, dummy_cit, dummy_ac)
        self.assertEqual(probs.shape, (8, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_experiment_tracking(self):
        """Verify experiment tracker logs run parameters and metrics."""
        tracker = ExperimentTracker(experiment_name="test_experiment", log_dir="reports/experiments")
        tracker.start_run("test_unit_run")
        tracker.log_params({"model_type": "xgboost_test", "max_depth": 5})
        tracker.log_metrics({"macro_f1": 0.88, "severe_recall": 0.97})
        tracker.end_run()

        df_runs = tracker.get_runs_dataframe()
        self.assertFalse(df_runs.empty, "Logged runs dataframe should not be empty")
        self.assertTrue((df_runs["run_name"] == "test_unit_run").any())

    def test_noaa_baseline_threshold_rule(self):
        """Verify NOAA Coral Reef Watch decision rule produces expected classes."""
        from src.models.baselines import NOAADHWThresholdRule
        rule = NOAADHWThresholdRule(watch_thresh=4.0, severe_thresh=8.0)
        df_dummy = pd.DataFrame({"dhw": [0.0, 2.5, 4.0, 6.5, 8.0, 12.0]})
        preds = rule.predict(df_dummy)
        np.testing.assert_array_equal(preds, [0, 0, 1, 1, 2, 2])
        probs = rule.predict_proba(df_dummy)
        self.assertEqual(probs.shape, (6, 3))
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_cli_load_inference_artifacts(self):
        """Verify CLI helper loads models and valid test dataset snapshot."""
        from src.cli import load_inference_artifacts, get_alert_tier
        pack, df = load_inference_artifacts(lead_days=14)
        self.assertIn("model", pack)
        self.assertIn("features", pack)
        self.assertFalse(df.empty)
        tier_normal = get_alert_tier(p_severe=0.05, p_bleach=0.10)
        tier_alert = get_alert_tier(p_severe=0.45, p_bleach=0.85)
        self.assertIn("NORMAL", tier_normal)
        self.assertIn("ALERT", tier_alert)

    def test_cli_audio_analysis(self):
        """Verify audio CLI feature extraction on sample audio clip."""
        from src.features.acoustic_features import extract_single_audio_features
        clip_path = "data/raw/audio/clip_00001_healthy.wav"
        if os.path.exists(clip_path):
            feats = extract_single_audio_features(clip_path)
            self.assertIn("shrimp_band_energy_ratio", feats)
            self.assertIn("fish_chorus_energy_ratio", feats)
            self.assertIn("aci", feats)
            self.assertIn("bioacoustic_index", feats)
            self.assertIn("ndsi", feats)
            self.assertGreaterEqual(feats["shrimp_band_energy_ratio"], 0.0)
            self.assertLessEqual(feats["shrimp_band_energy_ratio"], 1.0)

    def test_export_executive_html_report(self):
        """Verify standalone executive HTML report generation."""
        from src.export_report import generate_html_report
        test_out = "reports/test_executive_report.html"
        try:
            out = generate_html_report(test_out)
            self.assertTrue(os.path.exists(out), "Exported HTML report must exist")
            self.assertGreater(os.path.getsize(out), 10000, "Report must contain content and styles")
        finally:
            if os.path.exists(test_out):
                os.remove(test_out)

if __name__ == "__main__":
    unittest.main()

