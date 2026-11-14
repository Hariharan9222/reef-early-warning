"""
tests/test_backend_features.py
Unit tests for backend feature engineering and ingestion modules:
- Ecoacoustic feature extraction (Spectral Entropy, ACI, Bioacoustic Index, NDSI)
- Citizen diver trust score computation and spatial matching
- Multi-modal weekly feature aggregation pipeline
"""

import unittest
import numpy as np
from datetime import date
from backend.ingestion.acoustics_ingest import (
    compute_spectral_entropy, compute_aci, compute_bioacoustic_index, compute_ndsi
)
from backend.ingestion.citizen_ingest import (
    compute_trust_score, haversine_distance_km
)
from backend.models.schemas import CitizenReportCreate
from backend.database import SessionLocal
from backend.features.feature_pipeline import FeatureEngineeringPipeline

class TestEcoacousticFeatures(unittest.TestCase):
    def test_spectral_entropy_pure_tone_vs_noise(self):
        # Pure single frequency tone: power concentrated in one bin
        pure_spec = np.zeros((100, 20))
        pure_spec[10, :] = 10.0
        entropy_pure = compute_spectral_entropy(pure_spec)
        self.assertLess(entropy_pure, 0.05, "Pure tone should have near-zero spectral entropy.")

        # Uniform white noise: power equally distributed across all bins
        noise_spec = np.ones((100, 20)) + np.random.uniform(0.99, 1.01, (100, 20))
        entropy_noise = compute_spectral_entropy(noise_spec)
        self.assertGreater(entropy_noise, 0.95, "Uniform noise should have near-maximum spectral entropy (~1.0).")

    def test_aci_computation(self):
        # Constant signal (no temporal variability) -> ACI = 0
        const_spec = np.ones((50, 10)) * 5.0
        aci_const = compute_aci(const_spec)
        self.assertEqual(aci_const, 0.0, "Constant signal must have zero ACI.")

        # Modulated signal -> ACI > 0
        mod_spec = np.array([[1.0, 5.0, 1.0, 5.0, 1.0]] * 20)
        aci_mod = compute_aci(mod_spec)
        self.assertGreater(aci_mod, 0.0, "Time-varying signal must have positive ACI.")

    def test_ndsi_bounded(self):
        freqs = np.linspace(0, 12000, 120)
        spec = np.random.uniform(0.1, 1.0, (120, 10))
        ndsi = compute_ndsi(spec, freqs)
        self.assertTrue(-1.0 <= ndsi <= 1.0, "NDSI must be strictly within [-1.0, 1.0].")

class TestCitizenDiverTrustScore(unittest.TestCase):
    def test_trusted_observer_high_score(self):
        report = CitizenReportCreate(
            latitude=-14.668,
            longitude=145.460,
            observed_at=date.today(),
            observer_id="dr_jane_marine_bio",
            observer_type="marine_biologist",
            survey_program="Reef Check",
            depth_meters=12.0,
            bleaching_percent=35.0,
            has_photo=True,
            notes="Extensive branching Acropora bleaching observed."
        )
        score = compute_trust_score(report)
        self.assertGreaterEqual(score, 0.85, "Marine biologist with photo and Reef Check program should score >= 0.85")

    def test_unverified_observer_low_score(self):
        report = CitizenReportCreate(
            latitude=-14.668,
            longitude=145.460,
            observed_at=date.today(),
            observer_id="tourist_anon",
            observer_type="recreational_diver",
            survey_program="Direct Submission",
            depth_meters=45.0,  # Deep depth beyond optimal coral photosynthetic zone
            bleaching_percent=90.0,
            has_photo=False,
            notes=""
        )
        score = compute_trust_score(report)
        self.assertLess(score, 0.55, "Unverified diver at 45m depth with no photo should score low")

    def test_haversine_distance(self):
        # Lizard Island (-14.668, 145.460) to Ribbon Reef (-15.350, 145.800)
        dist = haversine_distance_km(-14.668, 145.460, -15.350, 145.800)
        self.assertTrue(70.0 < dist < 110.0, f"Distance should be ~85km, got {dist:.2f}km")

class TestFeaturePipeline(unittest.TestCase):
    def setUp(self):
        self.db = SessionLocal()

    def tearDown(self):
        self.db.close()

    def test_weekly_features_generation(self):
        features = FeatureEngineeringPipeline.compute_site_weekly_features(self.db, "gbr_lizard_island")
        self.assertIn("sst", features)
        self.assertIn("ssta", features)
        self.assertIn("dhw", features)
        self.assertIn("ssta_roll_7d", features)
        self.assertIn("acoustic_is_missing", features)
        self.assertIn("diver_has_reports", features)
        self.assertIn(features["acoustic_is_missing"], (0, 1))

if __name__ == "__main__":
    unittest.main()
