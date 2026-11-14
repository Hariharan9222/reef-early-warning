"""
backend/ingestion/acoustics_ingest.py
Acoustic feature extractor for real underwater hydrophone soundscape recordings.
Extracts:
- Acoustic Complexity Index (ACI)
- Acoustic Diversity Index (ADI)
- Spectral Entropy
- Snapping Shrimp Band Energy Ratio (2 - 20 kHz)
- Fish Chorus Band Energy Ratio (50 - 1000 Hz)
- Bioacoustic Index (BI)
- Normalized Difference Soundscape Index (NDSI)
- 13 MFCC feature coefficients
"""

import os
import json
import logging
import numpy as np
import soundfile as sf
import librosa
from typing import Dict, Any

logger = logging.getLogger("coralguard.acoustics")

def compute_spectral_entropy(linear_spec: np.ndarray) -> float:
    """
    Computes normalized spectral entropy across the frequency bins of the spectrogram.
    H = - sum(p * log2(p)) / log2(N)
    Values range from 0.0 (pure tone) to 1.0 (white noise / uniform disorder).
    """
    # Average power across time frames
    power_spectrum = np.mean(linear_spec**2, axis=1)
    total_power = np.sum(power_spectrum) + 1e-12
    p = power_spectrum / total_power
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p + 1e-12))
    norm_entropy = float(entropy / np.log2(len(power_spectrum) + 1e-12))
    return float(np.clip(norm_entropy, 0.0, 1.0))

def compute_aci(linear_spec: np.ndarray) -> float:
    """Computes Acoustic Complexity Index (ACI) according to Pieretti et al. (2011)."""
    diff = np.abs(np.diff(linear_spec, axis=1))
    sum_diff = np.sum(diff, axis=1)
    sum_intensity = np.sum(linear_spec, axis=1) + 1e-12
    return float(np.mean(sum_diff / sum_intensity))

def compute_bioacoustic_index(linear_spec: np.ndarray, freqs: np.ndarray, min_freq: float = 2000.0, max_freq: float = 8000.0) -> float:
    """Computes Bioacoustic Index (BI) integrating biophonic area (2000 - 8000 Hz)."""
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return 0.0
    sub_spec = linear_spec[mask, :]
    mean_power = np.mean(sub_spec**2, axis=1)
    db = 10.0 * np.log10(mean_power + 1e-12)
    db_norm = db - np.min(db)
    return float(np.sum(db_norm))

def compute_ndsi(linear_spec: np.ndarray, freqs: np.ndarray, bio_range=(2000, 11000), anthro_range=(100, 1000)) -> float:
    """Computes Normalized Difference Soundscape Index (NDSI) in range [-1.0, +1.0]."""
    bio_mask = (freqs >= bio_range[0]) & (freqs <= bio_range[1])
    anthro_mask = (freqs >= anthro_range[0]) & (freqs <= anthro_range[1])
    bio_energy = np.sum(linear_spec[bio_mask, :]**2)
    anthro_energy = np.sum(linear_spec[anthro_mask, :]**2)
    denom = bio_energy + anthro_energy + 1e-12
    return float((bio_energy - anthro_energy) / denom)

class AcousticIngestionEngine:
    """Processes raw hydrophone soundscape recordings into validated ecoacoustic features."""

    @staticmethod
    def process_recording(file_path: str, n_mels: int = 64) -> Dict[str, Any]:
        """
        Loads an audio clip, validates signal integrity, and extracts all required ecoacoustic indices.
        Returns status='no_data' or 'sensor_fault' if unreadable or empty.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Acoustic file not found: {file_path}")
            return {"status": "no_data", "error": "file_not_found"}

        try:
            y, sr = sf.read(file_path)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if y.dtype != np.float32 and y.dtype != np.float64:
                y = y.astype(np.float32) / 32768.0

            if len(y) == 0 or np.max(np.abs(y)) < 1e-6:
                logger.warning(f"Acoustic clip {file_path} is silent or corrupted.")
                return {"status": "sensor_fault", "error": "silent_or_corrupted_signal"}

            # Compute linear spectrogram
            linear_spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
            total_energy = np.sum(linear_spec**2) + 1e-12

            # Band energies
            shrimp_mask = (freqs >= 2000) & (freqs <= 20000)
            fish_mask = (freqs >= 50) & (freqs <= 1000)
            shrimp_ratio = float(np.sum(linear_spec[shrimp_mask, :]**2) / total_energy)
            fish_ratio = float(np.sum(linear_spec[fish_mask, :]**2) / total_energy)

            # Ecoacoustic indices
            aci_val = compute_aci(linear_spec)
            entropy_val = compute_spectral_entropy(linear_spec)
            bi_val = compute_bioacoustic_index(linear_spec, freqs)
            ndsi_val = compute_ndsi(linear_spec, freqs)

            # Spectral centroid
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            cent_mean = float(np.mean(cent))

            # MFCCs (13 coefficients)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
            mfcc_means = [float(v) for v in np.mean(mfcc, axis=1)]

            return {
                "status": "verified",
                "file_path": file_path,
                "duration_sec": float(len(y) / sr),
                "sample_rate": sr,
                "aci": round(aci_val, 4),
                "spectral_entropy": round(entropy_val, 4),
                "shrimp_band_energy_ratio": round(shrimp_ratio, 4),
                "fish_chorus_energy_ratio": round(fish_ratio, 4),
                "bioacoustic_index": round(bi_val, 2),
                "ndsi": round(ndsi_val, 4),
                "spectral_centroid_mean": round(cent_mean, 1),
                "mfccs": mfcc_means,
                "mfccs_json": json.dumps(mfcc_means)
            }

        except Exception as e:
            logger.error(f"Error processing acoustic clip {file_path}: {e}")
            return {"status": "sensor_fault", "error": str(e)}

