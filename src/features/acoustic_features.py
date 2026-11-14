"""
src/features/acoustic_features.py
Extracts bioacoustic features from reef soundscape recordings:
- Log-mel spectrogram
- MFCC (mean, std)
- Spectral centroid
- Acoustic Complexity Index (ACI)
- Acoustic Diversity Index (ADI)
- Snapping-shrimp band energy (2-20 kHz)
- Fish chorus band energy (50-1000 Hz)
- Learned embedding from a small CNN autoencoder
"""

import os
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.utils import get_logger, load_config, set_seed

logger = get_logger("acoustic_features")

class SmallAudioCNNAutoencoder(nn.Module):
    """Small CNN autoencoder to learn low-dimensional acoustic embeddings."""
    def __init__(self, embedding_dim: int = 16):
        super().__init__()
        # Input shape: (Batch, 1, 64, T)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # -> (16, 32, T/2)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 16, T/4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, 32 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (32, 4, 4)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((64, 128))
        )

    def forward(self, x):
        emb = self.encoder(x)
        recon = self.decoder(emb)
        return recon, emb

    def encode(self, x):
        with torch.no_grad():
            return self.encoder(x)

def compute_aci(spec: np.ndarray) -> float:
    """
    Computes Acoustic Complexity Index (ACI) over a spectrogram matrix (Freq x Time).
    Measures variability of sound intensity without being biased by constant anthropogenic noise.
    """
    diff = np.abs(np.diff(spec, axis=1))
    sum_spec = np.sum(spec, axis=1, keepdims=True) + 1e-8
    aci_per_band = np.sum(diff / sum_spec, axis=1)
    return float(np.mean(aci_per_band))

def compute_adi(spec: np.ndarray, num_bands: int = 8) -> float:
    """
    Computes Acoustic Diversity Index (ADI) using Shannon entropy across frequency bands.
    Higher diversity reflects healthier biophony.
    """
    band_energies = np.array_split(spec, num_bands, axis=0)
    totals = np.array([np.sum(b) for b in band_energies])
    p = totals / (np.sum(totals) + 1e-8)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p + 1e-8)))

def compute_bioacoustic_index(linear_spec: np.ndarray, freqs: np.ndarray, min_freq: float = 2000.0, max_freq: float = 8000.0) -> float:
    """
    Computes Bioacoustic Index (BI) according to Boelman et al. (2007).
    Calculates the area under the dB power spectrum between 2 kHz and 8 kHz.
    """
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return 0.0
    mean_spec = np.mean(linear_spec[mask, :], axis=1)
    db = 20.0 * np.log10(mean_spec + 1e-8)
    db_normalized = db - np.min(db)
    return float(np.sum(db_normalized))

def compute_ndsi(linear_spec: np.ndarray, freqs: np.ndarray, bio_range=(2000, 11000), anthro_range=(100, 1000)) -> float:
    """
    Computes Normalized Difference Soundscape Index (NDSI) according to Kasten et al. (2012).
    NDSI = (Biophony - Anthrophony) / (Biophony + Anthrophony)
    Range: [-1.0, 1.0]. Positive values indicate biophony-dominated healthy reef acoustics.
    """
    bio_mask = (freqs >= bio_range[0]) & (freqs <= bio_range[1])
    anthro_mask = (freqs >= anthro_range[0]) & (freqs <= anthro_range[1])

    bio_energy = np.sum(linear_spec[bio_mask, :]**2)
    anthro_energy = np.sum(linear_spec[anthro_mask, :]**2)

    denom = bio_energy + anthro_energy + 1e-8
    return float((bio_energy - anthro_energy) / denom)

def extract_single_audio_features(file_path: str, sr: int = 22050, n_mels: int = 64) -> dict:
    """Extracts standard bioacoustic and ecoacoustic indices from one audio file."""
    y, file_sr = sf.read(file_path)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if y.dtype != np.float32 and y.dtype != np.float64:
        y = y.astype(np.float32) / 32768.0
    sr = file_sr
    
    # 1. Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # 2. MFCCs
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # 3. Spectral Centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = float(np.mean(cent))
    cent_std = float(np.std(cent))

    # 4. Acoustic Complexity Index (ACI) & Acoustic Diversity Index (ADI)
    linear_spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    aci_val = compute_aci(linear_spec)
    adi_val = compute_adi(linear_spec)

    # 5. Frequency Band Energies:
    # Snapping shrimp band: 2000 - 20000 Hz
    # Fish chorus band: 50 - 1000 Hz
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    total_energy = np.sum(linear_spec**2) + 1e-8

    shrimp_mask = (freqs >= 2000) & (freqs <= 20000)
    fish_mask = (freqs >= 50) & (freqs <= 1000)

    shrimp_energy_ratio = float(np.sum(linear_spec[shrimp_mask, :]**2) / total_energy)
    fish_energy_ratio = float(np.sum(linear_spec[fish_mask, :]**2) / total_energy)

    # 6. Bioacoustic Index (BI) & Normalized Difference Soundscape Index (NDSI)
    bi_val = compute_bioacoustic_index(linear_spec, freqs)
    ndsi_val = compute_ndsi(linear_spec, freqs)

    features = {
        "spectral_centroid_mean": cent_mean,
        "spectral_centroid_std": cent_std,
        "aci": aci_val,
        "adi": adi_val,
        "bioacoustic_index": bi_val,
        "ndsi": ndsi_val,
        "shrimp_band_energy_ratio": shrimp_energy_ratio,
        "fish_chorus_energy_ratio": fish_energy_ratio,
        "log_mel_fixed": cvt_fixed_mel(log_mel, target_len=128)
    }

    for i in range(13):
        features[f"mfcc_mean_{i}"] = float(mfcc_mean[i])
        features[f"mfcc_std_{i}"] = float(mfcc_std[i])

    return features

def cvt_fixed_mel(log_mel: np.ndarray, target_len: int = 128) -> np.ndarray:
    """Pad or crop log_mel to fixed (64, target_len) for CNN input."""
    if log_mel.shape[1] < target_len:
        pad_width = target_len - log_mel.shape[1]
        return np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    else:
        return log_mel[:, :target_len]

def train_and_extract_cnn_embeddings(
    mel_matrices: np.ndarray,
    embedding_dim: int = 16,
    epochs: int = 15,
    seed: int = 42
) -> np.ndarray:
    """Trains a lightweight PyTorch CNN autoencoder and returns latent embeddings."""
    set_seed(seed)
    # Shape: (N, 1, 64, 128)
    x_tensor = torch.tensor(mel_matrices, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(x_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = SmallAudioCNNAutoencoder(embedding_dim=embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    model.train()
    for ep in range(epochs):
        for batch in loader:
            x_b = batch[0]
            optimizer.zero_grad()
            recon, _ = model(x_b)
            # Match target size
            target = nn.functional.adaptive_avg_pool2d(x_b, (64, 128))
            loss = criterion(recon, target)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        embeddings = model.encode(x_tensor).numpy()

    return embeddings

def process_all_acoustics(meta_parquet: str, config: dict, output_dir: str = "data/processed") -> str:
    """Processes all raw acoustic clips into structured features + CNN embeddings."""
    set_seed(config.get("seed", 42))
    df_meta = pd.read_parquet(meta_parquet)
    logger.info(f"Extracting bioacoustic features from {len(df_meta)} clips...")

    all_feat_dicts = []
    fixed_mels = []

    for _, row in df_meta.iterrows():
        fpath = row["file_path"]
        if not os.path.exists(fpath):
            fpath = os.path.join("data/raw/audio", os.path.basename(fpath))
        
        feats = extract_single_audio_features(
            file_path=fpath,
            sr=config["acoustics"]["sample_rate"],
            n_mels=config["acoustics"]["n_mels"]
        )
        fixed_mel = feats.pop("log_mel_fixed")
        fixed_mels.append(fixed_mel)
        feats["clip_id"] = row["clip_id"]
        feats["site_id"] = row["site_id"]
        feats["date"] = row["date"]
        all_feat_dicts.append(feats)

    features_df = pd.DataFrame(all_feat_dicts)
    mels_arr = np.array(fixed_mels)

    # Train small CNN autoencoder to extract learned embeddings
    logger.info("Training small CNN autoencoder for acoustic latent embeddings...")
    embeddings = train_and_extract_cnn_embeddings(
        mel_matrices=mels_arr,
        embedding_dim=16,
        epochs=15,
        seed=config.get("seed", 42)
    )

    for dim in range(embeddings.shape[1]):
        features_df[f"acoustic_cnn_emb_{dim}"] = embeddings[:, dim]

    # Save log-mel tensors for late-fusion neural network
    tensor_path = os.path.join(output_dir, "acoustic_spectrogram_tensors.pt")
    torch.save({
        "clip_ids": features_df["clip_id"].tolist(),
        "tensors": torch.tensor(mels_arr, dtype=torch.float32).unsqueeze(1)
    }, tensor_path)

    out_parquet = os.path.join(output_dir, "acoustics_features.parquet")
    features_df.to_parquet(out_parquet, index=False)
    logger.info(f"Acoustic features ({len(features_df)} records, {features_df.shape[1]} features) saved to {out_parquet}")
    return out_parquet

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    meta_p = os.path.join(cfg["paths"]["processed_dir"], "acoustics_metadata.parquet")
    if os.path.exists(meta_p):
        process_all_acoustics(meta_p, cfg, cfg["paths"]["processed_dir"])
    else:
        logger.warning(f"Metadata parquet not found at {meta_p}. Please run ingestion first.")
