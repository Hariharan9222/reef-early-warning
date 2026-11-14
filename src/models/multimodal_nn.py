"""
src/models/multimodal_nn.py
PyTorch Multimodal Late-Fusion Neural Network for Coral Bleaching Risk:
- Separate modality encoders for:
    1. SST thermal features
    2. Citizen diver reports
    3. Underwater acoustic soundscapes
- Concatenated fusion classification head
- Focal Loss with class-frequency balancing
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils import get_logger, set_seed

logger = get_logger("multimodal_nn")

class FocalLoss(nn.Module):
    """Multi-class Focal Loss to tackle severe class imbalance."""
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (B, C) logits, targets: (B,)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class MultimodalDataset(Dataset):
    def __init__(self, X_sst: np.ndarray, X_cit: np.ndarray, X_ac: np.ndarray, y: np.ndarray):
        self.sst = torch.tensor(X_sst, dtype=torch.float32)
        self.cit = torch.tensor(X_cit, dtype=torch.float32)
        self.ac = torch.tensor(X_ac, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.sst[idx], self.cit[idx], self.ac[idx], self.y[idx]

class MultimodalLateFusionNet(nn.Module):
    """
    Late fusion architecture:
    - SST branch: Linear(d_sst, 32) -> BatchNorm -> ReLU -> Linear(32, 16)
    - Citizen branch: Linear(d_cit, 16) -> BatchNorm -> ReLU -> Linear(16, 16)
    - Acoustic branch: Linear(d_ac, 32) -> BatchNorm -> ReLU -> Linear(32, 16)
    - Late Fusion Head: Concat(16 + 16 + 16) -> Linear(48, 32) -> Dropout(0.3) -> Linear(32, 3)
    """
    def __init__(self, sst_dim: int, cit_dim: int, ac_dim: int, num_classes: int = 3):
        super().__init__()
        self.sst_encoder = nn.Sequential(
            nn.Linear(sst_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.cit_encoder = nn.Sequential(
            nn.Linear(cit_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        self.ac_encoder = nn.Sequential(
            nn.Linear(ac_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(16 + 16 + 16, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x_sst, x_cit, x_ac):
        f_sst = self.sst_encoder(x_sst)
        f_cit = self.cit_encoder(x_cit)
        f_ac = self.ac_encoder(x_ac)

        fused = torch.cat([f_sst, f_cit, f_ac], dim=1)
        logits = self.fusion_head(fused)
        return logits

    def predict_proba(self, x_sst, x_cit, x_ac) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x_sst, x_cit, x_ac)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

def train_multimodal_nn(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    sst_cols: list,
    cit_cols: list,
    ac_cols: list,
    target_col: str = "target_t14",
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 0.001,
    seed: int = 42
):
    """Trains the late-fusion PyTorch model with Focal Loss."""
    set_seed(seed)
    
    # Preprocessing: Standard Scalers
    scaler_sst = StandardScaler()
    scaler_cit = StandardScaler()
    scaler_ac = StandardScaler()

    X_train_sst = scaler_sst.fit_transform(df_train[sst_cols].fillna(0.0))
    X_train_cit = scaler_cit.fit_transform(df_train[cit_cols].fillna(0.0))
    X_train_ac = scaler_ac.fit_transform(df_train[ac_cols].fillna(0.0))
    y_train = df_train[target_col].values

    X_val_sst = scaler_sst.transform(df_val[sst_cols].fillna(0.0))
    X_val_cit = scaler_cit.transform(df_val[cit_cols].fillna(0.0))
    X_val_ac = scaler_ac.transform(df_val[ac_cols].fillna(0.0))
    y_val = df_val[target_col].values

    train_ds = MultimodalDataset(X_train_sst, X_train_cit, X_train_ac, y_train)
    val_ds = MultimodalDataset(X_val_sst, X_val_cit, X_val_ac, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Class weights for focal loss
    class_counts = np.bincount(y_train, minlength=3)
    alpha_weights = 1.0 / (class_counts + 1e-6)
    alpha_weights = alpha_weights / alpha_weights.sum()
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32)

    model = MultimodalLateFusionNet(len(sst_cols), len(cit_cols), len(ac_cols), num_classes=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=alpha_tensor, gamma=2.0)

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for sst_b, cit_b, ac_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(sst_b, cit_b, ac_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y_b)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sst_b, cit_b, ac_b, y_b in val_loader:
                out = model(sst_b, cit_b, ac_b)
                v_loss = criterion(out, y_b)
                val_loss += v_loss.item() * len(y_b)

        train_loss /= len(train_ds)
        val_loss /= len(val_ds)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    if best_weights is not None:
        model.load_state_dict(best_weights)

    logger.info(f"Multimodal NN training complete. Best Val Focal Loss: {best_val_loss:.4f}")
    return model, {
        "scaler_sst": scaler_sst,
        "scaler_cit": scaler_cit,
        "scaler_ac": scaler_ac,
        "sst_cols": sst_cols,
        "cit_cols": cit_cols,
        "ac_cols": ac_cols
    }

