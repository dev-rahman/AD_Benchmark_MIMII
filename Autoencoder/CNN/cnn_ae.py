"""
Model 4 — Convolutional Autoencoder on log-mel spectrograms.
Deep Learning reconstruction-based anomaly detection.

Paper category : Deep Learning — Reconstruction
Input          : Log-mel spectrogram  (1 × N_MELS × T) — treated as 2-D image
Architecture   : Conv2d encoder (3 layers) → bottleneck → ConvTranspose2d decoder
Anomaly score  : Mean squared pixel reconstruction error

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python Autoencoder/CNN/cnn_ae.py [--machine fan|pump|valve] [--all]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES, AE_EPOCHS, AE_BATCH_SIZE, AE_LR, get_device
from utils.features import load_logmel_dataset
from utils.evaluate import evaluate, save_results

FIXED_LEN = 128   # number of time frames per clip


# ── Architecture ──────────────────────────────────────────────────────────────

class ConvAutoencoder(nn.Module):
    """
    2-D Convolutional Autoencoder.
    Input  : (B, 1, 128, 128) log-mel spectrogram
    Encoder: 3× Conv2d with stride-2 downsampling → latent (B, 64, 16, 16)
    Decoder: 3× ConvTranspose2d upsampling → reconstructed (B, 1, 128, 128)
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # → 16×64×64
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # → 32×32×32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # → 64×16×16
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # → 32×32×32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # → 16×64×64
            nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16,  1, kernel_size=4, stride=2, padding=1), # → 1×128×128
            nn.Tanh(),   # output in [-1, 1] — matches normalised log-mel
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise(X: np.ndarray) -> np.ndarray:
    """Scale log-mel spectrograms to [-1, 1] per dataset."""
    vmin, vmax = X.min(), X.max()
    return 2.0 * (X - vmin) / (vmax - vmin + 1e-8) - 1.0


def to_tensor(X: np.ndarray) -> torch.Tensor:
    """Add channel dimension and convert to float32 tensor."""
    return torch.from_numpy(X[:, np.newaxis, :, :].astype(np.float32))


def train_model(model: nn.Module, loader: DataLoader, device: str,
                epochs: int = AE_EPOCHS) -> None:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=AE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for (x,) in loader:
            x    = x.to(device)
            loss = criterion(model(x), x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if ep % 20 == 0 or ep == 1:
            print(f"    Epoch [{ep:3d}/{epochs}]  loss = {epoch_loss/len(loader):.6f}")


def reconstruction_error(model: nn.Module, X_tensor: torch.Tensor,
                         device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor.to(device)).cpu().numpy()
    orig = X_tensor.numpy()
    return np.mean((orig - recon) ** 2, axis=(1, 2, 3))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(machine: str) -> dict:
    device = get_device()
    print(f"
{'='*55}")
    print(f"  CNN AE | machine = {machine} | device = {device}")
    print(f"{'='*55}")

    train_dir = os.path.join(DATA_ROOT, machine, "train",  "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test",   "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test",   "anomaly")

    print("Loading log-mel spectrograms …")
    X_tr   = load_logmel_dataset(train_dir, fixed_len=FIXED_LEN)
    X_norm = load_logmel_dataset(test_norm,  fixed_len=FIXED_LEN)
    X_anom = load_logmel_dataset(test_anom,  fixed_len=FIXED_LEN)

    # Normalise together so test set uses same scale
    all_data = np.concatenate([X_tr, X_norm, X_anom], axis=0)
    vmin, vmax = all_data.min(), all_data.max()
    norm = lambda x: (2.0 * (x - vmin) / (vmax - vmin + 1e-8) - 1.0)
    X_tr   = norm(X_tr)
    X_norm = norm(X_norm)
    X_anom = norm(X_anom)

    print(f"  Train: {X_tr.shape}  |  Test normal: {X_norm.shape}  |  Test anomaly: {X_anom.shape}")

    loader = DataLoader(
        TensorDataset(to_tensor(X_tr)),
        batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=False,
    )
    model = ConvAutoencoder().to(device)
    print("Training CNN Autoencoder …")
    train_model(model, loader, device)

    X_test   = np.concatenate([X_norm, X_anom], axis=0)
    y_true   = np.array([0] * len(X_norm) + [1] * len(X_anom))
    T_test   = to_tensor(X_test)
    scores   = reconstruction_error(model, T_test, device)

    result = evaluate(y_true, scores, machine=machine, method="CNN_AE")
    save_results([result], os.path.join(RESULTS_DIR, f"cnn_ae_{machine}.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description="CNN Autoencoder")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    machines = MACHINES if args.all else [args.machine]
    results  = [run(m) for m in machines]
    if args.all:
        save_results(results, os.path.join(RESULTS_DIR, "cnn_ae_all.csv"))


if __name__ == "__main__":
    main()
