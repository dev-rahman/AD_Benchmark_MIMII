"""
Model 3 — Dense (Fully-Connected) Autoencoder.
Deep Learning reconstruction-based anomaly detection.

Paper category : Deep Learning — Reconstruction
Input          : Rich feature vector (87 dims)
Architecture   : Encoder 87→128→64→32  |  Decoder 32→64→128→87
Anomaly score  : Mean squared reconstruction error (higher = more anomalous)

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python Autoencoder/Dense/dense_ae.py [--machine fan|pump|valve] [--all]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES, AE_EPOCHS, AE_BATCH_SIZE, AE_LR, get_device
from utils.features import load_feature_dataset, rich_features
from utils.evaluate import evaluate, save_results


# ── Plot output directory (NEW) ────────────────────────────────────────────────
PLOT_DIR = os.path.join(os.path.dirname(__file__), "v1")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Architecture ──────────────────────────────────────────────────────────────

class DenseAutoencoder(nn.Module):
    """
    Symmetric fully-connected autoencoder.
    Encoder compresses input to a 32-dim bottleneck.
    Decoder reconstructs the original feature vector.
    """
    def __init__(self, input_dim: int, bottleneck: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 64),  nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Linear(64, 128),         nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model: nn.Module, loader: DataLoader, device: str,
                epochs: int = AE_EPOCHS, lr: float = AE_LR) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    model.train()
    history = []
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
        avg = epoch_loss / len(loader)
        history.append(avg)
        if ep % 20 == 0 or ep == 1:
            print(f"    Epoch [{ep:3d}/{epochs}]  loss = {avg:.6f}")
    return history


# ── Anomaly scoring ───────────────────────────────────────────────────────────

def reconstruction_error(model: nn.Module, X: np.ndarray, device: str) -> np.ndarray:
    """Return per-sample MSE between input and reconstruction."""
    model.eval()
    t = torch.from_numpy(X.astype(np.float32)).to(device)
    with torch.no_grad():
        recon = model(t).cpu().numpy()
    return np.mean((X.astype(np.float32) - recon) ** 2, axis=1)


# ── ROC curve plotting (NEW) ──────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, scores: np.ndarray, machine: str, save_dir: str = PLOT_DIR):
    """
    Plot and save a single binary ROC curve (normal vs anomaly)
    for one machine, using the continuous reconstruction-error scores.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"Dense AE (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Dense Autoencoder — {machine}")
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"roc_dense_ae_{machine}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved ROC plot: {save_path}")

    return roc_auc


# ── Main ──────────────────────────────────────────────────────────────────────

def run(machine: str) -> dict:
    device = get_device()
    print(f"{'='*55}")
    print(f"  Dense AE | machine = {machine} | device = {device}")
    print(f"{'='*55}")

    train_dir = os.path.join(DATA_ROOT, machine, "train",  "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test",   "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test",   "anomaly")

    print("Loading features …")
    X_tr   = load_feature_dataset(train_dir, rich_features)
    X_norm = load_feature_dataset(test_norm,  rich_features)
    X_anom = load_feature_dataset(test_anom,  rich_features)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr).astype(np.float32)
    X_norm = scaler.transform(X_norm).astype(np.float32)
    X_anom = scaler.transform(X_anom).astype(np.float32)

    print(f"  Train: {X_tr.shape}  |  Test normal: {X_norm.shape}  |  Test anomaly: {X_anom.shape}")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr)),
        batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=False,
    )
    model = DenseAutoencoder(input_dim=X_tr.shape[1]).to(device)
    print("Training Dense Autoencoder …")
    train_model(model, loader, device)

    X_test = np.vstack([X_norm, X_anom])
    y_true = np.array([0] * len(X_norm) + [1] * len(X_anom))
    scores = reconstruction_error(model, X_test, device)

    # Save ROC curve plot for this machine (NEW).
    plot_roc_curve(y_true, scores, machine)

    result = evaluate(y_true, scores, machine=machine, method="DenseAE")
    save_results([result], os.path.join(RESULTS_DIR, f"dense_ae_{machine}.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Dense Autoencoder")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    machines = MACHINES if args.all else [args.machine]
    results  = [run(m) for m in machines]
    if args.all:
        save_results(results, os.path.join(RESULTS_DIR, "dense_ae_all.csv"))


if __name__ == "__main__":
    main() 