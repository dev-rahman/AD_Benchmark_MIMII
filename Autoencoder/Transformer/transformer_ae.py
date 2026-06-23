"""
Model 5 — Transformer Autoencoder on log-mel spectrograms.
Deep Learning reconstruction-based anomaly detection.

Paper category : Deep Learning — Reconstruction
Input          : Log-mel spectrogram flattened to sequence of frequency bins
                 Each time frame is one token → shape (T, N_MELS) = (128, 128)
Architecture   : Linear projection → Transformer encoder (4 heads, 2 layers)
                 → Transformer decoder → linear projection back to N_MELS
Anomaly score  : Mean squared token reconstruction error

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python Autoencoder/Transformer/transformer_ae.py [--machine fan|pump|valve] [--all]
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES, AE_EPOCHS, AE_BATCH_SIZE, AE_LR, get_device
from utils.features import load_logmel_dataset
from utils.evaluate import evaluate, save_results

FIXED_LEN  = 128   # T — number of time frames (= sequence length)
N_MELS     = 128   # frequency bins (= token dimension)
D_MODEL    = 128   # transformer hidden size
NHEAD      = 4
NUM_LAYERS = 2
DIM_FF     = 256
DROPOUT    = 0.1


# ── Positional Encoding ───────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── Architecture ──────────────────────────────────────────────────────────────

class TransformerAutoencoder(nn.Module):
    """
    Sequence-to-sequence Transformer Autoencoder.

    The log-mel spectrogram (N_MELS × T) is treated as a sequence of T tokens,
    each of dimension N_MELS. The Transformer encoder compresses contextual
    information; the decoder reconstructs each token from this context.

    An anomaly results in high reconstruction error because the model
    has only learned the manifold of normal sounds.
    """
    def __init__(self, token_dim: int = N_MELS, d_model: int = D_MODEL,
                 nhead: int = NHEAD, num_layers: int = NUM_LAYERS,
                 dim_ff: int = DIM_FF, dropout: float = DROPOUT):
        super().__init__()
        self.input_proj  = nn.Linear(token_dim, d_model)
        self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True
        )
        self.decoder     = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, token_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, N_MELS)
        Returns reconstruction of same shape.
        """
        proj = self.pos_enc(self.input_proj(x))   # (B, T, D_MODEL)
        mem  = self.encoder(proj)                  # (B, T, D_MODEL)
        out  = self.decoder(proj, mem)             # (B, T, D_MODEL)
        return self.output_proj(out)               # (B, T, N_MELS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def to_sequence_tensor(X: np.ndarray) -> torch.Tensor:
    """
    Convert log-mel dataset (N, N_MELS, T) → (N, T, N_MELS) float32 tensor.
    Transpose so time is the sequence dimension.
    """
    X_t = X.transpose(0, 2, 1).astype(np.float32)  # (N, T, N_MELS)
    return torch.from_numpy(X_t)


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


def reconstruction_error(model: nn.Module, T_test: torch.Tensor,
                         device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        recon = model(T_test.to(device)).cpu().numpy()
    orig = T_test.numpy()
    return np.mean((orig - recon) ** 2, axis=(1, 2))


# ── Main ──────────────────────────────────────────────────────────────────────

def run(machine: str) -> dict:
    device = get_device()
    print(f"
{'='*55}")
    print(f"  Transformer AE | machine = {machine} | device = {device}")
    print(f"{'='*55}")

    train_dir = os.path.join(DATA_ROOT, machine, "train",  "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test",   "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test",   "anomaly")

    print("Loading log-mel spectrograms …")
    X_tr   = load_logmel_dataset(train_dir, fixed_len=FIXED_LEN)
    X_norm = load_logmel_dataset(test_norm,  fixed_len=FIXED_LEN)
    X_anom = load_logmel_dataset(test_anom,  fixed_len=FIXED_LEN)

    # Normalise to [0, 1] across entire dataset
    vmin = min(X_tr.min(), X_norm.min(), X_anom.min())
    vmax = max(X_tr.max(), X_norm.max(), X_anom.max())
    norm = lambda x: (x - vmin) / (vmax - vmin + 1e-8)
    X_tr   = norm(X_tr)
    X_norm = norm(X_norm)
    X_anom = norm(X_anom)

    print(f"  Train: {X_tr.shape}  |  Test normal: {X_norm.shape}  |  Test anomaly: {X_anom.shape}")

    T_tr = to_sequence_tensor(X_tr)
    loader = DataLoader(
        TensorDataset(T_tr),
        batch_size=AE_BATCH_SIZE, shuffle=True, drop_last=False,
    )
    model = TransformerAutoencoder().to(device)
    print("Training Transformer Autoencoder …")
    train_model(model, loader, device)

    X_test = np.concatenate([X_norm, X_anom], axis=0)
    y_true = np.array([0] * len(X_norm) + [1] * len(X_anom))
    T_test = to_sequence_tensor(X_test)
    scores = reconstruction_error(model, T_test, device)

    result = evaluate(y_true, scores, machine=machine, method="TransformerAE")
    save_results([result], os.path.join(RESULTS_DIR, f"transformer_ae_{machine}.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Transformer Autoencoder")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    machines = MACHINES if args.all else [args.machine]
    results  = [run(m) for m in machines]
    if args.all:
        save_results(results, os.path.join(RESULTS_DIR, "transformer_ae_all.csv"))


if __name__ == "__main__":
    main()
