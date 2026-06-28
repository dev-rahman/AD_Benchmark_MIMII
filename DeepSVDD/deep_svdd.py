"""
Model 8 — Deep SVDD (Support Vector Data Description)
Deep Learning one-class classification for anomaly detection.

Paper category : Deep Learning — One-Class Classification
Input          : Log-mel spectrogram frames (128 mel bins, 5 frames → 640-dim)
Architecture   : CNN encoder → 64-dim hypersphere embedding
Loss           : Minimize distance of normal embeddings to hypersphere centre c
Anomaly score  : Euclidean distance from centre c (higher = more anomalous)

Reference      : Ruff et al. "Deep One-Class Classification" ICML 2018
                 Applied to MIMII: AUC ~0.80–0.84 (fan/pump/valve @ 0dB)

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python DeepSVDD/deep_svdd.py [--machine fan|pump|valve] [--all]
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES, SAMPLE_RATE, get_device
from utils.features import wav_paths
from utils.evaluate import evaluate, save_results

# ── Hyperparameters ───────────────────────────────────────────────────────────

N_MELS      = 128
N_FFT       = 1024
HOP_LENGTH  = 512
N_FRAMES    = 5          # consecutive frames concatenated per sample
EMBED_DIM   = 64         # hypersphere dimension
EPOCHS      = 150
BATCH_SIZE  = 128
LR          = 1e-3
WEIGHT_DECAY= 1e-5


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_frames(folder: str, n_mels: int = N_MELS,
                   n_frames: int = N_FRAMES) -> np.ndarray:
    """
    Extract overlapping log-mel frame vectors from all .wav files in folder.
    Each sample = n_frames consecutive mel columns flattened → (n_frames * n_mels,) dim.
    This is the official MIMII baseline input format.
    """
    paths = wav_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No .wav files found in: {folder}")

    all_vectors = []
    for p in paths:
        y, sr = librosa.load(p, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
        )
        log_mel = librosa.power_to_db(mel, ref=np.max).T  # (T, n_mels)
        # Slide a window of n_frames across time axis — one vector per window
        for i in range(len(log_mel) - n_frames + 1):
            vec = log_mel[i:i + n_frames].flatten()       # (n_frames * n_mels,)
            all_vectors.append(vec)

    return np.array(all_vectors, dtype=np.float32)


def file_level_scores(folder: str, model: nn.Module, centre: torch.Tensor,
                      scaler: StandardScaler, device: str,
                      n_mels: int = N_MELS, n_frames: int = N_FRAMES) -> np.ndarray:
    """
    Return one anomaly score per file = mean frame-level distance to centre.
    """
    paths = wav_paths(folder)
    scores = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            y, sr = librosa.load(p, sr=SAMPLE_RATE, mono=True)
            mel = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels
            )
            log_mel = librosa.power_to_db(mel, ref=np.max).T
            vecs = []
            for i in range(len(log_mel) - n_frames + 1):
                vecs.append(log_mel[i:i + n_frames].flatten())
            if not vecs:
                scores.append(0.0)
                continue
            X = scaler.transform(np.array(vecs, dtype=np.float32))
            t = torch.tensor(X).to(device)
            emb = model(t)
            dist = torch.mean((emb - centre) ** 2, dim=1)
            scores.append(dist.mean().item())
    return np.array(scores)


# ── Architecture ──────────────────────────────────────────────────────────────

class SVDDEncoder(nn.Module):
    """
    Fully-connected encoder that maps input frames to a compact embedding.
    No bias in final layer (standard Deep SVDD requirement).
    Architecture: input → 512 → 256 → 128 → EMBED_DIM
    """
    def __init__(self, input_dim: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, embed_dim, bias=False),  # no bias — SVDD requirement
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── SVDD Training ──────────────────────────────────────────────────────────────

def init_centre(model: nn.Module, loader: DataLoader,
                device: str, embed_dim: int) -> torch.Tensor:
    """
    Initialise hypersphere centre c as mean of encoder outputs on training data.
    Clamp values close to 0 to avoid trivial solution (collapsing to zero).
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for (x,) in loader:
            embeddings.append(model(x.to(device)))
    c = torch.mean(torch.cat(embeddings, dim=0), dim=0)
    # Avoid trivial solution: clamp small values away from zero
    c[(c.abs() < 0.01) & (c >= 0)] =  0.01
    c[(c.abs() < 0.01) & (c <  0)] = -0.01
    return c.detach()


def train_svdd(model: nn.Module, loader: DataLoader,
               centre: torch.Tensor, device: str,
               epochs: int = EPOCHS, lr: float = LR) -> None:
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )
    model.train()
    centre = centre.to(device)

    for ep in range(1, epochs + 1):
        epoch_loss = 0.0
        for (x,) in loader:
            x   = x.to(device)
            emb = model(x)
            # SVDD loss: mean squared distance of embeddings to centre c
            loss = torch.mean(torch.sum((emb - centre) ** 2, dim=1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        scheduler.step(avg)

        if ep % 25 == 0 or ep == 1:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"    Epoch [{ep:3d}/{epochs}]  loss = {avg:.6f}  lr = {current_lr:.2e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run(machine: str) -> dict:
    device = get_device()
    print(f"\n{'='*57}")
    print(f"  Deep SVDD | machine = {machine} | device = {device}")
    print(f"{'='*57}")

    train_dir = os.path.join(DATA_ROOT, machine, "train",  "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test",   "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test",   "anomaly")

    print("Extracting log-mel frame vectors (train) …")
    X_tr = extract_frames(train_dir)
    print(f"  Train frames: {X_tr.shape}")

    # Fit scaler on training frames only
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr_scaled)),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )

    input_dim = X_tr_scaled.shape[1]
    model = SVDDEncoder(input_dim=input_dim, embed_dim=EMBED_DIM).to(device)

    print("Initialising hypersphere centre c …")
    centre = init_centre(model, loader, device, EMBED_DIM)

    print(f"Training Deep SVDD — {EPOCHS} epochs …")
    train_svdd(model, loader, centre, device, epochs=EPOCHS, lr=LR)

    print("Scoring test files …")
    scores_norm = file_level_scores(test_norm, model, centre.to(device), scaler, device)
    scores_anom = file_level_scores(test_anom, model, centre.to(device), scaler, device)

    scores = np.concatenate([scores_norm, scores_anom])
    y_true = np.array([0] * len(scores_norm) + [1] * len(scores_anom))

    print(f"  Test normal files: {len(scores_norm)}  |  Test anomaly files: {len(scores_anom)}")

    result = evaluate(y_true, scores, machine=machine, method="DeepSVDD")
    save_results([result], os.path.join(RESULTS_DIR, f"deep_svdd_{machine}.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Deep SVDD anomaly detection")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all", action="store_true", help="Run all machines")
    args = parser.parse_args()
    machines = MACHINES if args.all else [args.machine]
    results  = [run(m) for m in machines]
    if args.all:
        save_results(results, os.path.join(RESULTS_DIR, "deep_svdd_all.csv"))


if __name__ == "__main__":
    main()