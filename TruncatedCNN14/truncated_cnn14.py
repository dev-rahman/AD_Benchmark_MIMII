"""
Model 9 — Truncated Fine-Tuned CNN14 + kNN Anomaly Scorer
Hybrid DL anomaly detection.

Paper category : Deep Learning — Pretrained + Fine-Tuned Feature Extraction
Architecture   : CNN14 (6 conv blocks pretrained on AudioSet via PANNs)
                   - Blocks 1-3 + spec_augmenter : FROZEN  (pretrained AudioSet features)
                   - Blocks 4-6 + fc1            : TRAINABLE (fine-tuned on MIMII normals)
                   - projection_head              : TRAINABLE (2048 -> 128-dim embedding)
Anomaly scorer : kNN mean distance in fine-tuned embedding space
Cache          : Fine-tuned embeddings saved as .npy — reruns are instant

Reference      : Simin Mirzadeh — "truncated fine-tuned CNN14" strategy
Requires       : pip install panns-inference psutil
Authors        : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage          : python TruncatedCNN14/truncated_cnn14.py [--machine fan|pump|valve] [--all] [--k 5]
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES, SAMPLE_RATE, get_device
from utils.features import wav_paths
from utils.evaluate import evaluate, save_results

# ── Paths & constants ─────────────────────────────────────────────────────────

CACHE_DIR    = Path(os.path.abspath(__file__)).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

K_NEIGHBOURS = 5
EMBED_DIM    = 128
LR_FINETUNE  = 1e-4
LR_HEAD      = 1e-3
BATCH_SIZE   = 32
EPOCHS       = 10          # CPU-safe: ~5–8 min per machine


# ── Waveform dataset ──────────────────────────────────────────────────────────

class WaveformDataset(Dataset):
    def __init__(self, folder, sr=SAMPLE_RATE, duration=10.0):
        self.paths     = wav_paths(folder)
        if not self.paths:
            raise FileNotFoundError(f"No .wav files in: {folder}")
        self.sr        = sr
        self.n_samples = int(duration * sr)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y, _ = librosa.load(self.paths[idx], sr=self.sr, mono=True)
        if len(y) < self.n_samples:
            y = np.pad(y, (0, self.n_samples - len(y)))
        else:
            y = y[:self.n_samples]
        return torch.tensor(y, dtype=torch.float32)


# ── Truncated Fine-Tuned CNN14 ────────────────────────────────────────────────

class TruncatedCNN14(nn.Module):
    """
    CNN14 with selective layer freezing (Simin Mirzadeh strategy):
      FROZEN   (blocks 1-3): spectrogram_extractor, logmel_extractor,
                              spec_augmenter, bn0,
                              conv_block1, conv_block2, conv_block3
      TRAINABLE (blocks 4-6): conv_block4, conv_block5, conv_block6, fc1
      TRAINABLE (new head):   projection_head (2048 -> embed_dim)
    """
    def __init__(self, panns_model, embed_dim=EMBED_DIM):
        super().__init__()
        b = panns_model
        self.spectrogram_extractor = b.spectrogram_extractor
        self.logmel_extractor      = b.logmel_extractor
        self.spec_augmenter        = b.spec_augmenter
        self.bn0                   = b.bn0
        self.conv_block1           = b.conv_block1
        self.conv_block2           = b.conv_block2
        self.conv_block3           = b.conv_block3
        self.conv_block4           = b.conv_block4
        self.conv_block5           = b.conv_block5
        self.conv_block6           = b.conv_block6
        self.fc1                   = b.fc1

        self.projection_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embed_dim),
        )

        # Freeze blocks 1-3 and front-end
        for layer in [self.spectrogram_extractor, self.logmel_extractor,
                      self.spec_augmenter, self.bn0,
                      self.conv_block1, self.conv_block2, self.conv_block3]:
            for p in layer.parameters():
                p.requires_grad = False

        # Trainable: blocks 4-6, fc1, projection_head
        for layer in [self.conv_block4, self.conv_block5, self.conv_block6,
                      self.fc1, self.projection_head]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, waveform):
        with torch.no_grad():
            x = self.spectrogram_extractor(waveform)
            x = self.logmel_extractor(x)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=False)
            x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=False)

        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)

        x  = torch.mean(x, dim=3)
        x1 = torch.max(x, dim=2)[0]
        x2 = torch.mean(x, dim=2)
        x  = x1 + x2
        x  = F.dropout(x, p=0.5, training=self.training)
        x  = F.relu_(self.fc1(x))
        x  = F.dropout(x, p=0.5, training=self.training)

        emb = self.projection_head(x)
        return F.normalize(emb, p=2, dim=1)


# ── Contrastive loss ──────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature

    def forward(self, z1, z2):
        N   = z1.size(0)
        z   = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temp
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))
        labels = torch.cat([torch.arange(N, device=z.device) + N,
                             torch.arange(N, device=z.device)])
        return F.cross_entropy(sim, labels)


def augment(y):
    return y * torch.empty(1).uniform_(0.9, 1.1).item() + torch.randn_like(y) * 0.003


# ── Fine-tuning ───────────────────────────────────────────────────────────────

def fine_tune(model, train_dir, device, epochs):
    dataset = WaveformDataset(train_dir)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True, num_workers=0)
    loss_fn = NTXentLoss(temperature=0.1)

    param_groups = [
        {"params": list(model.conv_block4.parameters()) +
                   list(model.conv_block5.parameters()) +
                   list(model.conv_block6.parameters()) +
                   list(model.fc1.parameters()),
         "lr": LR_FINETUNE},
        {"params": list(model.projection_head.parameters()),
         "lr": LR_HEAD},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        for waveforms in loader:
            waveforms = waveforms.to(device)
            z1   = model(augment(waveforms))
            z2   = model(augment(waveforms))
            loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"    Epoch [{ep:2d}/{epochs}]  loss = {total/len(loader):.4f}")


# ── Embedding extraction with cache ──────────────────────────────────────────

def extract_embeddings(folder, model, device, cache_key):
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    if cache_path.exists():
        print(f"  [cache hit] {cache_key}")
        return np.load(str(cache_path))

    paths     = wav_paths(folder)
    n_samples = int(10.0 * SAMPLE_RATE)
    result    = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            y, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
            if len(y) < n_samples:
                y = np.pad(y, (0, n_samples - len(y)))
            else:
                y = y[:n_samples]
            wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
            emb = model(wav).cpu().numpy().squeeze()
            result.append(emb)

    arr = np.array(result)
    np.save(str(cache_path), arr)
    print(f"  [cached]  {cache_key}  shape={arr.shape}")
    return arr


# ── Main entry point ──────────────────────────────────────────────────────────

def run(machine, k=K_NEIGHBOURS):
    device = get_device()   # uses MPS on Apple Silicon, CUDA if available, else CPU

    print(f"\n{'='*60}")
    print(f"  Truncated Fine-Tuned CNN14 | machine={machine} | k={k}")
    print(f"  Frozen: blocks 1-3  |  Fine-tuned: blocks 4-6 + head")
    print(f"  Epochs: {EPOCHS}  |  device: {device}")
    print(f"{'='*60}")

    train_dir = os.path.join(DATA_ROOT, machine, "train",  "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test",   "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test",   "anomaly")

    print("Loading pretrained CNN14 ...")
    from panns_inference import AudioTagging
    at    = AudioTagging(checkpoint_path=None, device="cpu")
    model = TruncatedCNN14(at.model, embed_dim=EMBED_DIM).to(device)

    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {n_train:,}  |  Frozen: {n_frozen:,}")

    print(f"Fine-tuning on {machine} normal sounds ...")
    fine_tune(model, train_dir, device, epochs=EPOCHS)

    # Invalidate old cache when EPOCHS changes — tag includes epoch count
    tag = f"ftcnn14_ep{EPOCHS}_{machine}"
    print("Extracting fine-tuned embeddings ...")
    E_tr   = extract_embeddings(train_dir, model, device, f"{tag}_train_normal")
    E_norm = extract_embeddings(test_norm,  model, device, f"{tag}_test_normal")
    E_anom = extract_embeddings(test_anom,  model, device, f"{tag}_test_anomaly")

    scaler = StandardScaler()
    E_tr   = scaler.fit_transform(E_tr)
    E_norm = scaler.transform(E_norm)
    E_anom = scaler.transform(E_anom)

    print(f"  Train: {E_tr.shape}  |  Normal: {E_norm.shape}  |  Anomaly: {E_anom.shape}")

    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(E_tr)

    E_test = np.vstack([E_norm, E_anom])
    y_true = np.array([0] * len(E_norm) + [1] * len(E_anom))
    dists, _ = knn.kneighbors(E_test)
    scores   = dists.mean(axis=1)

    result = evaluate(y_true, scores, machine=machine,
                      method=f"TruncCNN14_kNN(k={k})")
    save_results([result],
                 os.path.join(RESULTS_DIR, f"truncated_cnn14_{machine}.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description="Truncated Fine-Tuned CNN14 + kNN")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all",     action="store_true")
    parser.add_argument("--k",       type=int, default=K_NEIGHBOURS)
    args     = parser.parse_args()
    machines = MACHINES if args.all else [args.machine]
    results  = [run(m, k=args.k) for m in machines]
    if args.all:
        save_results(results,
                     os.path.join(RESULTS_DIR, "truncated_cnn14_all.csv"))


if __name__ == "__main__":
    main()
