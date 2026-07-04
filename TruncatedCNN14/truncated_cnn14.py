"""
Model 9 — Truncated Fine-Tuned CNN14 (Block-4) + Segment-Level kNN Anomaly Scorer
Hybrid DL anomaly detection.


Paper category : Deep Learning — Pretrained + Fine-Tuned Feature Extraction
Architecture   : CNN14 (6 conv blocks pretrained on AudioSet via PANNs), TRUNCATED after Block 4
                 - Blocks 1-2 + spec_augmenter : FROZEN (pretrained AudioSet features)
                 - Blocks 3-4                  : TRAINABLE (fine-tuned on MIMII normals)
                 - projection_head             : TRAINABLE (GAP+GMP pooled -> 128-dim embedding)
Segmentation   : Each 10s clip split into short overlapping windows (default 2s, 1s hop)
                 -> one embedding per segment instead of one per clip
Pooling        : Global Average Pooling + Global Max Pooling, concatenated
Anomaly scorer : kNN mean distance per segment -> MAX over segments = clip anomaly score
Cache          : Fine-tuned embeddings saved as .npy — reruns are instant


Reference : Simin Mirzadeh — "truncated fine-tuned CNN14" strategy (Block-4 + segment-level variant)
Requires  : pip install panns-inference psutil
Authors   : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage     : python TruncatedCNN14/truncated_cnn14.py [--machine fan|pump|valve] [--all] [--k 5]

FIX NOTE (2026-07-03): conv_block4 in PANNs CNN14 outputs 512 channels, not 256.
GAP+GMP concatenation therefore yields a 1024-dim pooled vector. The
projection_head's block4_channels default and final Linear layer were
corrected to reflect this (512 -> *2 = 1024 in, 256 hidden, embed_dim out).
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
from utils.config import DATA_ROOT, RESULTS_DIR, MACHINES, SAMPLE_RATE, get_device
from utils.features import wav_paths
from utils.evaluate import evaluate, save_results


# ── Paths & constants ─────────────────────────────────────────────────────────


CACHE_DIR = Path(os.path.abspath(__file__)).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)


K_NEIGHBOURS = 5
EMBED_DIM = 128
LR_FINETUNE = 1e-4
LR_HEAD = 1e-3
BATCH_SIZE = 32
EPOCHS = 50                 # CPU-safe: ~5-8 min per machine


# Segment-level settings
SEGMENT_DURATION = 2.0      # seconds per segment (short window instead of full 10s clip)
SEGMENT_HOP = 1.0           # seconds between segment starts (overlap = SEGMENT_DURATION - HOP)
CLIP_DURATION = 10.0        # original clip duration in MIMII


# ── Waveform dataset (segment-level) ──────────────────────────────────────────


class SegmentWaveformDataset(Dataset):
    """
    Splits each 10s clip into overlapping short segments.
    Used only for FINE-TUNING (contrastive loss on normal segments).
    """
    def __init__(self, folder, sr=SAMPLE_RATE,
                 seg_duration=SEGMENT_DURATION, seg_hop=SEGMENT_HOP,
                 clip_duration=CLIP_DURATION):
        self.paths = wav_paths(folder)
        if not self.paths:
            raise FileNotFoundError(f"No .wav files in: {folder}")
        self.sr = sr
        self.seg_samples = int(seg_duration * sr)
        self.hop_samples = int(seg_hop * sr)
        self.clip_samples = int(clip_duration * sr)


    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        y, _ = librosa.load(self.paths[idx], sr=self.sr, mono=True)
        if len(y) < self.clip_samples:
            y = np.pad(y, (0, self.clip_samples - len(y)))
        else:
            y = y[:self.clip_samples]
        # Pick ONE random segment per call for fine-tuning speed/diversity
        max_start = max(0, len(y) - self.seg_samples)
        start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        seg = y[start:start + self.seg_samples]
        if len(seg) < self.seg_samples:
            seg = np.pad(seg, (0, self.seg_samples - len(seg)))
        return torch.tensor(seg, dtype=torch.float32)



def segment_clip(y, sr, seg_duration=SEGMENT_DURATION, seg_hop=SEGMENT_HOP):
    """Split one waveform into a list of fixed-length overlapping segments."""
    seg_samples = int(seg_duration * sr)
    hop_samples = int(seg_hop * sr)
    if len(y) < seg_samples:
        y = np.pad(y, (0, seg_samples - len(y)))
    segments = []
    start = 0
    while start + seg_samples <= len(y):
        segments.append(y[start:start + seg_samples])
        start += hop_samples
    if not segments:
        segments = [y[:seg_samples]]
    return segments


# ── Truncated Fine-Tuned CNN14 (Block-4 cutoff, GAP+GMP pooling) ─────────────


class TruncatedCNN14Block4(nn.Module):
    """
    CNN14 truncated after Block 4 (Simin Mirzadeh strategy, v2):
    FROZEN     (blocks 1-2): spectrogram_extractor, logmel_extractor,
                              spec_augmenter, bn0, conv_block1, conv_block2
    TRAINABLE  (blocks 3-4): conv_block3, conv_block4
    TRAINABLE  (new head)  : projection_head (GAP+GMP pooled feature -> embed_dim)


    Blocks 5-6 and fc1 are DROPPED entirely — we stop right after Block 4,
    keeping the fine-grained time-frequency detail that deeper blocks discard.

    NOTE: conv_block4 in PANNs CNN14 outputs 512 channels (channel progression
    is 64 -> 128 -> 256 -> 512 -> 1024 -> 2048 across blocks 1-6). GAP+GMP
    concatenation doubles this to 1024, which is why block4_channels defaults
    to 512 below (not 256).
    """
    def __init__(self, panns_model, embed_dim=EMBED_DIM, block4_channels=512):
        super().__init__()
        b = panns_model
        self.spectrogram_extractor = b.spectrogram_extractor
        self.logmel_extractor = b.logmel_extractor
        self.spec_augmenter = b.spec_augmenter
        self.bn0 = b.bn0
        self.conv_block1 = b.conv_block1
        self.conv_block2 = b.conv_block2
        self.conv_block3 = b.conv_block3
        self.conv_block4 = b.conv_block4
        # NOTE: conv_block5, conv_block6, fc1 intentionally NOT used


        # GAP+GMP pooled output of block4 has 2 * block4_channels features (1024 by default)
        self.projection_head = nn.Sequential(
            nn.Linear(block4_channels * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, embed_dim),
        )


        # Freeze blocks 1-2 and front-end
        for layer in [self.spectrogram_extractor, self.logmel_extractor,
                      self.spec_augmenter, self.bn0,
                      self.conv_block1, self.conv_block2]:
            for p in layer.parameters():
                p.requires_grad = False


        # Trainable: blocks 3-4, projection_head
        for layer in [self.conv_block3, self.conv_block4, self.projection_head]:
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
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        # x shape: (batch, channels, time, freq)


        # GAP + GMP pooling over time and frequency dims (replaces fc1 aggregation)
        gap = torch.mean(x, dim=(2, 3))
        gmp = torch.amax(x, dim=(2, 3))
        pooled = torch.cat([gap, gmp], dim=1)
        pooled = F.dropout(pooled, p=0.3, training=self.training)


        emb = self.projection_head(pooled)
        return F.normalize(emb, p=2, dim=1)


# ── Contrastive loss ──────────────────────────────────────────────────────────


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature


    def forward(self, z1, z2):
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.T) / self.temp
        mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))
        labels = torch.cat([torch.arange(N, device=z.device) + N,
                             torch.arange(N, device=z.device)])
        return F.cross_entropy(sim, labels)



def augment(y):
    return y * torch.empty(1).uniform_(0.9, 1.1).item() + torch.randn_like(y) * 0.003


# ── Fine-tuning (on short segments, not full 10s clips) ──────────────────────


def fine_tune(model, train_dir, device, epochs):
    dataset = SegmentWaveformDataset(train_dir)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, drop_last=True, num_workers=0)
    loss_fn = NTXentLoss(temperature=0.1)


    param_groups = [
        {"params": list(model.conv_block3.parameters()) +
                    list(model.conv_block4.parameters()),
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
            z1 = model(augment(waveforms))
            z2 = model(augment(waveforms))
            loss = loss_fn(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += loss.item()
        scheduler.step()
        print(f"  Epoch [{ep:2d}/{epochs}] loss = {total/len(loader):.4f}")


# ── Segment-level embedding extraction with cache ─────────────────────────────


def extract_segment_embeddings(folder, model, device, cache_key):
    """
    Returns:
        clip_embeddings : list of np.ndarray, one array of shape
                           (n_segments_in_clip, embed_dim) per audio file
    """
    cache_path = CACHE_DIR / f"{cache_key}.npz"
    if cache_path.exists():
        print(f"  [cache hit] {cache_key}")
        data = np.load(str(cache_path), allow_pickle=True)
        return list(data["clips"])


    paths = wav_paths(folder)
    result = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            y, _ = librosa.load(p, sr=SAMPLE_RATE, mono=True)
            segments = segment_clip(y, SAMPLE_RATE)
            batch = torch.tensor(np.stack(segments), dtype=torch.float32).to(device)
            embs = model(batch).cpu().numpy()   # (n_segments, embed_dim)
            result.append(embs)


    np.savez(str(cache_path), clips=np.array(result, dtype=object))
    print(f"  [cached] {cache_key} n_clips={len(result)}")
    return result


# ── Main entry point ──────────────────────────────────────────────────────────


def run(machine, k=K_NEIGHBOURS):
    device = get_device()  # uses MPS on Apple Silicon, CUDA if available, else CPU


    print(f"\n{'='*60}")
    print(f" Truncated Fine-Tuned CNN14 (Block-4) | machine={machine} | k={k}")
    print(f" Frozen: blocks 1-2 | Fine-tuned: blocks 3-4 + head")
    print(f" Segment: {SEGMENT_DURATION}s / hop {SEGMENT_HOP}s | Pooling: GAP+GMP")
    print(f" Epochs: {EPOCHS} | device: {device}")
    print(f"{'='*60}")


    train_dir = os.path.join(DATA_ROOT, machine, "train", "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test", "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test", "anomaly")


    print("Loading pretrained CNN14 ...")
    from panns_inference import AudioTagging
    at = AudioTagging(checkpoint_path=None, device="cpu")
    model = TruncatedCNN14Block4(at.model, embed_dim=EMBED_DIM).to(device)


    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable: {n_train:,} | Frozen: {n_frozen:,}")


    print(f"Fine-tuning on {machine} normal sounds (segment-level) ...")
    fine_tune(model, train_dir, device, epochs=EPOCHS)


    tag = f"trcnn14_b4_seg_ep{EPOCHS}_{machine}"
    print("Extracting segment-level fine-tuned embeddings ...")
    clips_tr = extract_segment_embeddings(train_dir, model, device, f"{tag}_train_normal")
    clips_norm = extract_segment_embeddings(test_norm, model, device, f"{tag}_test_normal")
    clips_anom = extract_segment_embeddings(test_anom, model, device, f"{tag}_test_anomaly")


    # Fit scaler on ALL training segments (pooled across clips)
    all_train_segments = np.vstack(clips_tr)
    scaler = StandardScaler()
    scaler.fit(all_train_segments)
    E_tr = scaler.transform(all_train_segments)


    print(f"  Train segments: {E_tr.shape}")


    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(E_tr)


    def clip_scores(clip_embeddings_list):
        """For each clip: score every segment, take MAX segment score as clip score."""
        scores = []
        for segs in clip_embeddings_list:
            segs_scaled = scaler.transform(segs)
            dists, _ = knn.kneighbors(segs_scaled)
            seg_scores = dists.mean(axis=1)   # mean distance to k neighbours, per segment
            scores.append(seg_scores.max())   # worst segment drives the clip score
        return np.array(scores)


    scores_norm = clip_scores(clips_norm)
    scores_anom = clip_scores(clips_anom)


    scores = np.concatenate([scores_norm, scores_anom])
    y_true = np.array([0] * len(scores_norm) + [1] * len(scores_anom))


    print(f"  Normal clips: {len(scores_norm)} | Anomaly clips: {len(scores_anom)}")


    result = evaluate(y_true, scores, machine=machine,
                       method=f"TruncCNN14Block4_SegKNN(k={k})")
    save_results([result],
                  os.path.join(RESULTS_DIR, f"truncated_cnn14_{machine}.csv"))
    return result



def main():
    parser = argparse.ArgumentParser(description="Truncated Fine-Tuned CNN14 (Block-4) + Segment kNN")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--k", type=int, default=K_NEIGHBOURS)
    args = parser.parse_args()
    machines = MACHINES if args.all else [args.machine]
    results = [run(m, k=args.k) for m in machines]
    if args.all:
        save_results(results,
                      os.path.join(RESULTS_DIR, "truncated_cnn14_all.csv"))



if __name__ == "__main__":
    main()
