"""
Model 6 — CNN14 Pretrained Embeddings + k-Nearest Neighbours.
Embedding-based (Hybrid) anomaly detection.

Paper category : Hybrid — DL Feature Extraction + Classical Scorer
Feature extractor : CNN14 (pretrained on AudioSet via PANNs)
                    Outputs a 2048-dim embedding per audio clip
Anomaly scorer    : kNN — mean distance to k nearest normal training neighbours
                    (higher distance = more anomalous)
Cache             : Embeddings are saved as .npy on first run to avoid re-inference

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python Embedding/CNN14_KNN/cnn14_knn.py [--machine fan|pump|valve] [--all] [--k 5]
Requires: pip install panns-inference
"""

import argparse
import os
import sys

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES, SAMPLE_RATE, get_device
from utils.features import wav_paths, load_audio
from utils.evaluate import evaluate, save_results

K_NEIGHBOURS = 5
CACHE_DIR    = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Embedding extraction ──────────────────────────────────────────────────────

def load_cnn14_model(device: str):
    """Load pretrained CNN14 from panns-inference. Downloads on first call."""
    from panns_inference import AudioTagging
    at = AudioTagging(checkpoint_path=None, device=device)
    at.model.eval()
    return at.model


def extract_embeddings(folder: str, model, device: str,
                       cache_key: str) -> np.ndarray:
    """
    Extract CNN14 2048-dim embeddings for every .wav in folder.
    Results are cached to disk (.npy) so second runs are instant.
    """
    from tqdm import tqdm
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.npy")
    if os.path.exists(cache_path):
        print(f"  [cache hit] {cache_key}")
        return np.load(cache_path)

    paths = wav_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No .wav files in: {folder}")

    embeddings = []
    for p in tqdm(paths, desc=os.path.basename(folder), leave=False):
        y = load_audio(p, sr=SAMPLE_RATE)
        waveform = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(waveform)
        emb = out["embedding"].squeeze().cpu().numpy()
        embeddings.append(emb)

    result = np.array(embeddings)
    np.save(cache_path, result)
    print(f"  [cached]    {cache_key}  shape={result.shape}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def run(machine: str, k: int = K_NEIGHBOURS) -> dict:
    # Select device — MPS fallback if panns-inference doesn't support it
    device = get_device()
    try:
        import panns_inference  # noqa: F401
    except ImportError:
        raise ImportError("Run: pip install panns-inference")

    print(f"
{'='*55}")
    print(f"  CNN14 + kNN | machine = {machine} | k = {k} | device = {device}")
    print(f"{'='*55}")

    train_dir = os.path.join(DATA_ROOT, machine, "train",  "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test",   "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test",   "anomaly")

    print("Loading CNN14 model (downloads ~200 MB on first run) …")
    model = load_cnn14_model(device)

    E_tr   = extract_embeddings(train_dir, model, device, f"{machine}_train_normal")
    E_norm = extract_embeddings(test_norm,  model, device, f"{machine}_test_normal")
    E_anom = extract_embeddings(test_anom,  model, device, f"{machine}_test_anomaly")

    scaler = StandardScaler()
    E_tr   = scaler.fit_transform(E_tr)
    E_norm = scaler.transform(E_norm)
    E_anom = scaler.transform(E_anom)

    print(f"  Train: {E_tr.shape}  |  Test normal: {E_norm.shape}  |  Test anomaly: {E_anom.shape}")

    # Fit kNN on normal training embeddings only
    knn = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=-1)
    knn.fit(E_tr)

    E_test = np.vstack([E_norm, E_anom])
    y_true = np.array([0] * len(E_norm) + [1] * len(E_anom))

    dists, _ = knn.kneighbors(E_test)
    scores   = dists.mean(axis=1)   # mean distance to k nearest normal neighbours

    result = evaluate(y_true, scores, machine=machine, method=f"CNN14_kNN(k={k})")
    save_results([result], os.path.join(RESULTS_DIR, f"cnn14_knn_k{k}_{machine}.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description="CNN14 + kNN (Hybrid)")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--k", type=int, default=K_NEIGHBOURS,
                        help="Number of nearest neighbours (default: 5)")
    args = parser.parse_args()
    machines = MACHINES if args.all else [args.machine]
    results  = [run(m, k=args.k) for m in machines]
    if args.all:
        save_results(results, os.path.join(RESULTS_DIR, f"cnn14_knn_k{args.k}_all.csv"))


if __name__ == "__main__":
    main()
