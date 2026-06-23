"""
Global configuration for the MIMII Anomaly Detection Benchmark.
Authors: Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
"""

import os
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT   = os.path.join(ROOT_DIR, "data", "MIMII")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
MACHINES    = ["fan", "pump", "valve"]
SAMPLE_RATE = 16000

# ── Audio features ────────────────────────────────────────────────────────────
N_MFCC     = 40
N_MELS     = 128
N_FFT      = 1024
HOP_LENGTH = 512

# ── Autoencoder training ──────────────────────────────────────────────────────
AE_EPOCHS     = 100
AE_BATCH_SIZE = 32
AE_LR         = 1e-3

# ── Device (M1 MPS / CUDA / CPU) ─────────────────────────────────────────────
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
