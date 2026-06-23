"""
Shared audio feature extraction utilities.
Authors: Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
"""

import os
import numpy as np
import librosa
from tqdm import tqdm

from utils.config import SAMPLE_RATE, N_MFCC, N_MELS, N_FFT, HOP_LENGTH


# ── Low-level loaders ─────────────────────────────────────────────────────────

def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a mono .wav file, resampled to `sr` Hz."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def wav_paths(folder: str) -> list[str]:
    """Return sorted list of .wav file paths in `folder`."""
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".wav")
    )


# ── Feature extractors ────────────────────────────────────────────────────────

def mfcc_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Mean + std of MFCCs → shape (2 * N_MFCC,) = (80,).
    Used by: OCSVM, Isolation Forest
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def rich_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Extended hand-crafted feature vector → shape (87,):
      MFCC mean + std       (80 dims)
      Spectral centroid      (2 dims)
      Spectral rolloff       (2 dims)
      RMS energy             (2 dims)
      Zero-crossing rate     (1 dim)
    Used by: OCSVM, Isolation Forest (improved baseline)
    """
    mfcc     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC,
                                    n_fft=N_FFT, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)
    rms      = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    zcr      = librosa.feature.zero_crossing_rate(y, hop_length=HOP_LENGTH)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        centroid.mean(axis=1), centroid.std(axis=1),
        rolloff.mean(axis=1),  rolloff.std(axis=1),
        rms.mean(axis=1),      rms.std(axis=1),
        zcr.mean(axis=1),
    ])


def logmel_spectrogram(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Log-mel spectrogram → shape (N_MELS, T).
    Used by: CNN Autoencoder, Transformer Autoencoder
    """
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                          n_fft=N_FFT, hop_length=HOP_LENGTH)
    return librosa.power_to_db(mel, ref=np.max)


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_feature_dataset(folder: str, feature_fn=rich_features) -> np.ndarray:
    """Load all .wav files in `folder`, extract features → 2-D matrix (N, D)."""
    paths = wav_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No .wav files found in: {folder}")
    feats = [feature_fn(load_audio(p)) for p in tqdm(paths, desc=os.path.basename(folder), leave=False)]
    return np.array(feats)


def load_logmel_dataset(folder: str, fixed_len: int = 128) -> np.ndarray:
    """
    Load all .wav files → fixed-length log-mel spectrograms.
    Pads or crops the time axis to `fixed_len` frames.
    Returns shape (N, N_MELS, fixed_len).
    Used by: CNN AE, Transformer AE
    """
    paths = wav_paths(folder)
    if not paths:
        raise FileNotFoundError(f"No .wav files found in: {folder}")
    specs = []
    for p in tqdm(paths, desc=os.path.basename(folder), leave=False):
        y   = load_audio(p)
        mel = logmel_spectrogram(y)  # (N_MELS, T)
        if mel.shape[1] < fixed_len:
            pad = fixed_len - mel.shape[1]
            mel = np.pad(mel, ((0, 0), (0, pad)), mode="constant")
        else:
            mel = mel[:, :fixed_len]
        specs.append(mel)
    return np.array(specs)  # (N, N_MELS, fixed_len)
