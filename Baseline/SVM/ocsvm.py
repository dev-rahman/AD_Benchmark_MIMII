"""
Model 1 — One-Class SVM with rich MFCC features.
Classical ML baseline for MIMII anomaly detection.

Produces window-level & event-level multi-class (fan/pump/valve) ROC plots
at 0.2s and 0.4s window lengths, saved to ./plotes/

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python ocsvm.py
"""

import os
import sys
import numpy as np
import librosa
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.config import DATA_ROOT, RESULTS_DIR, MACHINES

# ── Hyperparameters ──────────────────────────────────────────────
NU, GAMMA, KERNEL = 0.05, "scale", "rbf"
SR = 16000
WINDOW_LENS = [0.2, 0.4]
MACHINES_ORDERED = ["fan", "pump", "valve"]
CLASS_NAMES = ["A", "B", "C"]
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plotes")
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Feature extraction (self-contained, no external dependency) ─
def rich_features_from_signal(y, sr=SR, n_mfcc=13):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    n_frames = mfcc.shape[1]
    width = min(9, n_frames)
    if width % 2 == 0:
        width -= 1
    width = max(width, 3)

    d1 = librosa.feature.delta(mfcc, width=width)
    d2 = librosa.feature.delta(mfcc, order=2, width=width)
    rms = librosa.feature.rms(y=y)
    cen = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    feats = []
    for arr in [mfcc, d1, d2, rms, cen, bw, roll, zcr]:
        feats.append(arr.mean(axis=1))
        feats.append(arr.std(axis=1))
    return np.concatenate(feats)


def list_wavs(folder):
    if not os.path.isdir(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")]


def load_windows(path, win_len, sr=SR):
    y, _ = librosa.load(path, sr=sr)
    win_samples = int(win_len * sr)
    n_win = max(1, len(y) // win_samples)
    if len(y) < win_samples:
        return [y]
    return [y[i * win_samples:(i + 1) * win_samples] for i in range(n_win)]


def extract_dataset(files, win_len):
    X, event_ids = [], []
    for eid, f in enumerate(files):
        for seg in load_windows(f, win_len):
            X.append(rich_features_from_signal(seg))
            event_ids.append(eid)
    return np.array(X), np.array(event_ids)


def get_scores_for_machine(machine, win_len):
    train_dir = os.path.join(DATA_ROOT, machine, "train", "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test", "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test", "anomaly")

    train_files = list_wavs(train_dir)
    norm_files = list_wavs(test_norm)
    anom_files = list_wavs(test_anom)
    test_files = norm_files + anom_files

    print(f"  [{machine}] train={len(train_files)} test_norm={len(norm_files)} test_anom={len(anom_files)}")

    X_tr, _ = extract_dataset(train_files, win_len)
    X_test, event_ids = extract_dataset(test_files, win_len)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_test = scaler.transform(X_test)

    model = OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA)
    model.fit(X_tr)

    window_scores = -model.decision_function(X_test)
    n_events = len(test_files)
    event_scores = np.array([
        window_scores[event_ids == e].mean() if np.any(event_ids == e) else 0.0
        for e in range(n_events)
    ])
    return window_scores, event_ids, event_scores, len(test_files)


def build_multiclass_scores(win_len):
    per_machine = {m: get_scores_for_machine(m, win_len) for m in MACHINES_ORDERED}

    all_win_true, all_evt_true = [], []
    win_lengths, evt_lengths = [], []
    for ci, m in enumerate(MACHINES_ORDERED):
        w_scores, _, e_scores, n_events = per_machine[m]
        all_win_true += [ci] * len(w_scores)
        all_evt_true += [ci] * n_events
        win_lengths.append(len(w_scores))
        evt_lengths.append(n_events)

    win_score_matrix = np.zeros((sum(win_lengths), 3))
    evt_score_matrix = np.zeros((sum(evt_lengths), 3))
    ow, oe = 0, 0
    for ci, m in enumerate(MACHINES_ORDERED):
        w_scores, _, e_scores, n_events = per_machine[m]
        win_score_matrix[ow:ow + len(w_scores), ci] = -w_scores
        evt_score_matrix[oe:oe + n_events, ci] = -e_scores
        ow += len(w_scores)
        oe += n_events

    return (np.array(all_win_true), win_score_matrix,
            np.array(all_evt_true), evt_score_matrix)


def plot_roc(y_true, score_matrix, title, save_path):
    plt.figure(figsize=(5, 5))
    fprs, tprs = [], []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for c in range(3):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, score_matrix[:, c])
        fprs.append(fpr); tprs.append(tpr)
        a = auc(fpr, tpr)
        print(f"    Class {CLASS_NAMES[c]} ({MACHINES_ORDERED[c]}) AUC = {a:.3f}")
        plt.plot(fpr, tpr, color=colors[c],
                  label=f"Class {CLASS_NAMES[c]} vs Rest (AUC = {a:.3f})")

    all_fpr = np.unique(np.concatenate(fprs))
    mean_tpr = np.zeros_like(all_fpr)
    for fpr, tpr in zip(fprs, tprs):
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= 3
    macro_auc = auc(all_fpr, mean_tpr)
    print(f"    Macro-average AUC = {macro_auc:.3f}")
    plt.plot(all_fpr, mean_tpr, '--', color='red',
              label=f"Macro-average (AUC = {macro_auc:.3f})")
    plt.plot([0, 1], [0, 1], color='gray', lw=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    for win_len in WINDOW_LENS:
        print(f"\n=== Window length: {win_len}s ===")
        win_true, win_scores, evt_true, evt_scores = build_multiclass_scores(win_len)

        print("  -- Window-level results --")
        plot_roc(win_true, win_scores,
                  f"Window-level ({win_len}s)",
                  os.path.join(PLOT_DIR, f"roc_window_{win_len}s.png"))

        print("  -- Event-level results --")
        plot_roc(evt_true, evt_scores,
                  f"Event-level ({win_len}s)",
                  os.path.join(PLOT_DIR, f"roc_event_{win_len}s.png"))

    print(f"\nAll plots saved in: {PLOT_DIR}/")


if __name__ == "__main__":
    main()