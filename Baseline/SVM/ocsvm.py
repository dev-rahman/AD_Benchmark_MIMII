"""
Model 1 — One-Class SVM with rich MFCC features.
Classical ML baseline for MIMII anomaly detection.

Paper category : Classical ML
Features       : MFCC (mean + std) + spectral centroid, rolloff, RMS, ZCR  → 87 dims
Anomaly score  : Negative decision function of OC-SVM (higher = more anomalous)

Authors : Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
Usage   : python Baseline/SVM/ocsvm.py [--machine fan|pump|valve] [--all]
"""

import argparse
import os
import sys

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.config   import DATA_ROOT, RESULTS_DIR, MACHINES
from utils.features import load_feature_dataset, rich_features
from utils.evaluate import evaluate, save_results


# ── Hyperparameters ───────────────────────────────────────────────────────────
NU    = 0.05   # upper bound on training outlier fraction
GAMMA = "scale"
KERNEL = "rbf"


def run(machine: str) -> dict:
    print(f"
{'='*55}")
    print(f"  OCSVM | machine = {machine}")
    print(f"{'='*55}")

    train_dir = os.path.join(DATA_ROOT, machine, "train",  "normal")
    test_norm = os.path.join(DATA_ROOT, machine, "test",   "normal")
    test_anom = os.path.join(DATA_ROOT, machine, "test",   "anomaly")

    print("Loading features …")
    X_tr   = load_feature_dataset(train_dir, rich_features)
    X_norm = load_feature_dataset(test_norm,  rich_features)
    X_anom = load_feature_dataset(test_anom,  rich_features)

    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_norm = scaler.transform(X_norm)
    X_anom = scaler.transform(X_anom)

    print(f"  Train: {X_tr.shape}  |  Test normal: {X_norm.shape}  |  Test anomaly: {X_anom.shape}")

    model = OneClassSVM(kernel=KERNEL, nu=NU, gamma=GAMMA)
    print("Training OC-SVM …")
    model.fit(X_tr)

    X_test = np.vstack([X_norm, X_anom])
    y_true = np.array([0] * len(X_norm) + [1] * len(X_anom))
    scores = -model.decision_function(X_test)   # negate: higher = more anomalous

    result = evaluate(y_true, scores, machine=machine, method="OCSVM")
    save_results([result], os.path.join(RESULTS_DIR, f"ocsvm_{machine}.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description="One-Class SVM baseline")
    parser.add_argument("--machine", choices=MACHINES, default="fan")
    parser.add_argument("--all", action="store_true", help="Run all machine types")
    args = parser.parse_args()

    machines = MACHINES if args.all else [args.machine]
    results  = [run(m) for m in machines]

    if args.all:
        save_results(results, os.path.join(RESULTS_DIR, "ocsvm_all.csv"))


if __name__ == "__main__":
    main()
