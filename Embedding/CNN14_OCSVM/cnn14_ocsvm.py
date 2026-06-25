from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

CACHE_DIR = BASE_DIR / "Embedding" / "CNN14_KNN" / ".cache"

MACHINES = ["fan", "pump", "valve"]
NU_VALUES = [0.01, 0.05, 0.1, 0.2]


# ------------------------------------------------
# Load cached CNN14 embeddings (from cnn14_knn.py)
# ------------------------------------------------
def load_cached_embeddings(machine):
    cache_dir = BASE_DIR / "Embedding" / "CNN14_KNN" / ".cache"

    X_train     = np.load(cache_dir / f"{machine}_train_normal.npy")
    X_test_norm = np.load(cache_dir / f"{machine}_test_normal.npy")
    X_test_anom = np.load(cache_dir / f"{machine}_test_anomaly.npy")

    X_test = np.vstack([X_test_norm, X_test_anom])
    y_test = np.array([0] * len(X_test_norm) + [1] * len(X_test_anom))
    counts = {
        "train_normal_files": len(X_train),
        "test_normal_files":  len(X_test_norm),
        "test_anomaly_files": len(X_test_anom),
    }

    print(f"[{machine}] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_test, counts


# ------------------------------------------------
# Evaluate One-Class SVM on CNN14 embeddings
# ------------------------------------------------
def evaluate_ocsvm(machine, nu, X_train, X_test, y_test, counts):
    print(f"  nu={nu} ...", end=" ", flush=True)

    # Standardise — important for OCSVM with RBF kernel
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Fit on normal training embeddings only
    model = OneClassSVM(kernel="rbf", nu=nu)
    model.fit(X_train_scaled)

    # Anomaly score = negative decision function (higher = more anomalous)
    anomaly_scores = -model.decision_function(X_test_scaled)
    auc = roc_auc_score(y_test, anomaly_scores)

    # Hard predictions: -1 from OCSVM → 1 (anomaly), +1 → 0 (normal)
    raw_pred = model.predict(X_test_scaled)
    y_pred   = np.where(raw_pred == -1, 1, 0)

    cm     = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0
    )

    print(f"AUC={auc:.4f}  AnomalyRecall={report['anomaly']['recall']:.4f}")

    return {
        "machine":           machine,
        "nu":                nu,
        "roc_auc":           round(auc, 6),
        "accuracy":          round(report["accuracy"], 6),
        "normal_precision":  round(report["normal"]["precision"], 6),
        "normal_recall":     round(report["normal"]["recall"], 6),
        "anomaly_precision": round(report["anomaly"]["precision"], 6),
        "anomaly_recall":    round(report["anomaly"]["recall"], 6),
        "anomaly_f1":        round(report["anomaly"]["f1-score"], 6),
        "tn":  int(cm[0, 0]),
        "fp":  int(cm[0, 1]),
        "fn":  int(cm[1, 0]),
        "tp":  int(cm[1, 1]),
        "train_normal_files":  counts["train_normal_files"],
        "test_normal_files":   counts["test_normal_files"],
        "test_anomaly_files":  counts["test_anomaly_files"],
    }


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    print("=" * 60)
    print("CNN14 Embeddings + One-Class SVM")
    print("=" * 60)

    all_results = []

    for machine in MACHINES:
        print(f"\n--- {machine.upper()} ---")
        X_train, X_test, y_test, counts = load_cached_embeddings(machine)

        for nu in NU_VALUES:
            result = evaluate_ocsvm(machine, nu, X_train, X_test, y_test, counts)
            all_results.append(result)

    df = pd.DataFrame(all_results)

    # Save full results
    output_dir  = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cnn14_ocsvm_results.csv"
    df.to_csv(output_path, index=False)

    # Print best per machine
    print("\n" + "=" * 60)
    print("Best result per machine (highest ROC-AUC):")
    print("=" * 60)
    best = df.loc[df.groupby("machine")["roc_auc"].idxmax()]
    print(best[["machine", "nu", "roc_auc", "anomaly_recall", "anomaly_f1"]].to_string(index=False))

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()