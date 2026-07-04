"""
Dense Autoencoder with Log-Mel mean/std input.

Autoencoder family:
- Dense AE: Log-Mel mean/std vector
- CNN AE: Log-Mel spectrogram image
- Transformer AE: Log-Mel sequence

Usage:
    python Autoencoder/models/dense_autoencoder.py --machine fan
    python Autoencoder/models/dense_autoencoder.py --all
    python Autoencoder/models/dense_autoencoder.py --all --rebuild-cache
"""

from pathlib import Path
import argparse
import pickle
import random

import librosa
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers


# ------------------------------------------------
# Reproducibility
# ------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ------------------------------------------------
# Paths
# ------------------------------------------------
# This file is expected to be here:
# Autoencoder/models/dense_autoencoder.py
#
# BASE_DIR = Autoencoder/
# PROJECT_DIR = AD_Benchmark_MIMII/

BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BASE_DIR.parent

DATASET_DIR = PROJECT_DIR / "data" / "MIMII"

RESULT_DIR = BASE_DIR / "results"
CACHE_DIR = BASE_DIR / "feature_cache" / "dense"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# Configuration
# ------------------------------------------------

MACHINES = ["fan", "pump", "valve"]

N_MELS = 64
EPOCHS = 150
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.1

LEARNING_RATE = 1e-3
THRESHOLD_PERCENTILE = 95


# ------------------------------------------------
# Feature design notes
# ------------------------------------------------
# File-level experiment:
# one 10-second audio file -> one feature vector -> one anomaly score.
#
# Dense AE cannot directly process a 2D Log-Mel image.
# Therefore, we summarize the Log-Mel spectrogram over time.
#
# Pipeline:
# audio
# -> Log-Mel spectrogram
# -> mean over time for each Mel band
# -> std over time for each Mel band
# -> 128-dimensional vector
#
# With 64 Mel bands:
# 64 mean values + 64 std values = 128 features.
#
# Training:
# only normal sounds are used.
#
# Anomaly score:
# reconstruction error.
# Higher reconstruction error = more anomalous.
#
# Efficiency:
# extracted features are cached in:
# Autoencoder/feature_cache/dense/


# ------------------------------------------------
# Log-Mel feature extraction
# ------------------------------------------------

def extract_logmel_features(audio_path: Path, n_mels: int = N_MELS) -> np.ndarray:
    """
    Convert one audio file into one fixed-size Log-Mel feature vector.

    Output shape:
        (2 * n_mels,)

    Example:
        n_mels = 64 -> 128-dimensional vector
    """

    y, sr = librosa.load(audio_path, sr=None)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
    )

    logmel = librosa.power_to_db(
        mel,
        ref=np.max,
    )

    # Pool over time.
    logmel_mean = np.mean(logmel, axis=1)
    logmel_std = np.std(logmel, axis=1)

    feature_vector = np.concatenate([
        logmel_mean,
        logmel_std,
    ])

    return feature_vector.astype(np.float32)


# ------------------------------------------------
# Load one folder
# ------------------------------------------------

def load_features_from_folder(folder_path: Path, label: int):
    """
    Load all .wav files from one folder.

    label:
        0 = normal
        1 = anomaly
    """

    features = []
    labels = []

    wav_files = sorted(folder_path.glob("*.wav"))

    if len(wav_files) == 0:
        raise FileNotFoundError(f"No .wav files found in: {folder_path}")

    for wav_file in wav_files:
        features.append(extract_logmel_features(wav_file))
        labels.append(label)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return X, y


# ------------------------------------------------
# Feature cache
# ------------------------------------------------

def load_or_create_feature_cache(machine: str, rebuild_cache: bool = False):
    """
    Load cached features if available.
    Otherwise extract features and save them as .pkl.

    Cache contains:
        X_train
        X_test_normal
        X_test_anomaly
        y_test_normal
        y_test_anomaly
    """

    cache_file = CACHE_DIR / f"{machine}_dense_logmel_features.pkl"

    if cache_file.exists() and not rebuild_cache:
        print(f"Loading cached features: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print("Feature cache not found or rebuild requested.")
    print(f"Extracting Log-Mel mean/std features for: {machine}")

    train_normal_dir = DATASET_DIR / machine / "train" / "normal"
    test_normal_dir = DATASET_DIR / machine / "test" / "normal"
    test_anomaly_dir = DATASET_DIR / machine / "test" / "anomaly"

    for folder in [train_normal_dir, test_normal_dir, test_anomaly_dir]:
        if not folder.exists():
            raise FileNotFoundError(f"Expected folder not found: {folder}")

    X_train, _ = load_features_from_folder(train_normal_dir, label=0)
    X_test_normal, y_test_normal = load_features_from_folder(test_normal_dir, label=0)
    X_test_anomaly, y_test_anomaly = load_features_from_folder(test_anomaly_dir, label=1)

    cache = {
        "X_train": X_train,
        "X_test_normal": X_test_normal,
        "X_test_anomaly": X_test_anomaly,
        "y_test_normal": y_test_normal,
        "y_test_anomaly": y_test_anomaly,
    }

    with open(cache_file, "wb") as f:
        pickle.dump(cache, f)

    print(f"Saved feature cache to: {cache_file}")

    return cache


# ------------------------------------------------
# Build Dense Autoencoder
# ------------------------------------------------

def build_dense_autoencoder(input_dim: int) -> models.Model:
    """
    Dense Autoencoder architecture:

        input_dim
        -> 256
        -> 128
        -> 64 bottleneck
        -> 128
        -> 256
        -> input_dim
    """

    l2_reg = regularizers.l2(1e-5)

    input_layer = layers.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(input_layer)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.Dropout(0.1)(x)

    bottleneck = layers.Dense(
        64,
        activation="relu",
        name="bottleneck",
        kernel_regularizer=l2_reg,
    )(x)

    # Decoder
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(bottleneck)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(x)

    output_layer = layers.Dense(input_dim, activation="linear")(x)

    model = models.Model(
        inputs=input_layer,
        outputs=output_layer,
        name="Dense_Autoencoder_LogMel",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    return model


# ------------------------------------------------
# Reconstruction error
# ------------------------------------------------

def compute_reconstruction_error(model: models.Model, X: np.ndarray) -> np.ndarray:
    """
    Compute one reconstruction error per sample.
    """

    X_reconstructed = model.predict(X, verbose=0)

    errors = np.mean(
        np.square(X - X_reconstructed),
        axis=1,
    )

    return errors


# ------------------------------------------------
# Evaluation
# ------------------------------------------------

def evaluate_anomaly_detection(y_true, anomaly_scores, threshold):
    """
    ROC-AUC and PR-AUC use continuous anomaly scores.
    Precision, recall, F1 and confusion matrix use a fixed threshold.
    """

    y_pred = np.where(anomaly_scores > threshold, 1, 0)

    roc_auc = roc_auc_score(y_true, anomaly_scores)
    pr_auc = average_precision_score(y_true, anomaly_scores)

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0,
    )

    return roc_auc, pr_auc, cm, report


# ------------------------------------------------
# Run one machine
# ------------------------------------------------

def run_one_machine(machine: str, rebuild_cache: bool = False) -> dict:
    """
    Train and evaluate Dense AE for one machine.
    """

    print("\n" + "=" * 70)
    print(f"Dense Autoencoder | Machine: {machine}")
    print("=" * 70)

    cache = load_or_create_feature_cache(
        machine=machine,
        rebuild_cache=rebuild_cache,
    )

    X_train = cache["X_train"]

    X_test_normal = cache["X_test_normal"]
    y_test_normal = cache["y_test_normal"]

    X_test_anomaly = cache["X_test_anomaly"]
    y_test_anomaly = cache["y_test_anomaly"]

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    print("Train normal:", X_train.shape)
    print("Test normal:", X_test_normal.shape)
    print("Test anomaly:", X_test_anomaly.shape)

    # Fit scaler only on training normal data.
    # This avoids data leakage.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    input_dim = X_train_scaled.shape[1]

    model = build_dense_autoencoder(input_dim)

    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
        ),
    ]

    print("\nTraining Dense Autoencoder...")

    history = model.fit(
        X_train_scaled,
        X_train_scaled,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        shuffle=True,
        callbacks=training_callbacks,
        verbose=1,
    )

    # Threshold is computed only from training-normal reconstruction errors.
    train_errors = compute_reconstruction_error(model, X_train_scaled)
    threshold = np.percentile(train_errors, THRESHOLD_PERCENTILE)

    # Test anomaly scores.
    test_errors = compute_reconstruction_error(model, X_test_scaled)

    roc_auc, pr_auc, cm, report = evaluate_anomaly_detection(
        y_true=y_test,
        anomaly_scores=test_errors,
        threshold=threshold,
    )

    print("\nThreshold:", threshold)
    print("Confusion matrix:")
    print(cm)

    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    print(f"Accuracy:          {report['accuracy']:.4f}")
    print(f"Anomaly Precision: {report['anomaly']['precision']:.4f}")
    print(f"Anomaly Recall:    {report['anomaly']['recall']:.4f}")
    print(f"Anomaly F1:        {report['anomaly']['f1-score']:.4f}")
    print(f"Macro F1:          {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1:       {report['weighted avg']['f1-score']:.4f}")

    result = {
        "machine": machine,
        "method": "DenseAE",
        "feature_type": "logmel_mean_std",
        "n_mels": N_MELS,
        "input_dim": input_dim,
        "train_normal_files": len(X_train),
        "test_normal_files": len(X_test_normal),
        "test_anomaly_files": len(X_test_anomaly),
        "epochs_configured": EPOCHS,
        "epochs_trained": len(history.history["loss"]),
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": report["accuracy"],
        "normal_precision": report["normal"]["precision"],
        "normal_recall": report["normal"]["recall"],
        "normal_f1": report["normal"]["f1-score"],
        "anomaly_precision": report["anomaly"]["precision"],
        "anomaly_recall": report["anomaly"]["recall"],
        "anomaly_f1": report["anomaly"]["f1-score"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_f1": report["weighted avg"]["f1-score"],
        "tn": cm[0, 0],
        "fp": cm[0, 1],
        "fn": cm[1, 0],
        "tp": cm[1, 1],
    }

    return result


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dense Autoencoder with Log-Mel mean/std features",
    )

    parser.add_argument(
        "--machine",
        choices=MACHINES,
        default="fan",
        help="Machine type: fan, pump, or valve.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all machines.",
    )

    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Re-extract features and overwrite feature cache.",
    )

    args = parser.parse_args()

    selected_machines = MACHINES if args.all else [args.machine]

    all_results = []

    for machine in selected_machines:
        result = run_one_machine(
            machine=machine,
            rebuild_cache=args.rebuild_cache,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)

    print("\nFinal Dense Autoencoder results:")
    print(results_df)

    output_path = RESULT_DIR / "dense_autoencoder_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved results to: {output_path}")

    print("\nBest result per machine based on ROC-AUC:")
    print(results_df[[
        "machine",
        "roc_auc",
        "pr_auc",
        "anomaly_precision",
        "anomaly_recall",
        "anomaly_f1",
        "macro_f1",
    ]])


if __name__ == "__main__":
    main()