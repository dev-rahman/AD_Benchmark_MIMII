"""
Evaluation utilities: AUC, pAUC (DCASE), F1, confusion matrix.
Authors: Simin Mirzadeh, Muhammad Rahman — University of Bamberg, 2026
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


def compute_pauc(y_true: np.ndarray, scores: np.ndarray,
                 max_fpr: float = 0.1) -> float:
    """
    Partial AUC normalised to [0, 1], computed up to max_fpr (default 10%).
    This is the primary DCASE Challenge evaluation metric.
    Higher scores = more anomalous.
    """
    fpr, tpr, _ = roc_curve(y_true, scores)
    mask = fpr <= max_fpr
    if mask.sum() < 2:
        return float("nan")
    return auc(fpr[mask], tpr[mask]) / max_fpr


def best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Return the threshold that maximises F1 on the test set."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    f1s = []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        f1s.append(f1_score(y_true, preds, zero_division=0))
    return float(thresholds[np.argmax(f1s)])


def evaluate(y_true: np.ndarray, scores: np.ndarray,
             machine: str = "", method: str = "") -> dict:
    """
    Full evaluation dictionary.
    y_true: 0 = normal, 1 = anomaly
    scores: float array — higher means more anomalous
    """
    roc_auc = roc_auc_score(y_true, scores)
    pauc    = compute_pauc(y_true, scores)
    thresh  = best_threshold(y_true, scores)
    preds   = (scores >= thresh).astype(int)

    acc  = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    cm   = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (None, None, None, None)

    result = dict(
        machine=machine, method=method,
        AUC=round(roc_auc, 4), pAUC=round(pauc, 4),
        Accuracy=round(acc, 4), Precision=round(prec, 4),
        Recall=round(rec, 4), F1=round(f1, 4),
        Threshold=round(thresh, 6),
        TN=int(tn) if tn is not None else None,
        FP=int(fp) if fp is not None else None,
        FN=int(fn) if fn is not None else None,
        TP=int(tp) if tp is not None else None,
    )
    print(f"  [{method:25s} | {machine:5s}]  AUC={roc_auc:.4f}  pAUC={pauc:.4f}  F1={f1:.4f}")
    return result


def save_results(results: list[dict], csv_path: str) -> None:
    """Append results list to a CSV file."""
    pd.DataFrame(results).to_csv(csv_path, index=False)
    print(f"  → Saved: {csv_path}")
