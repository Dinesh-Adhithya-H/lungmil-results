"""Classification and survival evaluation metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score,
    confusion_matrix, roc_auc_score,
)


def _mcc(labels, preds) -> float:
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        num = tp * tn - fp * fn
        den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
        return float(num / den) if den > 0 else 0.0
    except Exception:
        return 0.0


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    """
    Binary classification metrics.

    Returns dict with keys: auc, auprc, bacc, mcc, sens, spec, threshold.
    Raises ValueError on NaN inputs.
    """
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    if np.any(np.isnan(labels)) or np.any(np.isnan(probs)):
        raise ValueError("NaN in compute_metrics")
    if len(np.unique(labels)) < 2:
        print("  [warn] single class in labels — neutral metrics returned")
        return dict(auc=0.5, auprc=0.0, bacc=0.5, mcc=0.0,
                    sens=0.0, spec=0.0, threshold=threshold)
    preds = (probs >= threshold).astype(int)
    m = dict(
        auc   = roc_auc_score(labels, probs),
        auprc = average_precision_score(labels, probs),
        bacc  = balanced_accuracy_score(labels, preds),
        mcc   = _mcc(labels, preds),
    )
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        m["sens"] = tp / max(tp + fn, 1)
        m["spec"] = tn / max(tn + fp, 1)
    except Exception:
        m["sens"] = m["spec"] = 0.0
    m["threshold"] = threshold
    return m
