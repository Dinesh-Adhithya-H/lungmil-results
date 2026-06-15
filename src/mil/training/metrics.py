"""Classification and survival evaluation metrics — all from sklearn / lifelines."""

from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
)


def compute_metrics(labels, probs, threshold: float = 0.5) -> dict:
    """Binary classification metrics via sklearn."""
    labels = np.asarray(labels)
    probs  = np.asarray(probs)
    if np.any(np.isnan(labels)) or np.any(np.isnan(probs)):
        raise ValueError("NaN in compute_metrics")
    if len(np.unique(labels)) < 2:
        print("  [warn] single class in labels — neutral metrics returned")
        return dict(auc=0.5, auprc=0.0, bacc=0.5, mcc=0.0,
                    sens=0.0, spec=0.0, threshold=threshold)
    preds = (probs >= threshold).astype(int)
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
    except Exception:
        sens = spec = 0.0
    return dict(
        auc       = roc_auc_score(labels, probs),
        auprc     = average_precision_score(labels, probs),
        bacc      = balanced_accuracy_score(labels, preds),
        mcc       = matthews_corrcoef(labels, preds),
        sens      = sens,
        spec      = spec,
        threshold = threshold,
    )


def compute_c_index(hazards, times, events) -> float:
    """Harrell's C-index via lifelines.utils.concordance_index."""
    from lifelines.utils import concordance_index
    try:
        return float(concordance_index(times, [-h for h in hazards], events))
    except Exception:
        return 0.5


def _plot_training_curves(history: Dict[str, List], out_dir: Path,
                           tag: str = "") -> None:
    """Save training-curve plots. Silently skips if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    if len(history) == 1:
        axes = [axes]
    for ax, (name, vals) in zip(axes, history.items()):
        ax.plot(vals); ax.set_title(name); ax.set_xlabel("eval step")
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(out_dir / f"curves_{tag}.png", dpi=100)
    plt.close(fig)
