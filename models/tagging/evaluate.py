"""
models/tagging/evaluate.py
=========================
Multi-Label Evaluation Suite: Macro mAP, Macro F1, and Macro ROC-AUC.
Compliance: Section 5.2, Table 5 of SANADA Base Paper.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def compute_tagging_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    k_guided: int = None
) -> Dict[str, float]:
    """
    Computes Paper Table 5 multi-label tagging metrics.
    Args:
        y_true:   Binary ground truth [N, num_classes] in {0, 1}
        y_scores: Continuous probabilities [N, num_classes] in [0, 1]
        threshold: Decision threshold for binarization (default: 0.5)
        k_guided: If provided, dynamically picks Top-K classes per sample
    Returns:
        {"mAP": float, "macro_f1": float, "macro_roc_auc": float}
    """
    n_samples, n_classes = y_true.shape

    # 1. Macro mAP (Mean Average Precision)
    aps = []
    for c in range(n_classes):
        if np.sum(y_true[:, c]) > 0:
            ap = average_precision_score(y_true[:, c], y_scores[:, c])
            aps.append(ap)
    m_ap = float(np.mean(aps)) if aps else 0.0

    # 2. Binary predictions (Count-Guided Top-K or Fixed Threshold)
    if k_guided is not None and k_guided > 0:
        y_pred = np.zeros_like(y_scores)
        for i in range(n_samples):
            top_k_indices = np.argsort(y_scores[i])[::-1][:k_guided]
            y_pred[i, top_k_indices] = 1.0
    else:
        y_pred = (y_scores >= threshold).astype(np.float32)

    # Macro F1
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    # 3. Macro ROC-AUC
    aucs = []
    for c in range(n_classes):
        if len(np.unique(y_true[:, c])) > 1:
            auc = roc_auc_score(y_true[:, c], y_scores[:, c])
            aucs.append(auc)
    macro_auc = float(np.mean(aucs)) if aucs else 0.5

    return {
        "mAP": round(m_ap, 4),
        "macro_f1": round(macro_f1, 4),
        "macro_roc_auc": round(macro_auc, 4)
    }
