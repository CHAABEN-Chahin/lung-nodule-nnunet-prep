"""
metrics.py — Evaluation metrics for Bloc 3.

Priority order (from design doc, Section 10.1):
  1. Sensitivity  (most critical — missing a malignant nodule is potentially fatal)
  2. AUC-ROC      (overall discrimination ability)
  3. F1-Score     (balance between precision and recall)
  4. MAE          (regression accuracy, average distance to true score)

Minimum thresholds for backbone selection (non-negotiable clinical floor):
  Sensitivity >= 0.85
  AUC-ROC     >= 0.80
  F1-Score    >= 0.75
  MAE         <= 0.60

These thresholds are applied BEFORE comparing backbones.
A backbone that fails ANY single threshold is eliminated regardless
of its performance on other metrics.
"""

import numpy as np
import torch
from typing import Dict, List
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    recall_score,       # sensitivity = recall for positive class
    confusion_matrix,
)


# Minimum thresholds for clinical validity (do not lower these)
THRESHOLDS = {
    "sensitivity": 0.85,
    "auc_roc":     0.80,
    "f1_score":    0.75,
    "mae":         0.60,   # note: mae must be BELOW threshold
}


def compute_metrics(
    true_labels:  np.ndarray,   # (N,)  binary ground truth {0, 1}
    true_scores:  np.ndarray,   # (N,)  continuous ground truth [1.0, 5.0]
    pred_probs:   np.ndarray,   # (N,)  P(malignant) from binary head
    pred_scores:  np.ndarray,   # (N,)  regression score predictions [1.0, 5.0]
    threshold:    float = 0.5,  # binarization threshold for sensitivity/F1
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for one epoch / evaluation pass.

    Args:
        true_labels:  Ground truth binary labels (0=benign, 1=malignant).
        true_scores:  Ground truth continuous malignancy scores.
        pred_probs:   Predicted probability of malignancy (from BinaryHead).
        pred_scores:  Predicted malignancy scores (from RegressionHead).
        threshold:    Confidence threshold for binary decision (default 0.5).

    Returns:
        Dict with keys: sensitivity, specificity, auc_roc, f1_score, mae,
                        and threshold_pass (bool — all minimums met).
    """
    # Binarize predictions
    pred_labels = (pred_probs >= threshold).astype(int)

    # ── Sensitivity (recall for malignant class) ──────────────────────────────
    # TP / (TP + FN) — fraction of true malignant cases correctly detected
    sensitivity = recall_score(true_labels, pred_labels, pos_label=1,
                               zero_division=0.0)

    # ── Specificity ───────────────────────────────────────────────────────────
    # TN / (TN + FP) — fraction of true benign cases correctly identified
    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0

    # ── AUC-ROC ───────────────────────────────────────────────────────────────
    try:
        auc_roc = roc_auc_score(true_labels, pred_probs)
    except ValueError:
        # Only one class present in batch (can happen early in training)
        auc_roc = 0.0

    # ── F1-Score ──────────────────────────────────────────────────────────────
    f1 = f1_score(true_labels, pred_labels, pos_label=1, zero_division=0.0)

    # ── MAE (regression) ──────────────────────────────────────────────────────
    mae = float(np.mean(np.abs(pred_scores - true_scores)))

    # ── Threshold check ───────────────────────────────────────────────────────
    passes_thresholds = (
        sensitivity >= THRESHOLDS["sensitivity"] and
        auc_roc     >= THRESHOLDS["auc_roc"]     and
        f1          >= THRESHOLDS["f1_score"]     and
        mae         <= THRESHOLDS["mae"]
    )

    return {
        "sensitivity":       round(sensitivity,  4),
        "specificity":       round(specificity,  4),
        "auc_roc":           round(auc_roc,      4),
        "f1_score":          round(f1,           4),
        "mae":               round(mae,          4),
        "threshold_pass":    passes_thresholds,
    }


def collect_epoch_outputs(
    all_true_labels:  List[torch.Tensor],
    all_true_scores:  List[torch.Tensor],
    all_pred_probs:   List[torch.Tensor],
    all_pred_scores:  List[torch.Tensor],
) -> Dict[str, float]:
    """
    Aggregate batch-level tensors collected during an epoch and compute metrics.

    Args:
        all_true_labels:  List of (B,) tensors from each batch.
        all_true_scores:  List of (B,) tensors from each batch.
        all_pred_probs:   List of (B,) tensors — P(malignant).
        all_pred_scores:  List of (B,) tensors — regression scores.

    Returns:
        Metrics dict (same as compute_metrics).
    """
    true_labels  = torch.cat(all_true_labels).numpy()
    true_scores  = torch.cat(all_true_scores).numpy()
    pred_probs   = torch.cat(all_pred_probs).numpy()
    pred_scores  = torch.cat(all_pred_scores).numpy()

    return compute_metrics(true_labels, true_scores, pred_probs, pred_scores)
