"""
Evaluation metrics for rejection prediction probes.

Computes AUROC, F1, precision, recall, calibration, and per-position
breakdown of probe accuracy.
"""

import logging
from typing import Dict, Optional, List

import torch
import numpy as np

logger = logging.getLogger(__name__)


def compute_probe_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics for the probe.

    Args:
        logits: Raw logits (before sigmoid), shape [N]
        labels: Binary labels (1=rejected, 0=accepted), shape [N]
        positions: Draft positions, shape [N]. If provided, computes per-position metrics.
        threshold: Classification threshold on sigmoid(logits)

    Returns:
        Dictionary of metrics
    """
    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()

    metrics = {}

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        metrics["auroc"] = float(roc_auc_score(labels_np, probs))
    except (ImportError, ValueError):
        # ValueError can happen if only one class present
        metrics["auroc"] = 0.0

    # Binary predictions
    preds = (probs >= threshold).astype(int)

    # Accuracy
    metrics["accuracy"] = float((preds == labels_np).mean())

    # Precision, Recall, F1
    tp = ((preds == 1) & (labels_np == 1)).sum()
    fp = ((preds == 1) & (labels_np == 0)).sum()
    fn = ((preds == 0) & (labels_np == 1)).sum()
    tn = ((preds == 0) & (labels_np == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)
    metrics["true_positive"] = int(tp)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_negative"] = int(tn)

    # Class distribution
    metrics["reject_rate"] = float(labels_np.mean())
    metrics["predicted_reject_rate"] = float(preds.mean())

    # Average predicted probability for accepted vs rejected
    if labels_np.sum() > 0:
        metrics["avg_prob_rejected"] = float(probs[labels_np == 1].mean())
    else:
        metrics["avg_prob_rejected"] = 0.0
    if (1 - labels_np).sum() > 0:
        metrics["avg_prob_accepted"] = float(probs[labels_np == 0].mean())
    else:
        metrics["avg_prob_accepted"] = 0.0

    # Calibration (Expected Calibration Error)
    metrics["ece"] = _compute_ece(probs, labels_np, n_bins=10)

    # Per-position metrics
    if positions is not None:
        positions_np = positions.numpy()
        unique_positions = sorted(np.unique(positions_np))
        per_pos_auroc = {}
        per_pos_reject_rate = {}
        per_pos_accuracy = {}
        for pos in unique_positions:
            mask = positions_np == pos
            if mask.sum() < 10:  # skip positions with too few samples
                continue
            pos_labels = labels_np[mask]
            pos_probs = probs[mask]
            pos_preds = preds[mask]

            per_pos_reject_rate[int(pos)] = float(pos_labels.mean())
            per_pos_accuracy[int(pos)] = float((pos_preds == pos_labels).mean())

            try:
                from sklearn.metrics import roc_auc_score
                if len(np.unique(pos_labels)) > 1:
                    per_pos_auroc[int(pos)] = float(roc_auc_score(pos_labels, pos_probs))
            except (ImportError, ValueError):
                pass

        metrics["per_position_auroc"] = per_pos_auroc
        metrics["per_position_reject_rate"] = per_pos_reject_rate
        metrics["per_position_accuracy"] = per_pos_accuracy

    return metrics


def _compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error.

    Measures how well predicted probabilities match empirical frequencies.

    Args:
        probs: Predicted probabilities, shape [N]
        labels: True labels, shape [N]
        n_bins: Number of calibration bins

    Returns:
        ECE value (lower is better)
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_conf - bin_acc)
    return float(ece)


def find_optimal_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    metric: str = "f1",
    thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Find the optimal classification threshold.

    Sweeps thresholds and finds the one that maximizes the target metric.

    Args:
        logits: Raw logits, shape [N]
        labels: Binary labels, shape [N]
        metric: Metric to optimize ("f1", "accuracy", "balanced_accuracy")
        thresholds: Optional list of thresholds to try

    Returns:
        Dict with "best_threshold" and "best_{metric}" values
    """
    if thresholds is None:
        thresholds = [i / 100 for i in range(5, 96)]

    probs = torch.sigmoid(logits).numpy()
    labels_np = labels.numpy()

    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        preds = (probs >= t).astype(int)

        if metric == "f1":
            tp = ((preds == 1) & (labels_np == 1)).sum()
            fp = ((preds == 1) & (labels_np == 0)).sum()
            fn = ((preds == 0) & (labels_np == 1)).sum()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        elif metric == "accuracy":
            score = (preds == labels_np).mean()
        elif metric == "balanced_accuracy":
            pos_acc = (preds[labels_np == 1] == 1).mean() if labels_np.sum() > 0 else 0
            neg_acc = (preds[labels_np == 0] == 0).mean() if (1 - labels_np).sum() > 0 else 0
            score = (pos_acc + neg_acc) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return {
        "best_threshold": best_threshold,
        f"best_{metric}": float(best_score),
    }
