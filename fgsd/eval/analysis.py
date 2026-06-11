"""
Analysis and visualization for FGSD experiments.

Produces figures and analysis for the paper:
1. Per-position rejection rate analysis
2. Feature importance analysis (which probe features matter)
3. Rejection taxonomy (what types of tokens get rejected)
4. Cross-model transfer analysis
5. Speedup vs threshold tradeoff curves
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


def analyze_rejection_patterns(
    data_dict: Dict[str, torch.Tensor],
    tokenizer=None,
) -> Dict:
    """Analyze patterns in token rejection across positions and types.

    Args:
        data_dict: Collected hidden state data with labels
        tokenizer: Optional tokenizer for token-level analysis

    Returns:
        Analysis results dictionary
    """
    labels = data_dict["labels"].numpy()
    positions = data_dict["positions"].numpy()

    results = {}

    # Per-position rejection rate
    unique_positions = sorted(np.unique(positions))
    per_pos_stats = {}
    for pos in unique_positions:
        mask = positions == pos
        reject_rate = labels[mask].mean()
        count = mask.sum()
        per_pos_stats[int(pos)] = {
            "reject_rate": float(reject_rate),
            "count": int(count),
        }
    results["per_position"] = per_pos_stats

    # Overall statistics
    results["overall"] = {
        "total_samples": int(len(labels)),
        "reject_rate": float(labels.mean()),
        "num_positions": len(unique_positions),
        "max_position": int(max(unique_positions)),
    }

    # Rejection rate trend (does it increase with position?)
    if len(unique_positions) >= 3:
        rates = [per_pos_stats[p]["reject_rate"] for p in unique_positions]
        # Linear regression of rejection rate vs position
        x = np.array(unique_positions, dtype=float)
        y = np.array(rates, dtype=float)
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            results["rejection_trend_slope"] = float(slope)
            results["rejection_trend_direction"] = "increasing" if slope > 0.01 else (
                "decreasing" if slope < -0.01 else "stable"
            )

    # Token-level analysis if tokenizer provided
    if tokenizer is not None and "draft_token_ids" in data_dict:
        token_ids = data_dict["draft_token_ids"].numpy()
        results["token_analysis"] = _analyze_token_types(token_ids, labels, tokenizer)

    logger.info(
        f"Rejection analysis: {results['overall']['total_samples']} samples, "
        f"reject_rate={results['overall']['reject_rate']:.3f}, "
        f"trend={results.get('rejection_trend_direction', 'N/A')}"
    )

    return results


def _analyze_token_types(
    token_ids: np.ndarray,
    labels: np.ndarray,
    tokenizer,
) -> Dict:
    """Analyze rejection rates by token type."""
    results = {}

    # Decode tokens for categorization
    decoded = []
    for tid in token_ids:
        try:
            decoded.append(tokenizer.decode([int(tid)]))
        except Exception:
            decoded.append("")

    # Categorize tokens
    categories = {
        "punctuation": [],
        "number": [],
        "whitespace": [],
        "short_word": [],  # 1-3 chars
        "long_word": [],   # 4+ chars
        "special": [],
    }

    for i, token_text in enumerate(decoded):
        stripped = token_text.strip()
        if not stripped or all(c in ' \t\n' for c in token_text):
            categories["whitespace"].append(i)
        elif all(c in '.,;:!?()[]{}"\'-/' for c in stripped):
            categories["punctuation"].append(i)
        elif stripped.isdigit() or stripped.replace('.', '').isdigit():
            categories["number"].append(i)
        elif len(stripped) <= 3:
            categories["short_word"].append(i)
        elif stripped.isalpha():
            categories["long_word"].append(i)
        else:
            categories["special"].append(i)

    for cat, indices in categories.items():
        if indices:
            cat_labels = labels[indices]
            results[cat] = {
                "count": len(indices),
                "reject_rate": float(cat_labels.mean()),
            }

    return results


def analyze_probe_features(
    probe: torch.nn.Module,
    data_dict: Dict[str, torch.Tensor],
    input_source: str = "draft_hidden",
    top_k: int = 20,
) -> Dict:
    """Analyze which features the probe uses most.

    For linear probes, this is straightforward (weight magnitudes).
    For MLPs, uses gradient-based feature importance.

    Args:
        probe: Trained probe model
        data_dict: Test data
        input_source: Feature source type
        top_k: Number of top features to report

    Returns:
        Feature importance analysis
    """
    results = {}

    # For linear probes: direct weight analysis
    if hasattr(probe, 'linear'):
        weights = probe.linear.weight.detach().cpu().squeeze()
        abs_weights = weights.abs()
        top_indices = abs_weights.topk(min(top_k, len(abs_weights))).indices
        results["method"] = "weight_magnitude"
        results["top_features"] = [
            {"index": int(idx), "weight": float(weights[idx]), "abs_weight": float(abs_weights[idx])}
            for idx in top_indices
        ]
        results["weight_stats"] = {
            "mean_abs": float(abs_weights.mean()),
            "std_abs": float(abs_weights.std()),
            "max_abs": float(abs_weights.max()),
            "sparsity": float((abs_weights < 0.01 * abs_weights.max()).float().mean()),
        }

    # For MLPs: gradient-based importance
    elif hasattr(probe, 'mlp'):
        results["method"] = "gradient_importance"
        if input_source in ["draft_hidden", "target_hidden", "combined"]:
            features_key = {
                "draft_hidden": "draft_hidden",
                "target_hidden": "target_hidden",
                "combined": None,  # handled separately
            }.get(input_source)

            if features_key and features_key in data_dict:
                features = data_dict[features_key][:1000].float().requires_grad_(True)
                features = features.to(next(probe.parameters()).device)

                logits = probe(features).squeeze()
                loss = logits.sum()
                loss.backward()

                grads = features.grad.abs().mean(dim=0).cpu()
                top_indices = grads.topk(min(top_k, len(grads))).indices
                results["top_features"] = [
                    {"index": int(idx), "importance": float(grads[idx])}
                    for idx in top_indices
                ]
                results["importance_stats"] = {
                    "mean": float(grads.mean()),
                    "std": float(grads.std()),
                    "max": float(grads.max()),
                }

    return results


def threshold_sweep(
    probe: torch.nn.Module,
    data_dict: Dict[str, torch.Tensor],
    input_source: str = "draft_hidden",
    thresholds: Optional[List[float]] = None,
    norm_stats: Optional[Dict[str, torch.Tensor]] = None,
) -> List[Dict]:
    """Sweep rejection thresholds to plot speedup vs quality tradeoff.

    For each threshold, computes:
    - What fraction of draft tokens would be stopped early
    - What fraction of those are true rejections (precision)
    - What fraction of true rejections are caught (recall)

    Args:
        probe: Trained probe
        data_dict: Test data
        input_source: Feature source
        thresholds: List of thresholds to try
        norm_stats: Normalization statistics

    Returns:
        List of dicts with metrics per threshold
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Get features and labels
    if input_source == "draft_hidden":
        features = data_dict["draft_hidden"].float()
    elif input_source == "target_hidden":
        features = data_dict["target_hidden"].float()
    elif input_source == "entropy":
        features = data_dict["draft_entropy"].float().unsqueeze(-1)
    else:
        raise ValueError(f"Unknown input_source: {input_source}")

    labels = data_dict["labels"]

    # Normalize
    if norm_stats is not None:
        features = (features - norm_stats["mean"]) / norm_stats["std"]

    # Get predictions
    probe.eval()
    device = next(probe.parameters()).device
    with torch.no_grad():
        all_probs = []
        bs = 512
        for i in range(0, len(features), bs):
            batch = features[i:i+bs].to(device)
            logits = probe(batch).squeeze(-1)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
        all_probs = torch.cat(all_probs)

    labels_np = labels.numpy()
    probs_np = all_probs.numpy()

    results = []
    for t in thresholds:
        preds = (probs_np >= t).astype(int)
        tp = ((preds == 1) & (labels_np == 1)).sum()
        fp = ((preds == 1) & (labels_np == 0)).sum()
        fn = ((preds == 0) & (labels_np == 1)).sum()
        tn = ((preds == 0) & (labels_np == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # "Draft savings": fraction of draft tokens we'd stop early
        stop_rate = preds.mean()

        # "Wasted savings": fraction of stops that were false positives
        # (we stopped on tokens that would have been accepted)
        false_stop_rate = fp / (tp + fp) if (tp + fp) > 0 else 0

        results.append({
            "threshold": t,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "stop_rate": float(stop_rate),
            "false_stop_rate": float(false_stop_rate),
            "true_positive": int(tp),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_negative": int(tn),
        })

    return results


def compute_cross_model_transfer(
    probe: torch.nn.Module,
    source_model: str,
    target_data_dirs: Dict[str, str],
    input_source: str = "draft_hidden",
    norm_stats: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate probe trained on one model pair against other pairs.

    Tests whether rejection patterns are model-specific or universal.

    Args:
        probe: Probe trained on source_model data
        source_model: Name of the model the probe was trained on
        target_data_dirs: Dict mapping model name -> data directory
        input_source: Feature source
        norm_stats: Normalization stats from source model training

    Returns:
        Dict mapping model name -> metrics
    """
    from ..collector.dataset import load_collected_data, ProbeDataset
    from ..probe.evaluate import compute_probe_metrics

    results = {}

    for model_name, data_dir in target_data_dirs.items():
        try:
            data = load_collected_data(data_dir)
            dataset = ProbeDataset(data, input_source=input_source, normalize=False)

            # Apply source model's normalization
            features = dataset.features
            if norm_stats is not None:
                features = (features - norm_stats["mean"]) / norm_stats["std"]

            # Evaluate
            probe.eval()
            device = next(probe.parameters()).device
            with torch.no_grad():
                all_logits = []
                bs = 512
                for i in range(0, len(features), bs):
                    batch = features[i:i+bs].to(device)
                    logits = probe(batch).squeeze(-1)
                    all_logits.append(logits.cpu())
                all_logits = torch.cat(all_logits)

            metrics = compute_probe_metrics(all_logits, dataset.labels)
            results[model_name] = {
                "auroc": metrics["auroc"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"],
                "is_source": model_name == source_model,
            }

            logger.info(
                f"Transfer {source_model} -> {model_name}: "
                f"AUROC={metrics['auroc']:.4f}, F1={metrics['f1']:.4f}"
            )

        except Exception as e:
            logger.warning(f"Failed to evaluate on {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    return results
