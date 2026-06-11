"""
Dataset management for FGSD probe training data.

Handles saving, loading, and creating PyTorch datasets from collected
hidden state + accept/reject data. Uses safetensors for efficient I/O.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.utils.data as data
import numpy as np

logger = logging.getLogger(__name__)


def save_collected_data(
    dataset: Dict[str, torch.Tensor],
    output_dir: str,
    chunk_id: int = 0,
    metadata: Optional[dict] = None,
) -> str:
    """Save collected hidden state data to disk.

    Args:
        dataset: Dictionary of tensors from HiddenStateCollector.collect_flat_dataset
        output_dir: Directory to save to
        chunk_id: Chunk index for this save (for incremental collection)
        metadata: Optional metadata dict to save alongside

    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)

    save_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}.pt")
    torch.save(dataset, save_path)

    if metadata is not None:
        meta_path = os.path.join(output_dir, f"chunk_{chunk_id:04d}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    # Update manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {"chunks": [], "total_samples": 0}

    chunk_info = {
        "chunk_id": chunk_id,
        "path": os.path.basename(save_path),
        "num_samples": int(dataset["labels"].shape[0]),
        "num_rejected": int((dataset["labels"] == 1).sum().item()),
        "num_accepted": int((dataset["labels"] == 0).sum().item()),
        "keys": list(dataset.keys()),
    }
    manifest["chunks"].append(chunk_info)
    manifest["total_samples"] += chunk_info["num_samples"]

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        f"Saved chunk {chunk_id} with {chunk_info['num_samples']} samples to {save_path}"
    )

    return save_path


def load_collected_data(
    data_dir: str,
    chunk_ids: Optional[List[int]] = None,
) -> Dict[str, torch.Tensor]:
    """Load collected data from disk, optionally selecting specific chunks.

    Args:
        data_dir: Directory containing saved chunks
        chunk_ids: Optional list of chunk IDs to load. None = load all.

    Returns:
        Merged dictionary of tensors
    """
    manifest_path = os.path.join(data_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        chunks_to_load = manifest["chunks"]
        if chunk_ids is not None:
            chunks_to_load = [c for c in chunks_to_load if c["chunk_id"] in chunk_ids]
    else:
        # Fallback: scan directory for chunk files
        chunk_files = sorted(
            [f for f in os.listdir(data_dir) if f.startswith("chunk_") and f.endswith(".pt")]
        )
        chunks_to_load = [{"path": f} for f in chunk_files]

    all_data: Dict[str, List[torch.Tensor]] = {}
    for chunk_info in chunks_to_load:
        chunk_path = os.path.join(data_dir, chunk_info["path"])
        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        for key, tensor in chunk_data.items():
            if key not in all_data:
                all_data[key] = []
            all_data[key].append(tensor)

    merged = {key: torch.cat(tensors, dim=0) for key, tensors in all_data.items()}

    logger.info(
        f"Loaded {len(chunks_to_load)} chunks with {merged['labels'].shape[0]} total samples"
    )

    return merged


class ProbeDataset(data.Dataset):
    """PyTorch dataset for probe training.

    Provides (features, label) pairs where features depend on the
    configured input source (draft hidden states, entropy, etc).
    """

    def __init__(
        self,
        data_dict: Dict[str, torch.Tensor],
        input_source: str = "draft_hidden",
        normalize: bool = True,
    ):
        """Initialize dataset.

        Args:
            data_dict: Dictionary from load_collected_data
            input_source: Which features to use:
                - "draft_hidden": draft model hidden states
                - "target_hidden": target model hidden states (3 layers concat)
                - "combined": draft_hidden + target_hidden + entropy
                - "entropy": entropy only (scalar feature)
            normalize: Whether to normalize features to zero mean, unit variance
        """
        self.labels = data_dict["labels"].float()
        self.positions = data_dict.get("positions", torch.zeros_like(self.labels)).long()
        self.input_source = input_source

        # Build feature tensor based on input source
        if input_source == "draft_hidden":
            if "draft_hidden" not in data_dict:
                raise ValueError("draft_hidden not found in data. Re-collect with collect_draft_hidden=True.")
            self.features = data_dict["draft_hidden"].float()

        elif input_source == "target_hidden":
            if "target_hidden" not in data_dict:
                raise ValueError("target_hidden not found in data. Re-collect with collect_target_hidden=True.")
            self.features = data_dict["target_hidden"].float()

        elif input_source == "entropy":
            if "draft_entropy" not in data_dict:
                raise ValueError("draft_entropy not found in data. Re-collect with collect_entropy=True.")
            self.features = data_dict["draft_entropy"].float().unsqueeze(-1)

        elif input_source == "combined":
            parts = []
            if "draft_hidden" in data_dict:
                parts.append(data_dict["draft_hidden"].float())
            if "target_hidden" in data_dict:
                parts.append(data_dict["target_hidden"].float())
            if "draft_entropy" in data_dict:
                parts.append(data_dict["draft_entropy"].float().unsqueeze(-1))
            if not parts:
                raise ValueError("No features found for combined input source.")
            self.features = torch.cat(parts, dim=-1)

        else:
            raise ValueError(f"Unknown input_source: {input_source}")

        # Normalize features
        self.mean = None
        self.std = None
        if normalize and self.features.shape[-1] > 1:
            self.mean = self.features.mean(dim=0)
            self.std = self.features.std(dim=0).clamp(min=1e-8)
            self.features = (self.features - self.mean) / self.std

        logger.info(
            f"ProbeDataset: {len(self)} samples, "
            f"feature_dim={self.features.shape[-1]}, "
            f"reject_rate={self.labels.mean():.3f}"
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (features, label, position)."""
        return self.features[idx], self.labels[idx], self.positions[idx]

    @property
    def feature_dim(self) -> int:
        return self.features.shape[-1]

    @property
    def reject_rate(self) -> float:
        return self.labels.mean().item()

    def get_normalization_stats(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get normalization statistics for applying to test data."""
        if self.mean is not None:
            return {"mean": self.mean, "std": self.std}
        return None


def create_train_val_test_splits(
    dataset: ProbeDataset,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[data.Subset, data.Subset, data.Subset]:
    """Split dataset into train/val/test sets.

    Uses stratified splitting to maintain class balance.

    Args:
        dataset: Full dataset
        val_fraction: Fraction for validation
        test_fraction: Fraction for test
        seed: Random seed

    Returns:
        Tuple of (train_subset, val_subset, test_subset)
    """
    n = len(dataset)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)

    # Stratified split: separate positive and negative indices
    pos_indices = (dataset.labels == 1).nonzero(as_tuple=True)[0]
    neg_indices = (dataset.labels == 0).nonzero(as_tuple=True)[0]

    # Shuffle each class
    pos_perm = pos_indices[torch.randperm(len(pos_indices), generator=generator)]
    neg_perm = neg_indices[torch.randperm(len(neg_indices), generator=generator)]

    # Split each class proportionally
    n_pos_test = max(1, int(len(pos_perm) * test_fraction))
    n_pos_val = max(1, int(len(pos_perm) * val_fraction))
    n_neg_test = max(1, int(len(neg_perm) * test_fraction))
    n_neg_val = max(1, int(len(neg_perm) * val_fraction))

    test_indices = torch.cat([pos_perm[:n_pos_test], neg_perm[:n_neg_test]])
    val_indices = torch.cat([
        pos_perm[n_pos_test:n_pos_test + n_pos_val],
        neg_perm[n_neg_test:n_neg_test + n_neg_val],
    ])
    train_indices = torch.cat([
        pos_perm[n_pos_test + n_pos_val:],
        neg_perm[n_neg_test + n_neg_val:],
    ])

    logger.info(
        f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    return (
        data.Subset(dataset, train_indices.tolist()),
        data.Subset(dataset, val_indices.tolist()),
        data.Subset(dataset, test_indices.tolist()),
    )
