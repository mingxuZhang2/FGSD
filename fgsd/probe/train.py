"""
Training loop for rejection prediction probes.

Trains a probe to predict P(rejection) from hidden states, with:
- Class-weighted BCE loss (to handle accept/reject imbalance)
- Early stopping on validation AUROC
- Checkpoint saving
- Training metrics logging
"""

import os
import json
import time
import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from .models import create_probe
from .evaluate import compute_probe_metrics
from ..collector.dataset import ProbeDataset, create_train_val_test_splits

logger = logging.getLogger(__name__)


class ProbeTrainer:
    """Trainer for rejection prediction probes."""

    def __init__(
        self,
        probe: nn.Module,
        train_set: Subset,
        val_set: Subset,
        test_set: Optional[Subset] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        num_epochs: int = 20,
        early_stopping_patience: int = 5,
        use_class_weights: bool = True,
        pos_weight_cap: float = 5.0,
        device: str = "cuda",
        output_dir: str = "results/probes",
        seed: int = 42,
    ):
        self.probe = probe.to(device)
        self.device = device
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.seed = seed

        os.makedirs(output_dir, exist_ok=True)

        # DataLoaders
        self.train_loader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_set, batch_size=batch_size * 2, shuffle=False,
            num_workers=4, pin_memory=True,
        )
        self.test_loader = None
        if test_set is not None:
            self.test_loader = DataLoader(
                test_set, batch_size=batch_size * 2, shuffle=False,
                num_workers=4, pin_memory=True,
            )

        # Compute class weights for imbalanced data
        if use_class_weights:
            # Count labels in training set
            all_labels = []
            for features, label, pos in train_set:
                all_labels.append(label)
            all_labels = torch.stack(all_labels)
            n_pos = (all_labels == 1).sum().float()
            n_neg = (all_labels == 0).sum().float()
            pos_weight = (n_neg / n_pos).clamp(max=pos_weight_cap)
            logger.info(f"Class weights: pos_weight={pos_weight:.2f} (neg/pos = {n_neg}/{n_pos})")
        else:
            pos_weight = torch.tensor(1.0)

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device)
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # LR scheduler: reduce on plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

        # Training state
        self.best_val_auroc = 0.0
        self.epochs_without_improvement = 0
        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_auroc": [],
            "val_f1": [],
            "lr": [],
        }

    def train_epoch(self) -> float:
        """Run one training epoch.

        Returns:
            Average training loss
        """
        self.probe.train()
        total_loss = 0.0
        n_batches = 0

        for features, labels, positions in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            positions = positions.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            if hasattr(self.probe, 'input_proj'):
                # PositionAwareProbe
                logits = self.probe(features, positions).squeeze(-1)
            else:
                logits = self.probe(features).squeeze(-1)

            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.probe.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate probe on a dataset.

        Args:
            data_loader: DataLoader to evaluate on

        Returns:
            Dictionary of metrics
        """
        self.probe.eval()
        all_logits = []
        all_labels = []
        all_positions = []
        total_loss = 0.0
        n_batches = 0

        for features, labels, positions in data_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            positions = positions.to(self.device)

            if hasattr(self.probe, 'input_proj'):
                logits = self.probe(features, positions).squeeze(-1)
            else:
                logits = self.probe(features).squeeze(-1)

            loss = self.criterion(logits, labels)
            total_loss += loss.item()
            n_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_positions.append(positions.cpu())

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)
        all_positions = torch.cat(all_positions)

        metrics = compute_probe_metrics(all_logits, all_labels, all_positions)
        metrics["loss"] = total_loss / n_batches

        return metrics

    def train(self) -> Dict[str, float]:
        """Run full training loop with early stopping.

        Returns:
            Test metrics (if test set provided) or best validation metrics
        """
        logger.info(
            f"Starting training: {self.num_epochs} epochs, "
            f"probe params: {self.probe.num_parameters if hasattr(self.probe, 'num_parameters') else sum(p.numel() for p in self.probe.parameters())}"
        )

        torch.manual_seed(self.seed)

        for epoch in range(self.num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_metrics = self.evaluate(self.val_loader)

            # LR scheduling
            self.scheduler.step(val_metrics["auroc"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_auroc"].append(val_metrics["auroc"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["lr"].append(current_lr)

            elapsed = time.time() - start_time

            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} ({elapsed:.1f}s): "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_AUROC={val_metrics['auroc']:.4f}, "
                f"val_F1={val_metrics['f1']:.4f}, "
                f"lr={current_lr:.2e}"
            )

            # Early stopping check
            if val_metrics["auroc"] > self.best_val_auroc:
                self.best_val_auroc = val_metrics["auroc"]
                self.epochs_without_improvement = 0
                # Save best checkpoint
                self._save_checkpoint("best.pt", epoch, val_metrics)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping at epoch {epoch+1} "
                        f"(best val AUROC: {self.best_val_auroc:.4f})"
                    )
                    break

        # Save final checkpoint
        self._save_checkpoint("final.pt", epoch, val_metrics)

        # Save training history
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

        # Evaluate on test set using best checkpoint
        if self.test_loader is not None:
            self._load_checkpoint("best.pt")
            test_metrics = self.evaluate(self.test_loader)
            logger.info(
                f"Test metrics: AUROC={test_metrics['auroc']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, "
                f"Precision={test_metrics['precision']:.4f}, "
                f"Recall={test_metrics['recall']:.4f}"
            )
            with open(os.path.join(self.output_dir, "test_metrics.json"), "w") as f:
                json.dump(test_metrics, f, indent=2)
            return test_metrics

        return {"best_val_auroc": self.best_val_auroc}

    def _save_checkpoint(self, filename: str, epoch: int, metrics: dict) -> None:
        """Save model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.probe.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_val_auroc": self.best_val_auroc,
        }, path)

    def _load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = os.path.join(self.output_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.probe.load_state_dict(checkpoint["model_state_dict"])


def train_probe_from_data(
    data_dir: str,
    probe_type: str = "mlp",
    input_source: str = "draft_hidden",
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    num_epochs: int = 20,
    early_stopping_patience: int = 5,
    use_class_weights: bool = True,
    pos_weight_cap: float = 5.0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    output_dir: str = "results/probes",
    device: str = "cuda",
    seed: int = 42,
    use_position: bool = False,
) -> Dict[str, float]:
    """End-to-end function to train a probe from collected data.

    Args:
        data_dir: Directory containing collected hidden state data
        probe_type: Probe architecture type
        input_source: Which features to use
        hidden_dim: MLP hidden dimension
        num_layers: Number of MLP layers
        dropout: Dropout rate
        learning_rate: Learning rate
        weight_decay: Weight decay
        batch_size: Batch size
        num_epochs: Maximum epochs
        early_stopping_patience: Patience for early stopping
        use_class_weights: Whether to use class-weighted loss
        pos_weight_cap: Maximum positive weight
        val_fraction: Validation set fraction
        test_fraction: Test set fraction
        output_dir: Output directory for checkpoints and logs
        device: Device to train on
        seed: Random seed
        use_position: Whether to use position-aware probe

    Returns:
        Test metrics dictionary
    """
    from ..collector.dataset import load_collected_data

    # Load data
    data_dict = load_collected_data(data_dir)

    # Subsample if dataset is too large (prevent OOM for high-dim features)
    max_train_samples = 200000
    n = data_dict["labels"].shape[0]
    if n > max_train_samples:
        logger.info(f"Subsampling {n} -> {max_train_samples} samples to fit in memory")
        torch.manual_seed(seed)
        indices = torch.randperm(n)[:max_train_samples]
        data_dict = {k: v[indices] for k, v in data_dict.items()}

    # Create dataset
    dataset = ProbeDataset(data_dict, input_source=input_source, normalize=True)

    # Save normalization stats for inference
    norm_stats = dataset.get_normalization_stats()
    if norm_stats is not None:
        os.makedirs(output_dir, exist_ok=True)
        torch.save(norm_stats, os.path.join(output_dir, "norm_stats.pt"))

    # Split
    train_set, val_set, test_set = create_train_val_test_splits(
        dataset, val_fraction=val_fraction, test_fraction=test_fraction, seed=seed
    )

    # Create probe
    probe = create_probe(
        probe_type=probe_type,
        input_dim=dataset.feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_position=use_position,
    )

    logger.info(f"Created {probe_type} probe with {sum(p.numel() for p in probe.parameters())} parameters")

    # Save config
    config = {
        "probe_type": probe_type,
        "input_source": input_source,
        "input_dim": dataset.feature_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "dropout": dropout,
        "use_position": use_position,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "dataset_size": len(dataset),
        "reject_rate": dataset.reject_rate,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train
    trainer = ProbeTrainer(
        probe=probe,
        train_set=train_set,
        val_set=val_set,
        test_set=test_set,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        batch_size=batch_size,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        use_class_weights=use_class_weights,
        pos_weight_cap=pos_weight_cap,
        device=device,
        output_dir=output_dir,
        seed=seed,
    )

    return trainer.train()
