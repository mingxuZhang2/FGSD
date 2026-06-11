"""
Adaptive draft controller using probe predictions.

Integrates into EAGLE-3's speculative decoding loop to dynamically
control draft length based on predicted rejection probability.

Two strategies:
1. Early stopping: If P(reject) > threshold at position k, stop drafting
   at k tokens (avoid wasting compute on positions likely to be rejected).
2. Tree pruning: Prune branches in EAGLE-3's draft tree whose cumulative
   P(reject) exceeds a threshold.
"""

import sys
import os
import time
import json
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

EAGLE_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "EAGLE")
if EAGLE_ROOT not in sys.path:
    sys.path.insert(0, EAGLE_ROOT)

from eagle.model.ea_model import EaModel
from eagle.model.utils import (
    prepare_logits_processor,
    initialize_tree,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    reset_tree_mode,
)
from eagle.model.kv_cache import initialize_past_key_values

from ..probe.models import create_probe

logger = logging.getLogger(__name__)


class AdaptiveDraftController:
    """Controls EAGLE-3 draft generation using probe predictions.

    After each draft step, evaluates the probe on the draft model's hidden
    states. If P(rejection) exceeds the threshold, stops drafting early
    to avoid wasting compute on likely-rejected positions.
    """

    def __init__(
        self,
        probe: nn.Module,
        threshold: float = 0.5,
        max_draft_length: int = 8,
        min_draft_length: int = 1,
        strategy: str = "threshold",
        norm_stats: Optional[Dict[str, torch.Tensor]] = None,
        max_extended_depth: Optional[int] = None,
        extend_threshold: float = 0.2,
        max_position: int = 10,
    ):
        """Initialize adaptive controller.

        Args:
            probe: Trained rejection probe
            threshold: P(reject) threshold for stopping
            max_draft_length: Maximum draft tokens to generate
            min_draft_length: Minimum draft tokens before stopping is allowed
            strategy: "threshold" for early stopping, "confidence" for
                      confidence-weighted stopping
            norm_stats: Feature normalization stats from training
            max_extended_depth: If set (> base depth), the adaptive draft may
                extend beyond the base depth while the probe predicts
                continued acceptance (bidirectional adaptive depth)
            extend_threshold: Mean P(reject) below which extension continues
        """
        self.probe = probe
        self.probe.eval()
        self.threshold = threshold
        self.max_draft_length = max_draft_length
        self.min_draft_length = min_draft_length
        self.strategy = strategy
        self.norm_stats = norm_stats
        self.max_extended_depth = max_extended_depth
        self.extend_threshold = extend_threshold
        self.max_position = max_position

        # Statistics tracking
        self._total_steps = 0
        self._early_stops = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0

    @torch.no_grad()
    def predict_rejection(
        self, hidden_state: torch.Tensor, position: Optional[int] = None
    ) -> float:
        """Predict rejection probability for a draft token.

        Args:
            hidden_state: Draft model hidden state, shape [hidden_dim] or [1, hidden_dim]
            position: Draft position index (if probe is position-aware)

        Returns:
            P(rejection) as a float
        """
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)

        # Normalize if stats available
        if self.norm_stats is not None:
            mean = self.norm_stats["mean"].to(hidden_state.device)
            std = self.norm_stats["std"].to(hidden_state.device)
            hidden_state = (hidden_state - mean) / std

        if hasattr(self.probe, 'input_proj') and position is not None:
            # PositionAwareProbe
            pos_tensor = torch.tensor([position], device=hidden_state.device)
            logit = self.probe(hidden_state, pos_tensor)
        else:
            logit = self.probe(hidden_state)

        return torch.sigmoid(logit).item()

    @torch.no_grad()
    def predict_rejection_batch(
        self, hidden_states: torch.Tensor, position: Optional[int] = None
    ) -> torch.Tensor:
        """Predict P(rejection) for a batch of hidden states in one forward.

        Args:
            hidden_states: Shape [n, hidden_dim]
            position: Draft depth for all n states (position-aware probes)

        Returns:
            P(rejection) tensor of shape [n]
        """
        if self.norm_stats is not None:
            mean = self.norm_stats["mean"].to(hidden_states.device)
            std = self.norm_stats["std"].to(hidden_states.device)
            hidden_states = (hidden_states - mean) / std
        hidden_states = hidden_states.float()

        if hasattr(self.probe, "input_proj") and position is not None:
            pos_tensor = torch.full(
                (hidden_states.shape[0],), position,
                dtype=torch.long, device=hidden_states.device,
            )
            logits = self.probe(hidden_states, pos_tensor)
        else:
            logits = self.probe(hidden_states)

        return torch.sigmoid(logits).flatten()

    @torch.no_grad()
    def should_stop_drafting(
        self,
        hidden_state: torch.Tensor,
        draft_position: int,
    ) -> bool:
        """Decide whether to stop drafting at current position.

        Args:
            hidden_state: Current draft hidden state
            draft_position: Current position in draft sequence (0-indexed)

        Returns:
            True if drafting should stop
        """
        # Always draft at least min_draft_length tokens
        if draft_position < self.min_draft_length:
            return False

        # Always stop at max_draft_length
        if draft_position >= self.max_draft_length:
            return True

        p_reject = self.predict_rejection(hidden_state, draft_position)

        if self.strategy == "threshold":
            return p_reject > self.threshold
        elif self.strategy == "confidence":
            # More aggressive stopping at later positions
            adjusted_threshold = self.threshold * (1.0 - 0.05 * draft_position)
            return p_reject > max(adjusted_threshold, 0.3)
        else:
            return p_reject > self.threshold

    @torch.no_grad()
    def predict_mean_rejection(
        self, hidden_states: torch.Tensor, max_candidates: int = 10
    ) -> float:
        """Predict mean P(rejection) across multiple hidden states.

        Args:
            hidden_states: Shape [n, hidden_dim]
            max_candidates: Max number of states to evaluate

        Returns:
            Mean P(rejection)
        """
        n = min(hidden_states.shape[0], max_candidates)
        total = 0.0
        for i in range(n):
            total += self.predict_rejection(hidden_states[i])
        return total / n if n > 0 else 0.0

    def get_adaptive_draft_length(
        self, hidden_states: torch.Tensor
    ) -> int:
        """Given a sequence of draft hidden states, determine optimal length.

        Scans positions sequentially and stops at the first position where
        P(reject) exceeds the threshold.

        Args:
            hidden_states: Draft hidden states, shape [seq_len, hidden_dim]

        Returns:
            Recommended draft length
        """
        for pos in range(hidden_states.shape[0]):
            if self.should_stop_drafting(hidden_states[pos], pos):
                return max(pos, self.min_draft_length)
        return hidden_states.shape[0]

    def update_stats(self, draft_length: int, accepted_length: int, early_stopped: bool):
        """Update running statistics."""
        self._total_steps += 1
        self._total_draft_tokens += draft_length
        self._total_accepted_tokens += accepted_length
        if early_stopped:
            self._early_stops += 1

    def get_stats(self) -> Dict[str, float]:
        """Get controller statistics."""
        if self._total_steps == 0:
            return {}
        return {
            "total_steps": self._total_steps,
            "early_stop_rate": self._early_stops / self._total_steps,
            "avg_draft_length": self._total_draft_tokens / self._total_steps,
            "avg_accepted_length": self._total_accepted_tokens / self._total_steps,
            "acceptance_rate": (
                self._total_accepted_tokens / self._total_draft_tokens
                if self._total_draft_tokens > 0 else 0
            ),
        }

    def reset_stats(self):
        """Reset statistics."""
        self._total_steps = 0
        self._early_stops = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: str = "cuda",
        threshold: float = 0.5,
        max_draft_length: int = 8,
        min_draft_length: int = 1,
        strategy: str = "threshold",
        max_extended_depth: Optional[int] = None,
        extend_threshold: float = 0.2,
        max_position: int = 10,
    ) -> "AdaptiveDraftController":
        """Load controller from a saved probe checkpoint.

        Args:
            checkpoint_dir: Directory containing best.pt and config.json
            device: Device to load probe on
            threshold: Rejection threshold
            max_draft_length: Max draft length
            min_draft_length: Min draft length
            strategy: Stopping strategy

        Returns:
            Initialized AdaptiveDraftController
        """
        # Load probe config
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create probe architecture
        probe = create_probe(
            probe_type=config["probe_type"],
            input_dim=config["input_dim"],
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            use_position=config.get("use_position", False),
        )

        # Load weights
        checkpoint = torch.load(
            os.path.join(checkpoint_dir, "best.pt"),
            map_location=device,
            weights_only=False,
        )
        probe.load_state_dict(checkpoint["model_state_dict"])
        probe.to(device)
        probe.eval()

        # Load normalization stats if available
        norm_stats_path = os.path.join(checkpoint_dir, "norm_stats.pt")
        norm_stats = None
        if os.path.exists(norm_stats_path):
            norm_stats = torch.load(norm_stats_path, map_location=device, weights_only=False)

        logger.info(
            f"Loaded probe from {checkpoint_dir}: "
            f"type={config['probe_type']}, "
            f"val_auroc={checkpoint['metrics'].get('auroc', 'N/A')}"
        )

        return cls(
            probe=probe,
            threshold=threshold,
            max_draft_length=max_draft_length,
            min_draft_length=min_draft_length,
            strategy=strategy,
            norm_stats=norm_stats,
            max_extended_depth=max_extended_depth,
            extend_threshold=extend_threshold,
            max_position=max_position,
        )
