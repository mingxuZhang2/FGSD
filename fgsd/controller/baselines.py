"""
Baseline draft controllers for comparison with FGSD.

Implements:
1. SVIPEntropyController: Reimplementation of SVIP's entropy-based stopping
   (no open-source code exists). Based on EMNLP 2024 paper.
2. FixedLengthController: Fixed draft length (standard EAGLE-3 behavior)
3. OracleController: Uses ground-truth accept/reject labels (upper bound)
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)


class SVIPEntropyController:
    """SVIP entropy-based adaptive draft stopping.

    Based on: "SVIP: Towards Efficient Speculative Decoding via
    Self-Verification with Integrated Probabilities" (EMNLP 2024)

    SVIP stops drafting when the draft model's output entropy exceeds a
    threshold, as high entropy indicates uncertainty about the next token
    (and thus likely rejection by the target model).

    This is our primary baseline since SVIP has no open-source code.
    """

    def __init__(
        self,
        entropy_threshold: float = 2.0,
        max_draft_length: int = 8,
        min_draft_length: int = 1,
        max_extended_depth: int = 0,
        extend_threshold: float = 0.5,
    ):
        """Initialize SVIP entropy controller.

        Args:
            entropy_threshold: Stop when entropy exceeds this value.
                Typical range: 1.0-3.0 (depends on vocab size and task).
                SVIP paper suggests tuning per model pair.
            max_draft_length: Maximum draft tokens
            min_draft_length: Minimum draft tokens before stopping
            max_extended_depth: If > base depth, extension is allowed while
                entropy stays below extend_threshold (mirrors FGSD bidir)
            extend_threshold: Entropy (nats) below which extension continues
        """
        self.entropy_threshold = entropy_threshold
        self.max_draft_length = max_draft_length
        self.min_draft_length = min_draft_length

        # Interface used by adaptive_topk_genrate (same control loop as the
        # FGSD probe; only the signal source differs)
        self.signal = "entropy"
        self.threshold = entropy_threshold
        self.max_extended_depth = max_extended_depth or None
        self.extend_threshold = extend_threshold

        # Statistics
        self._total_steps = 0
        self._early_stops = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0

    @torch.no_grad()
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute entropy of logit distribution.

        Args:
            logits: Shape [vocab_size] or [1, vocab_size]

        Returns:
            Entropy value (scalar)
        """
        if logits.dim() > 1:
            logits = logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum()
        return entropy.item()

    def should_stop_drafting(
        self,
        draft_logits: torch.Tensor,
        draft_position: int,
    ) -> bool:
        """Decide whether to stop drafting based on entropy.

        Args:
            draft_logits: Output logits of draft model at current position
            draft_position: Current position in draft sequence (0-indexed)

        Returns:
            True if drafting should stop
        """
        if draft_position < self.min_draft_length:
            return False
        if draft_position >= self.max_draft_length:
            return True

        entropy = self.compute_entropy(draft_logits)
        return entropy > self.entropy_threshold

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
            "entropy_threshold": self.entropy_threshold,
        }

    def reset_stats(self):
        """Reset statistics."""
        self._total_steps = 0
        self._early_stops = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0


class FixedLengthController:
    """Fixed draft length controller (standard EAGLE-3 behavior).

    Always generates a fixed number of draft tokens. This is the default
    behavior of EAGLE-3 and serves as the most basic baseline.
    """

    def __init__(self, draft_length: int = 5):
        """Initialize fixed length controller.

        Args:
            draft_length: Number of draft tokens to generate each step
        """
        self.draft_length = draft_length

        self._total_steps = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0

    def should_stop_drafting(self, draft_position: int, **kwargs) -> bool:
        """Stop when reaching the fixed draft length."""
        return draft_position >= self.draft_length

    def update_stats(self, draft_length: int, accepted_length: int, early_stopped: bool):
        self._total_steps += 1
        self._total_draft_tokens += draft_length
        self._total_accepted_tokens += accepted_length

    def get_stats(self) -> Dict[str, float]:
        if self._total_steps == 0:
            return {}
        return {
            "total_steps": self._total_steps,
            "fixed_draft_length": self.draft_length,
            "avg_accepted_length": self._total_accepted_tokens / self._total_steps,
            "acceptance_rate": (
                self._total_accepted_tokens / self._total_draft_tokens
                if self._total_draft_tokens > 0 else 0
            ),
        }

    def reset_stats(self):
        self._total_steps = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0


class OracleController:
    """Oracle controller that uses ground-truth accept/reject labels.

    This serves as an upper bound on performance: if we had perfect
    knowledge of which tokens would be rejected, what is the maximum
    speedup we could achieve?

    Note: This controller cannot be used in real inference. It requires
    running the full speculative decoding loop first to obtain labels,
    then simulating what would happen with perfect prediction.
    """

    def __init__(self, max_draft_length: int = 8):
        self.max_draft_length = max_draft_length
        self._total_steps = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0
        self._saved_draft_tokens = 0

    def compute_oracle_draft_length(
        self, accept_mask: torch.Tensor
    ) -> int:
        """Given ground-truth accept mask, return optimal draft length.

        The optimal length is the first rejection position, since all
        tokens after that are wasted.

        Args:
            accept_mask: Boolean tensor, True = accepted

        Returns:
            Optimal draft length
        """
        # Find first rejection
        reject_positions = (~accept_mask).nonzero(as_tuple=True)[0]
        if len(reject_positions) == 0:
            # All accepted: draft the full length
            return min(len(accept_mask), self.max_draft_length)
        else:
            # Stop at first rejection
            return reject_positions[0].item()

    def update_stats(
        self,
        accept_mask: torch.Tensor,
        actual_draft_length: int,
    ):
        """Update statistics given a ground-truth accept mask."""
        oracle_length = self.compute_oracle_draft_length(accept_mask)
        accepted = accept_mask.sum().item()

        self._total_steps += 1
        self._total_draft_tokens += oracle_length
        self._total_accepted_tokens += accepted
        self._saved_draft_tokens += (actual_draft_length - oracle_length)

    def get_stats(self) -> Dict[str, float]:
        if self._total_steps == 0:
            return {}
        return {
            "total_steps": self._total_steps,
            "avg_oracle_draft_length": self._total_draft_tokens / self._total_steps,
            "avg_accepted_length": self._total_accepted_tokens / self._total_steps,
            "total_saved_tokens": self._saved_draft_tokens,
            "avg_saved_per_step": self._saved_draft_tokens / self._total_steps,
        }

    def reset_stats(self):
        self._total_steps = 0
        self._total_draft_tokens = 0
        self._total_accepted_tokens = 0
        self._saved_draft_tokens = 0
