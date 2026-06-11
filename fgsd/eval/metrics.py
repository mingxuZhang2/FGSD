"""
Speculative decoding evaluation metrics.

Computes the standard metrics used in SD literature:
1. Speedup ratio: Wall-clock speedup vs vanilla autoregressive
2. Average acceptance length (tau): Mean tokens accepted per draft-verify cycle
3. Per-position acceptance rate (n-alpha): P(accept at position n)
4. Tokens per second: Raw throughput
5. Overhead: Time spent on probe inference
"""

import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StepMetrics:
    """Metrics for a single draft-verify step."""
    draft_length: int = 0          # number of draft tokens generated
    accepted_length: int = 0       # number of accepted tokens
    draft_time_ms: float = 0.0     # time for draft generation (ms)
    verify_time_ms: float = 0.0    # time for target model verification (ms)
    probe_time_ms: float = 0.0     # time for probe inference (ms)
    total_time_ms: float = 0.0     # total step time (ms)
    early_stopped: bool = False    # whether controller stopped early


@dataclass
class GenerationMetrics:
    """Metrics for a complete generation (one prompt)."""
    prompt_id: str = ""
    method: str = ""
    temperature: float = 0.0

    # Token counts
    total_tokens: int = 0          # total generated tokens
    input_length: int = 0          # prompt length

    # Timing
    wall_time_s: float = 0.0       # total wall-clock time
    tokens_per_second: float = 0.0 # throughput

    # Step-level metrics
    steps: List[StepMetrics] = field(default_factory=list)
    num_steps: int = 0

    # Aggregate metrics
    avg_acceptance_length: float = 0.0  # tau
    avg_draft_length: float = 0.0
    speedup_ratio: float = 0.0  # vs autoregressive baseline

    # Per-position acceptance rate
    per_position_accept_rate: Dict[int, float] = field(default_factory=dict)

    # Controller stats
    early_stop_rate: float = 0.0
    avg_probe_time_ms: float = 0.0


class MetricsTracker:
    """Tracks and aggregates metrics across multiple generations."""

    def __init__(self):
        self.generations: List[GenerationMetrics] = []
        self._baseline_tokens_per_second: Optional[float] = None

    def set_baseline_speed(self, tokens_per_second: float):
        """Set the autoregressive baseline speed for speedup computation."""
        self._baseline_tokens_per_second = tokens_per_second

    def add_generation(self, metrics: GenerationMetrics):
        """Add metrics for a completed generation."""
        # Compute derived metrics
        if metrics.wall_time_s > 0:
            metrics.tokens_per_second = metrics.total_tokens / metrics.wall_time_s

        if metrics.steps:
            accepted_lengths = [s.accepted_length for s in metrics.steps]
            draft_lengths = [s.draft_length for s in metrics.steps]
            probe_times = [s.probe_time_ms for s in metrics.steps]

            metrics.avg_acceptance_length = np.mean(accepted_lengths) if accepted_lengths else 0
            metrics.avg_draft_length = np.mean(draft_lengths) if draft_lengths else 0
            metrics.num_steps = len(metrics.steps)
            metrics.early_stop_rate = np.mean([s.early_stopped for s in metrics.steps])
            metrics.avg_probe_time_ms = np.mean(probe_times) if probe_times else 0

        if self._baseline_tokens_per_second is not None and self._baseline_tokens_per_second > 0:
            metrics.speedup_ratio = metrics.tokens_per_second / self._baseline_tokens_per_second

        self.generations.append(metrics)

    def compute_aggregate(self) -> Dict[str, float]:
        """Compute aggregate metrics across all generations.

        Returns:
            Dictionary with mean and std of all key metrics
        """
        if not self.generations:
            return {}

        result = {}

        # Collect arrays of each metric (ensure CPU floats)
        def _to_float(x):
            if hasattr(x, 'cpu'):
                return float(x.cpu())
            return float(x)

        tps = [_to_float(g.tokens_per_second) for g in self.generations]
        tau = [_to_float(g.avg_acceptance_length) for g in self.generations]
        speedup = [_to_float(g.speedup_ratio) for g in self.generations if _to_float(g.speedup_ratio) > 0]
        early_stop = [_to_float(g.early_stop_rate) for g in self.generations]
        probe_time = [_to_float(g.avg_probe_time_ms) for g in self.generations]

        mean_tps = float(np.mean(tps))
        result["tokens_per_second_mean"] = mean_tps
        result["tokens_per_second_std"] = float(np.std(tps))
        result["tau_mean"] = float(np.mean(tau))
        result["tau_std"] = float(np.std(tau))
        result["early_stop_rate_mean"] = float(np.mean(early_stop))
        result["avg_probe_time_ms"] = float(np.mean(probe_time))

        # Compute speedup from baseline (not per-generation, since baseline
        # may be set after generations are added)
        if self._baseline_tokens_per_second and self._baseline_tokens_per_second > 0:
            result["speedup_mean"] = mean_tps / self._baseline_tokens_per_second
        elif speedup:
            result["speedup_mean"] = float(np.mean(speedup))
            result["speedup_std"] = float(np.std(speedup))

        result["num_generations"] = len(self.generations)
        result["total_tokens"] = sum(g.total_tokens for g in self.generations)
        result["total_steps"] = sum(g.num_steps for g in self.generations)

        return result

    def compute_per_position_acceptance(self) -> Dict[int, float]:
        """Compute per-position acceptance rate (n-alpha) across all generations.

        Returns:
            Dictionary mapping position -> acceptance rate
        """
        position_accepted = {}
        position_total = {}

        for gen in self.generations:
            for step in gen.steps:
                for pos in range(step.draft_length):
                    if pos not in position_total:
                        position_total[pos] = 0
                        position_accepted[pos] = 0
                    position_total[pos] += 1
                    if pos < step.accepted_length:
                        position_accepted[pos] += 1

        result = {}
        for pos in sorted(position_total.keys()):
            if position_total[pos] > 0:
                result[pos] = position_accepted[pos] / position_total[pos]

        return result

    def to_table_row(self) -> Dict[str, str]:
        """Format aggregate metrics as a table row for paper results."""
        agg = self.compute_aggregate()
        return {
            "Speedup": f"{agg.get('speedup_mean', 0):.2f}x",
            "tau": f"{agg.get('tau_mean', 0):.2f}",
            "Tokens/s": f"{agg.get('tokens_per_second_mean', 0):.1f}",
            "Early Stop%": f"{agg.get('early_stop_rate_mean', 0)*100:.1f}%",
            "Probe (ms)": f"{agg.get('avg_probe_time_ms', 0):.3f}",
        }
