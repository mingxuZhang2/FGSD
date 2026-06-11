"""
Rejection prediction probe architectures.

Probes take hidden states (from the draft model, target model, or both)
and predict P(rejection) for each draft token position.

Three probe architectures:
1. LinearProbe: Single linear layer (minimal overhead, ~0 latency cost)
2. MLPProbe: Multi-layer perceptron (better accuracy, still very fast)
3. MultiLayerProbe: Aggregates multiple layers then MLP (if we have per-layer hidden states)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


class LinearProbe(nn.Module):
    """Linear probe for rejection prediction.

    Maps hidden states directly to P(rejection) via a single linear layer.
    This is the minimal-overhead option with near-zero latency cost.

    Input: [batch, hidden_dim]
    Output: [batch, 1] logits (use sigmoid for probabilities)
    """

    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape [batch, input_dim]

        Returns:
            Logits, shape [batch, 1]
        """
        return self.linear(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MLPProbe(nn.Module):
    """MLP probe for rejection prediction.

    Multi-layer perceptron with configurable depth and width.
    Still very lightweight compared to the LLM.

    Input: [batch, hidden_dim]
    Output: [batch, 1] logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        # Final projection to single logit
        layers.append(nn.Linear(current_dim, 1, bias=bias))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape [batch, input_dim]

        Returns:
            Logits, shape [batch, 1]
        """
        return self.mlp(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MultiLayerProbe(nn.Module):
    """Probe that aggregates hidden states from multiple layers.

    Takes per-layer hidden states, applies optional per-layer projection,
    then aggregates (concat or attention-weighted) before final MLP.

    This is for cases where we have access to individual layer outputs
    rather than a pre-concatenated representation.

    Input: [batch, num_layers, hidden_dim]
    Output: [batch, 1] logits
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        proj_dim: int = 128,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        aggregation: str = "concat",  # concat, mean, attention
    ):
        super().__init__()
        self.num_layers = num_layers
        self.aggregation = aggregation

        # Per-layer projections to reduce dimensionality
        self.layer_projs = nn.ModuleList([
            nn.Linear(hidden_dim, proj_dim) for _ in range(num_layers)
        ])

        # Aggregation
        if aggregation == "concat":
            agg_dim = proj_dim * num_layers
        elif aggregation == "mean":
            agg_dim = proj_dim
        elif aggregation == "attention":
            self.attn_weights = nn.Linear(proj_dim, 1)
            agg_dim = proj_dim
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(agg_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Per-layer hidden states, shape [batch, num_layers, hidden_dim]

        Returns:
            Logits, shape [batch, 1]
        """
        # Project each layer
        projected = []
        for i, proj in enumerate(self.layer_projs):
            projected.append(proj(x[:, i]))  # [batch, proj_dim]

        projected = torch.stack(projected, dim=1)  # [batch, num_layers, proj_dim]

        # Aggregate
        if self.aggregation == "concat":
            agg = projected.reshape(projected.shape[0], -1)  # [batch, num_layers * proj_dim]
        elif self.aggregation == "mean":
            agg = projected.mean(dim=1)  # [batch, proj_dim]
        elif self.aggregation == "attention":
            weights = self.attn_weights(projected).softmax(dim=1)  # [batch, num_layers, 1]
            agg = (projected * weights).sum(dim=1)  # [batch, proj_dim]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        return self.mlp(agg)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PositionAwareProbe(nn.Module):
    """Probe that additionally conditions on draft position.

    Wraps any base probe and adds position embeddings as additional input.
    Motivation: rejection probability varies by draft position (later positions
    are more likely to be rejected).

    Input: (hidden_states [batch, hidden_dim], positions [batch])
    Output: [batch, 1] logits
    """

    def __init__(
        self,
        base_probe: nn.Module,
        input_dim: int,
        max_positions: int = 16,
        position_dim: int = 16,
    ):
        super().__init__()
        self.position_embedding = nn.Embedding(max_positions, position_dim)

        # We need a new first layer that takes input_dim + position_dim
        # and maps to the base probe's expected input
        self.input_proj = nn.Linear(input_dim + position_dim, input_dim)
        self.base_probe = base_probe

    def forward(
        self, x: torch.Tensor, positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features, shape [batch, input_dim]
            positions: Draft positions, shape [batch]. If None, ignored.

        Returns:
            Logits, shape [batch, 1]
        """
        if positions is not None:
            pos_emb = self.position_embedding(positions.clamp(max=self.position_embedding.num_embeddings - 1))
            x = self.input_proj(torch.cat([x, pos_emb], dim=-1))

        return self.base_probe(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def create_probe(
    probe_type: str,
    input_dim: int,
    hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.1,
    use_position: bool = False,
    max_positions: int = 16,
    **kwargs,
) -> nn.Module:
    """Factory function to create a probe model.

    Args:
        probe_type: "linear", "mlp", or "multi_layer"
        input_dim: Input feature dimension
        hidden_dim: MLP hidden dimension
        num_layers: Number of MLP layers (for "mlp" type)
        dropout: Dropout rate
        use_position: Whether to add position-aware wrapper
        max_positions: Maximum draft positions to embed

    Returns:
        Probe model
    """
    if probe_type == "linear":
        probe = LinearProbe(input_dim)
    elif probe_type == "mlp":
        probe = MLPProbe(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
    elif probe_type == "multi_layer":
        n_model_layers = kwargs.get("n_model_layers", 3)
        per_layer_dim = input_dim // n_model_layers
        probe = MultiLayerProbe(
            hidden_dim=per_layer_dim,
            num_layers=n_model_layers,
            proj_dim=min(128, per_layer_dim),
            mlp_hidden=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")

    if use_position:
        probe = PositionAwareProbe(
            base_probe=probe,
            input_dim=input_dim,
            max_positions=max_positions,
        )

    return probe
