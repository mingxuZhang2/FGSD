"""
Configuration system for FGSD experiments.

Uses dataclasses for type safety and YAML for serialization.
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class ModelConfig:
    """Configuration for model paths."""
    base_model_path: str = ""
    ea_model_path: str = ""
    model_family: str = "llama"  # llama, qwen, vicuna
    use_eagle3: bool = True
    torch_dtype: str = "float16"  # float16, bfloat16
    device_map: str = "auto"

    # EAGLE-3 draft parameters
    total_token: int = 60
    depth: int = 5
    top_k: int = 10
    threshold: float = 1.0


@dataclass
class CollectorConfig:
    """Configuration for hidden state collection."""
    # Data sources
    benchmark: str = "mt_bench"  # mt_bench, humaneval, gsm8k, alpaca, cnn_dm
    data_path: str = ""
    max_samples: int = -1  # -1 means all

    # Generation parameters
    temperature: float = 0.0
    max_new_tokens: int = 512
    max_length: int = 2048

    # Collection parameters
    collect_draft_hidden: bool = True
    collect_target_hidden: bool = True
    collect_draft_logits: bool = True
    collect_entropy: bool = True

    # Which target model layers to capture (EAGLE-3 uses 3 specific layers)
    # These are indices relative to the target model layers.
    # Set to empty list to use EAGLE-3 defaults (layer 2, mid, n-3).
    target_layers: List[int] = field(default_factory=list)

    # Output
    output_dir: str = "data/hidden_states"
    save_format: str = "safetensors"  # safetensors, npz
    chunk_size: int = 5000  # save every N samples to manage memory


@dataclass
class ProbeConfig:
    """Configuration for rejection probe training."""
    # Architecture
    probe_type: str = "mlp"  # linear, mlp, multi_layer
    hidden_dim: int = 256  # MLP hidden dim
    num_mlp_layers: int = 2
    dropout: float = 0.1
    input_source: str = "draft_hidden"  # draft_hidden, target_hidden, combined, entropy

    # Which layer(s) to use as input
    # For single-layer probes: specify one layer index
    # For multi-layer: specify multiple or use -1 for all
    probe_layers: List[int] = field(default_factory=lambda: [-1])

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    num_epochs: int = 20
    early_stopping_patience: int = 5
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    # Class imbalance handling
    use_class_weights: bool = True
    pos_weight_cap: float = 5.0  # cap positive weight to avoid instability

    # Data
    data_dir: str = "data/hidden_states"
    output_dir: str = "results/probes"


@dataclass
class ControllerConfig:
    """Configuration for adaptive draft controller."""
    # Stopping strategy
    strategy: str = "threshold"  # threshold, top_k_prune, hybrid
    rejection_threshold: float = 0.5  # P(reject) above this => stop
    max_draft_length: int = 8
    min_draft_length: int = 1  # always draft at least this many tokens

    # Tree pruning (if strategy includes pruning)
    prune_threshold: float = 0.7  # prune branches with P(reject) > this
    min_tree_width: int = 2  # keep at least this many branches

    # Probe path (trained probe checkpoint)
    probe_checkpoint: str = ""

    # Baseline configs
    baseline: str = "none"  # none, svip_entropy, fixed_length
    svip_entropy_threshold: float = 2.0  # SVIP entropy stopping threshold
    fixed_draft_length: int = 5  # for fixed-length baseline


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Benchmarks to run
    benchmarks: List[str] = field(
        default_factory=lambda: ["mt_bench", "humaneval", "gsm8k", "alpaca", "cnn_dm"]
    )

    # Evaluation parameters
    temperatures: List[float] = field(default_factory=lambda: [0.0, 1.0])
    num_runs: int = 1  # number of runs for averaging (>1 only for temp > 0)
    max_new_tokens: int = 512
    max_length: int = 2048

    # Methods to evaluate
    methods: List[str] = field(
        default_factory=lambda: ["vanilla", "eagle3", "fgsd", "svip"]
    )

    # Output
    output_dir: str = "results/eval"
    save_generations: bool = False  # save generated text for quality check


@dataclass
class FGSDConfig:
    """Top-level configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    collector: CollectorConfig = field(default_factory=CollectorConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Global settings
    project_name: str = "fgsd"
    seed: int = 42
    log_level: str = "INFO"

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "FGSDConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "FGSDConfig":
        """Create config from a dictionary, handling nested dataclasses."""
        return cls(
            model=ModelConfig(**data.get("model", {})),
            collector=CollectorConfig(**data.get("collector", {})),
            probe=ProbeConfig(**data.get("probe", {})),
            controller=ControllerConfig(**data.get("controller", {})),
            eval=EvalConfig(**data.get("eval", {})),
            project_name=data.get("project_name", "fgsd"),
            seed=data.get("seed", 42),
            log_level=data.get("log_level", "INFO"),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)
