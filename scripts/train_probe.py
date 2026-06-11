"""
Script to train rejection prediction probes.

Phase 2 of FGSD: train lightweight probes on collected hidden state data
to predict which draft tokens will be rejected.

Usage:
    python scripts/train_probe.py \
        --data-dir data/hidden_states/llama31_8b \
        --probe-type mlp \
        --input-source draft_hidden \
        --output-dir results/probes/llama31_8b/mlp_draft \
        --num-epochs 20

    # Ablation: compare probe types
    for probe in linear mlp; do
        for source in draft_hidden target_hidden entropy combined; do
            python scripts/train_probe.py \
                --probe-type $probe --input-source $source ...
        done
    done
"""

import argparse
import logging
import os
import sys
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgsd.probe.train import train_probe_from_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train rejection prediction probe")

    # Data
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing collected hidden state data")

    # Probe architecture
    parser.add_argument("--probe-type", type=str, default="mlp",
                        choices=["linear", "mlp", "multi_layer"],
                        help="Probe architecture type")
    parser.add_argument("--input-source", type=str, default="draft_hidden",
                        choices=["draft_hidden", "target_hidden", "entropy", "combined"],
                        help="Which features to use as input")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="MLP hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="Number of MLP layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use-position", action="store_true", default=False,
                        help="Use position-aware probe")

    # Training
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--use-class-weights", action="store_true", default=True)
    parser.add_argument("--no-class-weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--pos-weight-cap", type=float, default=5.0)

    # Data splitting
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)

    # Output
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FGSD Probe Training")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Check CUDA
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Train
    metrics = train_probe_from_data(
        data_dir=args.data_dir,
        probe_type=args.probe_type,
        input_source=args.input_source,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        use_class_weights=args.use_class_weights,
        pos_weight_cap=args.pos_weight_cap,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
        use_position=args.use_position,
    )

    logger.info("=" * 60)
    logger.info("Training complete. Final metrics:")
    for k, v in metrics.items():
        if isinstance(v, (dict, list)):
            continue
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
