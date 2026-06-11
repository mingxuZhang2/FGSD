"""
Script to collect hidden state + accept/reject data from EAGLE-3 speculative decoding.

This is Phase 1 of FGSD: run EAGLE-3 on benchmark prompts and capture
intermediate states needed for training the rejection probe.

Usage:
    python scripts/collect_data.py \
        --base-model-path /path/to/LLaMA-3.1-8B-Instruct \
        --ea-model-path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
        --benchmark mt_bench \
        --output-dir data/hidden_states/llama31_8b/mt_bench \
        --temperature 0.0

    # Collect from multiple benchmarks:
    for bench in mt_bench humaneval gsm8k alpaca cnn_dm; do
        python scripts/collect_data.py --benchmark $bench ...
    done
"""

import argparse
import logging
import os
import sys
import time
import json

import torch

# Ensure our package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fgsd.collector.eagle_hooks import HiddenStateCollector
from fgsd.collector.dataset import save_collected_data
from fgsd.eval.benchmark import load_benchmark_prompts

# Add EAGLE to path
EAGLE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EAGLE")
sys.path.insert(0, EAGLE_ROOT)

from eagle.model.ea_model import EaModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Collect hidden states from EAGLE-3 SD")

    # Model paths
    parser.add_argument("--base-model-path", type=str, required=True,
                        help="Path to base (target) model")
    parser.add_argument("--ea-model-path", type=str, required=True,
                        help="Path to EAGLE-3 draft model checkpoint")
    parser.add_argument("--use-eagle3", action="store_true", default=True,
                        help="Use EAGLE-3 (default True)")

    # EAGLE parameters
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=1.0)

    # Benchmark
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["mt_bench", "humaneval", "gsm8k", "alpaca", "cnn_dm"],
                        help="Benchmark to run")
    parser.add_argument("--data-dir", type=str, default="data/benchmarks",
                        help="Directory containing benchmark data")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Maximum samples to process (-1 = all)")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=2048)

    # Collection parameters
    parser.add_argument("--collect-draft-hidden", action="store_true", default=True)
    parser.add_argument("--collect-target-hidden", action="store_true", default=True)
    parser.add_argument("--collect-entropy", action="store_true", default=True)
    parser.add_argument("--collect-draft-logits", action="store_true", default=False,
                        help="Collect full draft logits (large; default False)")

    # Output
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save collected data")
    parser.add_argument("--chunk-size", type=int, default=50,
                        help="Save checkpoint every N prompts")

    # Model loading
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--is-llama3", action="store_true", default=False,
                        help="Use LLaMA-3 stop tokens")

    # Sharding for multi-GPU parallelism
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard index for parallel collection")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards")

    args = parser.parse_args()

    # Log configuration
    logger.info("=" * 60)
    logger.info("FGSD Hidden State Collection")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    # Set environment for offline operation
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Load model
    logger.info("Loading model...")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = EaModel.from_pretrained(
        use_eagle3=args.use_eagle3,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        threshold=args.threshold,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    logger.info("Model loaded successfully")

    # Create collector
    collector = HiddenStateCollector(
        model=model,
        collect_draft_hidden=args.collect_draft_hidden,
        collect_target_hidden=args.collect_target_hidden,
        collect_draft_logits=args.collect_draft_logits,
        collect_entropy=args.collect_entropy,
    )

    # Load benchmark prompts
    prompts = load_benchmark_prompts(
        args.benchmark, data_dir=args.data_dir, max_samples=args.max_samples
    )

    # Apply sharding if multi-GPU
    if args.num_shards > 1:
        shard_size = (len(prompts) + args.num_shards - 1) // args.num_shards
        start = args.shard_id * shard_size
        end = min(start + shard_size, len(prompts))
        prompts = prompts[start:end]
        logger.info(f"Shard {args.shard_id}/{args.num_shards}: prompts [{start}:{end}]")

    logger.info(f"Loaded {len(prompts)} prompts from {args.benchmark}")

    # Warmup
    logger.info("Running warmup...")
    tokenizer = model.get_tokenizer()

    warmup_text = "Hello, how are you today?"
    messages = [{"role": "user", "content": warmup_text}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer([text], add_special_tokens=False).input_ids
    except Exception:
        input_ids = tokenizer([warmup_text]).input_ids

    for _ in range(2):
        model.eagenerate(
            torch.as_tensor(input_ids).cuda(),
            temperature=0.0, max_new_tokens=32, is_llama3=args.is_llama3,
        )
    logger.info("Warmup complete")

    # Collect data
    os.makedirs(args.output_dir, exist_ok=True)
    all_records = []
    chunk_id = 0
    total_samples = 0

    for i, prompt_data in enumerate(prompts):
        try:
            # Prepare input
            prompt_text = prompt_data["prompt"]
            messages = [{"role": "user", "content": prompt_text}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_ids = tokenizer([text], add_special_tokens=False).input_ids
            except Exception:
                input_ids = tokenizer([prompt_text]).input_ids

            input_ids_tensor = torch.as_tensor(input_ids).cuda()

            # Generate and collect
            record = collector.generate_and_collect(
                input_ids=input_ids_tensor,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                max_length=args.max_length,
                is_llama3=args.is_llama3,
                prompt_id=prompt_data["id"],
                prompt_text=prompt_text[:200],  # truncate for storage
            )

            all_records.append(record)

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Processed {i+1}/{len(prompts)} prompts, "
                    f"current: {record.total_tokens} tokens in {record.total_steps} steps "
                    f"({record.wall_time:.2f}s)"
                )

        except Exception as e:
            logger.warning(f"Error on prompt {i} ({prompt_data['id']}): {e}")
            continue

        # Save checkpoint
        if (i + 1) % args.chunk_size == 0 or (i + 1) == len(prompts):
            if all_records:
                flat_data = collector.collect_flat_dataset(all_records)
                n_samples = flat_data["labels"].shape[0]
                total_samples += n_samples

                metadata = {
                    "benchmark": args.benchmark,
                    "base_model": args.base_model_path,
                    "ea_model": args.ea_model_path,
                    "temperature": args.temperature,
                    "num_prompts": len(all_records),
                    "prompt_range": f"{i + 1 - len(all_records)}-{i}",
                }

                save_collected_data(
                    flat_data, args.output_dir, chunk_id=chunk_id, metadata=metadata
                )

                logger.info(
                    f"Saved chunk {chunk_id}: {n_samples} samples "
                    f"(total so far: {total_samples})"
                )

                all_records = []
                chunk_id += 1

    # Final summary
    logger.info("=" * 60)
    logger.info(f"Collection complete: {total_samples} total samples from {len(prompts)} prompts")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    # Clean up hooks
    collector.remove_hooks()


if __name__ == "__main__":
    main()
