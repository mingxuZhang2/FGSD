"""
Script to run end-to-end evaluation of FGSD vs baselines.

Phase 3 of FGSD: evaluate all methods on standard benchmarks and
collect the metrics needed for the paper's main results table.

Usage:
    python scripts/run_eval.py \
        --base-model-path /path/to/LLaMA-3.1-8B-Instruct \
        --ea-model-path /path/to/EAGLE3-LLaMA3.1-Instruct-8B \
        --probe-dir results/probes/llama31_8b/mlp_draft \
        --benchmarks mt_bench humaneval gsm8k \
        --methods vanilla eagle3 fgsd svip \
        --output-dir results/eval/llama31_8b
"""

import argparse
import logging
import os
import sys
import json
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EAGLE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EAGLE")
sys.path.insert(0, EAGLE_ROOT)

from eagle.model.ea_model import EaModel
from eagle.model.utils import prepare_logits_processor

from fgsd.eval.benchmark import BenchmarkRunner, load_benchmark_prompts, run_warmup
from fgsd.eval.metrics import MetricsTracker
from fgsd.controller.adaptive import AdaptiveDraftController
from fgsd.controller.baselines import SVIPEntropyController, FixedLengthController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="FGSD End-to-End Evaluation")

    # Model paths
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--ea-model-path", type=str, required=True)
    parser.add_argument("--use-eagle3", action="store_true", default=True)

    # EAGLE parameters
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)

    # Probe (for FGSD method)
    parser.add_argument("--probe-dir", type=str, default="",
                        help="Directory containing trained probe (for FGSD method)")
    parser.add_argument("--rejection-threshold", type=float, default=0.3,
                        help="Rejection probability threshold for FGSD")
    parser.add_argument("--max-extended-depth", type=int, default=0,
                        help="If > base depth, FGSD may extend draft depth up "
                             "to this while the probe predicts acceptance")
    parser.add_argument("--extend-threshold", type=float, default=0.2,
                        help="Mean P(reject) below which depth extension continues")
    parser.add_argument("--max-position", type=int, default=0,
                        help="Max probe position (0 = auto from max-extended-depth or depth)")
    parser.add_argument("--aggregation", type=str, default="mean",
                        choices=["mean", "score_weighted", "top1"],
                        help="How to aggregate P(reject) over top_k candidates")
    parser.add_argument("--extend-decay", type=float, default=0.0,
                        help="Exponential decay rate for extend_threshold at deeper levels")

    # Evaluation settings
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["mt_bench", "humaneval", "gsm8k"],
                        help="Benchmarks to evaluate")
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["vanilla", "eagle3", "fgsd"],
                        help="Methods to evaluate")
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0],
                        help="Temperatures to evaluate")
    parser.add_argument("--max-samples", type=int, default=-1,
                        help="Max samples per benchmark (-1 = all)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-runs", type=int, default=1,
                        help="Number of runs for averaging (use >1 for temp>0)")

    # SVIP baseline parameters
    parser.add_argument("--svip-entropy-threshold", type=float, default=2.0)
    parser.add_argument("--svip-extend-threshold", type=float, default=0.5,
                        help="Entropy (nats) below which SVIP extension continues")

    # Output
    parser.add_argument("--data-dir", type=str, default="data/benchmarks")
    parser.add_argument("--output-dir", type=str, required=True)

    # Model loading
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--is-llama3", action="store_true", default=False)
    parser.add_argument("--n-warmup", type=int, default=3)

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("FGSD End-to-End Evaluation")
    logger.info("=" * 60)
    for k, v in vars(args).items():
        logger.info(f"  {k}: {v}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    os.makedirs(args.output_dir, exist_ok=True)

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
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    logger.info("Model loaded")

    # Create benchmark runner
    runner = BenchmarkRunner(model=model, is_llama3=args.is_llama3)

    # Warmup
    run_warmup(model, tokenizer, is_llama3=args.is_llama3, n_warmup=args.n_warmup)

    # Load FGSD controller if needed
    fgsd_controller = None
    if "fgsd" in args.methods and args.probe_dir:
        max_pos = args.max_position or args.max_extended_depth or args.depth
        fgsd_controller = AdaptiveDraftController.from_checkpoint(
            checkpoint_dir=args.probe_dir,
            device="cuda",
            threshold=args.rejection_threshold,
            max_extended_depth=args.max_extended_depth or None,
            extend_threshold=args.extend_threshold,
            max_position=max_pos,
            aggregation=args.aggregation,
            extend_decay=args.extend_decay,
        )
        logger.info(f"Loaded FGSD controller from {args.probe_dir} "
                     f"(max_pos={max_pos}, agg={args.aggregation}, decay={args.extend_decay})")

    # SVIP controller: entropy signal through the SAME bidirectional control
    # loop as FGSD — isolates internal-features-vs-output-entropy comparison
    svip_controller = SVIPEntropyController(
        entropy_threshold=args.svip_entropy_threshold,
        max_extended_depth=args.max_extended_depth,
        extend_threshold=args.svip_extend_threshold,
    )

    # Run evaluations
    all_results = {}

    for benchmark in args.benchmarks:
        logger.info(f"\n{'='*40}")
        logger.info(f"Benchmark: {benchmark}")
        logger.info(f"{'='*40}")

        prompts = load_benchmark_prompts(
            benchmark, data_dir=args.data_dir, max_samples=args.max_samples
        )

        if not prompts:
            logger.warning(f"No prompts loaded for {benchmark}, skipping")
            continue

        for temperature in args.temperatures:
            baseline_tps = None  # reset per (benchmark, temperature) group

            for method in args.methods:
                for run_idx in range(args.num_runs):
                    key = f"{benchmark}_{method}_t{temperature}_run{run_idx}"
                    logger.info(f"\nRunning: {key}")

                    torch.manual_seed(run_idx)

                    try:
                        if method == "vanilla":
                            tracker = runner.run_autoregressive(
                                prompts, temperature=temperature,
                                max_new_tokens=args.max_new_tokens,
                            )
                        elif method == "eagle3":
                            tracker = runner.run_eagle3(
                                prompts, temperature=temperature,
                                max_new_tokens=args.max_new_tokens,
                            )
                        elif method == "fgsd":
                            if fgsd_controller is None:
                                logger.warning("FGSD controller not loaded, skipping")
                                continue
                            tracker = runner.run_fgsd(
                                prompts, controller=fgsd_controller,
                                temperature=temperature,
                                max_new_tokens=args.max_new_tokens,
                            )
                        elif method == "svip":
                            logger.info("SVIP baseline: entropy-gated adaptive depth")
                            tracker = runner.run_fgsd(
                                prompts, controller=svip_controller,
                                temperature=temperature,
                                max_new_tokens=args.max_new_tokens,
                                method_name="svip",
                            )
                        else:
                            logger.warning(f"Unknown method: {method}")
                            continue

                        # Set baseline speed from vanilla run
                        if method == "vanilla":
                            agg = tracker.compute_aggregate()
                            baseline_tps = agg.get("tokens_per_second_mean", 0)
                            logger.info(f"Baseline speed: {baseline_tps:.1f} tokens/s")

                        # Pass baseline to non-vanilla trackers for speedup computation
                        if baseline_tps and method != "vanilla":
                            tracker.set_baseline_speed(baseline_tps)

                        # Compute and store results
                        agg = tracker.compute_aggregate()
                        all_results[key] = agg
                        per_pos = tracker.compute_per_position_acceptance()
                        all_results[key]["per_position_alpha"] = per_pos

                        speedup_str = f"{agg['speedup_mean']:.2f}x" if 'speedup_mean' in agg else "N/A"
                        logger.info(
                            f"  tokens/s: {agg.get('tokens_per_second_mean', 0):.1f}, "
                            f"  tau: {agg.get('tau_mean', 0):.2f}, "
                            f"  speedup: {speedup_str}"
                        )

                        # Log controller stats for FGSD
                        if method == "fgsd" and fgsd_controller is not None:
                            ctrl_stats = fgsd_controller.get_stats()
                            logger.info(
                                f"  FGSD controller: early_stop={ctrl_stats.get('early_stop_rate', 0):.1%}, "
                                f"  avg_draft={ctrl_stats.get('avg_draft_length', 0):.1f}, "
                                f"  avg_accepted={ctrl_stats.get('avg_accepted_length', 0):.1f}"
                            )
                            all_results[key]["controller_stats"] = ctrl_stats

                    except Exception as e:
                        logger.error(f"Error running {key}: {e}", exc_info=True)
                        all_results[key] = {"error": str(e)}

    # Save all results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nAll results saved to {results_path}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)
    logger.info(f"{'Method':<15} {'Benchmark':<15} {'Temp':<6} {'Tokens/s':<12} {'Tau':<8} {'Speedup':<10}")
    logger.info("-" * 80)

    for key, result in all_results.items():
        if "error" in result:
            continue
        parts = key.split("_")
        # Parse benchmark_method_tX_runY
        benchmark = parts[0]
        method = parts[1] if len(parts) > 1 else "?"
        temp = parts[2] if len(parts) > 2 else "?"

        tps = result.get("tokens_per_second_mean", 0)
        tau = result.get("tau_mean", 0)
        speedup = result.get("speedup_mean", "N/A")
        if isinstance(speedup, float):
            speedup = f"{speedup:.2f}x"

        logger.info(f"{method:<15} {benchmark:<15} {temp:<6} {tps:<12.1f} {tau:<8.2f} {speedup:<10}")


if __name__ == "__main__":
    main()
