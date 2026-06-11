"""Detailed per-step analysis for paper figures.

Runs a representative prompt subset with all methods and records per-step:
  - accepted_length (for acceptance length distribution)
  - drafted_depth (for depth histogram)
  - probe_time_ms, verify_time_ms (for overhead breakdown)
  - oracle_optimal_depth (for oracle upper bound)

Outputs a single JSON with all raw per-step data + computed aggregates.
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
EAGLE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "EAGLE")
sys.path.insert(0, EAGLE_ROOT)

from eagle.model.ea_model import EaModel

from fgsd.eval.benchmark import BenchmarkRunner, load_benchmark_prompts, run_warmup
from fgsd.controller.adaptive import AdaptiveDraftController
from fgsd.controller.baselines import SVIPEntropyController
from fgsd.controller.adaptive_draft import install_adaptive_draft, uninstall_adaptive_draft

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_with_step_recording(runner, prompts, method, controller=None,
                            temperature=0.0, max_new_tokens=512):
    """Run inference and return per-step metrics for every prompt."""
    all_steps = []
    total_tokens = 0
    total_time = 0.0

    if controller is not None:
        install_adaptive_draft(runner.model, controller)
        controller.reset_stats()

    try:
        for p in prompts:
            input_ids = runner._prepare_input(p["prompt"])
            input_len = input_ids.shape[1]

            torch.cuda.synchronize()
            t0 = time.time()

            if method == "vanilla":
                out, new_tok, _ = runner.model.naivegenerate(
                    input_ids, temperature=temperature,
                    max_new_tokens=max_new_tokens, log=True,
                    is_llama3=runner.is_llama3)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                ntok = int(new_tok.item()) if isinstance(new_tok, torch.Tensor) else int(new_tok)
                total_tokens += ntok
                total_time += elapsed
                continue

            if method in ("eagle3",):
                out, new_tok, idx = runner.model.eagenerate(
                    input_ids, temperature=temperature,
                    max_new_tokens=max_new_tokens, log=True,
                    is_llama3=runner.is_llama3)
                torch.cuda.synchronize()
                elapsed = time.time() - t0
                ntok = int(new_tok.item()) if isinstance(new_tok, torch.Tensor) else int(new_tok)
                nsteps = int(idx.item()) + 1 if isinstance(idx, torch.Tensor) else int(idx) + 1
                total_tokens += ntok
                total_time += elapsed
                if nsteps > 0:
                    avg_tau = ntok / nsteps
                    all_steps.append({
                        "prompt_id": p["id"],
                        "total_tokens": ntok,
                        "num_steps": nsteps,
                        "avg_tau": avg_tau,
                        "wall_time_s": elapsed,
                    })
                continue

            # FGSD / SVIP — use the detailed _run_fgsd_single which records StepMetrics
            gen_metrics = runner._run_fgsd_single(
                input_ids=input_ids,
                controller=controller,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                prompt_id=p["id"],
                method_name=method,
            )
            total_tokens += gen_metrics.total_tokens
            total_time += gen_metrics.wall_time_s

            for s in gen_metrics.steps:
                all_steps.append({
                    "prompt_id": p["id"],
                    "accepted_length": s.accepted_length,
                    "draft_depth": s.draft_length,
                    "probe_time_ms": s.probe_time_ms,
                    "verify_time_ms": s.verify_time_ms,
                    "early_stopped": s.early_stopped,
                })
    finally:
        if controller is not None:
            uninstall_adaptive_draft(runner.model)

    tps = total_tokens / total_time if total_time > 0 else 0
    return {
        "method": method,
        "total_tokens": total_tokens,
        "total_time_s": total_time,
        "tokens_per_second": tps,
        "num_prompts": len(prompts),
        "steps": all_steps,
    }


def compute_oracle_bound(steps_data, draft_cost_ms=1.2, verify_cost_ms=21.0):
    """Estimate oracle throughput from per-step accepted_lengths.

    Oracle picks draft_depth = accepted_length for each step (no wasted drafts).
    At each step:
      - actual cost = draft_depth * draft_cost + verify_cost
      - oracle cost = min(accepted_length, max_depth) * draft_cost + verify_cost
    """
    if not steps_data:
        return {}

    total_actual_draft_ms = 0
    total_oracle_draft_ms = 0
    total_verify_ms = 0
    total_tokens = 0

    for s in steps_data:
        acc = s.get("accepted_length", 0)
        depth = s.get("draft_depth", 5)
        v_ms = s.get("verify_time_ms", verify_cost_ms)

        total_actual_draft_ms += depth * draft_cost_ms
        total_oracle_draft_ms += min(acc, depth) * draft_cost_ms
        total_verify_ms += v_ms
        total_tokens += acc

    actual_total_ms = total_actual_draft_ms + total_verify_ms
    oracle_total_ms = total_oracle_draft_ms + total_verify_ms

    return {
        "actual_tps_estimate": total_tokens / (actual_total_ms / 1000) if actual_total_ms > 0 else 0,
        "oracle_tps_estimate": total_tokens / (oracle_total_ms / 1000) if oracle_total_ms > 0 else 0,
        "draft_time_saved_pct": (total_actual_draft_ms - total_oracle_draft_ms) / total_actual_draft_ms * 100 if total_actual_draft_ms > 0 else 0,
        "total_tokens": total_tokens,
        "num_steps": len(steps_data),
    }


def compute_distributions(steps_data):
    """Compute acceptance length and depth distributions."""
    if not steps_data:
        return {}
    acc_lengths = [s["accepted_length"] for s in steps_data if "accepted_length" in s]
    depths = [s["draft_depth"] for s in steps_data if "draft_depth" in s]
    probe_times = [s["probe_time_ms"] for s in steps_data if "probe_time_ms" in s]
    verify_times = [s["verify_time_ms"] for s in steps_data if "verify_time_ms" in s]

    result = {}
    if acc_lengths:
        result["acceptance_length"] = {
            "mean": float(np.mean(acc_lengths)),
            "std": float(np.std(acc_lengths)),
            "median": float(np.median(acc_lengths)),
            "histogram": {str(k): int(v) for k, v in
                          zip(*np.unique(acc_lengths, return_counts=True))},
        }
    if depths:
        result["draft_depth"] = {
            "mean": float(np.mean(depths)),
            "std": float(np.std(depths)),
            "histogram": {str(k): int(v) for k, v in
                          zip(*np.unique(depths, return_counts=True))},
        }
    if probe_times:
        result["probe_time_ms"] = {
            "mean": float(np.mean(probe_times)),
            "std": float(np.std(probe_times)),
            "total": float(np.sum(probe_times)),
        }
    if verify_times:
        result["verify_time_ms"] = {
            "mean": float(np.mean(verify_times)),
            "std": float(np.std(verify_times)),
            "total": float(np.sum(verify_times)),
        }
        if probe_times:
            result["probe_overhead_pct"] = float(
                np.sum(probe_times) / (np.sum(probe_times) + np.sum(verify_times)) * 100
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--ea-model-path", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["mt_bench", "humaneval", "gsm8k"])
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--data-dir", default="data/benchmarks")
    parser.add_argument("--output", default="results/analysis/detailed_analysis.json")
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--is-llama3", action="store_true", default=False)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=60, depth=args.depth, top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    runner = BenchmarkRunner(model=model, is_llama3=args.is_llama3)
    run_warmup(model, model.get_tokenizer(), is_llama3=args.is_llama3, n_warmup=3)

    probe_controller = AdaptiveDraftController.from_checkpoint(
        checkpoint_dir=args.probe_dir, device="cuda",
        threshold=1.5, max_extended_depth=10, extend_threshold=0.5,
    )
    svip_controller = SVIPEntropyController(
        entropy_threshold=3.0, max_extended_depth=10, extend_threshold=0.7,
    )

    methods = [
        ("eagle3", None),
        ("fgsd", probe_controller),
        ("svip", svip_controller),
    ]

    results = {}
    for bench in args.benchmarks:
        prompts = load_benchmark_prompts(bench, data_dir=args.data_dir,
                                         max_samples=args.max_samples)
        if not prompts:
            continue

        results[bench] = {}
        for method_name, ctrl in methods:
            logger.info(f"Running {method_name} on {bench} ({len(prompts)} prompts)")
            data = run_with_step_recording(
                runner, prompts, method_name, controller=ctrl,
                max_new_tokens=args.max_new_tokens,
            )

            data["distributions"] = compute_distributions(data["steps"])

            if method_name in ("fgsd", "svip") and data["steps"]:
                data["oracle_bound"] = compute_oracle_bound(data["steps"])

            results[bench][method_name] = data
            logger.info(f"  {method_name}: {data['tokens_per_second']:.1f} tok/s, "
                        f"{len(data['steps'])} steps recorded")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved to {args.output}")

    for bench, bench_data in results.items():
        print(f"\n{'='*60}")
        print(f"  {bench}")
        print(f"{'='*60}")
        for method, mdata in bench_data.items():
            print(f"\n  {method}: {mdata['tokens_per_second']:.1f} tok/s")
            dist = mdata.get("distributions", {})
            if "acceptance_length" in dist:
                al = dist["acceptance_length"]
                print(f"    acceptance length: mean={al['mean']:.2f} std={al['std']:.2f}")
            if "draft_depth" in dist:
                dd = dist["draft_depth"]
                print(f"    draft depth: mean={dd['mean']:.2f} std={dd['std']:.2f}")
                print(f"    depth histogram: {dd['histogram']}")
            if "probe_overhead_pct" in dist:
                print(f"    probe overhead: {dist['probe_overhead_pct']:.1f}%")
            if "oracle_bound" in mdata:
                ob = mdata["oracle_bound"]
                print(f"    oracle bound: {ob['oracle_tps_estimate']:.1f} tok/s "
                      f"(draft time saved: {ob['draft_time_saved_pct']:.1f}%)")


if __name__ == "__main__":
    main()
