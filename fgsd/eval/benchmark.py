"""
Benchmark runner for end-to-end evaluation.

Loads benchmark datasets (MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DM),
runs speculative decoding with different methods (vanilla, EAGLE-3, FGSD, SVIP),
and collects timing/quality metrics.

Follows EAGLE-3's evaluation protocol for fair comparison.
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple

import torch

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

from .metrics import MetricsTracker, GenerationMetrics, StepMetrics

logger = logging.getLogger(__name__)


def _to_int(x):
    """Convert potentially GPU tensor to Python int."""
    if isinstance(x, torch.Tensor):
        return int(x.item())
    return int(x)


def prune_tree_to_depth(draft_tokens, tree_mask, tree_position_ids, retrieve_indices, max_depth):
    """Prune EAGLE-3 draft tree to a maximum depth.

    Removes tree nodes deeper than max_depth and updates all data structures.
    Saves target model verification compute on pruned branches.
    """
    device = draft_tokens.device
    n_total = tree_position_ids.shape[0]

    # Move to CPU for indexing (tree_mask is often on CPU, position_ids on GPU)
    keep_mask_cpu = (tree_position_ids.cpu() <= max_depth)
    n_kept = keep_mask_cpu.sum().item()

    if n_kept >= n_total:
        return draft_tokens, tree_mask, tree_position_ids, retrieve_indices

    old_to_new = torch.full((n_total,), -1, dtype=torch.long)
    new_i = 0
    for old_i in range(n_total):
        if keep_mask_cpu[old_i]:
            old_to_new[old_i] = new_i
            new_i += 1

    # Use device-matched masks for each tensor
    keep_gpu = keep_mask_cpu.to(device)
    keep_mask_m = keep_mask_cpu.to(tree_mask.device)

    pruned_tokens = draft_tokens[:, keep_gpu]
    pruned_mask = tree_mask[:, :, keep_mask_m, :][:, :, :, keep_mask_m]
    pruned_pos_ids = tree_position_ids[keep_gpu]

    seen_paths = set()
    new_rows = []
    for c in range(retrieve_indices.shape[0]):
        row = retrieve_indices[c]
        path = []
        for d in range(row.shape[0]):
            idx = row[d].item()
            if idx < 0:
                break
            if idx >= n_total or not keep_mask_cpu[idx]:
                break
            path.append(old_to_new[idx].item())
        if path:
            key = tuple(path)
            if key not in seen_paths:
                seen_paths.add(key)
                new_rows.append(path)

    if not new_rows:
        new_rows = [[0]]

    max_len = max(len(r) for r in new_rows)
    padded = [r + [-1] * (max_len - len(r)) for r in new_rows]
    pruned_indices = torch.tensor(padded, dtype=torch.long, device=retrieve_indices.device)

    return pruned_tokens, pruned_mask, pruned_pos_ids, pruned_indices


# ---------------------------------------------------------------------------
# Benchmark data loaders
# ---------------------------------------------------------------------------

def load_benchmark_prompts(
    benchmark: str,
    data_dir: str = "",
    max_samples: int = -1,
) -> List[Dict]:
    """Load prompts from a benchmark dataset.

    Args:
        benchmark: Name of the benchmark
        data_dir: Base directory for benchmark data
        max_samples: Maximum number of samples to load (-1 = all)

    Returns:
        List of dicts with keys: "id", "prompt", "turns" (for multi-turn)
    """
    prompts = []

    if benchmark == "mt_bench":
        prompts = _load_mt_bench(data_dir)
    elif benchmark == "humaneval":
        prompts = _load_humaneval(data_dir)
    elif benchmark == "gsm8k":
        prompts = _load_gsm8k(data_dir)
    elif benchmark == "alpaca":
        prompts = _load_alpaca(data_dir)
    elif benchmark == "cnn_dm":
        prompts = _load_cnn_dm(data_dir)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    if max_samples > 0:
        prompts = prompts[:max_samples]

    logger.info(f"Loaded {len(prompts)} prompts from {benchmark}")
    return prompts


def _load_mt_bench(data_dir: str) -> List[Dict]:
    """Load MT-Bench questions (80 multi-turn conversations)."""
    # Try EAGLE's data path first
    eagle_path = os.path.join(EAGLE_ROOT, "eagle", "data", "mt_bench", "question.jsonl")

    # Fall back to data_dir
    if not os.path.exists(eagle_path):
        eagle_path = os.path.join(data_dir, "mt_bench", "question.jsonl")
    if not os.path.exists(eagle_path):
        eagle_path = os.path.join(data_dir, "question.jsonl")

    if not os.path.exists(eagle_path):
        logger.warning(f"MT-Bench data not found. Creating placeholder prompts.")
        return [
            {"id": f"mt_bench_{i}", "prompt": f"Question {i}", "turns": [f"Question {i}"]}
            for i in range(80)
        ]

    prompts = []
    with open(eagle_path, "r") as f:
        for line in f:
            item = json.loads(line)
            prompts.append({
                "id": str(item["question_id"]),
                "prompt": item["turns"][0],
                "turns": item["turns"],
            })

    return prompts


def _load_local_dataset(data_dir: str, name: str):
    """Load a local dataset from Arrow or Parquet format."""
    import pandas as pd
    local_path = os.path.join(data_dir, name)
    if not os.path.exists(local_path):
        return None
    parquet = [f for f in os.listdir(local_path) if f.endswith(".parquet")]
    if parquet:
        return pd.read_parquet(os.path.join(local_path, parquet[0])).to_dict("records")
    try:
        from datasets import load_from_disk
        ds = load_from_disk(local_path)
        return list(ds)
    except Exception:
        return None


def _load_humaneval(data_dir: str) -> List[Dict]:
    """Load HumanEval coding problems (164 problems)."""
    try:
        records = _load_local_dataset(data_dir, "humaneval")
        if records is None:
            return []
        prompts = []
        for item in records:
            prompts.append({
                "id": item.get("task_id", f"humaneval_{len(prompts)}"),
                "prompt": item["prompt"],
                "turns": [item["prompt"]],
            })
        return prompts
    except Exception as e:
        logger.warning(f"Could not load HumanEval: {e}")
        return []


def _load_gsm8k(data_dir: str) -> List[Dict]:
    """Load GSM8K math problems (1319 test problems)."""
    try:
        from datasets import load_from_disk, load_dataset
        local_path = os.path.join(data_dir, "gsm8k")
        if os.path.exists(local_path):
            ds = load_from_disk(local_path)
        else:
            ds = load_dataset("gsm8k", "main", split="test")
            ds.save_to_disk(local_path)

        prompts = []
        for i, item in enumerate(ds):
            prompts.append({
                "id": f"gsm8k_{i}",
                "prompt": item["question"],
                "turns": [item["question"]],
            })
        return prompts
    except Exception as e:
        logger.warning(f"Could not load GSM8K: {e}")
        return []


def _load_alpaca(data_dir: str) -> List[Dict]:
    """Load Alpaca instruction-following dataset."""
    try:
        records = _load_local_dataset(data_dir, "alpaca")
        if records is None:
            return []
        prompts = []
        for i, item in enumerate(records):
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            if inp:
                instruction += f"\n{inp}"
            prompts.append({
                "id": f"alpaca_{i}",
                "prompt": instruction,
                "turns": [instruction],
            })
        return prompts
    except Exception as e:
        logger.warning(f"Could not load Alpaca: {e}")
        return []


def _load_cnn_dm(data_dir: str) -> List[Dict]:
    """Load CNN/DailyMail summarization dataset (test split)."""
    try:
        records = _load_local_dataset(data_dir, "cnn_dm")
        if records is None:
            return []
        prompts = []
        for i, item in enumerate(records):
            # Truncate long articles: input + 512 new tokens must fit the
            # fixed 2048-token KV cache that eagenerate allocates.
            article = item["article"][:2000]
            prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
            prompts.append({
                "id": f"cnn_dm_{i}",
                "prompt": prompt,
                "turns": [prompt],
            })
        return prompts
    except Exception as e:
        logger.warning(f"Could not load CNN/DM: {e}")
        return []


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs speculative decoding benchmarks with different methods."""

    def __init__(
        self,
        model: EaModel,
        is_llama3: bool = True,
        system_prompt: Optional[str] = None,
    ):
        """Initialize benchmark runner.

        Args:
            model: Loaded EaModel
            is_llama3: Whether model is LLaMA-3 (affects stop tokens)
            system_prompt: System prompt for chat models
        """
        self.model = model
        self.tokenizer = model.get_tokenizer()
        self.is_llama3 = is_llama3
        self.system_prompt = system_prompt

    def _prepare_input(self, prompt: str) -> torch.Tensor:
        """Prepare input_ids from a prompt string.

        Applies chat template if available. No system prompt by default:
        this must match data collection (collect_data.py), and DeepSeek-R1
        officially recommends against system prompts. A mismatched system
        prompt collapses EAGLE-3 tau from 5.3 to 1.9 (diag job 339301).
        """
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            messages = [{"role": "user", "content": prompt}]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            input_ids = self.tokenizer([text], add_special_tokens=False).input_ids
        except Exception:
            # Fallback for models without chat template
            input_ids = self.tokenizer([prompt]).input_ids

        return torch.as_tensor(input_ids).cuda()

    def run_autoregressive(
        self,
        prompts: List[Dict],
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> MetricsTracker:
        """Run vanilla autoregressive generation (no speculative decoding).

        This establishes the baseline speed for speedup computation.
        """
        tracker = MetricsTracker()

        for prompt_data in prompts:
            input_ids = self._prepare_input(prompt_data["prompt"])
            input_len = input_ids.shape[1]

            torch.cuda.synchronize()
            start = time.time()

            output_ids, new_token, idx = self.model.naivegenerate(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                log=True,
                is_llama3=self.is_llama3,
            )

            torch.cuda.synchronize()
            elapsed = time.time() - start

            metrics = GenerationMetrics(
                prompt_id=prompt_data["id"],
                method="autoregressive",
                temperature=temperature,
                total_tokens=_to_int(new_token),
                input_length=input_len,
                wall_time_s=elapsed,
            )
            tracker.add_generation(metrics)

        return tracker

    def run_eagle3(
        self,
        prompts: List[Dict],
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> MetricsTracker:
        """Run standard EAGLE-3 speculative decoding (no adaptive control)."""
        tracker = MetricsTracker()

        for prompt_data in prompts:
            input_ids = self._prepare_input(prompt_data["prompt"])
            input_len = input_ids.shape[1]

            torch.cuda.synchronize()
            start = time.time()

            output_ids, new_token, idx = self.model.eagenerate(
                input_ids,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                log=True,
                is_llama3=self.is_llama3,
            )

            torch.cuda.synchronize()
            elapsed = time.time() - start

            new_token_int = _to_int(new_token)
            idx_int = _to_int(idx)
            metrics = GenerationMetrics(
                prompt_id=prompt_data["id"],
                method="eagle3",
                temperature=temperature,
                total_tokens=new_token_int,
                input_length=input_len,
                wall_time_s=elapsed,
                num_steps=idx_int + 1,
                avg_acceptance_length=new_token_int / (idx_int + 1) if idx_int >= 0 else 0,
            )
            tracker.add_generation(metrics)

        return tracker

    def run_fgsd(
        self,
        prompts: List[Dict],
        controller,  # AdaptiveDraftController
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        method_name: str = "fgsd",
    ) -> MetricsTracker:
        """Run FGSD v3: EAGLE-3 with probe-gated draft depth.

        The probe runs INSIDE topK_genrate after each depth level: when mean
        P(reject) over the top_k candidates at the current depth exceeds the
        threshold, the remaining depth iterations are skipped — each skipped
        level is one saved draft-model forward. (The v2 design pruned the
        finished tree post-hoc and saved no draft compute at all.)
        """
        from ..controller.adaptive_draft import (
            install_adaptive_draft,
            uninstall_adaptive_draft,
        )

        tracker = MetricsTracker()
        controller.reset_stats()

        install_adaptive_draft(self.model, controller)
        try:
            for prompt_data in prompts:
                input_ids = self._prepare_input(prompt_data["prompt"])
                gen_metrics = self._run_fgsd_single(
                    input_ids=input_ids,
                    controller=controller,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    max_length=max_length,
                    prompt_id=prompt_data["id"],
                    method_name=method_name,
                )
                tracker.add_generation(gen_metrics)
        finally:
            uninstall_adaptive_draft(self.model)

        return tracker

    @torch.no_grad()
    def _run_fgsd_single(
        self,
        input_ids: torch.Tensor,
        controller,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        prompt_id: str = "",
        method_name: str = "fgsd",
    ) -> GenerationMetrics:
        """Run FGSD generation for a single prompt.

        Requires install_adaptive_draft() to have patched ea_layer beforehand;
        per-step draft depth and probe timing are read back from ea_layer.
        """
        model = self.model
        ea_layer = model.ea_layer

        if self.is_llama3:
            stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(temperature=temperature)
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        model.ea_layer.reset_kv()

        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            current_length_data.zero_()
        else:
            (past_key_values, past_key_values_data, current_length_data) = \
                initialize_past_key_values(model.base_model, max_length=max_length)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(model)

        # Timer includes initialize_tree for fair comparison with eagenerate()
        torch.cuda.synchronize()
        gen_start = time.time()

        # Prefill — the patched topK_genrate inside initialize_tree records
        # drafted depth / probe time on ea_layer
        probe_ms_prev = ea_layer._fgsd_probe_time_ms
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = \
            initialize_tree(input_ids, model, past_key_values, logits_processor)

        new_token = 0
        effective_max_length = max_length - model.ea_layer.total_tokens - 10
        steps = []

        for step_idx in range(effective_max_length):
            step_metrics = StepMetrics()

            # The current tree was produced by the most recent (patched)
            # topK_genrate call; read back its drafted depth and probe time.
            drafted_depth = ea_layer._fgsd_last_drafted_depth
            early_stopped = ea_layer._fgsd_last_early_stopped
            step_metrics.probe_time_ms = ea_layer._fgsd_probe_time_ms - probe_ms_prev
            probe_ms_prev = ea_layer._fgsd_probe_time_ms

            # Set (potentially pruned) tree mask for target model
            model.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            # --- Target model verification ---
            verify_start = time.time()
            target_logits, hidden_state_new, outputs = tree_decoding(
                model, draft_tokens, past_key_values, tree_position_ids,
                input_ids, retrieve_indices,
            )
            torch.cuda.synchronize()
            step_metrics.verify_time_ms = (time.time() - verify_start) * 1000

            # Verification
            draft_tokens_padded = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_padded[0, retrieve_indices]
            best_candidate, accept_length, sample_p = evaluate_posterior(
                target_logits, candidates, logits_processor
            )

            accept_len_int = _to_int(accept_length)
            # +1 for the bonus token from target model (consistent with EAGLE-3 tau)
            step_metrics.accepted_length = accept_len_int + 1
            # draft_length now records drafted DEPTH levels (1..depth), not
            # tree width — early exit saves (depth - draft_length) forwards
            step_metrics.draft_length = drafted_depth
            step_metrics.early_stopped = early_stopped
            step_metrics.total_time_ms = step_metrics.verify_time_ms + step_metrics.probe_time_ms
            steps.append(step_metrics)

            controller.update_stats(drafted_depth, accept_len_int + 1, early_stopped)

            # Update state (calls the patched topK_genrate for the next tree)
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = \
                update_inference_inputs(
                    input_ids, candidates, best_candidate, accept_length,
                    retrieve_indices, logits_processor, new_token,
                    past_key_values_data, current_length_data,
                    model, hidden_state_new, sample_p,
                )

            # Check stopping
            if self.is_llama3 and stop_token_id in input_ids[0, input_len:].tolist():
                break
            if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > effective_max_length:
                break

        torch.cuda.synchronize()
        gen_time = time.time() - gen_start

        return GenerationMetrics(
            prompt_id=prompt_id,
            method=method_name,
            temperature=temperature,
            total_tokens=_to_int(new_token),
            input_length=input_len,
            wall_time_s=gen_time,
            steps=steps,
        )


def run_warmup(model: EaModel, tokenizer, is_llama3: bool = True, n_warmup: int = 3):
    """Run warmup iterations to stabilize GPU timing."""
    logger.info(f"Running {n_warmup} warmup iterations...")
    prompt = "Write a short story about a robot learning to paint."
    messages = [{"role": "user", "content": prompt}]

    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer([text], add_special_tokens=False).input_ids
    except Exception:
        input_ids = tokenizer([prompt]).input_ids

    input_ids = torch.as_tensor(input_ids).cuda()

    for i in range(n_warmup):
        torch.cuda.synchronize()
        model.eagenerate(
            input_ids, temperature=0.0, max_new_tokens=128,
            log=False, is_llama3=is_llama3,
        )
        torch.cuda.synchronize()

    logger.info("Warmup complete")
