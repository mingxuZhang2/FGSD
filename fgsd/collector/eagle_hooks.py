"""
Hook into EAGLE-3 speculative decoding to capture hidden states and
accept/reject labels at each draft position.

Architecture notes (from reading EAGLE source):
- EAGLE-3's target model (modeling_llama_kv.py) captures hidden states at
  3 specific layers: layer 2, layer n//2, layer n-3 (where n = num_layers).
  These are concatenated (3 * hidden_size) and fed into the draft model.
- The draft model (cnets.py::Model) has a single decoder layer (`midlayer`)
  plus fc + lm_head. We hook into this to capture draft hidden states.
- Verification happens in evaluate_posterior() which compares target logits
  against draft tokens to determine acceptance.

We instrument the EAGLE eagenerate() loop to capture:
1. Draft model hidden states at each draft step
2. Draft model output logits/entropy
3. Accept/reject labels from verification
4. Position information within the draft sequence
"""

import sys
import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import numpy as np

# Add EAGLE to path so we can import it
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

logger = logging.getLogger(__name__)


@dataclass
class DraftStepRecord:
    """Record for a single draft-verify cycle."""
    # Per-depth hidden states from draft model midlayer.
    # List of tensors: depth_hidden_states[d] has shape [n_candidates, hidden_dim]
    # where d=0 is the context encoding (skipped for training), d=1..D are
    # the draft depth levels. Each depth level has top_k candidate hidden states.
    depth_hidden_states: List[torch.Tensor] = field(default_factory=list)

    # Legacy: single draft hidden tensor (deprecated, use depth_hidden_states)
    draft_hidden: Optional[torch.Tensor] = None

    # Target model hidden states at the 3 captured layers
    # Shape: [tree_size, 3 * hidden_dim]
    target_hidden: Optional[torch.Tensor] = None

    # Number of tokens accepted in this step (draft tokens only, excludes sample_token)
    accept_length: int = 0

    # Accept/reject label for each draft position (legacy)
    accept_mask: Optional[torch.Tensor] = None

    # Position of each draft token in the draft sequence (0-indexed)
    draft_positions: Optional[torch.Tensor] = None

    # The actual draft token ids
    draft_token_ids: Optional[torch.Tensor] = None

    # Step index in the generation
    step_idx: int = 0


@dataclass
class GenerationRecord:
    """Record for an entire generation (one prompt)."""
    prompt_id: str = ""
    prompt_text: str = ""
    steps: List[DraftStepRecord] = field(default_factory=list)
    total_tokens: int = 0
    total_steps: int = 0
    wall_time: float = 0.0


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy of logit distribution.

    Args:
        logits: Shape [..., vocab_size]

    Returns:
        Entropy tensor with shape [...]
    """
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


class HiddenStateCollector:
    """Instruments EAGLE-3 speculative decoding to collect hidden states
    and accept/reject labels.

    This class wraps an EaModel and modifies its generation loop to capture
    intermediate states needed for training rejection probes.

    Usage:
        model = EaModel.from_pretrained(...)
        collector = HiddenStateCollector(model, collect_draft_hidden=True)
        records = collector.generate_and_collect(input_ids, ...)
    """

    def __init__(
        self,
        model: EaModel,
        collect_draft_hidden: bool = True,
        collect_target_hidden: bool = True,
        collect_draft_logits: bool = True,
        collect_entropy: bool = True,
    ):
        self.model = model
        self.collect_draft_hidden = collect_draft_hidden
        self.collect_target_hidden = collect_target_hidden
        self.collect_draft_logits = collect_draft_logits
        self.collect_entropy = collect_entropy

        # Per-depth hidden state accumulation (appended by hook, cleared between steps)
        self._depth_hidden_states: List[torch.Tensor] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHook] = []

        # Install hook on draft model's midlayer to capture hidden states
        if self.collect_draft_hidden:
            self._install_draft_hooks()

    def _install_draft_hooks(self) -> None:
        """Install forward hook on the EAGLE draft model's decoder layer
        to capture its output hidden states."""
        ea_layer = self.model.ea_layer

        def midlayer_hook(module, input_args, output):
            h = output[0].detach() if isinstance(output, tuple) else output.detach()
            self._depth_hidden_states.append(h)

        handle = ea_layer.midlayer.register_forward_hook(midlayer_hook)
        self._hook_handles.append(handle)
        logger.info("Installed forward hook on EAGLE draft model midlayer")

    def remove_hooks(self) -> None:
        """Remove all installed hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    @torch.no_grad()
    def generate_and_collect(
        self,
        input_ids: torch.Tensor,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        is_llama3: bool = False,
        prompt_id: str = "",
        prompt_text: str = "",
    ) -> GenerationRecord:
        """Run EAGLE-3 generation with hidden state collection.

        This reimplements the eagenerate() loop from ea_model.py with
        additional instrumentation to capture hidden states and labels.

        Args:
            input_ids: Input token ids, shape [1, seq_len]
            temperature: Sampling temperature (0 = greedy)
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum new tokens to generate
            max_length: Maximum total sequence length
            is_llama3: Whether to use LLaMA-3 stop tokens
            prompt_id: Identifier for this prompt
            prompt_text: Text of the prompt

        Returns:
            GenerationRecord with all collected data
        """
        model = self.model
        record = GenerationRecord(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
        )

        if is_llama3:
            stop_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        if temperature > 1e-5:
            logits_processor = prepare_logits_processor(
                temperature=temperature, top_p=top_p, top_k=top_k
            )
        else:
            logits_processor = None

        padding = (torch.zeros(1, 1, dtype=torch.long) - 1).to(input_ids.device)
        input_ids = input_ids.clone()
        model.ea_layer.reset_kv()

        # Initialize KV cache
        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model, max_length=max_length)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        reset_tree_mode(model)

        # Prefill: run base model on full prompt, then initialize draft tree.
        # topK_genrate inside initialize_tree fires the midlayer hook D+1 times.
        self._depth_hidden_states.clear()
        draft_tokens, retrieve_indices, tree_mask, tree_position_ids, logits, hidden_state, sample_token = initialize_tree(
            input_ids, model, past_key_values, logits_processor
        )

        new_token = 0
        effective_max_length = max_length - model.ea_layer.total_tokens - 10
        start_time = time.time()

        # depth_hidden_states was populated by initialize_tree's topK_genrate call
        for step_idx in range(effective_max_length):
            step_record = DraftStepRecord(step_idx=step_idx)

            # Set tree mask for target model
            model.base_model.model.tree_mask = tree_mask
            draft_tokens = draft_tokens.to(input_ids.device)

            # --- Target model forward (verification) ---
            target_logits, hidden_state_new, outputs = tree_decoding(
                model,
                draft_tokens,
                past_key_values,
                tree_position_ids,
                input_ids,
                retrieve_indices,
            )

            # Capture target hidden states if requested
            if self.collect_target_hidden and hasattr(outputs, "hidden_states"):
                if outputs.hidden_states is not None:
                    target_hs = torch.cat(
                        [h.detach().cpu() for h in outputs.hidden_states], dim=-1
                    )
                    step_record.target_hidden = target_hs.squeeze(0)

            # Prepare candidates for verification
            draft_tokens_padded = torch.cat((draft_tokens, padding), dim=1)
            candidates = draft_tokens_padded[0, retrieve_indices]

            # --- Verification: determine accept/reject ---
            best_candidate, accept_length, sample_p = evaluate_posterior(
                target_logits, candidates, logits_processor
            )

            accept_length_int = accept_length.item() if isinstance(accept_length, torch.Tensor) else int(accept_length)
            step_record.accept_length = accept_length_int

            # Capture per-depth hidden states from the midlayer hook.
            # _depth_hidden_states was filled during the topK_genrate that
            # produced this step's draft tree. Entry [0] is context encoding,
            # entries [1..D] are per-depth draft candidates (each [1, top_k, H]).
            if self.collect_draft_hidden and self._depth_hidden_states:
                step_record.depth_hidden_states = [
                    h.cpu().squeeze(0) if h.dim() == 3 else h.cpu()
                    for h in self._depth_hidden_states
                ]

            record.steps.append(step_record)

            # Clear before update_inference_inputs, which calls topK_genrate
            # and will refill _depth_hidden_states for the next step
            self._depth_hidden_states.clear()

            # --- Update state for next step ---
            input_ids, draft_tokens, retrieve_indices, tree_mask, tree_position_ids, new_token, hidden_state, sample_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                retrieve_indices,
                logits_processor,
                new_token,
                past_key_values_data,
                current_length_data,
                model,
                hidden_state_new,
                sample_p,
            )

            # Check stopping conditions
            if is_llama3:
                if stop_token_id in input_ids[0, input_len:].tolist():
                    break
            if model.tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
            if input_ids.shape[1] > effective_max_length:
                break

        record.wall_time = time.time() - start_time
        record.total_tokens = new_token
        record.total_steps = step_idx + 1

        return record

    @torch.no_grad()
    def collect_flat_dataset(
        self,
        records: List[GenerationRecord],
    ) -> Dict[str, torch.Tensor]:
        """Flatten generation records into a dataset for probe training.

        Uses per-depth hidden states with depth-level acceptance labels.
        For each generation step, at each depth d (1..D):
          - Features: top_k candidate hidden states from depth d
          - Label: 0 (accepted) if d <= accept_length, 1 (rejected) otherwise

        This matches how the probe is used during eval: mean P(reject)
        across candidates at depth d determines whether to prune.

        Returns:
            Dictionary with tensors:
                - draft_hidden: [N, hidden_dim]
                - labels: [N] binary (1=rejected, 0=accepted)
                - positions: [N] depth index
                - prompt_ids: [N]
                - step_ids: [N]
        """
        all_draft_hidden = []
        all_labels = []
        all_positions = []
        all_prompt_ids = []
        all_step_ids = []

        for rec_idx, record in enumerate(records):
            for step in record.steps:
                if step.depth_hidden_states:
                    # New path: per-depth aligned data
                    for d_idx in range(1, len(step.depth_hidden_states)):
                        d_hidden = step.depth_hidden_states[d_idx]
                        if d_hidden.dim() == 1:
                            d_hidden = d_hidden.unsqueeze(0)
                        n_candidates = d_hidden.shape[0]

                        if d_idx <= step.accept_length:
                            labels_d = torch.zeros(n_candidates, dtype=torch.long)
                        else:
                            labels_d = torch.ones(n_candidates, dtype=torch.long)

                        all_draft_hidden.append(d_hidden)
                        all_labels.append(labels_d)
                        all_positions.append(
                            torch.full((n_candidates,), d_idx, dtype=torch.long)
                        )
                        all_prompt_ids.append(
                            torch.full((n_candidates,), rec_idx, dtype=torch.long)
                        )
                        all_step_ids.append(
                            torch.full((n_candidates,), step.step_idx, dtype=torch.long)
                        )

                elif step.accept_mask is not None and step.draft_hidden is not None:
                    # Legacy path: old-format data (backward compat)
                    n_tokens = len(step.accept_mask)
                    labels = (~step.accept_mask).long()
                    if step.draft_hidden.shape[0] >= n_tokens:
                        all_draft_hidden.append(step.draft_hidden[:n_tokens])
                        all_labels.append(labels)
                        all_positions.append(step.draft_positions[:n_tokens])
                        all_prompt_ids.append(
                            torch.full((n_tokens,), rec_idx, dtype=torch.long)
                        )
                        all_step_ids.append(
                            torch.full((n_tokens,), step.step_idx, dtype=torch.long)
                        )

        result = {
            "labels": torch.cat(all_labels, dim=0),
            "positions": torch.cat(all_positions, dim=0),
            "prompt_ids": torch.cat(all_prompt_ids, dim=0),
            "step_ids": torch.cat(all_step_ids, dim=0),
        }

        if all_draft_hidden:
            result["draft_hidden"] = torch.cat(all_draft_hidden, dim=0)

        n_accepted = (result["labels"] == 0).sum()
        n_rejected = (result["labels"] == 1).sum()
        logger.info(
            f"Collected {result['labels'].shape[0]} samples: "
            f"{n_accepted} accepted, {n_rejected} rejected "
            f"(reject rate: {n_rejected / (n_accepted + n_rejected):.1%})"
        )

        return result
