"""In-draft adaptive early exit for EAGLE-3 (FGSD v3).

The v2 design pruned the draft tree AFTER topK_genrate had already built it,
so no draft compute was ever saved (avg_draft_length was stuck at 59). This
module patches `ea_layer.topK_genrate` so the probe runs INSIDE the per-depth
draft loop: when the mean P(reject) over the top_k candidates at the current
depth exceeds the threshold, the remaining depth iterations are skipped
entirely — each skipped iteration is one full draft-model forward.

The probe input is `out_hidden` from the draft model forward, which is exactly
the midlayer output captured by the collection hook (Model.forward returns
layer_outputs[0] from midlayer), so train/eval features match. The depth index
(i+1) is passed as the probe position, matching the collection labels.
"""

import math
import time
import types
import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def adaptive_topk_genrate(self, hidden_states, input_ids, head, logits_processor):
    """Copy of eagle.model.cnets.Model.topK_genrate (EAGLE-3) with probe-based
    early exit inserted at the end of each depth iteration.

    Stats written to self: _fgsd_last_drafted_depth, _fgsd_last_early_stopped,
    _fgsd_probe_time_ms (cumulative).
    """
    controller = self._fgsd_controller

    input_ids = input_ids.to(hidden_states.device)
    total_tokens = self.total_tokens
    depth = self.depth
    top_k = self.top_k

    sample_token = input_ids[:, -1]

    scores_list = []
    parents_list = []
    ss_token = []

    input_ids = input_ids[:, 1:]
    input_ids = input_ids.to(hidden_states.device)

    len_posi = input_ids.shape[1]
    self.reset()

    if hasattr(self, "stable_kv") and self.stable_kv is not None:
        kv_len = self.stable_kv[0][0].shape[2]
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids[:, kv_len:],
                                           past_key_values=self.stable_kv, use_cache=True)
    else:
        out_hidden, past_key_values = self(hidden_states, input_ids=input_ids, use_cache=True)
    self.stable_kv = past_key_values
    last_hidden = out_hidden[:, -1]

    last_headout = self.lm_head(self.norm(last_hidden))

    last_p = self.logsoftmax(last_headout)
    top = torch.topk(last_p, top_k, dim=-1)
    topk_index, topk_p = top.indices, top.values
    scores = topk_p[0]
    scores_list.append(scores[None])
    parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
    if self.config.vocab_size == self.config.draft_vocab_size:
        ss_token.append(topk_index)
        input_ids = topk_index
    else:
        ss_token.append(topk_index + self.d2t[topk_index])
        input_ids = topk_index + self.d2t[topk_index]
    input_hidden = last_hidden[None].repeat(1, top_k, 1)
    tree_mask = self.tree_mask_init
    topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

    # Bidirectional adaptive depth: probe can stop early (skip remaining base
    # depth levels) AND extend beyond base depth while it predicts continued
    # acceptance. Extension is where the speedup comes from: one extra
    # accepted token saves a whole verify cycle (~21ms), while one extra
    # depth level costs ~1.2ms draft + ~0.6ms probe.
    max_total_depth = depth
    if controller is not None:
        max_total_depth = max(depth, getattr(controller, "max_extended_depth", 0) or 0)
    extend_threshold = getattr(controller, "extend_threshold", None) if controller is not None else None
    # Signal source: "probe" (FGSD, draft hidden states) or "entropy"
    # (SVIP-style baseline, draft output distribution). Same control logic,
    # different signal — isolates the value of internal features.
    signal = getattr(controller, "signal", "probe") if controller is not None else None

    drafted_depth = max_total_depth
    early_stopped = False

    for i in range(max_total_depth):
        self.tree_mask = tree_mask
        position_ids = len_posi + self.position_ids
        out_hidden, past_key_values = self(input_hidden, input_ids=input_ids, past_key_values=past_key_values,
                                           position_ids=position_ids, use_cache=True)
        len_posi += 1

        bias1 = top_k if i > 0 else 0
        bias2 = max(0, i - 1)
        bias = 1 + top_k ** 2 * bias2 + bias1
        parents = (topk_cs_index + bias)
        parents_list.append(parents)

        last_headout = self.lm_head(self.norm(out_hidden[0]))
        last_p = self.logsoftmax(last_headout)

        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values

        cu_scores = topk_p + scores[:, None]

        topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
        topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
        scores = topk_cs_p

        out_ids = topk_cs_index // top_k
        input_hidden = out_hidden[:, out_ids]

        input_ids = topk_index.view(-1)[topk_cs_index][None]

        if self.config.vocab_size == self.config.draft_vocab_size:
            ss_token.append(topk_index)
        else:
            input_ids = input_ids + self.d2t[input_ids]
            ss_token.append(topk_index + self.d2t[topk_index])
        scores_list.append(cu_scores)
        tree_mask = torch.cat((tree_mask[:, :, out_ids], self.tree_mask_init), dim=3)

        # ---- FGSD adaptive depth decision ----
        # out_hidden[0] holds the hidden states of the depth-(i+1) candidate
        # tokens (same tensor the collection hook captured). Probe positions
        # are clamped to the trained range (1..5). The depth-(i+2) candidates
        # appended above stay in the score pool even on a stop; their
        # cumulative scores rank them below surviving shallow nodes, and at
        # bs=1 extra verification width is essentially free.
        if controller is not None and i + 1 < max_total_depth:
            drafted = i + 1
            # Skip the depth-1 check: reject rate there is ~3%, so the
            # expected saving is below the cost of the probe call itself.
            if drafted == 1 and depth > 2:
                continue
            # Extension-only mode: skip base-depth checks entirely.
            # For probe signal: threshold > 1.0 can never fire (sigmoid range).
            # For entropy signal: threshold > 100 serves as the disable flag.
            skip_base = (signal != "entropy" and controller.threshold > 1.0) or \
                        (signal == "entropy" and controller.threshold > 100.0)
            if drafted < depth and skip_base:
                continue
            t0 = time.time()
            if signal == "entropy":
                # Mean entropy (nats) of the distributions the drafter would
                # sample the next depth from; high entropy = uncertain.
                mean_p = -(last_p.exp() * last_p).sum(-1).mean().item()
            else:
                max_pos = getattr(controller, "max_position", 5)
                p_reject = controller.predict_rejection_batch(
                    out_hidden[0], position=min(drafted, max_pos)
                )
                aggregation = getattr(controller, "aggregation", "mean")
                if aggregation == "score_weighted" and scores.numel() > 1:
                    w = torch.softmax(scores.float(), dim=0)
                    mean_p = (w * p_reject.float()).sum().item()
                elif aggregation == "top1":
                    top_idx = torch.argmax(scores)
                    mean_p = p_reject[top_idx].item()
                else:
                    mean_p = p_reject.mean().item()
            self._fgsd_probe_time_ms += (time.time() - t0) * 1000

            # Depth-dependent threshold: decay extend_threshold at deeper levels
            extend_decay = getattr(controller, "extend_decay", 0.0)
            if drafted < depth:
                # Within base depth: early exit on predicted rejection
                if mean_p > controller.threshold:
                    drafted_depth = drafted
                    early_stopped = True
                    break
            else:
                # At/beyond base depth: keep extending only while the probe
                # predicts continued acceptance
                extra = max(0, drafted - depth)
                eff_thr = extend_threshold * math.exp(-extend_decay * extra) if extend_decay > 0 else extend_threshold
                if eff_thr is None or mean_p > eff_thr:
                    drafted_depth = drafted
                    break

    self._fgsd_last_drafted_depth = drafted_depth
    self._fgsd_last_early_stopped = early_stopped

    scores_list = torch.cat(scores_list, dim=0).view(-1)
    ss_token_list = torch.cat(ss_token, dim=0).view(-1)
    top_scores = torch.topk(scores_list, total_tokens, dim=-1)
    top_scores_index = top_scores.indices
    top_scores_index = torch.sort(top_scores_index).values

    draft_tokens = ss_token_list[top_scores_index]
    draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

    draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
    mask_index = torch.searchsorted(top_scores_index, draft_parents - 1, right=False)
    mask_index[draft_parents == 0] = -1
    mask_index = mask_index + 1
    mask_index_list = mask_index.tolist()
    tree_mask = torch.eye(total_tokens + 1).bool()
    tree_mask[:, 0] = True
    for i in range(total_tokens):
        tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

    tree_position_ids = torch.sum(tree_mask, dim=1) - 1

    tree_mask = tree_mask.float()[None, None]
    draft_tokens = draft_tokens[None]

    del parents_list, scores_list, ss_token, ss_token_list, draft_parents

    max_depth = torch.max(tree_position_ids) + 1
    noleaf_index = torch.unique(mask_index).tolist()
    noleaf_num = len(noleaf_index) - 1
    leaf_num = total_tokens - noleaf_num

    retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
    retrieve_indices = retrieve_indices.tolist()

    rid = 0
    position_ids_list = tree_position_ids.tolist()

    for i in range(total_tokens + 1):
        if i not in noleaf_index:
            cid = i
            depth = position_ids_list[i]
            for j in reversed(range(depth + 1)):
                retrieve_indices[rid][j] = cid
                cid = mask_index_list[cid - 1]
            rid += 1

    if logits_processor is not None:
        maxitem = total_tokens + 5

        def custom_sort(lst):
            sort_keys = []
            for i in range(len(lst)):
                sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
            return sort_keys

        retrieve_indices = sorted(retrieve_indices, key=custom_sort)

    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    del mask_index, mask_index_list, noleaf_index, noleaf_num, leaf_num, max_depth, rid
    tree_position_ids = tree_position_ids.to(hidden_states.device)

    return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


def install_adaptive_draft(model, controller) -> None:
    """Replace ea_layer.topK_genrate with the probe-gated version."""
    ea_layer = model.ea_layer
    if getattr(ea_layer, "_fgsd_installed", False):
        ea_layer._fgsd_controller = controller
        return
    ea_layer._fgsd_controller = controller
    ea_layer._fgsd_probe_time_ms = 0.0
    ea_layer._fgsd_last_drafted_depth = ea_layer.depth
    ea_layer._fgsd_last_early_stopped = False
    ea_layer._orig_topk_genrate = ea_layer.topK_genrate
    ea_layer.topK_genrate = types.MethodType(adaptive_topk_genrate, ea_layer)
    ea_layer._fgsd_installed = True
    logger.info("Installed adaptive (probe-gated) topK_genrate on ea_layer")


def uninstall_adaptive_draft(model) -> None:
    """Restore the original topK_genrate."""
    ea_layer = model.ea_layer
    if not getattr(ea_layer, "_fgsd_installed", False):
        return
    ea_layer.topK_genrate = ea_layer._orig_topk_genrate
    ea_layer._fgsd_controller = None
    ea_layer._fgsd_installed = False
    logger.info("Restored original topK_genrate")
