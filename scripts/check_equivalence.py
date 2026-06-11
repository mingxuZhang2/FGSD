"""Output-equivalence check: speculative methods vs vanilla greedy decoding.

At temperature 0, EAGLE-3 verification guarantees the output token sequence
is identical to the target model's greedy decoding. Our adaptive-depth
controllers only change WHICH draft trees get proposed, never how candidates
are verified, so the guarantee must carry over. This script proves it
empirically: for each prompt, generate with vanilla greedy and with each
speculative method, then compare token-by-token.

Lengths can differ by up to one step's accepted tokens (eagenerate stops when
new_token exceeds max_new_tokens, possibly mid-step), so sequences are
truncated at the first EOS and compared over the common prefix length.
"""
import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
EAGLE_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EAGLE")
sys.path.insert(0, EAGLE_ROOT)

from eagle.model.ea_model import EaModel

from fgsd.eval.benchmark import BenchmarkRunner, load_benchmark_prompts
from fgsd.controller.adaptive import AdaptiveDraftController
from fgsd.controller.baselines import SVIPEntropyController
from fgsd.controller.adaptive_draft import install_adaptive_draft, uninstall_adaptive_draft

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def continuation(output_ids: torch.Tensor, input_len: int, eos_id: int) -> list:
    """New tokens after the prompt, truncated at (and including) first EOS."""
    toks = output_ids[0, input_len:].tolist()
    if eos_id in toks:
        toks = toks[: toks.index(eos_id) + 1]
    return toks


@torch.no_grad()
def divergence_logit_gap(model, input_ids, ref, pos, ref_tok, hyp_tok):
    """Teacher-force the common prefix through the target model and measure
    how contested the divergence position is.

    If the top-2 logit gap is at fp16 noise scale, the mismatch is a
    numerical argmax tie between the sequential (vanilla) and batched tree
    (verification) forward passes, not a correctness difference.
    """
    model.base_model.model.tree_mask = None
    prefix = torch.cat(
        [input_ids, torch.tensor([ref[:pos]], dtype=torch.long, device=input_ids.device)],
        dim=1,
    )
    logits = model.base_model(prefix).logits[0, -1].float()
    top2 = torch.topk(logits, 2)
    order = torch.argsort(logits, descending=True)
    rank = {int(t): int((order == t).nonzero()[0]) for t in (ref_tok, hyp_tok)}
    return {
        "top1_top2_gap": (top2.values[0] - top2.values[1]).item(),
        "ref_rank": rank[int(ref_tok)],
        "hyp_rank": rank[int(hyp_tok)],
        "ref_hyp_gap": abs(logits[ref_tok] - logits[hyp_tok]).item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--ea-model-path", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--benchmarks", nargs="+", default=["mt_bench", "humaneval", "gsm8k"])
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--data-dir", default="data/benchmarks")
    parser.add_argument("--output", default="results/analysis/equivalence_check.json")
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    model = EaModel.from_pretrained(
        use_eagle3=True,
        base_model_path=args.base_model_path,
        ea_model_path=args.ea_model_path,
        total_token=60, depth=5, top_k=10,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    tokenizer = model.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    runner = BenchmarkRunner(model=model, is_llama3=False)

    # Main-table configs: probe (linear d10, extension-only) and svip entropy
    probe_controller = AdaptiveDraftController.from_checkpoint(
        checkpoint_dir=args.probe_dir, device="cuda",
        threshold=1.5, max_extended_depth=10, extend_threshold=0.5,
    )
    svip_controller = SVIPEntropyController(
        entropy_threshold=3.0, max_extended_depth=10, extend_threshold=0.7,
    )

    methods = ["eagle3", "fgsd_probe", "svip"]
    results = {}

    for bench in args.benchmarks:
        prompts = load_benchmark_prompts(bench, data_dir=args.data_dir,
                                         max_samples=args.num_prompts)
        stats = {m: {"match": 0, "mismatch": 0, "tokens_compared": 0,
                     "first_mismatches": []} for m in methods}

        for p in prompts:
            input_ids = runner._prepare_input(p["prompt"])
            input_len = input_ids.shape[1]

            with torch.no_grad():
                out_v = model.naivegenerate(
                    input_ids, temperature=0.0,
                    max_new_tokens=args.max_new_tokens, log=False)
            ref = continuation(out_v, input_len, eos_id)

            for method in methods:
                if method == "fgsd_probe":
                    install_adaptive_draft(model, probe_controller)
                elif method == "svip":
                    install_adaptive_draft(model, svip_controller)
                try:
                    with torch.no_grad():
                        out_s = model.eagenerate(
                            input_ids, temperature=0.0,
                            max_new_tokens=args.max_new_tokens, log=False)
                finally:
                    if method != "eagle3":
                        uninstall_adaptive_draft(model)

                hyp = continuation(out_s, input_len, eos_id)
                n = min(len(ref), len(hyp))
                s = stats[method]
                s["tokens_compared"] += n
                if ref[:n] == hyp[:n]:
                    s["match"] += 1
                else:
                    s["mismatch"] += 1
                    pos = next(i for i in range(n) if ref[i] != hyp[i])
                    if len(s["first_mismatches"]) < 5:
                        entry = {
                            "prompt_id": p["id"], "position": pos,
                            "ref_token": ref[pos], "hyp_token": hyp[pos],
                            "ref_text": tokenizer.decode(ref[max(0, pos - 5): pos + 3]),
                            "hyp_text": tokenizer.decode(hyp[max(0, pos - 5): pos + 3]),
                        }
                        entry.update(divergence_logit_gap(
                            model, input_ids, ref, pos, ref[pos], hyp[pos]))
                        s["first_mismatches"].append(entry)

        results[bench] = stats
        for m in methods:
            s = stats[m]
            total = s["match"] + s["mismatch"]
            logger.info(f"[{bench}] {m}: {s['match']}/{total} prompts identical "
                        f"({s['tokens_compared']} tokens compared)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved to {args.output}")

    all_ok = all(s["mismatch"] == 0 for b in results.values() for s in b.values())
    if all_ok:
        print("EQUIVALENCE: PASS")
    else:
        gaps = [fm["ref_hyp_gap"] for b in results.values()
                for s in b.values() for fm in s["first_mismatches"]]
        rank2 = sum(1 for b in results.values() for s in b.values()
                    for fm in s["first_mismatches"] if fm["hyp_rank"] <= 1)
        print(f"EQUIVALENCE: MISMATCHES (max ref/hyp logit gap "
              f"{max(gaps):.4f}, {rank2}/{len(gaps)} divergent tokens are "
              f"the target's top-2) — fp16 argmax ties if gaps are small")


if __name__ == "__main__":
    main()
