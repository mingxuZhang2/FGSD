"""Offline comparison: probe vs draft-output entropy as rejection predictors.

Both signals are evaluated on the same held-out samples with the same
depth-level labels, answering: how much more discriminative are internal
hidden states than the output distribution's entropy?

Entropy is computed offline by pushing the saved draft hidden states through
the drafter's own norm + lm_head (the exact distribution topK_genrate samples
from), so no recollection is needed.
"""
import argparse
import json
import logging
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fgsd.collector.dataset import load_collected_data
from fgsd.probe.models import create_probe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_drafter_head(ea_model_path: str, device: str):
    """Load only norm weight + lm_head weight from the EAGLE-3 drafter."""
    import glob as _glob
    state = {}
    safetensors = _glob.glob(os.path.join(ea_model_path, "*.safetensors"))
    if safetensors:
        from safetensors.torch import load_file
        for f in safetensors:
            state.update(load_file(f))
    else:
        for f in _glob.glob(os.path.join(ea_model_path, "*.bin")):
            state.update(torch.load(f, map_location="cpu"))

    lm_head = state["lm_head.weight"].to(device).float()      # [draft_vocab, hidden]
    norm_w = state["norm.weight"].to(device).float()          # [hidden]

    cfg = json.load(open(os.path.join(ea_model_path, "config.json")))
    eps = cfg.get("rms_norm_eps", 1e-5)
    logger.info(f"Drafter head: lm_head {tuple(lm_head.shape)}, rms_eps={eps}")
    return lm_head, norm_w, eps


@torch.no_grad()
def compute_entropy(hidden: torch.Tensor, lm_head, norm_w, eps, batch_size=8192):
    """Entropy (nats) of softmax(lm_head(rmsnorm(h))) per sample."""
    out = []
    for i in range(0, hidden.shape[0], batch_size):
        h = hidden[i:i + batch_size].cuda().float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w
        logits = h @ lm_head.T
        logp = F.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        out.append(ent.cpu())
    return torch.cat(out)


@torch.no_grad()
def probe_scores(hidden, positions, probe_dir, device="cuda", batch_size=8192):
    """P(reject) from the trained probe on the same samples."""
    cfg = json.load(open(os.path.join(probe_dir, "config.json")))
    probe = create_probe(
        probe_type=cfg["probe_type"], input_dim=cfg["input_dim"],
        hidden_dim=cfg.get("hidden_dim", 256), num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.1), use_position=cfg.get("use_position", False),
    )
    ckpt = torch.load(os.path.join(probe_dir, "best.pt"), map_location=device, weights_only=False)
    probe.load_state_dict(ckpt["model_state_dict"])
    probe.to(device).eval()

    norm_stats = torch.load(os.path.join(probe_dir, "norm_stats.pt"),
                            map_location=device, weights_only=False)
    mean, std = norm_stats["mean"].float(), norm_stats["std"].float()

    out = []
    for i in range(0, hidden.shape[0], batch_size):
        h = hidden[i:i + batch_size].cuda().float()
        h = (h - mean) / std
        if cfg.get("use_position", False):
            pos = positions[i:i + batch_size].cuda().clamp(max=15)
            logit = probe(h, pos)
        else:
            logit = probe(h)
        out.append(torch.sigmoid(logit).flatten().cpu())
    return torch.cat(out)


def report_auroc(name, scores, labels, positions):
    from sklearn.metrics import roc_auc_score
    overall = roc_auc_score(labels.numpy(), scores.numpy())
    per_pos = {}
    for p in sorted(positions.unique().tolist()):
        m = positions == p
        if labels[m].sum() > 10 and (1 - labels[m]).sum() > 10:
            per_pos[p] = roc_auc_score(labels[m].numpy(), scores[m].numpy())
    pos_str = " ".join(f"d{p}={v:.3f}" for p, v in per_pos.items())
    logger.info(f"[{name}] AUROC overall={overall:.4f} | {pos_str}")
    return {"overall": overall, "per_position": per_pos}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ea-model-path", required=True)
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--train-subsample", type=int, default=200000,
                        help="Size of the training subsample (to exclude); "
                             "matches train_probe.py max_train_samples")
    parser.add_argument("--eval-samples", type=int, default=150000)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the training seed to exclude train rows")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    logger.info("Loading collected data...")
    data = load_collected_data(args.data_dir)
    n = data["labels"].shape[0]
    logger.info(f"{n} samples total")

    # Reproduce the training subsample permutation and take rows AFTER it
    # so the probe is evaluated on data it never saw.
    torch.manual_seed(args.seed)
    perm = torch.randperm(n)
    if n > args.train_subsample:
        eval_idx = perm[args.train_subsample:args.train_subsample + args.eval_samples]
    else:
        eval_idx = perm[:args.eval_samples]
    hidden = data["draft_hidden"][eval_idx]
    labels = data["labels"][eval_idx]
    positions = data["positions"][eval_idx]
    logger.info(f"Eval set: {len(eval_idx)} samples, reject rate {labels.float().mean():.3f}")

    lm_head, norm_w, eps = load_drafter_head(args.ea_model_path, "cuda")
    logger.info("Computing entropy from drafter head...")
    entropy = compute_entropy(hidden, lm_head, norm_w, eps)

    logger.info("Computing probe scores...")
    p_reject = probe_scores(hidden, positions, args.probe_dir)

    results = {
        "entropy": report_auroc("entropy", entropy, labels, positions),
        "probe": report_auroc("probe", p_reject, labels, positions),
    }

    # Combined signal: logistic regression on [entropy, p_reject]
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        X = torch.stack([entropy, p_reject], dim=1).numpy()
        y = labels.numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
        lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        comb = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
        ent_half = roc_auc_score(yte, Xte[:, 0])
        probe_half = roc_auc_score(yte, Xte[:, 1])
        logger.info(f"[combined] AUROC={comb:.4f} (same half: entropy={ent_half:.4f}, probe={probe_half:.4f})")
        results["combined"] = {"overall": comb, "entropy_same_half": ent_half,
                               "probe_same_half": probe_half}
    except Exception as e:
        logger.warning(f"Combined signal failed: {e}")

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
