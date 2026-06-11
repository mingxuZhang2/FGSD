#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err

# Detailed per-step analysis for paper figures:
#   acceptance length distribution, probe overhead breakdown, oracle bound.
# Runs 40 prompts per benchmark (representative subset).
#
# Usage:
#   sbatch scripts/slurm/analysis.sh
#   sbatch --export=ALL,IS_LLAMA3=1,TARGET=...,DRAFTER=...,PROBE_DIR=...,OUT=...,TAG=llama31 scripts/slurm/analysis.sh

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fgsd
cd /data/user/mzhang630/fgsd

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TARGET=${TARGET:-models/target/DeepSeek-R1-Distill-Llama-8B}
DRAFTER=${DRAFTER:-models/eagle3_drafters/EAGLE3-DeepSeek-R1-8B}
PROBE_DIR=${PROBE_DIR:-results/probes/deepseek_r1_8b_d10/linear_draft_hidden}
TAG=${TAG:-deepseek}
OUT=${OUT:-results/analysis/detailed_${TAG}.json}
EXTRA_ARGS=""
[ "${IS_LLAMA3:-0}" = "1" ] && EXTRA_ARGS="--is-llama3"

echo "=== Detailed Analysis [${TAG}] on $(hostname) — $(date) ==="
mkdir -p logs results/analysis

# Step 1: Detailed per-step analysis (acceptance length, overhead, oracle)
python -u scripts/analysis/run_detailed_analysis.py \
    --base-model-path "${TARGET}" \
    --ea-model-path "${DRAFTER}" \
    --probe-dir "${PROBE_DIR}" \
    --benchmarks mt_bench humaneval gsm8k \
    --max-samples 40 \
    --max-new-tokens 512 \
    --data-dir data/benchmarks \
    --output "${OUT}" \
    ${EXTRA_ARGS}

# Step 2: Threshold sensitivity (no GPU needed, reads existing results)
python -u scripts/analysis/threshold_sensitivity.py results/eval

# Step 3: Probe training cost summary
echo ""
echo "=== Probe Training Cost ==="
for probe_dir in results/probes/*/linear_draft_hidden results/probes/*/mlp_draft_hidden; do
    [ -f "${probe_dir}/config.json" ] || continue
    echo ""
    echo "  ${probe_dir}:"
    python3 -c "
import json, os
d = '${probe_dir}'
cfg = json.load(open(os.path.join(d, 'config.json')))
print(f\"    type: {cfg['probe_type']}\")
print(f\"    input_dim: {cfg['input_dim']}\")
print(f\"    hidden_dim: {cfg.get('hidden_dim', 'N/A')}\")
n_params = sum(1 for _ in open(os.path.join(d, 'config.json')))  # placeholder
# Count actual parameters from checkpoint
import torch
ckpt = torch.load(os.path.join(d, 'best.pt'), map_location='cpu', weights_only=False)
n_params = sum(p.numel() for p in ckpt['model_state_dict'].values())
metrics = ckpt.get('metrics', {})
print(f\"    parameters: {n_params:,}\")
print(f\"    val_auroc: {metrics.get('auroc', 'N/A')}\")
print(f\"    val_f1: {metrics.get('f1', 'N/A')}\")
print(f\"    epoch: {ckpt.get('epoch', 'N/A')}\")
"
done

echo ""
echo "=== Analysis complete: $(date) ==="
