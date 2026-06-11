#!/bin/bash
#SBATCH --job-name=l31-pipe
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --output=logs/l31_pipe_%j.out
#SBATCH --error=logs/l31_pipe_%j.err

# LLaMA-3.1-8B-Instruct probe pipeline (model generalization, P0):
# collect depth-10 labels -> merge -> train linear + mlp probes -> AUROC
# vs entropy. The final table runs afterwards via final_table.sh with
# TARGET/DRAFTER/PROBE_BASE/OUT_BASE/IS_LLAMA3 env overrides.
#
# Prompt format consistency: collection and eval both go through the
# tokenizer chat template with NO system prompt (the v2 root cause).

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fgsd
cd /data/user/mzhang630/fgsd

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TARGET=models/target/Llama-3.1-8B-Instruct
DRAFTER=models/eagle3_drafters/EAGLE3-LLaMA3.1-Instruct-8B
DATA_BASE=data/hidden_states/llama31_8b_d10
PROBE_BASE=results/probes/llama31_8b_d10

echo "=== LLaMA-3.1 pipeline on $(nvidia-smi --query-gpu=name --format=csv,noheader) — $(date) ==="
mkdir -p logs

echo "=== STEP 1: collect at depth 10 ==="
declare -A SAMPLES
SAMPLES[mt_bench]=80
SAMPLES[humaneval]=164
SAMPLES[gsm8k]=500
for BENCH in mt_bench humaneval gsm8k; do
    OUTDIR="${DATA_BASE}/${BENCH}/t0.0"
    [ -f "${OUTDIR}/manifest.json" ] && { echo "skip ${BENCH}"; continue; }
    python -u scripts/collect_data.py \
        --base-model-path "${TARGET}" --ea-model-path "${DRAFTER}" \
        --use-eagle3 --depth 10 --is-llama3 \
        --benchmark "${BENCH}" --data-dir data/benchmarks \
        --temperature 0.0 --max-new-tokens 512 --max-length 2048 \
        --collect-draft-hidden \
        --output-dir "${OUTDIR}" --chunk-size 100 \
        --dtype float16 --max-samples ${SAMPLES[$BENCH]}
done

echo "=== STEP 2: merge ==="
python -u scripts/merge_shards.py \
    --model-dir "${DATA_BASE}" \
    --output-dir "${DATA_BASE}/merged" \
    --benchmarks mt_bench humaneval gsm8k

echo "=== STEP 3: train linear probe ==="
python -u scripts/train_probe.py \
    --data-dir "${DATA_BASE}/merged/all" \
    --probe-type linear --input-source draft_hidden \
    --batch-size 512 --num-epochs 30 --learning-rate 1e-3 \
    --early-stopping-patience 7 \
    --output-dir "${PROBE_BASE}/linear_draft_hidden"

echo "=== STEP 4: train mlp+position probe ==="
python -u scripts/train_probe.py \
    --data-dir "${DATA_BASE}/merged/all" \
    --probe-type mlp --input-source draft_hidden \
    --hidden-dim 256 --num-layers 2 --dropout 0.1 \
    --batch-size 512 --num-epochs 30 --learning-rate 1e-3 \
    --early-stopping-patience 7 --use-position \
    --output-dir "${PROBE_BASE}/mlp_draft_hidden"

echo "=== STEP 5: probe vs entropy AUROC ==="
for P in linear_draft_hidden mlp_draft_hidden; do
    python -u scripts/compare_signals.py \
        --data-dir "${DATA_BASE}/merged/all" \
        --ea-model-path ${DRAFTER} \
        --probe-dir "${PROBE_BASE}/${P}" \
        --output results/analysis/signal_comparison_llama31_${P}.json
done

echo "ALL DONE: $(date)"
echo "Next: submit final_table.sh with the llama31 env overrides (see final_table.sh header)"
