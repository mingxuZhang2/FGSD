#!/bin/bash
#SBATCH --job-name=final-tab
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/final_%x_%j.out
#SBATCH --error=logs/final_%x_%j.err

# Paper main-table head-to-head, single benchmark per job:
#   sbatch --job-name=ft-mt scripts/slurm/final_table.sh mt_bench 80
#   sbatch --job-name=ft-he scripts/slurm/final_table.sh humaneval 164
#   sbatch --job-name=ft-gsm scripts/slurm/final_table.sh gsm8k 200
#
# ALL methods run inside this one job on one GPU, so throughput ratios are
# unaffected by node sharing (earlier cross-job numbers were skewed when two
# eval jobs landed on the same node). Configs:
#   vanilla / eagle3 d5 / eagle3 d8 / eagle3 d10 (static)
#   svip   s3.0_e0.7_d10              (entropy signal, best fixed config)
#   probe  linear ext0.5_d10, linear ext0.6_d10, mlp ext0.5_d10
# Probe runs are extension-only (rejection-threshold 1.5 disables early exit
# and skips in-base-depth probe calls).

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fgsd
cd /data/user/mzhang630/fgsd

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BENCH=$1
MAX=$2

# Model pair is overridable via environment for multi-model runs, e.g.:
#   sbatch --export=ALL,TARGET=models/target/Llama-3.1-8B-Instruct,\
#     DRAFTER=models/eagle3_drafters/EAGLE3-LLaMA3.1-Instruct-8B,\
#     PROBE_BASE=results/probes/llama31_8b_d10,\
#     OUT_BASE=results/eval/llama31_8b_final,IS_LLAMA3=1 \
#     --job-name=ft-l31-mt scripts/slurm/final_table.sh mt_bench 80
TARGET=${TARGET:-models/target/DeepSeek-R1-Distill-Llama-8B}
DRAFTER=${DRAFTER:-models/eagle3_drafters/EAGLE3-DeepSeek-R1-8B}
PROBE_BASE=${PROBE_BASE:-results/probes/deepseek_r1_8b_d10}
OUT_BASE=${OUT_BASE:-results/eval/deepseek_r1_8b_final}
TEMP=${TEMP:-0.0}
LINEAR_PROBE=${PROBE_BASE}/linear_draft_hidden
MLP_PROBE=${PROBE_BASE}/mlp_draft_hidden
EXTRA_ARGS=""
[ "${IS_LLAMA3:-0}" = "1" ] && EXTRA_ARGS="--is-llama3"

echo "=== Final table [${BENCH}, n=${MAX}] on $(nvidia-smi --query-gpu=name --format=csv,noheader) node $(hostname) — $(date) ==="
mkdir -p logs

run_eval () {
    local OUTDIR=$1; shift
    if [ -f "${OUTDIR}/results.json" ]; then
        echo "Skipping ${OUTDIR} (done)"
        return 0
    fi
    python -u scripts/run_eval.py \
        --base-model-path "${TARGET}" \
        --ea-model-path "${DRAFTER}" \
        --use-eagle3 \
        --temperatures ${TEMP} \
        --max-new-tokens 512 \
        --data-dir data/benchmarks \
        --dtype float16 \
        --n-warmup 3 \
        --benchmarks "${BENCH}" \
        --max-samples "${MAX}" \
        --output-dir "${OUTDIR}" \
        ${EXTRA_ARGS} \
        "$@"
}

# Main run: one model load, four methods sequentially on the same GPU.
# svip and the probe both use the bidirectional loop with d10 extension.
run_eval "${OUT_BASE}/${BENCH}/main" \
    --methods vanilla eagle3 svip fgsd \
    --depth 5 \
    --probe-dir "${LINEAR_PROBE}" \
    --rejection-threshold 1.5 \
    --extend-threshold 0.5 --max-extended-depth 10 \
    --svip-entropy-threshold 3.0 --svip-extend-threshold 0.7

# Probe variants
run_eval "${OUT_BASE}/${BENCH}/probe_linear_e0.6_d10" \
    --methods fgsd --depth 5 \
    --probe-dir "${LINEAR_PROBE}" \
    --rejection-threshold 1.5 \
    --extend-threshold 0.6 --max-extended-depth 10

run_eval "${OUT_BASE}/${BENCH}/probe_mlp_e0.5_d10" \
    --methods fgsd --depth 5 \
    --probe-dir "${MLP_PROBE}" \
    --rejection-threshold 1.5 \
    --extend-threshold 0.5 --max-extended-depth 10

# Static depth ablations (need their own model loads: depth is a load-time arg)
run_eval "${OUT_BASE}/${BENCH}/eagle3_d8"  --methods eagle3 --depth 8
run_eval "${OUT_BASE}/${BENCH}/eagle3_d10" --methods eagle3 --depth 10

echo "=== ${BENCH} complete: $(date) ==="
python3 - "${OUT_BASE}" "${BENCH}" <<'EOF'
import json, glob, sys
out_base, bench = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(f'{out_base}/{bench}/*/results.json')):
    sub = f.split('/')[-2]
    r = json.load(open(f))
    for k, v in r.items():
        if 'error' in v:
            print(f'{bench:<10s} {sub:<22s} {k}: ERROR {v["error"]}')
            continue
        cs = v.get('controller_stats', {})
        d = cs.get('avg_draft_length')
        d = f'{d:.2f}' if isinstance(d, float) else '-'
        sp = v.get('speedup_mean')
        sp = f'{sp:.2f}x' if isinstance(sp, float) else '-'
        print(f'{bench:<10s} {sub:<22s} {k:<32s} tps={v["tokens_per_second_mean"]:>7.1f} '
              f'tau={v["tau_mean"]:>5.2f} depth={d} speedup={sp}')
EOF
