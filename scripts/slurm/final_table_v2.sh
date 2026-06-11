#!/bin/bash
#SBATCH --job-name=ftv2
#SBATCH --account=d_yings_team
#SBATCH --partition=acd_u
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ftv2_%x_%j.out
#SBATCH --error=logs/ftv2_%x_%j.err

# Paper main-table v2: fair comparison + error bars.
#
# Fixes vs v1:
#   M3: --num-runs 3 for error bars (mean ± std)
#   M4: SVIP now extension-only (entropy_threshold=999 skips base-depth checks)
#       with its own threshold sweep
#
# Usage:
#   sbatch --job-name=ftv2-mt scripts/slurm/final_table_v2.sh mt_bench 80
#   sbatch --export=ALL,TARGET=...,DRAFTER=...,... scripts/slurm/final_table_v2.sh mt_bench 80

set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate fgsd
cd /data/user/mzhang630/fgsd

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BENCH=$1
MAX=$2
NRUNS=${NRUNS:-3}

TARGET=${TARGET:-models/target/DeepSeek-R1-Distill-Llama-8B}
DRAFTER=${DRAFTER:-models/eagle3_drafters/EAGLE3-DeepSeek-R1-8B}
PROBE_BASE=${PROBE_BASE:-results/probes/deepseek_r1_8b_d10}
OUT_BASE=${OUT_BASE:-results/eval/deepseek_r1_8b_v2}
TEMP=${TEMP:-0.0}
LINEAR_PROBE=${PROBE_BASE}/linear_draft_hidden
EXTRA_ARGS=""
[ "${IS_LLAMA3:-0}" = "1" ] && EXTRA_ARGS="--is-llama3"

echo "=== Final table v2 [${BENCH}, n=${MAX}, runs=${NRUNS}] on $(nvidia-smi --query-gpu=name --format=csv,noheader) — $(date) ==="
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
        --num-runs "${NRUNS}" \
        --output-dir "${OUTDIR}" \
        ${EXTRA_ARGS} \
        "$@"
}

# Main: vanilla + eagle3 d5 + SVIP (extension-only) + FGSD (extension-only)
# Both SVIP and FGSD are extension-only for fair comparison:
#   FGSD: rejection-threshold 1.5 (>1 disables early exit for sigmoid)
#   SVIP: entropy-threshold 999 (>100 disables early exit for entropy)
run_eval "${OUT_BASE}/${BENCH}/main" \
    --methods vanilla eagle3 svip fgsd \
    --depth 5 \
    --probe-dir "${LINEAR_PROBE}" \
    --rejection-threshold 1.5 \
    --extend-threshold 0.5 --max-extended-depth 10 \
    --svip-entropy-threshold 999.0 --svip-extend-threshold 0.7

# SVIP sweep (extension-only, different extend_thresholds)
for SVT in 0.5 0.9 1.1; do
    run_eval "${OUT_BASE}/${BENCH}/svip_ext${SVT}_d10" \
        --methods svip --depth 5 \
        --svip-entropy-threshold 999.0 --svip-extend-threshold ${SVT} \
        --max-extended-depth 10
done

# Static depth ablations
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
        std = v.get('tokens_per_second_std', 0)
        print(f'{bench:<10s} {sub:<22s} {k:<32s} '
              f'tps={v["tokens_per_second_mean"]:>7.1f}±{std:>4.1f} '
              f'tau={v["tau_mean"]:>5.2f}')
EOF
