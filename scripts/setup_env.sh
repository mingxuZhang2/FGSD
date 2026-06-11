#!/bin/bash
# Environment setup for FGSD on HPC3.
#
# Strategy: Clone the existing 'alphasteer' env (torch 2.6.0, transformers 4.57.6)
# and add missing packages from pre-downloaded wheels.
#
# Run this ONCE on HPC3 before submitting any SLURM jobs:
#   ssh hpc3login
#   cd /data/user/mzhang630/fgsd
#   bash scripts/setup_env.sh

set -e

echo "Setting up FGSD environment on HPC3..."

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh

# Check if fgsd env already exists
if conda env list | grep -q "fgsd"; then
    echo "Environment 'fgsd' already exists. Activating..."
    conda activate fgsd
else
    echo "Cloning 'alphasteer' environment to 'fgsd'..."
    conda create -n fgsd --clone alphasteer -y
    conda activate fgsd
    echo "Clone complete."

    # Install additional packages from pre-downloaded wheels (if available)
    if [ -d "wheels" ]; then
        echo "Installing packages from wheels..."
        pip install --no-index --find-links=./wheels/ \
            shortuuid \
            scikit-learn \
            2>/dev/null || echo "Some wheel installs skipped (already present or not found)"
    fi
fi

echo ""
echo "Environment ready:"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
echo ""

# Create project directory structure
echo "Creating directory structure..."
mkdir -p data/benchmarks data/hidden_states models/target models/eagle3_drafters
mkdir -p results/probes results/eval results/ablations results/analysis
mkdir -p logs

# Create symlinks to existing models on HPC3
echo "Setting up model symlinks..."

# LLaMA-3.1-8B-Instruct
if [ -d "/data/user/mzhang630/models/camenduru/Meta-Llama-3.1-8B-Instruct" ]; then
    ln -sfn /data/user/mzhang630/models/camenduru/Meta-Llama-3.1-8B-Instruct models/target/Meta-Llama-3.1-8B-Instruct
    echo "  Linked: Meta-Llama-3.1-8B-Instruct"
fi

# Vicuna-7B
if [ -d "/data/user/mzhang630/models/lmsys/vicuna-7b-v1.5" ]; then
    ln -sfn /data/user/mzhang630/models/lmsys/vicuna-7b-v1.5 models/target/vicuna-7b-v1.5
    echo "  Linked: vicuna-7b-v1.5"
fi

# Qwen2.5-7B-Instruct
if [ -d "/data/user/mzhang630/models/Qwen/Qwen2.5-7B-Instruct" ]; then
    ln -sfn /data/user/mzhang630/models/Qwen/Qwen2.5-7B-Instruct models/target/Qwen2.5-7B-Instruct
    echo "  Linked: Qwen2.5-7B-Instruct"
fi

# DeepSeek-R1-Distill-Llama-8B
if [ -d "/data/user/mzhang630/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B" ]; then
    ln -sfn /data/user/mzhang630/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B models/target/DeepSeek-R1-Distill-Llama-8B
    echo "  Linked: DeepSeek-R1-Distill-Llama-8B"
fi

echo ""
echo "Setup complete. Next steps:"
echo "  1. Download EAGLE-3 draft model checkpoints (on HPC2, then rsync)"
echo "  2. Download benchmark datasets (on HPC2, then rsync)"
echo "  3. Submit jobs: bash scripts/slurm/run_all.sh"
