# FGSD: Feature-Guided Speculative Decoding

Adaptive draft-tree depth control for speculative decoding via lightweight hidden-state probing. Built on [EAGLE-3](https://github.com/SafeAILab/EAGLE).

## Key Idea

In speculative decoding, a fixed draft depth is suboptimal: some steps could accept more tokens (wasted opportunity), while others waste compute on tokens that will be rejected. FGSD uses a **4,097-parameter linear probe** on drafter hidden states to predict token rejection, then **dynamically extends** the draft tree beyond the base depth when continued acceptance is likely.

**Extension-only design**: At batch size 1, early exit is EV-negative (probe cost exceeds draft savings). FGSD operates in extension-only mode (rejection threshold > 1.0), where the probe is only consulted at/beyond the base depth to decide whether to extend further.

## Results

### Main Table: Tokens/s (higher is better)

**DeepSeek-R1-Distill-LLaMA-8B**

| Method | MT-Bench | HumanEval | GSM8K | Alpaca | CNN/DM |
|--------|----------|-----------|-------|--------|--------|
| **T=0** | | | | | |
| Vanilla | 45.0 | 46.2 | 42.4 | 46.7 | 48.2 |
| EAGLE-3 (d=5) | 168.9 | 192.2 | 180.5 | 171.6 | 162.9 |
| SVIP | 173.3 | 195.9 | 189.2 | 181.0 | 161.2 |
| **FGSD (ours)** | **179.4** | **202.7** | **202.2** | **182.7** | **162.8** |
| **T=1** | | | | | |
| Vanilla | 42.0 | 49.2 | 50.9 | 45.4 | 45.4 |
| EAGLE-3 (d=5) | 128.4 | 168.2 | 191.2 | 136.5 | 131.0 |
| SVIP | 134.8 | 172.0 | 194.6 | 138.9 | 131.4 |
| **FGSD (ours)** | **134.5** | **170.5** | **202.4** | **138.7** | **131.5** |

**LLaMA-3.1-8B-Instruct**

| Method | MT-Bench | HumanEval | GSM8K | Alpaca | CNN/DM |
|--------|----------|-----------|-------|--------|--------|
| **T=0** | | | | | |
| Vanilla | 48.9 | 43.3 | 44.3 | 48.5 | 48.1 |
| EAGLE-3 (d=5) | 187.1 | 170.4 | 167.7 | 173.7 | 157.1 |
| SVIP | 184.5 | 175.0 | 172.2 | 190.1 | 159.8 |
| **FGSD (ours)** | **186.7** | **186.9** | **178.2** | **205.1** | **161.6** |
| **T=1** | | | | | |
| Vanilla | 45.3 | 45.1 | 44.0 | 43.9 | 51.4 |
| EAGLE-3 (d=5) | 136.1 | 162.0 | 140.0 | 135.7 | 136.1 |
| SVIP | 133.6 | 165.2 | 143.4 | 145.4 | 142.3 |
| **FGSD (ours)** | **136.3** | **169.8** | **145.6** | **151.5** | **140.9** |

### Signal Quality (AUROC for rejection prediction)

| Model | Probe (Linear) | Probe (MLP) | Entropy |
|-------|---------------|-------------|---------|
| DeepSeek-R1-8B | 0.858 | 0.876 | 0.630 |
| LLaMA-3.1-8B | 0.882 | 0.911 | 0.690 |

### Probe Overhead

Probe inference adds only **1.1--1.7%** to per-step wall time.

### Threshold Sensitivity

Throughput is stable across extend_threshold 0.2--0.6 (< 4% variation), requiring no per-task tuning.

## Method

### Pipeline

```
1. COLLECT:  Run EAGLE-3 at depth 10, capture drafter hidden states + accept/reject labels
2. TRAIN:    Train a linear probe (4,097 params) on collected data
3. EVAL:     Run FGSD with probe-gated tree extension (extension-only mode)
```

### Configuration (fixed across all models and tasks)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Base depth | 5 | EAGLE-3 default tree depth |
| Probe type | Linear | Single linear layer on draft hidden states |
| Rejection threshold | 1.5 | > 1.0 disables early exit (extension-only) |
| Extend threshold | 0.5 | Extend if mean P(reject) <= 0.5 |
| Max extended depth | 10 | Maximum tree depth after extension |

## Project Structure

```
fgsd/
  collector/        # Hidden state + accept/reject data collection
    eagle_hooks.py  # Hooks into EAGLE-3 draft loop
    dataset.py      # Data saving/loading
  probe/            # Rejection prediction probes
    models.py       # LinearProbe, MLPProbe, PositionAwareProbe
    train.py        # Training with class-weighted BCE + early stopping
    evaluate.py     # AUROC, F1, calibration metrics
  controller/       # Adaptive draft control
    adaptive.py     # Probe-based draft controller
    adaptive_draft.py  # Patched EAGLE-3 topK_genrate with probe-gated depth
    baselines.py    # SVIP entropy controller, fixed-length, oracle
  eval/             # End-to-end evaluation
    benchmark.py    # BenchmarkRunner (MT-Bench, HumanEval, GSM8K, Alpaca, CNN/DM)
    metrics.py      # StepMetrics, MetricsTracker
scripts/
  collect_data.py   # Phase 1: collect hidden states
  train_probe.py    # Phase 2: train probe
  run_eval.py       # Phase 3: evaluate
  merge_shards.py   # Merge collected data shards
  compare_signals.py # Probe vs entropy signal comparison
  analysis/         # Paper analysis scripts
  slurm/            # SLURM job scripts
```

## Setup

```bash
# 1. Clone EAGLE-3
git clone https://github.com/SafeAILab/EAGLE.git

# 2. Install dependencies
pip install torch transformers accelerate safetensors

# 3. Download models (target + EAGLE-3 drafter)
python scripts/download_models.py --all

# 4. Run full pipeline
# Collect data -> Train probe -> Evaluate
sbatch scripts/slurm/llama31_pipeline.sh
# Then run paper table
sbatch --export=ALL,TARGET=models/target/Llama-3.1-8B-Instruct,\
  DRAFTER=models/eagle3_drafters/EAGLE3-LLaMA3.1-Instruct-8B,\
  PROBE_BASE=results/probes/llama31_8b_d10,\
  OUT_BASE=results/eval/llama31_8b_final,IS_LLAMA3=1 \
  scripts/slurm/final_table.sh mt_bench 80
```

## Supported Model Pairs

| Target Model | EAGLE-3 Drafter |
|---|---|
| DeepSeek-R1-Distill-LLaMA-8B | yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B |
| LLaMA-3.1-8B-Instruct | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B |
| Vicuna-13B-v1.3 | yuhuili/EAGLE3-Vicuna1.3-13B |
| Qwen3-8B | AngelSlim/Qwen3-8B_eagle3 (community) |

## Citation

Coming soon.

## Acknowledgements

Built on [EAGLE-3](https://github.com/SafeAILab/EAGLE) by SafeAILab.
