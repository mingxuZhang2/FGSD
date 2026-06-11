"""
Download EAGLE-3 draft model checkpoints and benchmark datasets.

Run this on HPC2 (which has internet), then rsync to HPC3.

Usage:
    # Download everything
    python scripts/download_models.py --all

    # Download only EAGLE-3 drafters
    python scripts/download_models.py --drafters

    # Download only benchmarks
    python scripts/download_models.py --benchmarks
"""

import argparse
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def download_eagle3_drafters(output_dir: str):
    """Download EAGLE-3 draft model checkpoints from HuggingFace."""
    from huggingface_hub import snapshot_download

    drafters = {
        "EAGLE3-LLaMA3.1-Instruct-8B": "yuhuili/EAGLE3-LLaMA3.1-Instruct-8B",
        "EAGLE3-Vicuna-7B": "yuhuili/EAGLE3-Vicuna1.3-13B",  # Note: may need 7B variant
        "EAGLE3-DeepSeek-R1-8B": "yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B",
        # Qwen drafters may be from SpecForge community
    }

    for name, repo_id in drafters.items():
        local_dir = os.path.join(output_dir, name)
        if os.path.exists(local_dir) and len(os.listdir(local_dir)) > 1:
            logger.info(f"  {name}: already downloaded, skipping")
            continue

        logger.info(f"  Downloading {name} from {repo_id}...")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
            )
            logger.info(f"  {name}: download complete")
        except Exception as e:
            logger.warning(f"  {name}: failed to download: {e}")


def download_benchmarks(output_dir: str):
    """Download benchmark datasets."""
    from datasets import load_dataset

    benchmarks = {
        "humaneval": ("openai_humaneval", None, "test"),
        "gsm8k": ("gsm8k", "main", "test"),
        "alpaca": ("tatsu-lab/alpaca", None, "train"),
        "cnn_dm": ("cnn_dailymail", "3.0.0", "test"),
    }

    for name, (dataset_name, config, split) in benchmarks.items():
        local_dir = os.path.join(output_dir, name)
        if os.path.exists(local_dir):
            logger.info(f"  {name}: already downloaded, skipping")
            continue

        logger.info(f"  Downloading {name}...")
        try:
            if config:
                ds = load_dataset(dataset_name, config, split=split)
            else:
                ds = load_dataset(dataset_name, split=split)
            ds.save_to_disk(local_dir)
            logger.info(f"  {name}: {len(ds)} samples saved to {local_dir}")
        except Exception as e:
            logger.warning(f"  {name}: failed to download: {e}")

    # MT-Bench: copy from EAGLE repo if available
    mt_bench_dir = os.path.join(output_dir, "mt_bench")
    eagle_mt_bench = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "EAGLE", "eagle", "data", "mt_bench"
    )
    if not os.path.exists(mt_bench_dir) and os.path.exists(eagle_mt_bench):
        import shutil
        shutil.copytree(eagle_mt_bench, mt_bench_dir)
        logger.info(f"  mt_bench: copied from EAGLE repo")
    elif not os.path.exists(mt_bench_dir):
        logger.warning("  mt_bench: not found in EAGLE repo. Download manually.")


def main():
    parser = argparse.ArgumentParser(description="Download models and benchmarks")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument("--drafters", action="store_true", help="Download EAGLE-3 drafters")
    parser.add_argument("--benchmarks", action="store_true", help="Download benchmarks")
    parser.add_argument("--drafter-dir", type=str,
                        default="implementation/models/eagle3_drafters")
    parser.add_argument("--benchmark-dir", type=str,
                        default="implementation/data/benchmarks")
    args = parser.parse_args()

    if not any([args.all, args.drafters, args.benchmarks]):
        args.all = True

    if args.all or args.drafters:
        logger.info("=== Downloading EAGLE-3 draft model checkpoints ===")
        os.makedirs(args.drafter_dir, exist_ok=True)
        download_eagle3_drafters(args.drafter_dir)

    if args.all or args.benchmarks:
        logger.info("=== Downloading benchmark datasets ===")
        os.makedirs(args.benchmark_dir, exist_ok=True)
        download_benchmarks(args.benchmark_dir)

    logger.info("\nDone! Next step: rsync to HPC3:")
    logger.info(
        "  rsync -avz -e 'ssh -i /hpc2hdd/home/mzhang630/data/id_rsa' "
        "implementation/ mzhang630@hpc3login.hpc.hkust-gz.edu.cn:/data/user/mzhang630/fgsd/"
    )


if __name__ == "__main__":
    main()
