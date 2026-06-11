"""Merge shard directories into a single flat directory for probe training."""
import os
import sys
import json
import shutil
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def merge_shards(base_dir: str, output_dir: str):
    """Merge all shard_* subdirs and the root dir into output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    all_chunks = []
    chunk_counter = 0
    total_samples = 0

    # Collect from root (non-sharded benchmarks like mt_bench, gsm8k)
    dirs_to_scan = []
    if os.path.exists(os.path.join(base_dir, "manifest.json")):
        dirs_to_scan.append(base_dir)

    # Collect from shard subdirs
    for d in sorted(os.listdir(base_dir)):
        shard_path = os.path.join(base_dir, d)
        if os.path.isdir(shard_path) and d.startswith("shard_"):
            if os.path.exists(os.path.join(shard_path, "manifest.json")):
                dirs_to_scan.append(shard_path)

    for src_dir in dirs_to_scan:
        manifest = json.load(open(os.path.join(src_dir, "manifest.json")))
        for chunk_info in manifest["chunks"]:
            src_file = os.path.join(src_dir, chunk_info["path"])
            dst_name = f"chunk_{chunk_counter:04d}.pt"
            dst_file = os.path.join(output_dir, dst_name)
            shutil.copy2(src_file, dst_file)

            all_chunks.append({
                "chunk_id": chunk_counter,
                "path": dst_name,
                "num_samples": chunk_info.get("num_samples", 0),
                "source": os.path.basename(src_dir),
            })
            total_samples += chunk_info.get("num_samples", 0)
            chunk_counter += 1

    manifest_out = {
        "chunks": all_chunks,
        "total_samples": total_samples,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest_out, f, indent=2)

    logger.info(f"Merged {chunk_counter} chunks ({total_samples} samples) -> {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Base dir, e.g. data/hidden_states/deepseek_r1_8b")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output merged dir, e.g. data/hidden_states/deepseek_r1_8b/merged")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["mt_bench", "gsm8k", "humaneval", "alpaca", "cnn_dm"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    grand_total = 0

    for bench in args.benchmarks:
        bench_dir = os.path.join(args.model_dir, bench, "t0.0")
        if not os.path.exists(bench_dir):
            logger.warning(f"Skipping {bench}: {bench_dir} not found")
            continue
        bench_out = os.path.join(args.output_dir, bench)
        merge_shards(bench_dir, bench_out)

    # Also create an all-benchmarks merged dir
    all_out = os.path.join(args.output_dir, "all")
    os.makedirs(all_out, exist_ok=True)
    all_chunks = []
    chunk_counter = 0
    total = 0

    for bench in args.benchmarks:
        bench_merged = os.path.join(args.output_dir, bench)
        if not os.path.exists(os.path.join(bench_merged, "manifest.json")):
            continue
        manifest = json.load(open(os.path.join(bench_merged, "manifest.json")))
        for chunk_info in manifest["chunks"]:
            src = os.path.join(bench_merged, chunk_info["path"])
            dst_name = f"chunk_{chunk_counter:04d}.pt"
            # Symlink instead of copy to save space
            os.symlink(os.path.abspath(src), os.path.join(all_out, dst_name))
            all_chunks.append({
                "chunk_id": chunk_counter,
                "path": dst_name,
                "num_samples": chunk_info.get("num_samples", 0),
                "source": bench,
            })
            total += chunk_info.get("num_samples", 0)
            chunk_counter += 1

    with open(os.path.join(all_out, "manifest.json"), "w") as f:
        json.dump({"chunks": all_chunks, "total_samples": total}, indent=2, fp=f)

    logger.info(f"All-benchmarks merged: {total} samples in {chunk_counter} chunks")


if __name__ == "__main__":
    main()
