"""Assemble threshold sensitivity data from probe sweep results.

Reads results/eval/<model>_probe_d10/<bench>/<config>/results.json and
the final-table baselines, outputs a single JSON suitable for plotting
tok/s and tau vs extend_threshold for each (probe_type, max_depth, bench).
"""
import json
import glob
import os
import re
import sys


def parse_config(dirname):
    """Parse 'linear_ext0.5_d10' -> (probe_type, extend_threshold, max_depth)."""
    m = re.match(r"(linear|mlp)_ext([\d.]+)_d(\d+)", dirname)
    if not m:
        return None
    return m.group(1), float(m.group(2)), int(m.group(3))


def extract_metrics(results_json, bench):
    """Pull tok/s, tau, avg_draft_length from results.json for the first matching key."""
    with open(results_json) as f:
        data = json.load(f)
    for key, v in data.items():
        if bench in key and "error" not in v:
            return {
                "tps": v["tokens_per_second_mean"],
                "tau": v["tau_mean"],
                "avg_draft_length": v.get("controller_stats", {}).get("avg_draft_length"),
                "avg_probe_time_ms": v.get("avg_probe_time_ms", 0),
            }
    return None


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else "results/eval"

    sweep_dirs = sorted(glob.glob(os.path.join(base, "*_probe_d10")))
    if not sweep_dirs:
        print("No sweep dirs found under", base)
        sys.exit(1)

    all_data = {}

    for sweep_dir in sweep_dirs:
        model_tag = os.path.basename(sweep_dir).replace("_probe_d10", "")
        all_data[model_tag] = {}

        for bench in ["mt_bench", "humaneval", "gsm8k"]:
            bench_dir = os.path.join(sweep_dir, bench)
            if not os.path.isdir(bench_dir):
                continue

            points = []
            for cfg_dir in sorted(os.listdir(bench_dir)):
                parsed = parse_config(cfg_dir)
                if not parsed:
                    continue
                probe_type, ext_thr, max_d = parsed
                rpath = os.path.join(bench_dir, cfg_dir, "results.json")
                if not os.path.exists(rpath):
                    continue
                metrics = extract_metrics(rpath, bench)
                if metrics is None:
                    continue
                points.append({
                    "probe_type": probe_type,
                    "extend_threshold": ext_thr,
                    "max_depth": max_d,
                    **metrics,
                })

            # Add baselines from final table if available
            final_dir = os.path.join(base, model_tag + "_final", bench, "main")
            if os.path.exists(os.path.join(final_dir, "results.json")):
                with open(os.path.join(final_dir, "results.json")) as f:
                    fdata = json.load(f)
                for key, v in fdata.items():
                    if "error" in v:
                        continue
                    for tag in ("eagle3", "svip", "vanilla"):
                        if f"_{tag}_" in key:
                            method = tag
                            break
                    else:
                        continue
                    if method in ("eagle3", "svip", "vanilla"):
                        points.append({
                            "probe_type": "baseline_" + method,
                            "extend_threshold": None,
                            "max_depth": 5 if method == "eagle3" else None,
                            "tps": v["tokens_per_second_mean"],
                            "tau": v["tau_mean"],
                            "avg_draft_length": v.get("controller_stats", {}).get("avg_draft_length"),
                        })

            all_data[model_tag][bench] = points

    out_path = os.path.join(base, "threshold_sensitivity.json")
    with open(out_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved {out_path}")

    for model_tag, benchmarks in all_data.items():
        print(f"\n{'='*60}")
        print(f"  {model_tag}")
        print(f"{'='*60}")
        for bench, points in benchmarks.items():
            print(f"\n  {bench}:")
            print(f"  {'config':<25s} {'tps':>7s} {'tau':>6s} {'depth':>6s}")
            print(f"  {'-'*50}")
            for p in sorted(points, key=lambda x: (x["probe_type"], x.get("extend_threshold") or 0, x.get("max_depth") or 0)):
                cfg = p["probe_type"]
                if p.get("extend_threshold") is not None:
                    cfg += f" e{p['extend_threshold']}"
                if p.get("max_depth") is not None:
                    cfg += f" d{p['max_depth']}"
                d = f"{p['avg_draft_length']:.2f}" if isinstance(p.get("avg_draft_length"), float) else "-"
                print(f"  {cfg:<25s} {p['tps']:>7.1f} {p['tau']:>6.2f} {d:>6s}")


if __name__ == "__main__":
    main()
