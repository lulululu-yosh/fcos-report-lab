from __future__ import annotations

import argparse
import subprocess
import sys


def run(args: list[str]) -> None:
    print("$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all post-inference FCOS report analyses.")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-coco-ap", action="store_true", help="Skip per-class pycocotools AP extraction.")
    parser.add_argument("--skip-cases", action="store_true")
    parser.add_argument("--skip-internals", action="store_true")
    args = parser.parse_args()
    base = [sys.executable, "-m"]
    no_ap = ["--no-ap"] if args.skip_coco_ap else []
    for model in ["fcos", "retinanet"]:
        run([*base, "fcos_report.analyze_sizes", "--model", model, "--split", args.split, "--config", args.config])
        run([*base, "fcos_report.analyze_classes", "--model", model, "--split", args.split, "--config", args.config, *no_ap])
        run([*base, "fcos_report.threshold_sweep", "--model", model, "--split", args.split, "--config", args.config])
        run([*base, "fcos_report.analyze_crowded_scenes", "--model", model, "--split", args.split, "--config", args.config])
    run([*base, "fcos_report.compare_models", "--kind", "class", "--split", args.split, "--config", args.config])
    run([*base, "fcos_report.compare_models", "--kind", "size", "--split", args.split, "--config", args.config])
    if not args.skip_cases:
        run([*base, "fcos_report.export_case_studies", "--split", args.split, "--config", args.config])
    if not args.skip_internals:
        run([*base, "fcos_report.analyze_fcos_internals", "--split", args.split, "--config", args.config])


if __name__ == "__main__":
    main()
