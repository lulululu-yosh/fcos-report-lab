from __future__ import annotations

import argparse
from pathlib import Path

from .coco import prepare_coco, run_inference
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export COCO test-dev style detection JSON.")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--prepare", action="store_true", help="Download test2017 and image info before export.")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-export only the first N test images.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.prepare:
        prepare_coco(Path(config["data"]["root"]), "test2017")
    result_path, summary = run_inference(args.model, "test2017", args.config, limit=args.limit)
    final_path = result_path.with_name(f"{args.model}_testdev.json")
    result_path.replace(final_path)
    print(f"Wrote {final_path}")
    print(summary)


if __name__ == "__main__":
    main()
