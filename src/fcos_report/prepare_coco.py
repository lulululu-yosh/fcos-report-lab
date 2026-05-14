from __future__ import annotations

import argparse
from pathlib import Path

from .coco import prepare_coco
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract COCO images/annotations.")
    parser.add_argument("--split", choices=["val2017", "test2017"], default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    prepare_coco(Path(config["data"]["root"]), args.split)
    print(f"COCO {args.split} is ready under {config['data']['root']}")


if __name__ == "__main__":
    main()
