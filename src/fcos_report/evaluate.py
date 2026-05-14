from __future__ import annotations

import argparse
import json
from pathlib import Path

from .coco import evaluate_coco, run_inference
from .config import ensure_output_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run torchvision detector inference and optional COCO evaluation.")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--split", choices=["val2017", "test2017"], default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--full", action="store_true", help="Evaluate all images in the split.")
    parser.add_argument("--limit", type=int, default=None, help="Override number of images for smoke runs.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    limit = None if args.full else (args.limit or int(config["data"]["smoke_limit"]))
    result_path, runtime = run_inference(args.model, args.split, args.config, limit=limit)
    metrics = {"runtime": runtime}
    if args.split == "val2017":
        metrics["coco_bbox"] = evaluate_coco(result_path, args.split, args.config)
    else:
        metrics["note"] = "test2017 has no public bbox annotations; upload JSON to the COCO server for mAP."

    suffix = "full" if args.full else f"limit{limit}"
    out = paths["metrics"] / f"{args.model}_{args.split}_{suffix}.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
