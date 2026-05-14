from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .analysis_common import write_csv
from .coco import image_dir, iter_coco_images
from .config import ensure_output_dirs, load_config
from .models import load_model, preprocess_image


@torch.inference_mode()
def benchmark_runtime(
    model: str,
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    limit: int | None = None,
    warmup: int = 5,
) -> pd.DataFrame:
    config = load_config(config_path)
    root = Path(config["data"]["root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, spec = load_model(model, device=device, score_threshold=config["models"][model]["score_threshold"])
    items = list(iter_coco_images(root, split, limit=limit))
    for item in items[:warmup]:
        image = Image.open(image_dir(root, split) / item["file_name"]).convert("RGB")
        _ = net([preprocess_image(image, device)])
    if device.type == "cuda":
        torch.cuda.synchronize()

    rows = []
    for item in tqdm(items, desc=f"benchmark {spec.key}"):
        image = Image.open(image_dir(root, split) / item["file_name"]).convert("RGB")
        tensor = preprocess_image(image, device)
        start = time.perf_counter()
        output = net([tensor])[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        scores = output["scores"].detach().cpu()
        rows.append({
            "model": model,
            "split": split,
            "image_id": int(item["id"]),
            "file_name": item["file_name"],
            "device": str(device),
            "seconds": elapsed,
            "fps": 1.0 / elapsed if elapsed else 0.0,
            "detections": int(len(scores)),
            "detections_score_0_5": int((scores >= 0.5).sum().item()),
            "detections_score_0_7": int((scores >= 0.7).sum().item()),
            "detections_score_0_9": int((scores >= 0.9).sum().item()),
            "max_score": float(scores.max().item()) if len(scores) else 0.0,
            "mean_score": float(scores.mean().item()) if len(scores) else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark per-image detector runtime and final output counts.")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    df = benchmark_runtime(args.model, args.split, args.config, args.limit, args.warmup)
    suffix = f"limit{args.limit}" if args.limit else "full"
    out = write_csv(df, paths["tables"] / f"runtime_{args.model}_{args.split}_{suffix}.csv")
    summary = df[["seconds", "fps", "detections", "detections_score_0_5", "detections_score_0_7"]].mean(numeric_only=True)
    print(out)
    print(summary.to_string())


if __name__ == "__main__":
    main()
