from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .analysis_common import infer_fpn_level_from_box, load_results, result_path_for, write_csv
from .coco import image_dir, iter_coco_images
from .config import ensure_output_dirs, load_config
from .models import load_model, preprocess_image


@torch.inference_mode()
def inspect_fcos_head(
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    limit: int = 20,
) -> pd.DataFrame:
    """Inspect raw torchvision FCOS head tensors for a small image subset."""
    config = load_config(config_path)
    root = Path(config["data"]["root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_model("fcos", device=device, score_threshold=config["models"]["fcos"]["score_threshold"])
    rows = []
    for item in tqdm(list(iter_coco_images(root, split, limit=limit)), desc="inspect fcos head"):
        image = Image.open(image_dir(root, split) / item["file_name"]).convert("RGB")
        tensors, _ = model.transform([preprocess_image(image, device)])
        features = model.backbone(tensors.tensors)
        features_list = list(features.values()) if isinstance(features, dict) else list(features)
        outputs = model.head(features_list)
        for output_name, levels in outputs.items():
            for level_idx, tensor in enumerate(levels):
                values = tensor.detach().float()
                rows.append({
                    "image_id": int(item["id"]),
                    "file_name": item["file_name"],
                    "output": output_name,
                    "level": f"P{level_idx + 3}",
                    "shape": "x".join(str(dim) for dim in values.shape),
                    "mean": float(values.mean().item()),
                    "std": float(values.std().item()),
                    "min": float(values.min().item()),
                    "max": float(values.max().item()),
                    "positive_fraction": float((values > 0).float().mean().item()),
                })
    return pd.DataFrame(rows)


def analyze_detection_scale_distribution(
    model: str,
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    result_path: str | Path | None = None,
    score_threshold: float = 0.05,
) -> pd.DataFrame:
    """Approximate FPN-level responsibility from final box size."""
    results = load_results(result_path or result_path_for(model, split, config_path))
    rows = []
    for det in results:
        if float(det["score"]) < score_threshold:
            continue
        rows.append({
            "model": model,
            "split": split,
            "image_id": int(det["image_id"]),
            "category_id": int(det["category_id"]),
            "score": float(det["score"]),
            "bbox_width": float(det["bbox"][2]),
            "bbox_height": float(det["bbox"][3]),
            "area": float(det["bbox"][2]) * float(det["bbox"][3]),
            "approx_fpn_level": infer_fpn_level_from_box(det["bbox"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FCOS internals and approximate FPN detection distribution.")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--score", type=float, default=0.05)
    parser.add_argument("--skip-head", action="store_true", help="Only analyze final detection box scale distribution.")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)

    scale_df = analyze_detection_scale_distribution("fcos", args.split, args.config, None, args.score)
    scale_out = write_csv(scale_df, paths["tables"] / f"fcos_detection_scale_distribution_{args.split}.csv")
    print(scale_out)
    if not args.skip_head:
        head_df = inspect_fcos_head(args.split, args.config, args.limit)
        head_out = write_csv(head_df, paths["tables"] / f"fcos_head_tensor_stats_{args.split}.csv")
        print(head_out)


if __name__ == "__main__":
    main()
