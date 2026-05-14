from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis_common import (
    detections_by_image,
    load_coco_data,
    load_results,
    match_image,
    overlap_stats,
    result_path_for,
    write_csv,
)
from .config import ensure_output_dirs, load_config


def analyze_crowded_scenes(
    model: str,
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    result_path: str | Path | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    min_gt: int = 6,
    min_overlap_pairs: int = 1,
) -> pd.DataFrame:
    config = load_config(config_path)
    coco = load_coco_data(config["data"]["root"], split)
    results = load_results(result_path or result_path_for(model, split, config_path))
    by_image = detections_by_image(results, score_threshold)
    rows = []
    for image_id, gt_anns in coco.annotations_by_image.items():
        stats = overlap_stats(gt_anns)
        if len(gt_anns) < min_gt and stats["overlap_pairs_iou_0_3"] < min_overlap_pairs:
            continue
        matches, false_positives, false_negatives = match_image(gt_anns, by_image.get(image_id, []), iou_threshold)
        image = coco.images[image_id]
        rows.append({
            "model": model,
            "split": split,
            "image_id": image_id,
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "gt_count": len(gt_anns),
            "detection_count": len(by_image.get(image_id, [])),
            "matched": len(matches),
            "false_positives": len(false_positives),
            "false_negatives": len(false_negatives),
            "recall_at_iou": len(matches) / len(gt_anns) if gt_anns else 0.0,
            **stats,
        })
    return pd.DataFrame(rows).sort_values(["overlap_pairs_iou_0_3", "gt_count", "false_negatives"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze crowded and overlapped COCO scenes.")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score", type=float, default=0.05)
    parser.add_argument("--min-gt", type=int, default=6)
    parser.add_argument("--min-overlap-pairs", type=int, default=1)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    df = analyze_crowded_scenes(
        args.model,
        args.split,
        args.config,
        args.result_json,
        args.iou,
        args.score,
        args.min_gt,
        args.min_overlap_pairs,
    )
    out = write_csv(df, paths["tables"] / f"crowded_scene_analysis_{args.model}_{args.split}.csv")
    print(out)


if __name__ == "__main__":
    main()
