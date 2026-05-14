from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis_common import (
    area_group,
    detections_by_image,
    load_coco_data,
    load_results,
    match_image,
    result_path_for,
    write_csv,
)
from .config import ensure_output_dirs, load_config


def analyze_size_groups(
    model: str,
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    result_path: str | Path | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> pd.DataFrame:
    config = load_config(config_path)
    coco = load_coco_data(config["data"]["root"], split)
    results = load_results(result_path or result_path_for(model, split, config_path))
    by_image = detections_by_image(results, score_threshold)

    rows_by_group = {
        group: {
            "model": model,
            "split": split,
            "size_group": group,
            "gt": 0,
            "matched": 0,
            "false_negatives": 0,
            "false_positives": 0,
            "iou_sum": 0.0,
            "score_sum": 0.0,
        }
        for group in ["small", "medium", "large"]
    }
    for image_id, gt_anns in coco.annotations_by_image.items():
        matches, false_positives, false_negatives = match_image(gt_anns, by_image.get(image_id, []), iou_threshold)
        for ann in gt_anns:
            rows_by_group[area_group(float(ann["area"]))]["gt"] += 1
        for item in matches:
            group = area_group(float(item["gt"]["area"]))
            rows_by_group[group]["matched"] += 1
            rows_by_group[group]["iou_sum"] += float(item["iou"])
            rows_by_group[group]["score_sum"] += float(item["det"]["score"])
        for ann in false_negatives:
            rows_by_group[area_group(float(ann["area"]))]["false_negatives"] += 1
        for det in false_positives:
            group = area_group(float(det["bbox"][2]) * float(det["bbox"][3]))
            rows_by_group[group]["false_positives"] += 1

    rows = []
    for row in rows_by_group.values():
        matched = row["matched"]
        gt = row["gt"]
        fp = row["false_positives"]
        row["recall_at_iou"] = matched / gt if gt else 0.0
        row["precision_at_iou"] = matched / (matched + fp) if matched + fp else 0.0
        row["mean_iou_matched"] = row.pop("iou_sum") / matched if matched else 0.0
        row["mean_score_matched"] = row.pop("score_sum") / matched if matched else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze FCOS/RetinaNet by COCO small/medium/large object groups.")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score", type=float, default=0.05)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    df = analyze_size_groups(args.model, args.split, args.config, args.result_json, args.iou, args.score)
    out = write_csv(df, paths["tables"] / f"size_group_analysis_{args.model}_{args.split}.csv")
    print(out)


if __name__ == "__main__":
    main()
