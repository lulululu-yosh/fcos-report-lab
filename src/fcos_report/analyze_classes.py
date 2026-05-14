from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_common import detections_by_image, load_coco_data, load_results, match_image, result_path_for, write_csv
from .coco import annotation_path
from .config import ensure_output_dirs, load_config


def class_ap_from_cocoeval(result_path: str | Path, split: str, config_path: str | Path) -> dict[int, float]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    config = load_config(config_path)
    coco_gt = COCO(str(annotation_path(Path(config["data"]["root"]), split)))
    coco_dt = coco_gt.loadRes(str(result_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    precision = evaluator.eval["precision"]  # T x R x K x A x M
    cat_ids = evaluator.params.catIds
    ap_by_cat = {}
    for idx, cat_id in enumerate(cat_ids):
        values = precision[:, :, idx, 0, -1]
        values = values[values > -1]
        ap_by_cat[int(cat_id)] = float(np.mean(values)) if values.size else float("nan")
    return ap_by_cat


def analyze_classes(
    model: str,
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    result_path: str | Path | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
    include_ap: bool = True,
) -> pd.DataFrame:
    config = load_config(config_path)
    result_path = Path(result_path or result_path_for(model, split, config_path))
    coco = load_coco_data(config["data"]["root"], split)
    results = load_results(result_path)
    by_image = detections_by_image(results, score_threshold)
    rows = {
        cat_id: {
            "model": model,
            "split": split,
            "category_id": cat_id,
            "category": name,
            "gt": 0,
            "detections": 0,
            "matched": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "score_sum": 0.0,
            "iou_sum": 0.0,
        }
        for cat_id, name in coco.categories.items()
    }
    for det in results:
        if float(det["score"]) >= score_threshold and int(det["category_id"]) in rows:
            row = rows[int(det["category_id"])]
            row["detections"] += 1
            row["score_sum"] += float(det["score"])
    for image_id, gt_anns in coco.annotations_by_image.items():
        for ann in gt_anns:
            rows[int(ann["category_id"])]["gt"] += 1
        matches, false_positives, false_negatives = match_image(gt_anns, by_image.get(image_id, []), iou_threshold)
        for item in matches:
            row = rows[int(item["gt"]["category_id"])]
            row["matched"] += 1
            row["iou_sum"] += float(item["iou"])
        for det in false_positives:
            if int(det["category_id"]) in rows:
                rows[int(det["category_id"])]["false_positives"] += 1
        for ann in false_negatives:
            rows[int(ann["category_id"])]["false_negatives"] += 1
    ap_by_cat = class_ap_from_cocoeval(result_path, split, config_path) if include_ap else {}
    output = []
    for row in rows.values():
        gt = row["gt"]
        matched = row["matched"]
        detections = row["detections"]
        row["recall_at_iou"] = matched / gt if gt else 0.0
        row["precision_at_iou"] = matched / detections if detections else 0.0
        row["mean_score"] = row.pop("score_sum") / detections if detections else 0.0
        row["mean_iou_matched"] = row.pop("iou_sum") / matched if matched else 0.0
        row["ap_coco"] = ap_by_cat.get(row["category_id"], float("nan"))
        output.append(row)
    return pd.DataFrame(output).sort_values(["ap_coco", "recall_at_iou"], ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-class detection behavior.")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score", type=float, default=0.05)
    parser.add_argument("--no-ap", action="store_true", help="Skip pycocotools per-class AP extraction.")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    df = analyze_classes(args.model, args.split, args.config, args.result_json, args.iou, args.score, not args.no_ap)
    out = write_csv(df, paths["tables"] / f"per_class_metrics_{args.model}_{args.split}.csv")
    print(out)


if __name__ == "__main__":
    main()
