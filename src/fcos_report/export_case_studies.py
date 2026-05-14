from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image

from .analysis_common import (
    detections_by_image,
    load_coco_data,
    load_results,
    match_image,
    overlap_stats,
    result_path_for,
    write_csv,
)
from .coco import image_dir
from .config import ensure_output_dirs, load_config
from .visualization import PALETTE, draw_detections, save_side_by_side


def _to_drawable(detections: list[dict], categories: dict[int, str], top_k: int) -> list[dict]:
    drawable = []
    for det in detections[:top_k]:
        x, y, w, h = det["bbox"]
        cat_id = int(det["category_id"])
        drawable.append({
            "box": [x, y, x + w, y + h],
            "score": float(det["score"]),
            "label": categories.get(cat_id, str(cat_id)),
            "label_id": cat_id,
        })
    return drawable


def score_image(gt_anns: list[dict], detections: list[dict], iou_threshold: float) -> dict[str, float]:
    matches, fps, fns = match_image(gt_anns, detections, iou_threshold)
    return {
        "matched": float(len(matches)),
        "false_positives": float(len(fps)),
        "false_negatives": float(len(fns)),
        "recall": len(matches) / len(gt_anns) if gt_anns else 0.0,
        "mean_iou": sum(float(item["iou"]) for item in matches) / len(matches) if matches else 0.0,
    }


def export_case_studies(
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    fcos_json: str | Path | None = None,
    retinanet_json: str | Path | None = None,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.35,
    cases_per_type: int = 8,
    top_k_boxes: int = 30,
) -> pd.DataFrame:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    root = Path(config["data"]["root"])
    coco = load_coco_data(root, split)
    fcos_by_image = detections_by_image(load_results(fcos_json or result_path_for("fcos", split, config_path)), score_threshold)
    retina_by_image = detections_by_image(load_results(retinanet_json or result_path_for("retinanet", split, config_path)), score_threshold)
    case_root = paths["report_assets"] / "case_studies"
    case_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_id, gt_anns in coco.annotations_by_image.items():
        fcos_stats = score_image(gt_anns, fcos_by_image.get(image_id, []), iou_threshold)
        retina_stats = score_image(gt_anns, retina_by_image.get(image_id, []), iou_threshold)
        overlaps = overlap_stats(gt_anns)
        delta = fcos_stats["recall"] - retina_stats["recall"]
        if delta >= 0.25:
            case_type = "fcos_better"
        elif delta <= -0.25:
            case_type = "retinanet_better"
        elif fcos_stats["recall"] >= 0.8 and retina_stats["recall"] >= 0.8:
            case_type = "both_good"
        elif fcos_stats["recall"] <= 0.3 and retina_stats["recall"] <= 0.3:
            case_type = "both_hard"
        elif overlaps["overlap_pairs_iou_0_3"] > 0:
            case_type = "crowded_overlap"
        else:
            continue
        rows.append({
            "image_id": image_id,
            "file_name": coco.images[image_id]["file_name"],
            "case_type": case_type,
            "gt_count": len(gt_anns),
            "fcos_recall": fcos_stats["recall"],
            "retinanet_recall": retina_stats["recall"],
            "recall_delta_fcos_minus_retinanet": delta,
            "fcos_mean_iou": fcos_stats["mean_iou"],
            "retinanet_mean_iou": retina_stats["mean_iou"],
            "overlap_pairs_iou_0_3": overlaps["overlap_pairs_iou_0_3"],
            "max_pair_iou": overlaps["max_pair_iou"],
        })
    index = pd.DataFrame(rows)
    if index.empty:
        return index
    selected = []
    sort_cols = {
        "fcos_better": "recall_delta_fcos_minus_retinanet",
        "retinanet_better": "recall_delta_fcos_minus_retinanet",
        "both_good": "fcos_recall",
        "both_hard": "gt_count",
        "crowded_overlap": "overlap_pairs_iou_0_3",
    }
    ascending = {"retinanet_better": True}
    for case_type, group in index.groupby("case_type"):
        chosen = group.sort_values(sort_cols[case_type], ascending=ascending.get(case_type, False)).head(cases_per_type)
        selected.append(chosen)
    selected_df = pd.concat(selected, ignore_index=True)
    for _, row in selected_df.iterrows():
        image_path = image_dir(root, split) / row["file_name"]
        image = Image.open(image_path).convert("RGB")
        image_id = int(row["image_id"])
        left = draw_detections(image, _to_drawable(retina_by_image.get(image_id, []), coco.categories, top_k_boxes), PALETTE["retinanet"])
        right = draw_detections(image, _to_drawable(fcos_by_image.get(image_id, []), coco.categories, top_k_boxes), PALETTE["fcos"])
        out_dir = case_root / row["case_type"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{image_id}_retinanet_vs_fcos.jpg"
        save_side_by_side(left, right, ("RetinaNet", "FCOS"), out_path)
        metadata_path = out_dir / f"{image_id}.json"
        metadata_path.write_text(json.dumps(row.to_dict(), indent=2, default=_json_default), encoding="utf-8")
    return selected_df


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export qualitative FCOS vs RetinaNet case studies.")
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--fcos-json", default=None)
    parser.add_argument("--retinanet-json", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--score", type=float, default=0.35)
    parser.add_argument("--cases-per-type", type=int, default=8)
    parser.add_argument("--top-k-boxes", type=int, default=30)
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    df = export_case_studies(
        args.split,
        args.config,
        args.fcos_json,
        args.retinanet_json,
        args.iou,
        args.score,
        args.cases_per_type,
        args.top_k_boxes,
    )
    out = write_csv(df, paths["tables"] / f"case_study_index_{args.split}.csv")
    print(out)


if __name__ == "__main__":
    main()
