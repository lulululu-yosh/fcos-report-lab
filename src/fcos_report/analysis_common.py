from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .coco import annotation_path
from .config import ensure_output_dirs, load_config


@dataclass(frozen=True)
class CocoData:
    images: dict[int, dict[str, Any]]
    categories: dict[int, str]
    annotations_by_image: dict[int, list[dict[str, Any]]]


def load_coco_data(root: str | Path, split: str = "val2017") -> CocoData:
    data = json.loads(annotation_path(Path(root), split).read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data["images"]}
    categories = {int(item["id"]): item["name"] for item in data.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        annotations_by_image[int(ann["image_id"])].append(ann)
    return CocoData(images=images, categories=categories, annotations_by_image=dict(annotations_by_image))


def load_results(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result_path_for(model: str, split: str, config_path: str | Path = "configs/default.yaml") -> Path:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    candidates = [
        paths["coco_results"] / f"{model}_{split}_results.json",
        paths["coco_results"] / f"{model}_testdev.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No result JSON found for {model} {split}. Run fcos_report.evaluate first.")


def detections_by_image(results: list[dict[str, Any]], score_threshold: float = 0.0) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for det in results:
        if float(det["score"]) >= score_threshold:
            by_image[int(det["image_id"])].append(det)
    for image_dets in by_image.values():
        image_dets.sort(key=lambda item: float(item["score"]), reverse=True)
    return dict(by_image)


def xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in box]
    return np.array([x, y, x + max(0.0, w), y + max(0.0, h)], dtype=float)


def box_iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = xywh_to_xyxy(b)
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def area_group(area: float) -> str:
    if area < 32 * 32:
        return "small"
    if area < 96 * 96:
        return "medium"
    return "large"


def infer_fpn_level_from_box(box_xywh: list[float]) -> str:
    _, _, w, h = box_xywh
    size = max(float(w), float(h))
    if size <= 64:
        return "P3"
    if size <= 128:
        return "P4"
    if size <= 256:
        return "P5"
    if size <= 512:
        return "P6"
    return "P7"


def match_image(
    gt_anns: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Greedy same-class matching for analysis, not a replacement for COCO mAP."""
    matched_gt: set[int] = set()
    matches: list[dict[str, Any]] = []
    false_positives: list[dict[str, Any]] = []
    for det in sorted(detections, key=lambda item: float(item["score"]), reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, ann in enumerate(gt_anns):
            if idx in matched_gt:
                continue
            if int(ann["category_id"]) != int(det["category_id"]):
                continue
            iou = box_iou_xywh(ann["bbox"], det["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            ann = gt_anns[best_idx]
            matches.append({"gt": ann, "det": det, "iou": best_iou})
        else:
            false_positives.append(det)
    false_negatives = [ann for idx, ann in enumerate(gt_anns) if idx not in matched_gt]
    return matches, false_positives, false_negatives


def overlap_stats(gt_anns: list[dict[str, Any]]) -> dict[str, float]:
    pairs = 0
    overlap_pairs = 0
    max_iou = 0.0
    iou_sum = 0.0
    for i, left in enumerate(gt_anns):
        for right in gt_anns[i + 1:]:
            pairs += 1
            iou = box_iou_xywh(left["bbox"], right["bbox"])
            iou_sum += iou
            if iou > max_iou:
                max_iou = iou
            if iou >= 0.3:
                overlap_pairs += 1
    return {
        "gt_pairs": float(pairs),
        "overlap_pairs_iou_0_3": float(overlap_pairs),
        "mean_pair_iou": iou_sum / pairs if pairs else 0.0,
        "max_pair_iou": max_iou,
    }


def write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def summarize_detection_counts(results: list[dict[str, Any]], thresholds: list[float]) -> pd.DataFrame:
    rows = []
    all_image_ids = sorted({int(item["image_id"]) for item in results})
    for threshold in thresholds:
        by_image = detections_by_image(results, threshold)
        counts = [len(by_image.get(image_id, [])) for image_id in all_image_ids]
        scores = [float(item["score"]) for item in results if float(item["score"]) >= threshold]
        rows.append({
            "score_threshold": threshold,
            "images": len(all_image_ids),
            "detections": int(sum(counts)),
            "mean_detections_per_image": float(np.mean(counts)) if counts else 0.0,
            "median_detections_per_image": float(np.median(counts)) if counts else 0.0,
            "mean_score": float(np.mean(scores)) if scores else 0.0,
        })
    return pd.DataFrame(rows)
