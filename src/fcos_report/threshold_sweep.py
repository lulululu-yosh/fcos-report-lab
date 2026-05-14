from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis_common import (
    detections_by_image,
    load_coco_data,
    load_results,
    match_image,
    result_path_for,
    summarize_detection_counts,
    write_csv,
)
from .coco import evaluate_coco
from .config import ensure_output_dirs, load_config


def filter_results(results: list[dict], threshold: float) -> list[dict]:
    return [item for item in results if float(item["score"]) >= threshold]


def approximate_pr(
    results: list[dict],
    model: str,
    split: str,
    config_path: str | Path,
    thresholds: list[float],
    iou_threshold: float,
) -> pd.DataFrame:
    config = load_config(config_path)
    coco = load_coco_data(config["data"]["root"], split)
    rows = []
    for threshold in thresholds:
        by_image = detections_by_image(results, threshold)
        matched = false_positives = false_negatives = 0
        iou_sum = score_sum = 0.0
        for image_id, gt_anns in coco.annotations_by_image.items():
            matches, fps, fns = match_image(gt_anns, by_image.get(image_id, []), iou_threshold)
            matched += len(matches)
            false_positives += len(fps)
            false_negatives += len(fns)
            iou_sum += sum(float(item["iou"]) for item in matches)
            score_sum += sum(float(item["det"]["score"]) for item in matches)
        rows.append({
            "model": model,
            "split": split,
            "score_threshold": threshold,
            "iou_threshold": iou_threshold,
            "matched": matched,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision_approx": matched / (matched + false_positives) if matched + false_positives else 0.0,
            "recall_approx": matched / (matched + false_negatives) if matched + false_negatives else 0.0,
            "mean_iou_matched": iou_sum / matched if matched else 0.0,
            "mean_score_matched": score_sum / matched if matched else 0.0,
        })
    return pd.DataFrame(rows)


def threshold_sweep(
    model: str,
    split: str = "val2017",
    config_path: str | Path = "configs/default.yaml",
    result_path: str | Path | None = None,
    thresholds: list[float] | None = None,
    iou_threshold: float = 0.5,
    run_coco_eval: bool = False,
) -> pd.DataFrame:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    result_path = Path(result_path or result_path_for(model, split, config_path))
    results = load_results(result_path)
    thresholds = thresholds or [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]

    counts = summarize_detection_counts(results, thresholds)
    approx = approximate_pr(results, model, split, config_path, thresholds, iou_threshold)
    df = counts.merge(approx, on="score_threshold", how="left")
    if run_coco_eval and split == "val2017":
        metric_rows = []
        for threshold in thresholds:
            filtered_path = paths["coco_results"] / f"{model}_{split}_score_{threshold:.2f}.json"
            filtered_path.write_text(json.dumps(filter_results(results, threshold)), encoding="utf-8")
            metrics = evaluate_coco(filtered_path, split, config_path)
            metrics["score_threshold"] = threshold
            metric_rows.append(metrics)
        df = df.merge(pd.DataFrame(metric_rows), on="score_threshold", how="left")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze score threshold behavior from an existing COCO result JSON.")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--thresholds", default="0.05,0.1,0.2,0.3,0.5,0.7")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--coco-eval", action="store_true", help="Run full COCOeval for each filtered threshold.")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    df = threshold_sweep(args.model, args.split, args.config, args.result_json, thresholds, args.iou, args.coco_eval)
    out = write_csv(df, paths["tables"] / f"threshold_sweep_{args.model}_{args.split}.csv")
    print(out)


if __name__ == "__main__":
    main()
