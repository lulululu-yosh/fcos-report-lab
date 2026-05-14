from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analysis_common import write_csv
from .config import ensure_output_dirs, load_config


def compare_metric_tables(
    fcos_csv: str | Path,
    retinanet_csv: str | Path,
    key_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    fcos = pd.read_csv(fcos_csv)
    retina = pd.read_csv(retinanet_csv)
    merged = fcos.merge(retina, on=key_cols, suffixes=("_fcos", "_retinanet"))
    for metric in metric_cols:
        left = f"{metric}_fcos"
        right = f"{metric}_retinanet"
        if left in merged.columns and right in merged.columns:
            merged[f"{metric}_delta_fcos_minus_retinanet"] = merged[left] - merged[right]
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FCOS and RetinaNet analysis CSV files.")
    parser.add_argument("--kind", choices=["class", "size"], required=True)
    parser.add_argument("--split", default="val2017")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = ensure_output_dirs(config)
    if args.kind == "class":
        df = compare_metric_tables(
            paths["tables"] / f"per_class_metrics_fcos_{args.split}.csv",
            paths["tables"] / f"per_class_metrics_retinanet_{args.split}.csv",
            ["category_id", "category"],
            ["ap_coco", "recall_at_iou", "precision_at_iou", "mean_iou_matched", "mean_score"],
        )
        out = paths["tables"] / f"per_class_delta_fcos_vs_retinanet_{args.split}.csv"
    else:
        df = compare_metric_tables(
            paths["tables"] / f"size_group_analysis_fcos_{args.split}.csv",
            paths["tables"] / f"size_group_analysis_retinanet_{args.split}.csv",
            ["size_group"],
            ["recall_at_iou", "precision_at_iou", "mean_iou_matched", "mean_score_matched"],
        )
        out = paths["tables"] / f"size_group_delta_fcos_vs_retinanet_{args.split}.csv"
    print(write_csv(df, out))


if __name__ == "__main__":
    main()
