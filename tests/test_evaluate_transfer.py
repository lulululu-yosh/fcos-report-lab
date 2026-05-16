import numpy as np
import pandas as pd

from fcos_report.evaluate_transfer import (
    VISDRONE_TO_COCO,
    VOC_TO_COCO,
    continuous_ap,
    upsert_csv,
    xyxy_iou,
)


def test_transfer_class_mappings():
    assert VOC_TO_COCO["aeroplane"] == "airplane"
    assert VOC_TO_COCO["sofa"] == "couch"
    assert VISDRONE_TO_COCO["pedestrian"] == "person"
    assert "tricycle" not in VISDRONE_TO_COCO


def test_transfer_iou_and_ap():
    assert xyxy_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert xyxy_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    ap = continuous_ap(np.array([0.5, 1.0]), np.array([1.0, 0.5]))
    assert round(ap, 3) == 0.75


def test_upsert_csv_replaces_matching_key(tmp_path):
    path = tmp_path / "table.csv"
    upsert_csv(path, pd.DataFrame([{"dataset": "voc", "model": "fcos", "ap50": 0.1}]), ["dataset", "model"])
    upsert_csv(path, pd.DataFrame([{"dataset": "voc", "model": "fcos", "ap50": 0.2}]), ["dataset", "model"])
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["ap50"] == 0.2
