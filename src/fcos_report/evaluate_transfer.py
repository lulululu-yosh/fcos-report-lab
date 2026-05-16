from __future__ import annotations

import argparse
import json
import tarfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from .config import load_config
from .models import load_model, preprocess_image

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

VOC_TO_COCO = {
    "aeroplane": "airplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "diningtable": "dining table",
    "dog": "dog",
    "horse": "horse",
    "motorbike": "motorcycle",
    "person": "person",
    "pottedplant": "potted plant",
    "sheep": "sheep",
    "sofa": "couch",
    "train": "train",
    "tvmonitor": "tv",
}

VISDRONE_CLASSES = {
    0: "ignored regions",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

VISDRONE_TO_COCO = {
    "pedestrian": "person",
    "people": "person",
    "bicycle": "bicycle",
    "car": "car",
    "van": "truck",
    "truck": "truck",
    "bus": "bus",
    "motor": "motorcycle",
}

VOC_URLS = {
    ("2007", "trainval"): "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
    ("2007", "test"): "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
    ("2012", "trainval"): "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
}

SOURCE_COCO_AP50 = {
    "fcos": 0.5834783636177595,
    "retinanet": 0.5542430763065064,
}


@dataclass
class TargetAnnotation:
    bbox_xyxy: list[float]
    category_name: str
    coco_category_name: str
    coco_category_id: int
    ignore: bool = False


@dataclass
class TargetImage:
    image_id: str
    file_name: str
    width: int
    height: int
    annotations: list[TargetAnnotation]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def coco_name_to_id(categories: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(categories) if name != "__background__"}


def download_file(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, path)
    return path


def extract_archive(path: Path, out_dir: Path) -> None:
    marker = out_dir / f".extracted_{path.stem}"
    if marker.exists():
        return
    if path.suffix == ".zip":
        with ZipFile(path) as zf:
            zf.extractall(out_dir)
    else:
        with tarfile.open(path) as tf:
            tf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")


def download_voc(data_root: Path, year: str, split: str) -> None:
    kind = "test" if split == "test" and year == "2007" else "trainval"
    key = (year, kind)
    if key not in VOC_URLS:
        raise ValueError(f"No automatic VOC download URL for year={year}, split={split}.")
    archive = download_file(VOC_URLS[key], data_root / "downloads" / Path(VOC_URLS[key]).name)
    extract_archive(archive, data_root)


def visdrone_download_message(data_root: Path) -> str:
    return (
        "VisDrone is not automatically downloaded. Place the DET split under "
        f"{data_root / 'VisDrone'}, e.g. "
        "data/VisDrone/VisDrone2019-DET-val/images and annotations. "
        "Download from the official VisDrone project page or Kaggle mirrors, then unzip preserving the split directory."
    )


def parse_voc_object(obj: ET.Element, name_to_id: dict[str, int], include_difficult: bool) -> TargetAnnotation | None:
    voc_name = (obj.findtext("name") or "").strip()
    if voc_name not in VOC_TO_COCO:
        return None
    difficult = int(obj.findtext("difficult") or 0) == 1
    if difficult and not include_difficult:
        return None
    coco_name = VOC_TO_COCO[voc_name]
    if coco_name not in name_to_id:
        return None
    box = obj.find("bndbox")
    if box is None:
        return None
    # VOC coordinates are conventionally 1-based inclusive.
    x1 = max(0.0, float(box.findtext("xmin") or 0) - 1.0)
    y1 = max(0.0, float(box.findtext("ymin") or 0) - 1.0)
    x2 = float(box.findtext("xmax") or 0)
    y2 = float(box.findtext("ymax") or 0)
    return TargetAnnotation([x1, y1, x2, y2], voc_name, coco_name, name_to_id[coco_name], difficult)


def load_voc_annotations(
    root: str | Path,
    year: str = "2007",
    split: str = "test",
    name_to_id: dict[str, int] | None = None,
    include_difficult: bool = False,
) -> list[TargetImage]:
    name_to_id = name_to_id or {}
    voc_root = Path(root) / "VOCdevkit" / f"VOC{year}"
    split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"VOC split file not found: {split_file}. Use --download or check --data-root.")
    image_ids = [line.strip().split()[0] for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    images: list[TargetImage] = []
    for image_id in image_ids:
        xml_path = voc_root / "Annotations" / f"{image_id}.xml"
        tree = ET.parse(xml_path)
        root_el = tree.getroot()
        size = root_el.find("size")
        width = int(size.findtext("width") or 0) if size is not None else 0
        height = int(size.findtext("height") or 0) if size is not None else 0
        filename = root_el.findtext("filename") or f"{image_id}.jpg"
        annotations = [
            ann
            for ann in (parse_voc_object(obj, name_to_id, include_difficult) for obj in root_el.findall("object"))
            if ann is not None
        ]
        images.append(TargetImage(image_id, str(voc_root / "JPEGImages" / filename), width, height, annotations))
    return images


def load_visdrone_annotations(
    root: str | Path,
    split: str = "val",
    name_to_id: dict[str, int] | None = None,
    map_tricycle: bool = False,
) -> list[TargetImage]:
    name_to_id = name_to_id or {}
    mapping = dict(VISDRONE_TO_COCO)
    if map_tricycle:
        mapping.update({"tricycle": "bicycle", "awning-tricycle": "bicycle"})
    split_dir = Path(root) / "VisDrone" / f"VisDrone2019-DET-{split}"
    image_root = split_dir / "images"
    ann_root = split_dir / "annotations"
    if not image_root.exists() or not ann_root.exists():
        raise FileNotFoundError(f"VisDrone split not found under {split_dir}. {visdrone_download_message(Path(root))}")
    images: list[TargetImage] = []
    for image_path in sorted(image_root.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue
        ann_path = ann_root / f"{image_path.stem}.txt"
        annotations: list[TargetAnnotation] = []
        if ann_path.exists():
            for raw in ann_path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                parts = [p.strip() for p in raw.split(",")]
                if len(parts) < 6:
                    continue
                left, top, width, height = [float(v) for v in parts[:4]]
                score = int(float(parts[4]))
                category_id = int(float(parts[5]))
                source_name = VISDRONE_CLASSES.get(category_id, "unknown")
                if score == 0 or category_id == 0 or source_name not in mapping:
                    continue
                coco_name = mapping[source_name]
                if coco_name not in name_to_id:
                    continue
                annotations.append(TargetAnnotation(
                    [left, top, left + max(0.0, width), top + max(0.0, height)],
                    source_name,
                    coco_name,
                    name_to_id[coco_name],
                ))
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(TargetImage(image_path.stem, str(image_path), width, height, annotations))
    return images


def xyxy_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@torch.inference_mode()
def predict_transfer(
    images: list[TargetImage],
    model: torch.nn.Module,
    model_key: str,
    categories: list[str],
    allowed_ids: set[int],
    score_threshold: float,
    device: torch.device,
) -> tuple[list[dict], dict[str, float]]:
    predictions: list[dict] = []
    start = time.perf_counter()
    for item in tqdm(images, desc=f"{model_key} transfer"):
        image = Image.open(item.file_name).convert("RGB")
        output = model([preprocess_image(image, device)])[0]
        boxes = output["boxes"].detach().cpu().tolist()
        labels = output["labels"].detach().cpu().tolist()
        scores = output["scores"].detach().cpu().tolist()
        for box, label, score in zip(boxes, labels, scores):
            label = int(label)
            if score < score_threshold or label not in allowed_ids:
                continue
            predictions.append({
                "image_id": item.image_id,
                "file_name": item.file_name,
                "bbox_xyxy": [float(v) for v in box],
                "score": float(score),
                "coco_category_id": label,
                "coco_category_name": categories[label],
            })
    elapsed = time.perf_counter() - start
    return predictions, {
        "total_time": elapsed,
        "fps": len(images) / elapsed if elapsed else 0.0,
    }


def continuous_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    changed = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def evaluate_predictions(
    images: list[TargetImage],
    predictions: list[dict],
    dataset: str,
    split: str,
    model_key: str,
    iou_threshold: float,
) -> tuple[dict, pd.DataFrame]:
    gt_by_class_image: dict[str, dict[str, list[TargetAnnotation]]] = defaultdict(lambda: defaultdict(list))
    pred_by_class: dict[str, list[dict]] = defaultdict(list)
    class_to_coco_name: dict[str, str] = {}
    for image in images:
        for ann in image.annotations:
            if ann.ignore:
                continue
            class_to_coco_name[ann.coco_category_name] = ann.coco_category_name
            gt_by_class_image[ann.coco_category_name][image.image_id].append(ann)
    for pred in predictions:
        pred_by_class[pred["coco_category_name"]].append(pred)
        class_to_coco_name[pred["coco_category_name"]] = pred["coco_category_name"]

    per_class_rows = []
    total_tp = total_fp = total_fn = 0
    total_iou = 0.0
    ap_values = []
    for class_name in sorted(class_to_coco_name):
        gt_for_class = gt_by_class_image.get(class_name, {})
        preds = sorted(pred_by_class.get(class_name, []), key=lambda item: item["score"], reverse=True)
        num_gt = sum(len(items) for items in gt_for_class.values())
        matched: dict[str, set[int]] = defaultdict(set)
        tp = np.zeros(len(preds), dtype=float)
        fp = np.zeros(len(preds), dtype=float)
        matched_ious: list[float] = []
        for idx, pred in enumerate(preds):
            gt_items = gt_for_class.get(pred["image_id"], [])
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, ann in enumerate(gt_items):
                if gt_idx in matched[pred["image_id"]]:
                    continue
                iou = xyxy_iou(pred["bbox_xyxy"], ann.bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                tp[idx] = 1.0
                matched[pred["image_id"]].add(best_gt_idx)
                matched_ious.append(best_iou)
            else:
                fp[idx] = 1.0
        class_tp = int(tp.sum())
        class_fp = int(fp.sum())
        class_fn = int(max(0, num_gt - class_tp))
        if num_gt > 0 and preds:
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            recalls = cum_tp / num_gt
            precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(float).eps)
            ap50 = continuous_ap(recalls, precisions)
        else:
            ap50 = 0.0
        if num_gt > 0:
            ap_values.append(ap50)
        total_tp += class_tp
        total_fp += class_fp
        total_fn += class_fn
        total_iou += sum(matched_ious)
        per_class_rows.append({
            "dataset": dataset,
            "split": split,
            "model": model_key,
            "class_name": class_name,
            "coco_class_name": class_name,
            "num_gt": num_gt,
            "num_predictions": len(preds),
            "ap50": ap50,
            "recall50": class_tp / num_gt if num_gt else 0.0,
            "precision50": class_tp / (class_tp + class_fp) if class_tp + class_fp else 0.0,
            "mean_iou_matched": float(np.mean(matched_ious)) if matched_ious else 0.0,
            "false_positives": class_fp,
            "false_negatives": class_fn,
        })

    summary = {
        "ap50": float(np.mean(ap_values)) if ap_values else 0.0,
        "recall50": total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0,
        "precision50": total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0,
        "mean_iou_matched": total_iou / total_tp if total_tp else 0.0,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "num_gt": total_tp + total_fn,
        "num_predictions": total_tp + total_fp,
        "mapped_classes": sum(1 for row in per_class_rows if row["num_gt"] > 0),
    }
    return summary, pd.DataFrame(per_class_rows)


def upsert_csv(path: Path, row_df: pd.DataFrame, keys: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        if not old.empty:
            merged = old.merge(row_df[keys].drop_duplicates(), on=keys, how="left", indicator=True)
            old = old[merged["_merge"].eq("left_only").to_numpy()]
            row_df = pd.concat([old, row_df], ignore_index=True)
    row_df.to_csv(path, index=False)


def write_transfer_delta(tables_dir: Path, dataset: str, split: str) -> None:
    per_class_path = tables_dir / f"transfer_per_class_{dataset}.csv"
    if not per_class_path.exists():
        return
    df = pd.read_csv(per_class_path)
    df = df[df["split"].eq(split)]
    fcos = df[df["model"].eq("fcos")]
    retina = df[df["model"].eq("retinanet")]
    if fcos.empty or retina.empty:
        return
    merged = fcos.merge(retina, on=["dataset", "split", "class_name", "coco_class_name"], suffixes=("_fcos", "_retinanet"))
    delta = pd.DataFrame({
        "dataset": merged["dataset"],
        "split": merged["split"],
        "class_name": merged["class_name"],
        "ap50_fcos": merged["ap50_fcos"],
        "ap50_retinanet": merged["ap50_retinanet"],
        "ap50_delta": merged["ap50_fcos"] - merged["ap50_retinanet"],
        "recall50_fcos": merged["recall50_fcos"],
        "recall50_retinanet": merged["recall50_retinanet"],
        "recall_delta": merged["recall50_fcos"] - merged["recall50_retinanet"],
        "precision50_fcos": merged["precision50_fcos"],
        "precision50_retinanet": merged["precision50_retinanet"],
        "precision_delta": merged["precision50_fcos"] - merged["precision50_retinanet"],
    })
    upsert_csv(tables_dir / "transfer_delta.csv", delta, ["dataset", "split", "class_name"])


def source_coco_ap50(model: str, config_path: str | Path) -> float:
    config = load_config(config_path)
    metrics_path = Path(config["project"]["output_dir"]) / "metrics" / f"{model}_val2017_full.json"
    if metrics_path.exists():
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        return float(data["coco_bbox"]["AP50"])
    return SOURCE_COCO_AP50[model]


def evaluate_transfer(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    tables_dir = Path(config["project"]["output_dir"]) / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    model, spec = load_model(args.model, device=device, score_threshold=args.score_threshold)
    name_to_id = coco_name_to_id(spec.categories)

    if args.download and args.dataset == "voc":
        download_voc(Path(args.data_root), args.voc_year, args.split)
    elif args.download and args.dataset == "visdrone":
        print(visdrone_download_message(Path(args.data_root)))

    if args.dataset == "voc":
        images = load_voc_annotations(Path(args.data_root), args.voc_year, args.split, name_to_id, args.include_difficult)
    elif args.dataset == "visdrone":
        images = load_visdrone_annotations(Path(args.data_root), args.split, name_to_id, args.visdrone_map_tricycle)
    else:
        raise ValueError(args.dataset)
    if args.limit:
        images = images[:args.limit]
    allowed_ids = {ann.coco_category_id for image in images for ann in image.annotations if not ann.ignore}
    if not allowed_ids:
        raise RuntimeError(f"No mapped ground-truth classes found for {args.dataset} {args.split}.")

    predictions, runtime = predict_transfer(images, model, args.model, spec.categories, allowed_ids, args.score_threshold, device)
    limit_part = f"_limit{args.limit}" if args.limit else ""
    pred_path = output_dir / f"{args.dataset}_{args.model}_{args.split}{limit_part}_predictions.json"
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    summary, per_class = evaluate_predictions(images, predictions, args.dataset, args.split, args.model, args.iou_threshold)
    source_ap50 = source_coco_ap50(args.model, args.config)
    row = {
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit or "",
        "model": args.model,
        "num_images": len(images),
        "num_gt": summary["num_gt"],
        "num_predictions": summary["num_predictions"],
        "mapped_classes": summary["mapped_classes"],
        "ap50": summary["ap50"],
        "recall50": summary["recall50"],
        "precision50": summary["precision50"],
        "mean_iou_matched": summary["mean_iou_matched"],
        "false_positives": summary["false_positives"],
        "false_negatives": summary["false_negatives"],
        "fps": runtime["fps"],
        "total_time": runtime["total_time"],
        "source_coco_ap50": source_ap50,
        "transfer_retention": summary["ap50"] / source_ap50 if source_ap50 else 0.0,
        "iou_threshold": args.iou_threshold,
        "score_threshold": args.score_threshold,
        "predictions_json": str(pred_path),
    }
    upsert_csv(tables_dir / "transfer_summary.csv", pd.DataFrame([row]), ["dataset", "split", "limit", "model"])
    upsert_csv(tables_dir / f"transfer_per_class_{args.dataset}.csv", per_class, ["dataset", "split", "model", "class_name"])
    retention = pd.DataFrame([{
        "dataset": args.dataset,
        "split": args.split,
        "limit": args.limit or "",
        "model": args.model,
        "target_ap50": summary["ap50"],
        "source_coco_ap50": source_ap50,
        "transfer_retention": summary["ap50"] / source_ap50 if source_ap50 else 0.0,
        "note": "Approximate retention: target mAP50 divided by overall COCO val2017 AP50.",
    }])
    upsert_csv(tables_dir / "transfer_retention.csv", retention, ["dataset", "split", "limit", "model"])
    write_transfer_delta(tables_dir, args.dataset, args.split)
    print(json.dumps(row, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset transfer evaluation for COCO-pretrained FCOS and RetinaNet.")
    parser.add_argument("--dataset", choices=["voc", "visdrone"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--model", choices=["fcos", "retinanet"], required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/transfer")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--voc-year", choices=["2007", "2012"], default="2007")
    parser.add_argument("--include-difficult", action="store_true")
    parser.add_argument("--visdrone-map-tricycle", action="store_true")
    args = parser.parse_args()
    evaluate_transfer(args)


if __name__ == "__main__":
    main()
