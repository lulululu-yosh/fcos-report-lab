from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlretrieve
from zipfile import ZipFile

import torch
from PIL import Image
from tqdm import tqdm

from .config import ensure_output_dirs, load_config
from .models import load_model

COCO_URLS = {
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "test2017": "http://images.cocodataset.org/zips/test2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "image_info_test2017": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
}


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    urlretrieve(url, destination)
    return destination


def extract_zip(path: Path, destination: Path) -> None:
    marker = destination / f".extracted_{path.stem}"
    if marker.exists():
        return
    with ZipFile(path) as zf:
        zf.extractall(destination)
    marker.write_text("ok\n", encoding="utf-8")


def prepare_coco(root: str | Path, split: str) -> None:
    root = Path(root)
    if split not in {"val2017", "test2017"}:
        raise ValueError("split must be val2017 or test2017")
    image_zip = download_file(COCO_URLS[split], root / "zips" / f"{split}.zip")
    extract_zip(image_zip, root)
    ann_key = "annotations" if split == "val2017" else "image_info_test2017"
    ann_zip = download_file(COCO_URLS[ann_key], root / "zips" / f"{ann_key}.zip")
    extract_zip(ann_zip, root)


def annotation_path(root: Path, split: str) -> Path:
    if split == "val2017":
        return root / "annotations" / "instances_val2017.json"
    if split == "test2017":
        return root / "annotations" / "image_info_test2017.json"
    raise ValueError(split)


def image_dir(root: Path, split: str) -> Path:
    return root / split


def iter_coco_images(root: Path, split: str, limit: int | None = None) -> Iterable[dict[str, Any]]:
    ann_file = annotation_path(root, split)
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    images = data["images"][:limit]
    for item in images:
        yield item


def coco_result_from_output(image_id: int, output: dict[str, torch.Tensor], max_dets: int = 100) -> list[dict[str, Any]]:
    results = []
    for box, score, label in zip(output["boxes"][:max_dets], output["scores"][:max_dets], output["labels"][:max_dets]):
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        results.append({
            "image_id": int(image_id),
            "category_id": int(label.item()),
            "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
            "score": float(score.item()),
        })
    return results


@torch.inference_mode()
def run_inference(model_key: str, split: str, config_path: str | Path = "configs/default.yaml", limit: int | None = None) -> tuple[Path, dict[str, Any]]:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    root = Path(config["data"]["root"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_threshold = config["models"][model_key]["score_threshold"]
    model, spec = load_model(model_key, device=device, score_threshold=score_threshold)
    if limit is None and split == "val2017":
        limit = config["evaluation"].get("full_limit")
    max_dets = int(config["evaluation"]["max_detections_per_image"])

    results: list[dict[str, Any]] = []
    start = time.perf_counter()
    image_items = list(iter_coco_images(root, split, limit=limit))
    for item in tqdm(image_items, desc=f"{spec.key} {split}"):
        path = image_dir(root, split) / item["file_name"]
        image = Image.open(path).convert("RGB")
        from torchvision.transforms import functional as F
        output = model([F.to_tensor(image).to(device)])[0]
        results.extend(coco_result_from_output(item["id"], {k: v.cpu() for k, v in output.items()}, max_dets=max_dets))
    elapsed = time.perf_counter() - start

    result_path = paths["coco_results"] / f"{model_key}_{split}_results.json"
    result_path.write_text(json.dumps(results), encoding="utf-8")
    summary = {
        "model": spec.display_name,
        "split": split,
        "images": len(image_items),
        "detections": len(results),
        "seconds": elapsed,
        "fps": len(image_items) / elapsed if elapsed else 0.0,
        "result_path": str(result_path),
    }
    return result_path, summary


def evaluate_coco(result_path: str | Path, split: str, config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    config = load_config(config_path)
    root = Path(config["data"]["root"])
    coco_gt = COCO(str(annotation_path(root, split)))
    coco_dt = coco_gt.loadRes(str(result_path))
    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
    return {name: float(value) for name, value in zip(names, evaluator.stats)}
