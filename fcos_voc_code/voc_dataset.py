from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from torchvision.transforms import functional as F

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor",
]
VOC_CLASS_TO_IDX = {name: idx for idx, name in enumerate(VOC_CLASSES)}


@dataclass(frozen=True)
class VOCSpec:
    root: str
    year: str = "2012"
    image_set: str = "trainval"
    download: bool = False


class VOCDataset(Dataset):
    """Pascal VOC detection dataset for torchvision detection models.

    Labels are contiguous 0..19 because torchvision FCOS uses sigmoid multi-label
    classification internally and does not need an explicit background class.
    """

    def __init__(self, spec: VOCSpec, train: bool = True) -> None:
        self.dataset = VOCDetection(
            root=spec.root,
            year=spec.year,
            image_set=spec.image_set,
            download=spec.download,
        )
        self.train = train

    def __len__(self) -> int:
        return len(self.dataset)

    def _parse_target(self, annotation: dict[str, Any]) -> dict[str, torch.Tensor | str]:
        ann = annotation["annotation"]
        image_id = Path(ann["filename"]).stem
        objects = ann.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        boxes: list[list[float]] = []
        labels: list[int] = []
        difficult: list[int] = []
        for obj in objects:
            name = obj["name"].lower().strip()
            if name not in VOC_CLASS_TO_IDX:
                continue
            bbox = obj["bndbox"]
            # VOC XML coordinates are 1-based inclusive. Convert to 0-based xyxy.
            xmin = float(bbox["xmin"]) - 1.0
            ymin = float(bbox["ymin"]) - 1.0
            xmax = float(bbox["xmax"]) - 1.0
            ymax = float(bbox["ymax"]) - 1.0
            if xmax <= xmin or ymax <= ymin:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASS_TO_IDX[name])
            difficult.append(int(obj.get("difficult", 0)))

        target: dict[str, torch.Tensor | str] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "difficult": torch.as_tensor(difficult, dtype=torch.int64),
            "image_id": image_id,
        }
        return target

    def __getitem__(self, index: int):
        image, annotation = self.dataset[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        target = self._parse_target(annotation)
        return F.to_tensor(image), target


def detection_collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
