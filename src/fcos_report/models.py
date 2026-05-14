from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class ModelSpec:
    key: str
    display_name: str
    categories: list[str]


def load_model(model_key: str, device: str | torch.device | None = None, score_threshold: float | None = None):
    """Load a torchvision FCOS or RetinaNet model with official COCO weights."""
    import torchvision
    from torchvision.models.detection import (
        FCOS_ResNet50_FPN_Weights,
        RetinaNet_ResNet50_FPN_Weights,
        fcos_resnet50_fpn,
        retinanet_resnet50_fpn,
    )

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    key = model_key.lower()
    if key == "fcos":
        weights = FCOS_ResNet50_FPN_Weights.DEFAULT
        model = fcos_resnet50_fpn(weights=weights, score_thresh=score_threshold or 0.35)
        display = "FCOS ResNet-50-FPN"
    elif key == "retinanet":
        weights = RetinaNet_ResNet50_FPN_Weights.DEFAULT
        model = retinanet_resnet50_fpn(weights=weights, score_thresh=score_threshold or 0.35)
        display = "RetinaNet ResNet-50-FPN"
    else:
        raise ValueError(f"Unsupported model: {model_key}. Use 'fcos' or 'retinanet'.")

    model.eval().to(device)
    categories = weights.meta.get("categories", [])
    return model, ModelSpec(key=key, display_name=display, categories=categories)


def preprocess_image(image, device: torch.device):
    from torchvision.transforms import functional as F

    return F.to_tensor(image).to(device)


@torch.inference_mode()
def predict_image(model: torch.nn.Module, image, device: torch.device, top_k: int = 30) -> dict[str, Any]:
    tensor = preprocess_image(image, device)
    output = model([tensor])[0]
    keep = torch.arange(len(output["scores"]), device=device)[:top_k]
    return {k: v[keep].detach().cpu() for k, v in output.items()}


def output_to_detections(output: dict[str, Any], categories: list[str], min_score: float = 0.35) -> list[dict[str, Any]]:
    detections = []
    boxes = output["boxes"].tolist()
    scores = output["scores"].tolist()
    labels = output["labels"].tolist()
    for box, score, label_id in zip(boxes, scores, labels):
        if score < min_score:
            continue
        label = categories[label_id] if 0 <= label_id < len(categories) else str(label_id)
        detections.append({
            "box": [float(x) for x in box],
            "score": float(score),
            "label": label,
            "label_id": int(label_id),
        })
    return detections
