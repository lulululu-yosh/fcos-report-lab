from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .config import ensure_output_dirs, load_config
from .models import load_model, output_to_detections, predict_image
from .visualization import PALETTE, draw_detections, save_side_by_side

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(image_dir: str | Path) -> list[Path]:
    root = Path(image_dir)
    return sorted(path for path in root.iterdir() if path.suffix.lower() in IMAGE_EXTS)


@torch.inference_mode()
def save_fcos_head_maps(model, image: Image.Image, output_dir: Path, stem: str) -> list[Path]:
    """Save coarse FCOS head activation maps when torchvision internals are available."""
    from torchvision.transforms import functional as F

    device = next(model.parameters()).device
    tensors, _ = model.transform([F.to_tensor(image).to(device)])
    features = model.backbone(tensors.tensors)
    features_list = list(features.values()) if isinstance(features, dict) else list(features)
    head_outputs = model.head(features_list)
    written = []
    for name, values in head_outputs.items():
        if not values:
            continue
        fmap = values[0][0].detach().float().cpu()
        if fmap.ndim == 3:
            fmap = fmap.abs().mean(dim=0)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(fmap, cmap="magma")
        ax.set_title(f"FCOS {name} P3 activation")
        ax.axis("off")
        path = output_dir / f"{stem}_fcos_{name}_p3.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def run_demo(image_dir: str | Path, config_path: str | Path = "configs/default.yaml", save_head_maps: bool = False) -> list[Path]:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    images = collect_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}. Add jpg/png files or point --image-dir elsewhere.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fcos, fcos_spec = load_model("fcos", device=device, score_threshold=config["models"]["fcos"]["score_threshold"])
    retina, retina_spec = load_model("retinanet", device=device, score_threshold=config["models"]["retinanet"]["score_threshold"])
    written = []
    for path in images:
        image = Image.open(path).convert("RGB")
        fcos_out = predict_image(fcos, image, device)
        retina_out = predict_image(retina, image, device)
        fcos_dets = output_to_detections(fcos_out, fcos_spec.categories, config["models"]["fcos"]["score_threshold"])
        retina_dets = output_to_detections(retina_out, retina_spec.categories, config["models"]["retinanet"]["score_threshold"])
        left = draw_detections(image, retina_dets, PALETTE["retinanet"])
        right = draw_detections(image, fcos_dets, PALETTE["fcos"])
        out = paths["figures"] / f"{path.stem}_retinanet_vs_fcos.png"
        written.append(save_side_by_side(left, right, ("RetinaNet", "FCOS"), out))
        if save_head_maps:
            try:
                written.extend(save_fcos_head_maps(fcos, image, paths["figures"], path.stem))
            except Exception as exc:  # noqa: BLE001 - optional visualization should not break demo images.
                print(f"Skipping FCOS head maps for {path.name}: {exc}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FCOS/RetinaNet side-by-side detection visualizations.")
    parser.add_argument("--image-dir", default="samples")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--save-head-maps", action="store_true")
    args = parser.parse_args()
    for path in run_demo(args.image_dir, args.config, args.save_head_maps):
        print(path)


if __name__ == "__main__":
    main()
