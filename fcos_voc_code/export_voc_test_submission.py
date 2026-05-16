from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modeling import build_fcos_voc
from voc_dataset import VOC_CLASSES, VOCSpec, VOCDataset, detection_collate


def parse_args():
    p = argparse.ArgumentParser(description="Export Pascal VOC test server detection files.")
    p.add_argument("--voc-root", required=True)
    p.add_argument("--year", default="2012", choices=["2007", "2012"])
    p.add_argument("--image-set", default="test")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="outputs/voc_submit")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--score-thresh", type=float, default=0.001)
    p.add_argument("--no-centerness", action="store_true")
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = VOCDataset(VOCSpec(args.voc_root, args.year, args.image_set, False), train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        collate_fn=detection_collate, pin_memory=torch.cuda.is_available())

    model = build_fcos_voc(no_centerness=args.no_centerness, min_size=args.min_size, max_size=args.max_size).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    model.score_thresh = args.score_thresh

    rows = {cls: [] for cls in VOC_CLASSES}
    for images, targets in tqdm(loader, desc="export VOC test detections"):
        image_ids = [str(t["image_id"]) for t in targets]
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model(images)
        for image_id, output in zip(image_ids, outputs):
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()
            for box, score, label in zip(boxes, scores, labels):
                label_i = int(label)
                if label_i < 0 or label_i >= len(VOC_CLASSES):
                    continue
                if float(score) < args.score_thresh:
                    continue
                # VOC result files use 1-based coordinates.
                xmin, ymin, xmax, ymax = (box + 1.0).tolist()
                cls = VOC_CLASSES[label_i]
                rows[cls].append(f"{image_id} {float(score):.6f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}\n")

    prefix = "comp4_det_test"
    for cls, cls_rows in rows.items():
        with open(out_dir / f"{prefix}_{cls}.txt", "w", encoding="utf-8") as f:
            f.writelines(cls_rows)
    print(f"Wrote {len(VOC_CLASSES)} VOC submission txt files to {out_dir}")
    print("Zip this directory and upload the txt files to the Pascal VOC evaluation server.")


if __name__ == "__main__":
    main()
