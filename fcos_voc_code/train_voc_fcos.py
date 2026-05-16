from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modeling import build_fcos_voc
from voc_dataset import VOCSpec, VOCDataset, detection_collate


def parse_args():
    p = argparse.ArgumentParser(description="Train FCOS on Pascal VOC trainval.")
    p.add_argument("--voc-root", required=True, help="Root that contains or will contain VOCdevkit/")
    p.add_argument("--year", default="2012", choices=["2007", "2012"])
    p.add_argument("--image-set", default="trainval")
    p.add_argument("--download", action="store_true")
    p.add_argument("--output-dir", default="outputs/voc_fcos")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--step-size", type=int, default=8)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--no-centerness", action="store_true", help="Train FCOS ablation without centerness loss/inference score.")
    p.add_argument("--min-size", type=int, default=800)
    p.add_argument("--max-size", type=int, default=1333)
    return p.parse_args()


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "args": vars(args),
    }, path)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = VOCDataset(VOCSpec(args.voc_root, args.year, args.image_set, args.download), train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=detection_collate,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_fcos_voc(no_centerness=args.no_centerness, min_size=args.min_size, max_size=args.max_size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt.get("scaler") and scaler is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = {}
        pbar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, targets in pbar:
            images = [img.to(device, non_blocking=True) for img in images]
            train_targets = []
            for t in targets:
                train_targets.append({
                    "boxes": t["boxes"].to(device, non_blocking=True),
                    "labels": t["labels"].to(device, non_blocking=True),
                })

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                loss_dict = model(images, train_targets)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + float(v.detach().cpu())
            running["total"] = running.get("total", 0.0) + float(loss.detach().cpu())
            pbar.set_postfix({k: f"{v / (pbar.n + 1):.4f}" for k, v in running.items()})

        scheduler.step()
        epoch_log = {k: v / max(1, len(loader)) for k, v in running.items()}
        epoch_log["epoch"] = epoch
        epoch_log["lr"] = optimizer.param_groups[0]["lr"]
        history.append(epoch_log)
        with open(Path(args.output_dir) / "train_log.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_checkpoint(Path(args.output_dir) / "last.pth", model, optimizer, scheduler, scaler, epoch, args)

    save_checkpoint(Path(args.output_dir) / "model_final.pth", model, optimizer, scheduler, scaler, args.epochs - 1, args)
    print(f"Saved final checkpoint to {Path(args.output_dir) / 'model_final.pth'}")


if __name__ == "__main__":
    main()
