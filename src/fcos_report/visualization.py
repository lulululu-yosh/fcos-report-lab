from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PALETTE = {
    "fcos": "#1f77b4",
    "retinanet": "#d62728",
    "accent": "#2ca02c",
    "muted": "#6b7280",
    "gold": "#ffbf00",
}


def apply_style(style: str = "seaborn-v0_8-whitegrid") -> None:
    plt.style.use(style)
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.size": 10,
        "axes.titleweight": "bold",
    })


def save_bar(df: pd.DataFrame, x: str, y: str, title: str, path: Path, color: str = "#1f77b4") -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(df[x].astype(str), df[y], color=color)
    ax.set_title(title)
    ax.set_ylabel(y.upper())
    ax.tick_params(axis="x", rotation=25)
    for i, value in enumerate(df[y]):
        ax.text(i, value + 0.3, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_grouped_metrics(df: pd.DataFrame, id_col: str, metrics: list[str], title: str, path: Path) -> Path:
    plot_df = df[[id_col, *metrics]].set_index(id_col)
    ax = plot_df.plot(kind="bar", figsize=(10, 5), width=0.78)
    ax.set_title(title)
    ax.set_ylabel("score")
    ax.legend(loc="upper left", ncols=min(3, len(metrics)))
    ax.tick_params(axis="x", rotation=20)
    ax.figure.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)
    return path


def save_radar(df: pd.DataFrame, label_col: str, metrics: list[str], title: str, path: Path) -> Path:
    labels = [m.upper() for m in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for _, row in df.iterrows():
        values = [float(row[m]) for m in metrics]
        values += values[:1]
        color = PALETTE["fcos"] if "FCOS" in str(row[label_col]) else PALETTE["retinanet"]
        ax.plot(angles, values, label=row[label_col], linewidth=2, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title(title, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_centerness_heatmap(path: Path, size: int = 241) -> Path:
    xs = np.linspace(-1, 1, size)
    ys = np.linspace(-1, 1, size)
    grid = np.zeros((size, size), dtype=float)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            l, r = x + 1, 1 - x
            t, b = y + 1, 1 - y
            grid[iy, ix] = math.sqrt((min(l, r) / max(l, r)) * (min(t, b) / max(t, b)))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(grid, cmap="viridis", origin="lower", extent=[-1, 1, -1, 1], vmin=0, vmax=1)
    ax.set_title("FCOS Center-ness Target")
    ax.set_xlabel("normalized x offset")
    ax.set_ylabel("normalized y offset")
    fig.colorbar(im, ax=ax, label="center-ness")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def save_fpn_assignment(path: Path) -> Path:
    levels = ["P3", "P4", "P5", "P6", "P7"]
    ranges = [(0, 64), (64, 128), (128, 256), (256, 512), (512, 700)]
    fig, ax = plt.subplots(figsize=(9, 2.8))
    for i, (level, (start, end)) in enumerate(zip(levels, ranges)):
        ax.broken_barh([(start, end - start)], (i - 0.35, 0.7), facecolors=PALETTE["fcos"])
        label = f"{level}: [{start}, {'inf' if level == 'P7' else end}]"
        ax.text(start + 6, i, label, va="center", ha="left", color="white", weight="bold", fontsize=9)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels(levels)
    ax.set_xlabel("max(l, t, r, b) target distance")
    ax.set_title("FCOS FPN Regression Range Assignment")
    ax.set_xlim(0, 700)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def draw_detections(image: Image.Image, detections: Iterable[dict], color: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for det in detections:
        box = det["box"]
        label = det.get("label", "object")
        score = det.get("score", 0.0)
        draw.rectangle(box, outline=color, width=3)
        text = f"{label} {score:.2f}"
        left, top = box[0], max(0, box[1] - 16)
        draw.rectangle([left, top, left + max(70, len(text) * 7), top + 16], fill=color)
        draw.text((left + 2, top + 1), text, fill="white", font=font)
    return out


def save_side_by_side(left: Image.Image, right: Image.Image, labels: tuple[str, str], path: Path) -> Path:
    w, h = left.size
    canvas = Image.new("RGB", (w * 2, h + 36), "white")
    canvas.paste(left, (0, 36))
    canvas.paste(right, (w, 36))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), labels[0], fill=PALETTE["retinanet"])
    draw.text((w + 12, 10), labels[1], fill=PALETTE["fcos"])
    canvas.save(path)
    return path
