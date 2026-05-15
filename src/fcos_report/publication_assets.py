from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from textwrap import dedent

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import ensure_output_dirs, load_config


COLORS = {
    "fcos": "#0072B2",
    "retinanet": "#D55E00",
    "delta_pos": "#009E73",
    "delta_neg": "#CC79A7",
    "neutral": "#4D4D4D",
    "grid": "#E6E6E6",
}


def publication_dirs(config_path: str | Path = "configs/default.yaml") -> dict[str, Path]:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    paths["publication_figures"] = paths["root"] / "publication_figures"
    paths["latex_tables"] = paths["root"] / "latex_tables"
    paths["publication_figures"].mkdir(parents=True, exist_ok=True)
    paths["latex_tables"].mkdir(parents=True, exist_ok=True)
    return paths


def set_publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, stem: str, out_dir: Path) -> list[Path]:
    written = []
    for ext in ["pdf", "svg"]:
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", transparent=False)
        written.append(path)
    plt.close(fig)
    return written


def pct(value: float) -> float:
    return float(value) * 100.0


def read_metric_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_main_metrics(paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for model, label in [("fcos", "FCOS"), ("retinanet", "RetinaNet")]:
        data = read_metric_json(paths["metrics"] / f"{model}_val2017_full.json")
        row = {"model": label, "fps": data["runtime"]["fps"], "seconds": data["runtime"]["seconds"]}
        row.update({key: pct(value) for key, value in data["coco_bbox"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def latex_escape(text: object) -> str:
    value = str(text)
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def latex_cell(value: object) -> str:
    if isinstance(value, str) and ("\\" in value or "$" in value):
        return value
    return latex_escape(value)


def write_booktabs_table(
    df: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    column_format: str | None = None,
    float_format: str = "{:.2f}",
) -> Path:
    column_format = column_format or ("l" + "r" * (len(df.columns) - 1))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(latex_cell(col) for col in df.columns) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                cells.append(float_format.format(float(value)))
            else:
                cells.append(latex_cell(value))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_main_metric_figure(paths: dict[str, Path]) -> list[Path]:
    df = load_main_metrics(paths)
    metrics = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
    fcos = df[df["model"] == "FCOS"].iloc[0]
    retina = df[df["model"] == "RetinaNet"].iloc[0]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(x - width / 2, [retina[m] for m in metrics], width, label="RetinaNet", color=COLORS["retinanet"])
    ax.bar(x + width / 2, [fcos[m] for m in metrics], width, label="FCOS", color=COLORS["fcos"])
    for i, metric in enumerate(metrics):
        delta = fcos[metric] - retina[metric]
        y = max(fcos[metric], retina[metric]) + 1.0
        ax.text(i, y, f"{delta:+.1f}", ha="center", va="bottom", fontsize=8, color=COLORS["delta_pos"] if delta >= 0 else COLORS["delta_neg"])
    ax.set_xticks(x, metrics)
    ax.set_ylabel("COCO bbox metric (%)")
    ax.set_title("FCOS vs. RetinaNet on COCO val2017")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.set_ylim(0, max(df[metrics].max()) + 8)
    fig.tight_layout()
    return save_figure(fig, "main_coco_metrics", paths["publication_figures"])


def make_main_metric_table(paths: dict[str, Path]) -> Path:
    df = load_main_metrics(paths)
    metrics = ["model", "AP", "AP50", "AP75", "APs", "APm", "APl", "AR100", "fps"]
    table = df[metrics].copy()
    table.columns = ["Model", "AP", "AP50", "AP75", "APs", "APm", "APl", "AR100", "FPS"]
    delta = table.iloc[0].copy()
    delta["Model"] = r"$\Delta$ FCOS--RetinaNet"
    for col in table.columns[1:]:
        delta[col] = float(table.iloc[0][col]) - float(table.iloc[1][col])
    table = pd.concat([table, pd.DataFrame([delta])], ignore_index=True)
    return write_booktabs_table(table, paths["latex_tables"] / "main_coco_metrics.tex", "COCO val2017 comparison using torchvision pretrained detectors.", "tab:main-coco-metrics", "lrrrrrrrr")


def make_size_table(paths: dict[str, Path]) -> Path:
    src = pd.read_csv(paths["tables"] / "size_group_delta_fcos_vs_retinanet_val2017.csv")
    table = pd.DataFrame({
        "Size": src["size_group"].str.title(),
        "GT": src["gt_fcos"],
        "Recall FCOS": src["recall_at_iou_fcos"] * 100,
        "Recall RetinaNet": src["recall_at_iou_retinanet"] * 100,
        r"$\Delta$ Recall": src["recall_at_iou_delta_fcos_minus_retinanet"] * 100,
        "mIoU FCOS": src["mean_iou_matched_fcos"] * 100,
        "mIoU RetinaNet": src["mean_iou_matched_retinanet"] * 100,
    })
    return write_booktabs_table(table, paths["latex_tables"] / "size_group_analysis.tex", "Size-stratified detection analysis at IoU 0.50.", "tab:size-analysis", "lrrrrrr")


def make_threshold_figure(paths: dict[str, Path]) -> list[Path]:
    fcos = pd.read_csv(paths["tables"] / "threshold_sweep_fcos_val2017.csv")
    retina = pd.read_csv(paths["tables"] / "threshold_sweep_retinanet_val2017.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for df, label, color, marker in [(retina, "RetinaNet", COLORS["retinanet"], "s"), (fcos, "FCOS", COLORS["fcos"], "o")]:
        axes[0].plot(df["recall_approx"] * 100, df["precision_approx"] * 100, marker=marker, color=color, label=label, linewidth=1.8)
        for _, row in df.iterrows():
            if row["score_threshold"] in [0.05, 0.3, 0.7]:
                axes[0].annotate(f"{row['score_threshold']:.2f}", (row["recall_approx"] * 100, row["precision_approx"] * 100), xytext=(3, 3), textcoords="offset points", fontsize=7)
        axes[1].plot(df["score_threshold"], df["mean_detections_per_image"], marker=marker, color=color, label=label, linewidth=1.8)
    axes[0].set_xlabel("Approx. recall @ IoU 0.50 (%)")
    axes[0].set_ylabel("Approx. precision @ IoU 0.50 (%)")
    axes[0].set_title("Score-threshold trade-off")
    axes[1].set_xlabel("Score threshold")
    axes[1].set_ylabel("Detections / image")
    axes[1].set_title("Prediction density")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, "threshold_tradeoff", paths["publication_figures"])


def make_threshold_table(paths: dict[str, Path]) -> Path:
    rows = []
    for model, label in [("fcos", "FCOS"), ("retinanet", "RetinaNet")]:
        df = pd.read_csv(paths["tables"] / f"threshold_sweep_{model}_val2017.csv")
        df = df[df["score_threshold"].isin([0.05, 0.3, 0.5, 0.7])].copy()
        for _, row in df.iterrows():
            rows.append({
                "Model": label,
                "Thr.": row["score_threshold"],
                "Det./img": row["mean_detections_per_image"],
                "Prec.": row["precision_approx"] * 100,
                "Rec.": row["recall_approx"] * 100,
                "mIoU": row["mean_iou_matched"] * 100,
            })
    return write_booktabs_table(pd.DataFrame(rows), paths["latex_tables"] / "threshold_sweep.tex", "Score threshold sweep on COCO val2017.", "tab:threshold-sweep", "lrrrrr")


def make_per_class_delta_figure(paths: dict[str, Path], top_k: int = 12) -> list[Path]:
    df = pd.read_csv(paths["tables"] / "per_class_delta_fcos_vs_retinanet_val2017.csv")
    col = "ap_coco_delta_fcos_minus_retinanet"
    selected = pd.concat([df.nlargest(top_k, col), df.nsmallest(top_k, col)]).sort_values(col)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    y = np.arange(len(selected))
    colors = [COLORS["delta_pos"] if value >= 0 else COLORS["delta_neg"] for value in selected[col]]
    ax.barh(y, selected[col] * 100, color=colors)
    ax.axvline(0, color="#222222", linewidth=0.8)
    ax.set_yticks(y, selected["category"])
    ax.set_xlabel(r"$\Delta$ AP (FCOS - RetinaNet, percentage points)")
    ax.set_title("Categories with largest AP changes")
    fig.tight_layout()
    return save_figure(fig, "per_class_ap_delta_top_bottom", paths["publication_figures"])


def make_per_class_table(paths: dict[str, Path], top_k: int = 8) -> Path:
    df = pd.read_csv(paths["tables"] / "per_class_delta_fcos_vs_retinanet_val2017.csv")
    col = "ap_coco_delta_fcos_minus_retinanet"
    best = df.nlargest(top_k, col).assign(Group="FCOS higher")
    worst = df.nsmallest(top_k, col).assign(Group="RetinaNet higher")
    src = pd.concat([best, worst], ignore_index=True)
    table = pd.DataFrame({
        "Group": src["Group"],
        "Category": src["category"],
        "AP FCOS": src["ap_coco_fcos"] * 100,
        "AP RetinaNet": src["ap_coco_retinanet"] * 100,
        r"$\Delta$ AP": src[col] * 100,
    })
    return write_booktabs_table(table, paths["latex_tables"] / "per_class_ap_delta.tex", "Largest category-level AP differences.", "tab:per-class-delta", "llrrr")


def make_crowded_figure(paths: dict[str, Path]) -> list[Path]:
    fcos = pd.read_csv(paths["tables"] / "crowded_scene_analysis_fcos_val2017.csv")
    retina = pd.read_csv(paths["tables"] / "crowded_scene_analysis_retinanet_val2017.csv")
    merged = fcos.merge(retina, on=["image_id", "file_name"], suffixes=("_fcos", "_retinanet"))
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    sc = ax.scatter(
        merged["recall_at_iou_retinanet"] * 100,
        merged["recall_at_iou_fcos"] * 100,
        c=merged["overlap_pairs_iou_0_3_fcos"],
        s=np.clip(merged["gt_count_fcos"] * 5, 20, 220),
        cmap="viridis",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.plot([0, 100], [0, 100], linestyle="--", color=COLORS["neutral"], linewidth=1)
    ax.set_xlabel("RetinaNet recall @ IoU 0.50 (%)")
    ax.set_ylabel("FCOS recall @ IoU 0.50 (%)")
    ax.set_title("Crowded / overlapped scene recall")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("GT pairs with IoU >= 0.30")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    return save_figure(fig, "crowded_scene_recall_scatter", paths["publication_figures"])


def make_fpn_distribution_figure(paths: dict[str, Path]) -> list[Path]:
    df = pd.read_csv(paths["tables"] / "fcos_detection_scale_distribution_val2017.csv", usecols=["score", "approx_fpn_level", "area"])
    thresholds = [0.05, 0.3, 0.5, 0.7]
    levels = ["P3", "P4", "P5", "P6", "P7"]
    rows = []
    for thr in thresholds:
        counts = df[df["score"] >= thr]["approx_fpn_level"].value_counts(normalize=True)
        rows.append([counts.get(level, 0.0) * 100 for level in levels])
    data = np.array(rows)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    bottom = np.zeros(len(thresholds))
    cmap = plt.get_cmap("Blues")
    for i, level in enumerate(levels):
        values = data[:, i]
        ax.bar([str(t) for t in thresholds], values, bottom=bottom, label=level, color=cmap(0.35 + i * 0.12))
        bottom += values
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Detection share (%)")
    ax.set_title("Approximate FPN-level distribution of FCOS detections")
    ax.legend(frameon=False, ncols=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    return save_figure(fig, "fcos_fpn_level_distribution", paths["publication_figures"])


def make_head_stats_figure(paths: dict[str, Path]) -> list[Path]:
    df = pd.read_csv(paths["tables"] / "fcos_head_tensor_stats_val2017.csv")
    outputs = ["cls_logits", "bbox_regression", "bbox_ctrness"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    for ax, output in zip(axes, outputs):
        sub = df[df["output"] == output]
        ax.boxplot([sub["mean"], sub["std"]], tick_labels=["mean", "std"], patch_artist=True, boxprops={"facecolor": "#DCEEFF", "edgecolor": COLORS["fcos"]}, medianprops={"color": COLORS["retinanet"]})
        ax.set_title(output.replace("_", " "))
        ax.grid(axis="x", visible=False)
    fig.suptitle("FCOS head tensor statistics on sampled images", y=1.02)
    fig.tight_layout()
    return save_figure(fig, "fcos_head_tensor_stats", paths["publication_figures"])


def make_case_contact_sheet(paths: dict[str, Path], per_group: int = 2) -> Path | None:
    case_root = paths["report_assets"] / "case_studies"
    if not case_root.exists():
        return None
    groups = ["fcos_better", "retinanet_better", "both_good", "both_hard", "crowded_overlap"]
    selected: list[tuple[str, Path]] = []
    for group in groups:
        for path in sorted((case_root / group).glob("*_retinanet_vs_fcos.jpg"))[:per_group]:
            selected.append((group.replace("_", " "), path))
    if not selected:
        return None
    thumb_w = 460
    label_h = 34
    gap = 14
    cols = 2
    rows = math.ceil(len(selected) / cols)
    thumbs = []
    for label, path in selected:
        image = Image.open(path).convert("RGB")
        ratio = thumb_w / image.width
        thumb = image.resize((thumb_w, int(image.height * ratio)))
        thumbs.append((label, thumb))
    cell_h = max(thumb.height for _, thumb in thumbs) + label_h
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap, rows * cell_h + (rows + 1) * gap), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    for idx, (label, thumb) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = gap + col * (thumb_w + gap)
        y = gap + row * (cell_h + gap)
        draw.text((x, y), label.title(), fill=COLORS["neutral"], font=font)
        sheet.paste(thumb, (x, y + label_h))
    out = paths["publication_figures"] / "case_study_contact_sheet.jpg"
    sheet.save(out, quality=94)
    return out


def make_latex_bundle(paths: dict[str, Path], table_paths: list[Path]) -> Path:
    content = [
        "% Auto-generated table bundle.",
        "% Requires: \\usepackage{booktabs}",
        "",
    ]
    for path in table_paths:
        content.append(rf"\input{{{path.as_posix()}}}")
    bundle = paths["latex_tables"] / "all_tables.tex"
    bundle.write_text("\n\n".join(content), encoding="utf-8")
    return bundle


def generate_publication_assets(config_path: str | Path = "configs/default.yaml") -> list[Path]:
    set_publication_style()
    paths = publication_dirs(config_path)
    written: list[Path] = []
    table_paths = [
        make_main_metric_table(paths),
        make_size_table(paths),
        make_threshold_table(paths),
        make_per_class_table(paths),
    ]
    written.extend(table_paths)
    written.append(make_latex_bundle(paths, table_paths))
    for maker in [
        make_main_metric_figure,
        make_threshold_figure,
        make_per_class_delta_figure,
        make_crowded_figure,
        make_fpn_distribution_figure,
        make_head_stats_figure,
    ]:
        written.extend(maker(paths))
    contact = make_case_contact_sheet(paths)
    if contact:
        written.append(contact)
    manifest = paths["root"] / "publication_assets_manifest.md"
    manifest.write_text(
        dedent(
            """\
            # Publication Assets

            Generated by `python -m fcos_report.publication_assets`.

            - `outputs/publication_figures/`: PDF/SVG figures and case-study contact sheet.
            - `outputs/latex_tables/`: booktabs-style LaTeX tables.
            """
        ),
        encoding="utf-8",
    )
    written.append(manifest)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication-quality figures and LaTeX tables from experiment outputs.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    for path in generate_publication_assets(args.config):
        print(path)


if __name__ == "__main__":
    main()
