from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import ensure_output_dirs, load_config
from .paper_tables import export_all_tables, load_table
from .visualization import (
    PALETTE,
    apply_style,
    save_bar,
    save_centerness_heatmap,
    save_fpn_assignment,
    save_grouped_metrics,
    save_radar,
)


def _write_summary(paths: dict[str, Path]) -> Path:
    summary = """# FCOS Report Assets

Generated assets include paper-table visualizations, FCOS/RetinaNet comparison charts, center-ness heatmap, and FPN assignment diagram.

Use these figures as PPT experiment material. Full COCO evaluation metrics, when generated, are stored in `outputs/metrics`.
"""
    path = paths["report_assets"] / "summary.md"
    path.write_text(summary, encoding="utf-8")
    return path


def make_all(config_path: str | Path = "configs/default.yaml") -> list[Path]:
    config = load_config(config_path)
    paths = ensure_output_dirs(config)
    apply_style(config["figures"]["style"])
    dpi = int(config["figures"]["dpi"])
    written: list[Path] = []

    export_all_tables(paths["tables"])

    bpr = load_table("bpr")
    bpr["label"] = bpr["method"] + " " + bpr["matching_rule"].fillna("")
    written.append(save_bar(bpr, "label", "bpr", "Best Possible Recall (paper Table 1)", paths["figures"] / "paper_bpr.png", PALETTE["fcos"]))

    amb = load_table("ambiguity")
    amb["FPN"] = amb["with_fpn"].map({True: "with FPN", False: "no FPN"})
    written.append(save_grouped_metrics(amb, "FPN", ["ambiguous_samples", "ambiguous_samples_diff_class"], "Ambiguous Positive Samples (paper Table 2)", paths["figures"] / "paper_ambiguity.png"))

    minival = load_table("fcos_retinanet_minival")
    written.append(save_grouped_metrics(minival.head(5), "method", ["ap", "ap50", "ap75", "aps", "apm", "apl"], "FCOS vs RetinaNet on minival (paper Table 3)", paths["figures"] / "paper_minival_compare.png"))
    written.append(save_radar(minival.head(2), "method", ["ap", "ap50", "ap75", "aps", "apm", "apl"], "FCOS vs RetinaNet Metric Shape", paths["figures"] / "paper_fcos_retinanet_radar.png"))

    cent = load_table("centerness_ablation")
    written.append(save_grouped_metrics(cent, "variant", ["ap", "ap50", "ap75", "aps", "apm", "apl"], "Center-ness Ablation (paper Table 4)", paths["figures"] / "paper_centerness_ablation.png"))

    sota = load_table("sota_testdev").sort_values("ap", ascending=False).head(10)
    written.append(save_bar(sota, "method", "ap", "SOTA Single-scale COCO test-dev AP (paper Table 5)", paths["figures"] / "paper_sota_ap.png", PALETTE["accent"]))

    rpn = load_table("rpn_extension")
    written.append(save_grouped_metrics(rpn, "method", ["ar100", "ar1k"], "FCOS as RPN Proposal Generator (paper Table 6)", paths["figures"] / "paper_rpn_extension.png"))

    pr = load_table("class_agnostic_pr")
    written.append(save_grouped_metrics(pr, "method", ["ap50", "ap75", "ap90"], "Class-agnostic AP by IoU Threshold (paper Table 7)", paths["figures"] / "paper_pr_thresholds.png"))

    written.append(save_centerness_heatmap(paths["figures"] / "mechanism_centerness_heatmap.png"))
    written.append(save_fpn_assignment(paths["figures"] / "mechanism_fpn_assignment.png"))
    written.append(_write_summary(paths))

    manifest = {"figures": [str(p) for p in written], "dpi": dpi}
    manifest_path = paths["report_assets"] / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(manifest_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PPT-ready FCOS report figures.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--all", action="store_true", help="Generate all figures.")
    args = parser.parse_args()
    if not args.all:
        raise SystemExit("Use --all to generate the full figure set.")
    for path in make_all(args.config):
        print(path)


if __name__ == "__main__":
    main()
