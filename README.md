# FCOS Report Lab

Teaching-oriented FCOS experiments for a presentation on **FCOS: Fully Convolutional One-Stage Object Detection**. The project does not train from scratch. It uses official `torchvision` pretrained FCOS and RetinaNet models to generate COCO evaluation results and PPT-ready visualizations.

## What This Project Produces

- COCO `val2017` bbox metrics for FCOS and RetinaNet.
- COCO `test2017` / test-dev result JSON export for server submission.
- Paper-table visualizations for BPR, ambiguity, center-ness ablation, SOTA comparison, RPN extension, and AP50/AP75/AP90.
- Side-by-side qualitative detection images for FCOS vs RetinaNet.
- Mechanism visuals: center-ness heatmap and FPN regression range assignment.

Generated outputs are written under `outputs/` and ignored by Git by default.

## Quick Start

```bash
python -m pip install -e ".[dev]"
python -m fcos_report.make_figures --all
```

Add a few `.jpg` or `.png` files to `samples/`, then run:

```bash
python -m fcos_report.demo_images --image-dir samples
```

The first model inference downloads pretrained weights, so run it in Colab or another machine with a good network connection.

## Colab Workflow

Open `notebooks/FCOS_Report_Colab.ipynb` in Colab. For a private repository, authenticate in Colab and clone:

```bash
git clone https://github.com/lulululu-yosh/fcos-report-lab.git
cd fcos-report-lab
python -m pip install -e .
```

Smoke run:

```bash
python -m fcos_report.make_figures --all
python -m fcos_report.prepare_coco --split val2017
python -m fcos_report.evaluate --model fcos --split val2017 --limit 8
python -m fcos_report.evaluate --model retinanet --split val2017 --limit 8
```

Full `val2017` evaluation:

```bash
python -m fcos_report.evaluate --model fcos --split val2017 --full
python -m fcos_report.evaluate --model retinanet --split val2017 --full
```

Export test-dev style JSON:

```bash
python -m fcos_report.export_testdev --model fcos --prepare
```

`test2017` has no public bbox annotations, so this command only exports JSON. Upload the JSON to the COCO evaluation server if test-dev metrics are required.

## Fixed CLI Interfaces

```bash
python -m fcos_report.prepare_coco --split val2017
python -m fcos_report.evaluate --model fcos --split val2017 --full
python -m fcos_report.evaluate --model retinanet --split val2017 --full
python -m fcos_report.export_testdev --model fcos
python -m fcos_report.make_figures --all
python -m fcos_report.demo_images --image-dir samples
python -m fcos_report.benchmark_runtime --model fcos --split val2017 --limit 200
python -m fcos_report.analyze_sizes --model fcos --split val2017
python -m fcos_report.analyze_classes --model fcos --split val2017
python -m fcos_report.threshold_sweep --model fcos --split val2017
python -m fcos_report.analyze_crowded_scenes --model fcos --split val2017
python -m fcos_report.export_case_studies --split val2017
python -m fcos_report.analyze_fcos_internals --split val2017
python -m fcos_report.run_analysis --split val2017
```

## Experiment Code

Run full inference first so the analysis modules can reuse the same result JSON:

```bash
python -m fcos_report.evaluate --model fcos --split val2017 --full
python -m fcos_report.evaluate --model retinanet --split val2017 --full
python -m fcos_report.run_analysis --split val2017
```

The analysis modules generate CSVs and case-study images only when executed:

- `analyze_sizes`: small/medium/large GT matching, recall, precision, mean IoU.
- `analyze_classes`: per-class GT/detection/match stats and optional COCO per-class AP.
- `threshold_sweep`: score threshold sweep from a low-threshold result JSON.
- `analyze_crowded_scenes`: dense and overlapped image statistics.
- `export_case_studies`: FCOS-better, RetinaNet-better, both-good, both-hard, crowded examples.
- `analyze_fcos_internals`: FCOS head tensor stats and approximate FPN-level detection scale distribution.
- `benchmark_runtime`: per-image runtime and final output count statistics.

## Maintenance Notes

- Do not commit COCO images, annotations, pretrained weights, or generated outputs.
- Keep dependency versions compatible with the Colab runtime; prefer updating `torch` and `torchvision` together.
- `assets/paper_tables/` stores extracted paper numbers as CSV so figures can be regenerated deterministically.
- `configs/default.yaml` is the single place for thresholds, paths, and smoke/full evaluation defaults.
- Use `python -m pytest tests` before pushing changes.

## Project Layout

```text
assets/paper_tables/       Paper experiment numbers as CSV
configs/default.yaml       Shared runtime config
notebooks/                 Colab entrypoint
scripts/                   Convenience scripts
src/fcos_report/           Reusable package and CLI modules
tests/                     Lightweight local tests
outputs/                   Generated artifacts, ignored by Git
samples/                   Optional demo images, ignored except .gitkeep
```
