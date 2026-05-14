#!/usr/bin/env bash
set -euo pipefail

python -m fcos_report.prepare_coco --split val2017
python -m fcos_report.evaluate --model fcos --split val2017 --full
python -m fcos_report.evaluate --model retinanet --split val2017 --full
python -m fcos_report.make_figures --all
