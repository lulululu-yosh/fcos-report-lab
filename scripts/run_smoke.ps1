$ErrorActionPreference = "Stop"
python -m fcos_report.make_figures --all
python -m fcos_report.demo_images --image-dir samples
