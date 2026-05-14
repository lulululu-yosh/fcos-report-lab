from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/default.yaml")


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML project config."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    root = Path(config["project"]["output_dir"])
    paths = {
        "root": root,
        "metrics": root / "metrics",
        "tables": root / "tables",
        "figures": root / "figures",
        "coco_results": root / "coco_results",
        "report_assets": root / "report_assets",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
