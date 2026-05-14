from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLE_DIR = Path("assets/paper_tables")


def load_table(name: str, table_dir: Path = TABLE_DIR) -> pd.DataFrame:
    path = table_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Unknown paper table: {path}")
    return pd.read_csv(path)


def available_tables(table_dir: Path = TABLE_DIR) -> list[str]:
    return sorted(path.stem for path in table_dir.glob("*.csv"))


def export_all_tables(output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name in available_tables():
        df = load_table(name)
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)
    return written
