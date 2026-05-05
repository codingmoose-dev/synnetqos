from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f: json.dump(obj, f, indent=4)

def sha256_of_file(path: str | Path) -> str:
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): sha.update(chunk)
    return sha.hexdigest()

def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

def read_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".csv": raise ValueError(f"read_csv() expected a .csv file, got: {path}")
    return pd.read_csv(path)

def read_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() != ".parquet": raise ValueError(f"read_parquet() expected a .parquet file, got: {path}")
    return pd.read_parquet(path)

def save_plot(fig: plt.Figure, path: str | Path, dpi: int = 300) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, format=path.suffix.lstrip("."), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def path_exists(path: str | Path) -> bool:
    return Path(path).exists()
