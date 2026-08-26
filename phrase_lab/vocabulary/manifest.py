from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def hash_dict(data: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def hash_file(path: str | Path) -> str | None:
    path = Path(path)
    if not path.exists():
        return None
    return hashlib.sha1(path.read_bytes()).hexdigest()


def vocabulary_root(root: str | Path, version: str = "003") -> Path:
    return Path(root) / "vocabulary" / version


def space_root(root: str | Path, space: str, version: str = "003") -> Path:
    return vocabulary_root(root, version) / space


def codebook_root(root: str | Path, space: str, k: int, codebook_id: str, version: str = "003") -> Path:
    return space_root(root, space, version) / f"k{k}" / codebook_id


def atomic_write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def atomic_write_parquet(path: str | Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

