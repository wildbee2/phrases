from __future__ import annotations

from pathlib import Path


def raw_root(root: str | Path) -> Path:
    return Path(root)


def extracted_root(root: str | Path) -> Path:
    return Path(root) / "extracted"


def index_root(root: str | Path) -> Path:
    return Path(root) / "index"


def reviews_root(root: str | Path) -> Path:
    return Path(root) / "reviews"

