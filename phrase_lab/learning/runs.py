from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def learning_root(root: str | Path) -> Path:
    return Path(root) / "learning"


def version_root(root: str | Path, version: str = "voice_v1") -> Path:
    return learning_root(root) / version


def runs_root(root: str | Path) -> Path:
    return Path(root) / "runs" / "002_contrastive_encoder"


def make_run_id(config_hash: str, timestamp: datetime | None = None) -> str:
    timestamp = timestamp or datetime.now(timezone.utc)
    return f"{timestamp.strftime('%Y%m%dT%H%M%SZ')}_{config_hash[:8]}"


def run_dir(root: str | Path, run_id: str) -> Path:
    return runs_root(root) / run_id


def discover_runs(root: str | Path) -> list[Path]:
    base = runs_root(root)
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name)

