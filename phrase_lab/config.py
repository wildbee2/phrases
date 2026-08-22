from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path("configs/default.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_get(mapping: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def merge_cli_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = yaml.safe_load(yaml.safe_dump(cfg))
    for dotted, value in overrides.items():
        if value is None:
            continue
        cur = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value
    return out


@dataclass(frozen=True)
class ConfigHashable:
    data: dict[str, Any]

    def to_hashable(self) -> str:
        return yaml.safe_dump(self.data, sort_keys=True)

