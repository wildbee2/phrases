from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device | None = None) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def checkpoint_signature(payload: dict[str, Any]) -> str:
    keys = {k: payload.get(k) for k in ["dataset_manifest_hash", "experiment_config_hash", "tokenizer_hash", "model_config_hash", "seed"]}
    return json.dumps(keys, sort_keys=True)

