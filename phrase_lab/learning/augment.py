from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tokenize import MASK_TOKEN, PAD_TOKEN


@dataclass(frozen=True)
class AugmentationConfig:
    note_mask_probability: float = 0.10
    feature_mask_probability: float = 0.05
    timing_jitter_probability: float = 0.15
    duration_jitter_probability: float = 0.15
    ornament_dropout_probability: float = 0.05
    max_dropout_fraction: float = 0.20


def _real_note_rows(tokens: np.ndarray) -> np.ndarray:
    return np.flatnonzero(tokens[:, 0] != PAD_TOKEN)


def augment_tokens(tokens: np.ndarray, rng: np.random.Generator, cfg: AugmentationConfig | dict[str, Any]) -> np.ndarray:
    if isinstance(cfg, dict):
        cfg = AugmentationConfig(**{k: cfg.get(k, getattr(AugmentationConfig, k)) for k in AugmentationConfig.__annotations__})
    out = np.array(tokens, copy=True)
    rows = _real_note_rows(out)
    if len(rows) <= 2:
        return out
    internal = rows[1:-1]
    if not len(internal):
        return out

    # Whole-note masking.
    for row in internal:
        if rng.random() < cfg.note_mask_probability:
            out[row, :] = MASK_TOKEN

    # Feature masking.
    for row in internal:
        for col in range(out.shape[1]):
            if rng.random() < cfg.feature_mask_probability:
                out[row, col] = MASK_TOKEN

    # Small jitter on timing channels.
    for row in internal:
        if rng.random() < cfg.timing_jitter_probability:
            out[row, 2] = max(2, int(out[row, 2]) + int(rng.choice([-1, 1])))
        if rng.random() < cfg.duration_jitter_probability:
            out[row, 3] = max(2, int(out[row, 3]) + int(rng.choice([-1, 1])))

    # Conservative ornament dropout on internal notes only.
    if rng.random() < cfg.ornament_dropout_probability:
        keep = [rows[0]]
        max_drop = max(0, int(np.floor(len(internal) * cfg.max_dropout_fraction)))
        if max_drop > 0:
            drop_count = int(rng.integers(0, max_drop + 1))
            drop_positions = set(int(x) for x in rng.choice(internal, size=drop_count, replace=False))
        else:
            drop_positions = set()
        keep.extend([row for row in internal if row not in drop_positions])
        keep.append(rows[-1])
        packed = np.zeros_like(out)
        write_idx = 0
        for row in rows:
            if row in drop_positions:
                continue
            packed[write_idx] = out[row]
            write_idx += 1
        out = packed
    return out
