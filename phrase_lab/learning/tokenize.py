from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

PAD_TOKEN = 0
MASK_TOKEN = 1
REAL_TOKEN_OFFSET = 2


@dataclass(frozen=True)
class TokenizerConfig:
    max_notes: int = 96
    relative_pitch_clip: int = 48
    interval_clip: int = 24
    onset_bins: int = 128
    duration_bins: int = 64
    ioi_bins: int = 64


def _scalar_to_python(value: Any) -> Any:
    if hasattr(value, "as_py"):
        try:
            return value.as_py()
        except Exception:
            pass
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.item()
        except Exception:
            pass
    return value


def notes_from_any(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    value = _scalar_to_python(value)
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return notes_from_any(json.loads(value))
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        return notes_from_any(value.tolist())
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        item = _scalar_to_python(item)
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except Exception:
                continue
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if not isinstance(item, dict):
            continue
        note = {}
        for key in ("p", "o", "d", "v"):
            if key not in item:
                break
            note[key] = _scalar_to_python(item[key])
        else:
            try:
                note["p"] = int(note["p"])
                note["o"] = float(note["o"])
                note["d"] = float(note["d"])
                note["v"] = int(note["v"]) if note["v"] is not None else 80
            except Exception:
                continue
            out.append(note)
    out.sort(key=lambda n: (float(n["o"]), int(n["p"]), float(n["d"])))
    return out


def phrase_duration(notes: list[dict[str, Any]]) -> float:
    return float(max((float(n["o"]) + float(n["d"]) for n in notes), default=0.0))


def _quantize_signed(value: float, clip: int, overflow_low: int, overflow_high: int) -> int:
    if value < -clip:
        return overflow_low
    if value > clip:
        return overflow_high
    return REAL_TOKEN_OFFSET + 2 + (int(value) + clip)


def _quantize_unit(value: float, bins: int) -> int:
    if bins <= 1:
        return REAL_TOKEN_OFFSET
    clipped = min(max(float(value), 0.0), 1.0)
    idx = int(round(clipped * (bins - 1)))
    return REAL_TOKEN_OFFSET + min(max(idx, 0), bins - 1)


def tokenize_phrase(notes_value: Any, cfg: TokenizerConfig | dict[str, Any]) -> tuple[np.ndarray, int, float]:
    if isinstance(cfg, dict):
        cfg = TokenizerConfig(
            max_notes=int(cfg.get("max_notes", 96)),
            relative_pitch_clip=int(cfg.get("relative_pitch_clip", 48)),
            interval_clip=int(cfg.get("interval_clip", 24)),
            onset_bins=int(cfg.get("onset_bins", 128)),
            duration_bins=int(cfg.get("duration_bins", 64)),
            ioi_bins=int(cfg.get("ioi_bins", 64)),
        )
    notes = notes_from_any(notes_value)
    if len(notes) > cfg.max_notes:
        raise ValueError(f"phrase length {len(notes)} exceeds max_notes={cfg.max_notes}")
    duration = phrase_duration(notes)
    if not notes or duration <= 0:
        tokens = np.zeros((cfg.max_notes, 5), dtype=np.uint16)
        return tokens, 0, 0.0
    tokens = np.zeros((cfg.max_notes, 5), dtype=np.uint16)
    prev_onset = float(notes[0]["o"])
    prev_pitch = int(notes[0]["p"])
    for idx, note in enumerate(notes):
        rel_pitch = int(note["p"]) - int(notes[0]["p"])
        interval = 0 if idx == 0 else int(note["p"]) - prev_pitch
        onset = (float(note["o"]) - float(notes[0]["o"])) / duration
        dur = float(note["d"]) / duration
        ioi = 0.0 if idx == 0 else (float(note["o"]) - prev_onset) / duration
        prev_onset = float(note["o"])
        prev_pitch = int(note["p"])
        tokens[idx, 0] = _quantize_signed(rel_pitch, cfg.relative_pitch_clip, 2, 3)
        tokens[idx, 1] = _quantize_signed(interval, cfg.interval_clip, 2, 3)
        tokens[idx, 2] = _quantize_unit(onset, cfg.onset_bins)
        tokens[idx, 3] = _quantize_unit(dur, cfg.duration_bins)
        tokens[idx, 4] = _quantize_unit(ioi, cfg.ioi_bins)
    return tokens, len(notes), duration


def token_manifest_hash(cfg: TokenizerConfig | dict[str, Any]) -> str:
    if isinstance(cfg, TokenizerConfig):
        payload = cfg.__dict__
    else:
        payload = dict(cfg)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))
