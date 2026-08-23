from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _notes_list(notes_json: Any) -> list[dict[str, Any]]:
    if notes_json is None:
        return []
    if isinstance(notes_json, list):
        return notes_json
    return list(notes_json)


def contour_vector(notes_json: list[dict[str, Any]], steps: int = 32) -> np.ndarray:
    notes = _notes_list(notes_json)
    if not notes:
        return np.zeros(steps, dtype=np.float32)
    pitches = np.array([float(n["p"]) for n in notes], dtype=np.float32)
    rel = pitches - pitches[0]
    times = np.array([float(n["o"]) for n in notes], dtype=np.float32)
    end = max((float(n["o"]) + float(n["d"]) for n in notes), default=0.0)
    grid = np.linspace(0, max(end, 1e-6), steps)
    vals = np.interp(grid, times, rel, left=rel[0], right=rel[-1])
    return vals.astype(np.float32)


def _phrase_duration(notes_json: list[dict[str, Any]]) -> float:
    notes = _notes_list(notes_json)
    return max((float(n["o"]) + float(n["d"]) for n in notes), default=0.0)


def interval_features(notes_json: list[dict[str, Any]], clip: int = 12) -> np.ndarray:
    notes = _notes_list(notes_json)
    if len(notes) < 2:
        return np.zeros((2 * clip + 5,), dtype=np.float32)
    pitches = np.array([int(n["p"]) for n in notes], dtype=np.int32)
    iv = np.diff(pitches)
    hist = np.zeros(2 * clip + 1, dtype=np.float32)
    for v in iv:
        hist[int(np.clip(v, -clip, clip)) + clip] += 1.0
    hist /= max(1.0, hist.sum())
    mean_abs = np.mean(np.abs(iv))
    max_leap = np.max(np.abs(iv))
    step_frac = np.mean(np.abs(iv) <= 2)
    dir_change = np.mean(np.sign(iv[:-1]) != np.sign(iv[1:])) if len(iv) > 1 else 0.0
    extra = np.array([mean_abs, max_leap, step_frac, dir_change, len(iv)], dtype=np.float32)
    return np.concatenate([hist, extra]).astype(np.float32)


def relative_pitch_class_profile(notes_json: list[dict[str, Any]]) -> np.ndarray:
    notes = _notes_list(notes_json)
    if not notes:
        return np.zeros(12, dtype=np.float32)
    pitches = np.array([int(n["p"]) for n in notes], dtype=np.int32)
    rel = (pitches - pitches[0]) % 12
    weights = np.array([float(n["d"]) for n in notes], dtype=np.float32)
    hist = np.zeros(12, dtype=np.float32)
    for pc, w in zip(rel, weights):
        hist[int(pc)] += float(w)
    hist /= max(1e-6, hist.sum())
    return hist.astype(np.float32)


def rhythm_vector(notes_json: list[dict[str, Any]], steps: int = 32) -> np.ndarray:
    notes = _notes_list(notes_json)
    if not notes:
        return np.zeros(steps * 2 + 8, dtype=np.float32)
    end = _phrase_duration(notes)
    grid = np.linspace(0, max(end, 1e-6), steps)
    onset = np.zeros(steps, dtype=np.float32)
    activity = np.zeros(steps, dtype=np.float32)
    for n in notes:
        o = float(n["o"])
        d = float(n["d"])
        onset[min(steps - 1, int(np.floor(o / max(end, 1e-6) * (steps - 1))))] += 1.0
        start = int(np.floor(o / max(end, 1e-6) * (steps - 1)))
        stop = int(np.ceil((o + d) / max(end, 1e-6) * (steps - 1)))
        activity[max(0, start):min(steps, stop + 1)] += 1.0
    onset /= max(1.0, onset.sum())
    activity /= max(1.0, activity.max())
    onset_times = np.array([float(n["o"]) for n in notes], dtype=np.float32)
    iois = np.diff(onset_times) / max(end, 1e-6) if len(onset_times) > 1 else np.array([1.0], dtype=np.float32)
    dur = np.array([float(n["d"]) for n in notes], dtype=np.float32) / max(end, 1e-6)
    rest_fraction = max(0.0, 1.0 - float(np.sum(dur)) / max(end, 1e-6))
    stats = np.array(
        [
            float(np.mean(iois)) if len(iois) else 0.0,
            float(np.std(iois)) if len(iois) else 0.0,
            float(np.mean(dur)),
            float(np.std(dur)),
            rest_fraction,
            float(len(notes)) / max(end, 1e-6),
            float(np.mean(np.abs(np.diff(onset_times)))) if len(onset_times) > 2 else 0.0,
            float(np.mean(onset > 0)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([onset, activity, stats]).astype(np.float32)


def phrase_shape_descriptors(notes_json: list[dict[str, Any]], n_bars: float) -> np.ndarray:
    notes = _notes_list(notes_json)
    if not notes:
        return np.zeros(7, dtype=np.float32)
    pitches = np.array([int(n["p"]) for n in notes], dtype=np.float32)
    iv = np.diff(pitches) if len(pitches) > 1 else np.array([0], dtype=np.float32)
    return np.array(
        [
            np.log1p(len(notes)),
            float(np.max(pitches) - np.min(pitches)),
            float(np.mean(np.abs(iv))),
            float(np.mean(np.sign(iv[:-1]) != np.sign(iv[1:]))) if len(iv) > 1 else 0.0,
            float(max(0.0, 1.0 - np.sum([float(n["d"]) for n in notes]) / max(1e-6, max(float(n["o"]) + float(n["d"]) for n in notes)))),
            float(n_bars),
            float(notes[-1]["d"]) / max(1e-6, float(notes[0]["d"])),
        ],
        dtype=np.float32,
    )
