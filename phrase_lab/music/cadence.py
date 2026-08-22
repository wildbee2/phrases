from __future__ import annotations

from typing import Optional

import music21 as m21


def estimate_key_from_notes(notes: list[int]) -> tuple[str | None, str | None]:
    if not notes:
        return None, None
    try:
        s = m21.stream.Stream([m21.note.Note(n) for n in notes[:64]])
        ks = s.analyze("key")
        return ks.tonicPitchNameWithCase, ks.mode
    except Exception:
        return None, None


def cadence_strength(prev_pitch: int | None, end_pitch: int | None, inferred_key: str | None, inferred_mode: str | None) -> tuple[float, str | None]:
    if prev_pitch is None or end_pitch is None or inferred_key is None:
        return 0.0, None
    try:
        tonic_pc = m21.pitch.Pitch(inferred_key).pitchClass
    except Exception:
        return 0.0, None
    end_pc = end_pitch % 12
    prev_pc = prev_pitch % 12
    hits = {
        "2->1": prev_pc == (tonic_pc + 2) % 12 and end_pc == tonic_pc,
        "7->1": prev_pc == (tonic_pc + 11) % 12 and end_pc == tonic_pc,
        "4->3": prev_pc == (tonic_pc + 5) % 12 and end_pc == (tonic_pc + 4) % 12,
        "5->1": prev_pc == (tonic_pc + 7) % 12 and end_pc == tonic_pc,
    }
    for label, ok in hits.items():
        if ok:
            return 1.0, label
    if end_pc == tonic_pc:
        return 0.5, "tonic"
    if end_pc == (tonic_pc + 7) % 12:
        return 0.3, "dominant"
    return 0.0, None

