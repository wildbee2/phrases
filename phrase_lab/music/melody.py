from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable

import music21 as m21

from .parse import iter_score_parts, part_identity
from .types import MelodicLine, NoteEvent


def _note_event_from_element(el: m21.note.NotRest, measure_number: int | None = None) -> NoteEvent:
    tie_type = None
    if getattr(el, "tie", None) is not None:
        tie_type = el.tie.type
    pitch = int(el.pitch.midi) if hasattr(el, "pitch") else None
    return NoteEvent(
        pitch=pitch,
        onset_q=float(el.offset),
        duration_q=float(el.quarterLength),
        velocity=int(getattr(el.volume, "velocity", 80) or 80),
        measure_number=measure_number,
        beat=float(getattr(el, "beat", 0.0)) if getattr(el, "beat", None) is not None else None,
        tie_type=tie_type,
        is_grace=bool(getattr(el, "isGrace", False)),
    )


def _merge_ties(events: list[NoteEvent]) -> list[NoteEvent]:
    if not events:
        return events
    merged: list[NoteEvent] = []
    current = events[0]
    for nxt in events[1:]:
        if current.pitch == nxt.pitch and nxt.tie_type in {"continue", "stop"} and nxt.onset_q <= current.onset_q + current.duration_q + 1e-6:
            current = replace(current, duration_q=(nxt.onset_q + nxt.duration_q) - current.onset_q)
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def extract_explicit_voices(part: m21.stream.Part, score_id: str, part_id: str, part_name: str | None, instrument_name: str | None, min_notes_per_line: int) -> list[MelodicLine]:
    lines: list[MelodicLine] = []
    voices = list(part.recurse().getElementsByClass(m21.stream.Voice))
    if not voices:
        return lines
    for v_idx, voice in enumerate(voices):
        events: list[NoteEvent] = []
        for el in voice.recurse().notesAndRests:
            if isinstance(el, m21.note.Note):
                events.append(_note_event_from_element(el, getattr(el.activeSite, "measureNumber", None)))
            elif isinstance(el, m21.chord.Chord):
                highest = max(el.notes, key=lambda n: n.pitch.midi)
                events.append(_note_event_from_element(highest, getattr(el.activeSite, "measureNumber", None)))
        events = [e for e in events if e.pitch is not None]
        events = _merge_ties(sorted(events, key=lambda e: (e.onset_q, -e.pitch)))
        if len(events) >= min_notes_per_line:
            lines.append(
                MelodicLine(
                    score_id=score_id,
                    part_id=part_id,
                    part_name=part_name,
                    instrument_name=instrument_name,
                    voice_id=f"voice_{v_idx}",
                    extraction_mode="explicit_voice",
                    notes=events,
                    source_path=None,
                )
            )
    return lines


def extract_skyline(part: m21.stream.Part, score_id: str, part_id: str, part_name: str | None, instrument_name: str | None, min_notes_per_line: int) -> list[MelodicLine]:
    events: list[NoteEvent] = []
    by_onset: dict[float, list[m21.note.NotRest]] = defaultdict(list)
    for el in part.recurse().notesAndRests:
        if isinstance(el, m21.note.Note):
            by_onset[float(el.offset)].append(el)
        elif isinstance(el, m21.chord.Chord):
            by_onset[float(el.offset)].append(el)
    current_end = None
    last_pitch = None
    for onset in sorted(by_onset):
        sounding = by_onset[onset]
        pitches = []
        durations = []
        for el in sounding:
            if isinstance(el, m21.note.Note):
                pitches.append(el.pitch.midi)
                durations.append(el.quarterLength)
            else:
                pitches.append(max(n.pitch.midi for n in el.notes))
                durations.append(el.quarterLength)
        pitch = int(max(pitches))
        dur = float(max(durations))
        if current_end is None:
            current_start = onset
            current_end = onset + dur
            last_pitch = pitch
            continue
        if onset <= current_end + 1e-6:
            if pitch == last_pitch:
                current_end = max(current_end, onset + dur)
                continue
            events.append(NoteEvent(pitch=last_pitch, onset_q=float(current_start), duration_q=float(current_end - current_start), velocity=80))
            current_start = onset
            current_end = onset + dur
            last_pitch = pitch
            continue
        events.append(NoteEvent(pitch=last_pitch, onset_q=float(current_start), duration_q=float(current_end - current_start), velocity=80))
        current_start = onset
        current_end = onset + dur
        last_pitch = pitch
    if current_end is not None and last_pitch is not None:
        events.append(NoteEvent(pitch=last_pitch, onset_q=float(current_start), duration_q=float(current_end - current_start), velocity=80))
    events = [e for e in events if e.pitch is not None]
    events = _merge_ties(events)
    if len(events) < min_notes_per_line:
        return []
    return [
        MelodicLine(
            score_id=score_id,
            part_id=part_id,
            part_name=part_name,
            instrument_name=instrument_name,
            voice_id="skyline",
            extraction_mode="skyline",
            notes=events,
            source_path=None,
        )
    ]


def extract_melodic_lines(score: m21.stream.Score, score_id: str, min_notes_per_line: int = 12) -> list[MelodicLine]:
    lines: list[MelodicLine] = []
    for idx, part in enumerate(iter_score_parts(score)):
        part_id, part_name, instrument_name = part_identity(part, idx)
        explicit = extract_explicit_voices(part, score_id, part_id, part_name, instrument_name, min_notes_per_line)
        if explicit:
            lines.extend(explicit)
            continue
        lines.extend(extract_skyline(part, score_id, part_id, part_name, instrument_name, min_notes_per_line))
    return lines
