from __future__ import annotations

import math
from functools import lru_cache
from io import BytesIO
from typing import Any

import numpy as np
import mido

try:
    import pretty_midi
except Exception:  # pragma: no cover
    pretty_midi = None


def notes_to_pretty_midi(notes_json: list[dict[str, Any]], bpm: float = 100.0, transpose: int = 0, target_start_pitch: int | None = None) -> pretty_midi.PrettyMIDI:
    if pretty_midi is None:
        raise RuntimeError("pretty_midi is not available")
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)
    if not notes_json:
        pm.instruments.append(inst)
        return pm
    pitch_shift = transpose
    if target_start_pitch is not None:
        pitch_shift = target_start_pitch - int(notes_json[0]["p"])
    for n in notes_json:
        pitch = int(n["p"]) + pitch_shift
        pitch = max(0, min(127, pitch))
        start = float(n["o"]) * 60.0 / bpm
        end = start + float(n["d"]) * 60.0 / bpm
        inst.notes.append(pretty_midi.Note(velocity=int(n.get("v") or 80), pitch=pitch, start=start, end=max(end, start + 0.05)))
    pm.instruments.append(inst)
    return pm


def synthesize_phrase(notes_json: list[dict[str, Any]], sample_rate: int = 22050, bpm: float = 100.0, transpose: int = 0, target_start_pitch: int | None = None) -> tuple[np.ndarray, int]:
    duration = (max((n["o"] + n["d"] for n in notes_json), default=1.0) * 60.0 / bpm) + 0.4
    if pretty_midi is not None:
        pm = notes_to_pretty_midi(notes_json, bpm=bpm, transpose=transpose, target_start_pitch=target_start_pitch)
        try:
            audio = pm.synthesize(fs=sample_rate)
        except Exception:
            audio = None
    else:
        audio = None
    if audio is None:
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        audio = np.zeros_like(t, dtype=np.float32)
        pitch_shift = transpose if target_start_pitch is None else target_start_pitch - int(notes_json[0]["p"])
        for n in notes_json:
            start = int(float(n["o"]) * 60.0 / bpm * sample_rate)
            end = int((float(n["o"]) + float(n["d"])) * 60.0 / bpm * sample_rate)
            pitch = int(n["p"]) + pitch_shift
            freq = 440.0 * (2 ** ((pitch - 69) / 12))
            seg = np.sin(2 * np.pi * freq * t[: max(1, end - start)])
            audio[start:end] += 0.1 * seg[: end - start]
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size:
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak
    pad = int(sample_rate * 0.2)
    audio = np.pad(audio, (pad, pad), mode="constant")
    return audio.astype(np.float32), sample_rate


def phrase_to_midi_bytes(notes_json: list[dict[str, Any]], bpm: float = 100.0) -> bytes:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    ticks_per_second = midi.ticks_per_beat * bpm / 60.0
    events: list[tuple[int, int, str, int]] = []
    for n in notes_json:
        start = int(round(float(n["o"]) * ticks_per_second))
        end = int(round((float(n["o"]) + float(n["d"])) * ticks_per_second))
        pitch = max(0, min(127, int(n["p"])))
        vel = max(1, min(127, int(n.get("v") or 80)))
        events.append((start, 1, "note_on", pitch, vel))  # type: ignore[arg-type]
        events.append((end, 0, "note_off", pitch, 0))  # type: ignore[arg-type]
    events.sort(key=lambda item: (item[0], item[1]))
    last_tick = 0
    for tick, _, kind, pitch, vel in events:
        delta = max(0, tick - last_tick)
        last_tick = tick
        if kind == "note_on":
            track.append(mido.Message("note_on", note=pitch, velocity=vel, time=delta))
        else:
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=delta))
    track.append(mido.MetaMessage("end_of_track", time=0))
    buf = BytesIO()
    midi.save(file=buf)
    return buf.getvalue()
