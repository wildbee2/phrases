from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from phrase_lab.music.piano_roll import figure_to_png_bytes, phrase_piano_roll
from phrase_lab.music.render import synthesize_phrase
from phrase_lab.storage.phrase_store import PhraseStore
from phrase_lab.index.search import search_neighbors


@lru_cache(maxsize=128)
def _cached_audio(key: tuple) -> tuple[Any, int]:
    notes_json, bpm, transpose, target_start_pitch = key
    return synthesize_phrase(list(notes_json), bpm=bpm, transpose=transpose, target_start_pitch=target_start_pitch)


def render_phrase_audio(notes_json, bpm: float, transpose: int = 0, target_start_pitch: int | None = None):
    return synthesize_phrase(notes_json, bpm=bpm, transpose=transpose, target_start_pitch=target_start_pitch)


def render_phrase_plot(notes_json, title: str):
    return figure_to_png_bytes(phrase_piano_roll(notes_json, title=title))


def get_phrase_panel(store: PhraseStore, phrase_id: str) -> dict[str, Any]:
    row = store.get_phrase(phrase_id)
    return row


def nearest_neighbors(store: PhraseStore, phrase_id: str, embeddings: dict[str, Any], mode: str, k: int, exclude_same_score: bool, same_instrument: bool, length_ratio: tuple[float, float] | None):
    return search_neighbors(phrase_id, store.get_dataframe(), embeddings, mode=mode, k=k, exclude_same_score=exclude_same_score, same_instrument=same_instrument, length_ratio=length_ratio)

