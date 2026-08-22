from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class PhraseStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.phrases_path = self.root / "extracted" / "phrases.parquet"
        self._phrases: pd.DataFrame | None = None

    def _load(self) -> pd.DataFrame:
        if self._phrases is None:
            self._phrases = pd.read_parquet(self.phrases_path)
        return self._phrases

    def get_phrase(self, phrase_id: str) -> dict[str, Any]:
        df = self._load()
        row = df[df["phrase_id"] == phrase_id]
        if row.empty:
            raise KeyError(phrase_id)
        return row.iloc[0].to_dict()

    def get_score_phrases(self, score_id: str) -> pd.DataFrame:
        df = self._load()
        return df[df["score_id"] == score_id].sort_values(["start_q", "phrase_id"]).reset_index(drop=True)

    def search_metadata(self, title: str = "", composer: str = "", instrument: str = "", genre: str = "") -> pd.DataFrame:
        df = self._load()
        mask = pd.Series(True, index=df.index)
        if title:
            mask &= df["title"].fillna("").str.contains(title, case=False, na=False)
        if composer:
            mask &= df["composer_name"].fillna("").str.contains(composer, case=False, na=False)
        if instrument:
            mask &= df["instrument_name"].fillna("").str.contains(instrument, case=False, na=False)
        if genre:
            mask &= df["genres"].fillna("").astype(str).str.contains(genre, case=False, na=False)
        return df[mask].reset_index(drop=True)

    def get_notes(self, phrase_id: str) -> list[dict[str, Any]]:
        return self.get_phrase(phrase_id)["notes_json"]

    def get_dataframe(self) -> pd.DataFrame:
        return self._load().copy()

