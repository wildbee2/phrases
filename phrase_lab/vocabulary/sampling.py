from __future__ import annotations

import numpy as np
import pandas as pd

from phrase_lab.music.render import synthesize_phrase


def sample_cluster_members(assignments: pd.DataFrame, token: str, mode: str = "random", n: int = 5, seed: int = 42) -> pd.DataFrame:
    df = assignments[assignments["token"].astype(str) == str(token)].copy()
    if df.empty:
        return df.head(0)
    if mode == "centroid-nearest":
        return df.sort_values(["cosine_to_centroid", "phrase_id"], ascending=[False, True], kind="mergesort").head(int(n)).reset_index(drop=True)
    if mode == "low-confidence":
        return df.sort_values(["assignment_margin", "cosine_to_centroid", "phrase_id"], ascending=[True, True, True], kind="mergesort").head(int(n)).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=min(int(n), len(df)), replace=False)
    return df.iloc[idx].reset_index(drop=True)


def build_audio_montage(rows: pd.DataFrame, bpm: float = 100.0, match_start_pitch: bool = True) -> tuple[np.ndarray, int]:
    if rows.empty:
        return np.zeros(0, dtype=np.float32), 22050
    waves = []
    sr = 22050
    silence = np.zeros(int(0.5 * sr), dtype=np.float32)
    for _, row in rows.iterrows():
        notes = row.get("notes_json") if isinstance(row.get("notes_json"), list) else []
        target = int(notes[0]["p"]) if match_start_pitch and notes else None
        audio = synthesize_phrase(notes, bpm=bpm, target_start_pitch=target)
        sr = int(audio[1])
        waves.append(audio[0].astype(np.float32))
        waves.append(silence)
    return np.concatenate(waves) if waves else np.zeros(0, dtype=np.float32), sr


def select_hard_negative(assignments: pd.DataFrame, query_phrase_id: str, query_token: str, phrase_df: pd.DataFrame | None = None, seed: int = 42) -> pd.Series:
    df = assignments[assignments["token"].astype(str) != str(query_token)].copy()
    if phrase_df is not None:
        phrase_df = phrase_df.copy()
        phrase_df["phrase_id"] = phrase_df["phrase_id"].astype(str)
        if "score_id" in phrase_df.columns:
            df = df.merge(phrase_df[["phrase_id", "score_id", "n_bars", "n_notes"]], on="phrase_id", how="left", suffixes=("", "_meta"))
            q_row = phrase_df[phrase_df["phrase_id"] == str(query_phrase_id)].iloc[0]
            df = df[df["score_id"].astype(str) != str(q_row["score_id"])]
            if "n_bars" in df.columns:
                df = df.iloc[(df["n_bars"] - float(q_row["n_bars"])).abs().argsort(kind="mergesort")]
            if "n_notes" in df.columns:
                df = df.iloc[(df["n_notes"] - float(q_row["n_notes"])).abs().argsort(kind="mergesort")]
    if df.empty:
        raise ValueError("no hard negative candidate found")
    return df.sample(n=1, random_state=seed).iloc[0]
