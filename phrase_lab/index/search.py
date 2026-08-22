from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


def load_embeddings(index_dir: str | Path) -> dict[str, np.ndarray]:
    index_dir = Path(index_dir)
    return {
        "melody": np.load(index_dir / "melody_embeddings.npy"),
        "rhythm": np.load(index_dir / "rhythm_embeddings.npy"),
        "combined": np.load(index_dir / "combined_embeddings.npy"),
        "phrase_ids": np.load(index_dir / "phrase_ids.npy", allow_pickle=True),
    }


def load_index(index_dir: str | Path, name: str):
    index_dir = Path(index_dir)
    path = index_dir / f"{name}.faiss"
    if faiss is None or not path.exists():
        return np.load(index_dir / f"{name}_embeddings.npy")
    return faiss.read_index(str(path))


def search_neighbors(
    query_phrase_id: str,
    phrase_df: pd.DataFrame,
    embeddings: dict[str, np.ndarray],
    mode: str = "combined",
    k: int = 10,
    exclude_same_score: bool = True,
    same_instrument: bool = False,
    length_ratio: tuple[float, float] | None = None,
) -> pd.DataFrame:
    ids = list(embeddings["phrase_ids"])
    try:
        idx = ids.index(query_phrase_id)
    except ValueError:
        raise KeyError(query_phrase_id)
    vecs = embeddings[mode]
    q = vecs[idx:idx + 1]
    sims = vecs @ q.T
    order = np.argsort(-sims[:, 0])
    rows = phrase_df.reset_index(drop=True)
    out = []
    for i in order:
        if ids[i] == query_phrase_id:
            continue
        r = rows.iloc[i]
        if exclude_same_score and r["score_id"] == rows.iloc[idx]["score_id"]:
            continue
        if same_instrument and str(r["instrument_name"]) != str(rows.iloc[idx]["instrument_name"]):
            continue
        if length_ratio is not None:
            lo, hi = length_ratio
            ratio = float(r["n_bars"]) / max(1e-6, float(rows.iloc[idx]["n_bars"]))
            if not (lo <= ratio <= hi):
                continue
        out.append(
            {
                "rank": len(out) + 1,
                "similarity": float(sims[i, 0]),
                "phrase_id": r["phrase_id"],
                "title": r.get("title"),
                "composer": r.get("composer_name"),
                "instrument": r.get("instrument_name"),
                "measures": f'{r.get("start_measure")} - {r.get("end_measure")}',
                "bars": float(r.get("n_bars", 0.0)),
                "notes": int(r.get("n_notes", 0)),
                "score_id": r.get("score_id"),
            }
        )
        if len(out) >= k:
            break
    return pd.DataFrame(out)

