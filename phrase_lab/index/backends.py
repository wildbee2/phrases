from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from phrase_lab.index.search import load_embeddings, search_neighbors
from phrase_lab.learning.runs import discover_runs
from phrase_lab.storage.manifest import load_json
from phrase_lab.storage.phrase_store import PhraseStore


class RetrievalBackend(Protocol):
    name: str

    def contains(self, phrase_id: str) -> bool: ...

    def search(
        self,
        phrase_id: str,
        k: int,
        exclude_same_score: bool = True,
        same_instrument: bool = False,
        length_ratio: tuple[float, float] | None = None,
        candidate_split: str | None = None,
    ) -> pd.DataFrame: ...


@dataclass
class HandcraftedRetrievalBackend:
    root: Path
    name: str = "handcrafted"

    def __post_init__(self) -> None:
        self.store = PhraseStore(self.root)
        self.phrase_df = self.store.get_dataframe()
        self.embeddings = load_embeddings(self.root / "index")

    def contains(self, phrase_id: str) -> bool:
        return phrase_id in set(self.embeddings["phrase_ids"].tolist())

    def search(
        self,
        phrase_id: str,
        k: int,
        exclude_same_score: bool = True,
        same_instrument: bool = False,
        length_ratio: tuple[float, float] | None = None,
        candidate_split: str | None = None,
    ) -> pd.DataFrame:
        df = self.phrase_df
        if candidate_split is not None and "split" in df.columns:
            df = df[df["split"].astype(str) == candidate_split].reset_index(drop=True)
            allowed_ids = set(df["phrase_id"].astype(str))
            embeddings = {
                "combined": self.embeddings["combined"][np.isin(self.embeddings["phrase_ids"].astype(str), list(allowed_ids))],
                "melody": self.embeddings["melody"][np.isin(self.embeddings["phrase_ids"].astype(str), list(allowed_ids))],
                "rhythm": self.embeddings["rhythm"][np.isin(self.embeddings["phrase_ids"].astype(str), list(allowed_ids))],
                "phrase_ids": self.embeddings["phrase_ids"][np.isin(self.embeddings["phrase_ids"].astype(str), list(allowed_ids))],
            }
        else:
            embeddings = self.embeddings
        return search_neighbors(
            phrase_id,
            df,
            embeddings,
            mode="combined",
            k=k,
            exclude_same_score=exclude_same_score,
            same_instrument=same_instrument,
            length_ratio=length_ratio,
        )


@dataclass
class LearnedRetrievalBackend:
    root: Path
    run_id: str
    name: str = "learned"

    def __post_init__(self) -> None:
        self.run_dir = self.root / "runs" / "002_contrastive_encoder" / self.run_id
        self.retrieval_dir = self.run_dir / "retrieval"
        self.phrase_df = pd.read_parquet(self.retrieval_dir / "phrase_metadata.parquet")
        self.embeddings = np.load(self.retrieval_dir / "embeddings.npy")
        self.phrase_ids = np.load(self.retrieval_dir / "phrase_ids.npy", allow_pickle=True).astype(str)
        self.split_labels = np.load(self.retrieval_dir / "split_labels.npy", allow_pickle=True).astype(str)
        self.id_to_index = {phrase_id: idx for idx, phrase_id in enumerate(self.phrase_ids.tolist())}
        self.manifest = load_json(self.retrieval_dir / "index_manifest.json")

    def contains(self, phrase_id: str) -> bool:
        return str(phrase_id) in self.id_to_index

    def search(
        self,
        phrase_id: str,
        k: int,
        exclude_same_score: bool = True,
        same_instrument: bool = False,
        length_ratio: tuple[float, float] | None = None,
        candidate_split: str | None = None,
    ) -> pd.DataFrame:
        if phrase_id not in self.id_to_index:
            raise KeyError(phrase_id)
        query_idx = self.id_to_index[str(phrase_id)]
        query_vec = self.embeddings[query_idx]
        rows = self.phrase_df.copy()
        if candidate_split is not None and "split" in rows.columns:
            rows = rows[rows["split"].astype(str) == candidate_split]
        allowed_ids = set(rows["phrase_id"].astype(str))
        query_row = self.phrase_df[self.phrase_df["phrase_id"].astype(str) == str(phrase_id)].iloc[0]
        order = np.argsort(-(self.embeddings @ query_vec))
        out = []
        for idx in order:
            cand_id = self.phrase_ids[idx]
            if cand_id == str(phrase_id):
                continue
            if cand_id not in allowed_ids:
                continue
            r = rows[rows["phrase_id"].astype(str) == cand_id].iloc[0]
            if exclude_same_score and r["score_id"] == query_row["score_id"]:
                continue
            if same_instrument and str(r["instrument_name"]) != str(query_row["instrument_name"]):
                continue
            if length_ratio is not None:
                lo, hi = length_ratio
                ratio = float(r["n_bars"]) / max(1e-6, float(query_row["n_bars"]))
                if not (lo <= ratio <= hi):
                    continue
            out.append(
                {
                    "rank": len(out) + 1,
                    "similarity": float(self.embeddings[idx] @ query_vec),
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


def available_learned_runs(root: str | Path) -> list[str]:
    return [p.name for p in discover_runs(root) if (p / "retrieval" / "index_manifest.json").exists()]
