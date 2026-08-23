from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass
class PreparedLearningDataset:
    root: Path
    version: str
    phrase_metadata: pd.DataFrame
    split_assignments: pd.DataFrame
    tokens: np.ndarray
    dataset_manifest: dict[str, Any]
    token_manifest: dict[str, Any]

    @property
    def phrase_ids(self) -> np.ndarray:
        return self.phrase_metadata["phrase_id"].to_numpy()

    @property
    def score_ids(self) -> np.ndarray:
        return self.phrase_metadata["score_id"].to_numpy()

    @property
    def splits(self) -> np.ndarray:
        return self.phrase_metadata["split"].to_numpy()

    def split_mask(self, split: str) -> np.ndarray:
        return self.phrase_metadata["split"].astype(str).to_numpy() == split

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "version": self.version,
            "n_phrases": int(len(self.phrase_metadata)),
        }


def load_prepared_dataset(root: str | Path, version: str = "voice_v1") -> PreparedLearningDataset:
    root = Path(root)
    base = root / "learning" / version
    phrase_metadata = pd.read_parquet(base / "phrase_metadata.parquet")
    split_assignments = pd.read_parquet(base / "split_assignments.parquet")
    tokens = np.load(base / "tokens.npy", allow_pickle=False)
    import json

    with open(base / "dataset_manifest.json", "r", encoding="utf-8") as f:
        dataset_manifest = json.load(f)
    with open(base / "token_manifest.json", "r", encoding="utf-8") as f:
        token_manifest = json.load(f)
    return PreparedLearningDataset(
        root=root,
        version=version,
        phrase_metadata=phrase_metadata,
        split_assignments=split_assignments,
        tokens=tokens,
        dataset_manifest=dataset_manifest,
        token_manifest=token_manifest,
    )


def score_aware_batches(metadata: pd.DataFrame, batch_size: int, seed: int = 42) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    indices = list(range(len(metadata)))
    rng.shuffle(indices)
    score_to_indices: dict[str, list[int]] = {}
    for idx in indices:
        score_to_indices.setdefault(str(metadata.iloc[idx]["score_id"]), []).append(idx)
    batches: list[list[int]] = []
    current: list[int] = []
    current_scores: set[str] = set()
    for score_id, score_indices in sorted(score_to_indices.items(), key=lambda item: (len(item[1]), item[0])):
        for idx in score_indices:
            if len(current) >= batch_size or (score_id in current_scores and len(current_scores) < len(score_to_indices)):
                batches.append(current)
                current = []
                current_scores = set()
            current.append(idx)
            current_scores.add(score_id)
            if len(current) >= batch_size:
                batches.append(current)
                current = []
                current_scores = set()
    if current:
        batches.append(current)
    return batches

