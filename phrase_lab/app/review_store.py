from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def append_review(root: str | Path, phrase_id: str, label: str, note: str, segmentation_config_hash: str) -> None:
    root = Path(root)
    path = root / "reviews" / "phrase_reviews.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phrase_id": phrase_id,
                "label": label,
                "note": note,
                "segmentation_config_hash": segmentation_config_hash,
            }
        ]
    )
    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, index=False)


def append_blind_vote(root: str | Path, row: dict[str, Any]) -> None:
    root = Path(root)
    path = root / "reviews" / "encoder_blind_votes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def append_similarity_review(root: str | Path, row: dict[str, Any]) -> None:
    root = Path(root)
    path = root / "reviews" / "phrase_similarity_reviews.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def append_vocabulary_cluster_review(root: str | Path, row: dict[str, Any]) -> None:
    root = Path(root)
    path = root / "reviews" / "vocabulary_cluster_reviews.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)


def append_vocabulary_blind_trial(root: str | Path, row: dict[str, Any]) -> None:
    root = Path(root)
    path = root / "reviews" / "vocabulary_blind_trials.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        frame.to_csv(path, mode="a", header=False, index=False)
    else:
        frame.to_csv(path, index=False)
