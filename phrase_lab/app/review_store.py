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

