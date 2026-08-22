from __future__ import annotations

import numpy as np
import pandas as pd

from phrase_lab.index.search import search_neighbors


def test_self_result_excluded_and_same_score_works():
    df = pd.DataFrame(
        [
            {"phrase_id": "q", "score_id": "s1", "instrument_name": "Piano", "title": "T", "composer_name": "C", "start_measure": 1, "end_measure": 2, "n_bars": 2.0, "n_notes": 4},
            {"phrase_id": "a", "score_id": "s1", "instrument_name": "Piano", "title": "T", "composer_name": "C", "start_measure": 3, "end_measure": 4, "n_bars": 2.0, "n_notes": 4},
            {"phrase_id": "b", "score_id": "s2", "instrument_name": "Piano", "title": "U", "composer_name": "D", "start_measure": 5, "end_measure": 6, "n_bars": 2.0, "n_notes": 4},
        ]
    )
    emb = np.eye(3, dtype=np.float32)
    embeddings = {"combined": emb, "melody": emb, "rhythm": emb, "phrase_ids": np.array(["q", "a", "b"])}
    out = search_neighbors("q", df, embeddings, mode="combined", k=2, exclude_same_score=True)
    assert list(out["phrase_id"]) == ["b"]

