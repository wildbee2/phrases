from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phrase_lab.learning.positive_mining import mine_positive_pairs


def _cfg():
    return {
        "positive_mining": {
            "enabled": True,
            "baseline_space": "melody",
            "minimum_similarity": 0.92,
            "reciprocal_top_k": 3,
            "min_length_ratio": 0.75,
            "max_length_ratio": 1.33,
            "same_part_only": True,
            "max_pairs_per_phrase": 2,
        }
    }


def test_positive_mining_finds_recurrent_pair(tmp_path: Path):
    root = tmp_path
    phrases = pd.DataFrame(
        [
            {"phrase_id": "a", "score_id": "s1", "part_id": "p1", "split": "train", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "start_q": 0.0, "end_q": 2.0, "n_bars": 2.0, "notes_json": [{"p": 60, "o": 0.0, "d": 1.0, "v": 80}]},
            {"phrase_id": "b", "score_id": "s1", "part_id": "p1", "split": "train", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "start_q": 4.0, "end_q": 6.0, "n_bars": 2.0, "notes_json": [{"p": 60, "o": 0.0, "d": 1.0, "v": 80}]},
            {"phrase_id": "c", "score_id": "s1", "part_id": "p2", "split": "train", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "start_q": 8.0, "end_q": 10.0, "n_bars": 2.0, "notes_json": [{"p": 72, "o": 0.0, "d": 1.0, "v": 80}]},
        ]
    )
    (root / "extracted").mkdir(parents=True)
    phrases.to_parquet(root / "extracted" / "phrases.parquet", index=False)
    (root / "index").mkdir(parents=True)
    np.save(root / "index" / "melody_embeddings.npy", np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "index" / "rhythm_embeddings.npy", np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "index" / "combined_embeddings.npy", np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]], dtype=np.float32))
    np.save(root / "index" / "phrase_ids.npy", np.array(["a", "b", "c"], dtype=object))

    out = mine_positive_pairs(root, _cfg())
    assert {("a", "b"), ("b", "a")} & set(zip(out.get("phrase_id_a", []), out.get("phrase_id_b", [])))
    assert not ((out["phrase_id_a"] == "a") & (out["phrase_id_b"] == "c")).any()
