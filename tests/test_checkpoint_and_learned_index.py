from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from phrase_lab.index.backends import LearnedRetrievalBackend
from phrase_lab.learning.checkpoint import load_checkpoint, save_checkpoint
from phrase_lab.learning.model import EncoderConfig, LearnedPhraseEncoder


def test_checkpoint_round_trip(tmp_path: Path):
    model = LearnedPhraseEncoder(EncoderConfig(d_model=16, n_layers=1, n_heads=2, ff_multiplier=2, dropout=0.0, embedding_dim=8, max_notes=8))
    batch = torch.zeros((1, 8, 5), dtype=torch.long)
    before = model(batch).detach().clone()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, {"model_state": model.state_dict(), "model_config": model.cfg.__dict__})
    loaded = load_checkpoint(path, map_location="cpu")
    restored = LearnedPhraseEncoder(loaded["model_config"])
    restored.load_state_dict(loaded["model_state"])
    after = restored(batch).detach().clone()
    assert torch.allclose(before, after)


def test_learned_retrieval_is_row_order_independent(tmp_path: Path):
    root = tmp_path
    run_id = "run1"
    retrieval = root / "runs" / "002_contrastive_encoder" / run_id / "retrieval"
    retrieval.mkdir(parents=True)
    phrase_df = pd.DataFrame(
        [
            {"phrase_id": "c", "score_id": "s1", "instrument_name": "Piano", "title": "C", "composer_name": "X", "start_measure": 5, "end_measure": 6, "n_bars": 2.0, "n_notes": 4, "split": "test"},
            {"phrase_id": "a", "score_id": "s1", "instrument_name": "Piano", "title": "A", "composer_name": "X", "start_measure": 1, "end_measure": 2, "n_bars": 2.0, "n_notes": 4, "split": "test"},
            {"phrase_id": "b", "score_id": "s2", "instrument_name": "Piano", "title": "B", "composer_name": "X", "start_measure": 3, "end_measure": 4, "n_bars": 2.0, "n_notes": 4, "split": "test"},
        ]
    )
    phrase_df.to_parquet(retrieval / "phrase_metadata.parquet", index=False)
    np.save(retrieval / "embeddings.npy", np.array([[0.0, 0.1], [1.0, 0.0], [0.99, 0.01]], dtype=np.float32))
    np.save(retrieval / "phrase_ids.npy", np.array(["a", "b", "c"], dtype=object))
    np.save(retrieval / "split_labels.npy", np.array(["test", "test", "test"], dtype=object))
    (retrieval / "index_manifest.json").write_text("{}", encoding="utf-8")

    backend = LearnedRetrievalBackend(root, run_id)
    out = backend.search("a", k=2, exclude_same_score=True, candidate_split="test")
    assert list(out["phrase_id"]) == ["b"]
