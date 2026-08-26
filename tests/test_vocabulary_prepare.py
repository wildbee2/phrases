from __future__ import annotations

import pandas as pd

from phrase_lab.vocabulary.prepare import prepare_vocabulary_data


def test_prepare_vocabulary_filters_and_alignment(synthetic_vocab_root):
    root = synthetic_vocab_root["root"]
    cfg = synthetic_vocab_root["cfg"]
    eligible = prepare_vocabulary_data(root, cfg)
    assert set(eligible["phrase_id"]) == {"p1", "p2", "p3"}
    assert "p4" not in set(eligible["phrase_id"])
    assert "p5" not in set(eligible["phrase_id"])
    assert "p6" not in set(eligible["phrase_id"])
    out_dir = root / "vocabulary" / "003"
    assert (out_dir / "eligible_phrases.parquet").exists()
    assert (out_dir / "embedding_alignment.json").exists()
    saved = pd.read_parquet(out_dir / "eligible_phrases.parquet")
    assert set(saved["phrase_id"]) == {"p1", "p2", "p3"}

