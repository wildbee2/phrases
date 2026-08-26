from __future__ import annotations

from pathlib import Path

import pandas as pd

from phrase_lab.vocabulary.codebook import build_codebook
from phrase_lab.vocabulary.export import export_phrase_sequences
from phrase_lab.vocabulary.prepare import prepare_vocabulary_data


def test_codebook_assignment_is_phrase_id_stable(synthetic_vocab_root):
    root = synthetic_vocab_root["root"]
    cfg = synthetic_vocab_root["cfg"]
    prepare_vocabulary_data(root, cfg)
    eligible = pd.read_parquet(root / "vocabulary" / "003" / "eligible_phrases.parquet").sample(frac=1.0, random_state=7).reset_index(drop=True)
    eligible.to_parquet(root / "vocabulary" / "003" / "eligible_phrases.parquet", index=False)
    result = build_codebook(root, cfg, "melody", 2)
    assignments = pd.read_parquet(Path(result["path"]) / "assignments.parquet")
    assert set(assignments["phrase_id"]) == {"p1", "p2", "p3"}
    assert set(assignments["token"]) <= {"M_0000", "M_0001"}


def test_phrase_sequence_export_groups_and_sorts():
    df = pd.DataFrame(
        [
            {"score_id": "s1", "part_id": "b", "voice_id": "2", "phrase_id": "p2", "melody_token": "M_0002", "rhythm_token": "R_0001", "start_q": 3.0, "end_q": 4.0},
            {"score_id": "s1", "part_id": "b", "voice_id": "2", "phrase_id": "p1", "melody_token": "M_0001", "rhythm_token": "R_0000", "start_q": 1.0, "end_q": 2.0},
            {"score_id": "s1", "part_id": "a", "voice_id": "1", "phrase_id": "p3", "melody_token": "M_0003", "rhythm_token": "R_0002", "start_q": 0.0, "end_q": 1.0},
        ]
    )
    out = export_phrase_sequences("/tmp", df)
    assert list(out["part_id"]) == ["a", "b"]
    seq = out[out["part_id"] == "b"].iloc[0]
    assert seq["phrase_ids"] == ["p1", "p2"]
    assert seq["melody_tokens"] == ["M_0001", "M_0002"]
