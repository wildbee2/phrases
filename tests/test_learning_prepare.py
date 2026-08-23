from __future__ import annotations

from pathlib import Path

import pandas as pd

from phrase_lab.learning.prepare import prepare_encoder_dataset


def _config():
    return {
        "experiment": {"name": "002_contrastive_encoder", "seed": 42},
        "dataset": {"extraction_modes": ["explicit_voice"], "require_no_license_conflict": True, "min_notes": 2, "max_notes": 8, "min_bars": 1.0, "max_bars": 12.0, "max_phrases": None, "sampling": "random"},
        "split": {"train_fraction": 0.5, "val_fraction": 0.25, "test_fraction": 0.25, "split_unit": "score_id", "seed": 42},
        "tokenizer": {"max_notes": 8, "relative_pitch_clip": 48, "interval_clip": 24, "onset_bins": 16, "duration_bins": 16, "ioi_bins": 16},
        "augmentation": {},
        "positive_mining": {"enabled": True, "baseline_space": "melody", "minimum_similarity": 0.92, "reciprocal_top_k": 3, "min_length_ratio": 0.75, "max_length_ratio": 1.33, "same_part_only": True, "max_pairs_per_phrase": 2},
        "model": {"d_model": 16, "n_layers": 2, "n_heads": 2, "ff_multiplier": 2, "dropout": 0.1, "embedding_dim": 8, "max_notes": 8},
        "training": {"epochs": 1, "batch_size": 2, "gradient_accumulation_steps": 1, "learning_rate": 1e-3, "min_learning_rate": 1e-5, "weight_decay": 0.0, "temperature": 0.07, "mined_positive_probability": 0.3, "num_workers": 0, "amp": False, "gradient_clip_norm": 1.0, "early_stopping_patience": 1, "checkpoint_every_epochs": 1},
        "evaluation": {"fixed_query_count": 10, "neighbors": 5, "exclude_same_score": True, "formal_candidate_scope": "test", "baseline_space": "combined", "human_primary_neighbors": 5},
        "index": {"exact_index_max_phrases": 100, "hnsw_m": 16},
    }


def test_dataset_filtering_and_split_leakage(tmp_path: Path):
    root = tmp_path
    df = pd.DataFrame(
        [
            {"phrase_id": "p1", "score_id": "s1", "part_id": "1", "voice_id": "1", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "n_bars": 2.0, "n_notes": 3, "notes_json": '[{"p": 60, "o": 0.0, "d": 1.0, "v": 80}, {"p": 62, "o": 1.0, "d": 1.0, "v": 80}, {"p": 64, "o": 2.0, "d": 1.0, "v": 80}]'},
            {"phrase_id": "p2", "score_id": "s1", "part_id": "1", "voice_id": "1", "subset:no_license_conflict": True, "extraction_mode": "skyline", "n_bars": 2.0, "n_notes": 3, "notes_json": '[{"p": 60, "o": 0.0, "d": 1.0, "v": 80}]'},
            {"phrase_id": "p3", "score_id": "s2", "part_id": "1", "voice_id": "1", "subset:no_license_conflict": False, "extraction_mode": "explicit_voice", "n_bars": 2.0, "n_notes": 3, "notes_json": '[{"p": 60, "o": 0.0, "d": 1.0, "v": 80}]'},
            {"phrase_id": "p4", "score_id": "s3", "part_id": "1", "voice_id": "1", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "n_bars": 2.0, "n_notes": 3, "notes_json": "not json"},
            {"phrase_id": "p5", "score_id": "s4", "part_id": "1", "voice_id": "1", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "n_bars": 2.0, "n_notes": 3, "notes_json": '[{"p": 60, "o": 0.0, "d": 1.0, "v": 80}, {"p": 62, "o": 1.0, "d": 1.0, "v": 80}, {"p": 64, "o": 2.0, "d": 1.0, "v": 80}]'},
        ]
    )
    (root / "extracted").mkdir(parents=True)
    df.to_parquet(root / "extracted" / "phrases.parquet", index=False)

    prepare_encoder_dataset(root, _config())

    base = root / "learning" / "voice_v1"
    meta = pd.read_parquet(base / "phrase_metadata.parquet")
    rejected = pd.read_csv(base / "rejected_phrases.csv")
    assert set(meta["phrase_id"]) == {"p1", "p5"}
    assert "p2" not in set(meta["phrase_id"])
    assert "p3" not in set(meta["phrase_id"])
    assert set(rejected["phrase_id"]) == {"p4"}
    assert "missing_or_malformed_notes_json" in set(rejected.loc[rejected["phrase_id"] == "p4", "reason"])
    split_by_score = meta.groupby("score_id")["split"].nunique()
    assert split_by_score.max() == 1
    assignments = pd.read_parquet(base / "split_assignments.parquet")
    assert set(assignments["phrase_id"]) == set(meta["phrase_id"])
