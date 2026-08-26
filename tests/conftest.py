from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def make_vocabulary_config() -> dict:
    return {
        "experiment": {"name": "003_discrete_phrase_vocabulary", "seed": 42},
        "dataset": {
            "extraction_modes": ["explicit_voice"],
            "require_no_license_conflict": True,
            "min_notes": 2,
            "max_notes": 8,
            "min_bars": 1.0,
            "max_bars": 12.0,
            "max_phrases": None,
        },
        "spaces": {
            "melody": {"enabled": True, "cluster_sizes": [2], "primary_cluster_size": 2},
            "rhythm": {"enabled": True, "cluster_sizes": [2], "primary_cluster_size": 2},
            "combined": {"enabled": True, "cluster_sizes": [2], "primary_cluster_size": 2, "role": "comparison_only"},
        },
        "clustering": {"algorithm": "spherical_minibatch_kmeans", "batch_size": 4, "max_iter": 20, "n_init": 2, "reassignment_ratio": 0.01, "min_cluster_size_for_eval": 1, "fit_sample_size": None},
        "stability": {"enabled": True, "repeated_seeds": [42, 43, 44], "sample_size": 100},
        "evaluation": {"random_members_per_cluster": 2, "clusters_to_review_per_size": 4, "centroid_members_per_cluster": 2, "hard_negative_centroid_rank_max": 10, "blind_trials_target": 10},
        "human_gate": {"min_reviewed_clusters": 2, "minimum_coherent_fraction": 0.65, "minimum_decisive_trials": 5},
        "export": {"token_prefix_melody": "M", "token_prefix_rhythm": "R", "token_prefix_combined": "C", "token_width": 4},
    }


@pytest.fixture()
def synthetic_vocab_root(tmp_path: Path):
    root = tmp_path / "pdmx"
    (root / "extracted").mkdir(parents=True)
    (root / "index").mkdir(parents=True)
    phrases = pd.DataFrame(
        [
            {"phrase_id": "p1", "score_id": "s1", "part_id": "A", "voice_id": "1", "title": "T1", "composer_name": "C1", "instrument_name": "Piano", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "start_q": 0.0, "end_q": 2.0, "start_measure": 1, "end_measure": 2, "n_bars": 2.0, "n_notes": 3, "genres": "classical", "notes_json": json.dumps([{"p": 60, "o": 0.0, "d": 1.0, "v": 80}, {"p": 62, "o": 1.0, "d": 1.0, "v": 80}, {"p": 64, "o": 2.0, "d": 1.0, "v": 80}])},
            {"phrase_id": "p2", "score_id": "s1", "part_id": "A", "voice_id": "1", "title": "T1", "composer_name": "C1", "instrument_name": "Piano", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "start_q": 3.0, "end_q": 5.0, "start_measure": 3, "end_measure": 4, "n_bars": 2.0, "n_notes": 3, "genres": "classical", "notes_json": json.dumps([{"p": 61, "o": 0.0, "d": 1.0, "v": 80}, {"p": 63, "o": 1.0, "d": 1.0, "v": 80}, {"p": 65, "o": 2.0, "d": 1.0, "v": 80}])},
            {"phrase_id": "p3", "score_id": "s2", "part_id": "B", "voice_id": "1", "title": "T2", "composer_name": "C2", "instrument_name": "Violin", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "start_q": 0.0, "end_q": 2.0, "start_measure": 1, "end_measure": 2, "n_bars": 2.0, "n_notes": 3, "genres": "jazz", "notes_json": json.dumps([{"p": 72, "o": 0.0, "d": 1.0, "v": 80}, {"p": 74, "o": 1.0, "d": 1.0, "v": 80}, {"p": 76, "o": 2.0, "d": 1.0, "v": 80}])},
            {"phrase_id": "p4", "score_id": "s3", "part_id": "C", "voice_id": "1", "title": "T3", "composer_name": "C3", "instrument_name": "Flute", "subset:no_license_conflict": False, "extraction_mode": "explicit_voice", "start_q": 0.0, "end_q": 2.0, "start_measure": 1, "end_measure": 2, "n_bars": 2.0, "n_notes": 3, "genres": "folk", "notes_json": json.dumps([{"p": 55, "o": 0.0, "d": 1.0, "v": 80}, {"p": 57, "o": 1.0, "d": 1.0, "v": 80}, {"p": 59, "o": 2.0, "d": 1.0, "v": 80}])},
            {"phrase_id": "p5", "score_id": "s4", "part_id": "D", "voice_id": "1", "title": "T4", "composer_name": "C4", "instrument_name": "Piano", "subset:no_license_conflict": True, "extraction_mode": "skyline", "start_q": 0.0, "end_q": 2.0, "start_measure": 1, "end_measure": 2, "n_bars": 2.0, "n_notes": 3, "genres": "folk", "notes_json": json.dumps([{"p": 50, "o": 0.0, "d": 1.0, "v": 80}, {"p": 52, "o": 1.0, "d": 1.0, "v": 80}, {"p": 54, "o": 2.0, "d": 1.0, "v": 80}])},
            {"phrase_id": "p6", "score_id": "s5", "part_id": "E", "voice_id": "1", "title": "T5", "composer_name": "C5", "instrument_name": "Piano", "subset:no_license_conflict": True, "extraction_mode": "explicit_voice", "start_q": 0.0, "end_q": 2.0, "start_measure": 1, "end_measure": 2, "n_bars": 2.0, "n_notes": 3, "genres": "folk", "notes_json": "not json"},
        ]
    )
    phrases.to_parquet(root / "extracted" / "phrases.parquet", index=False)
    ids = np.array(["p1", "p2", "p3"], dtype=object)
    melody = np.array([[1.0, 0.0], [0.98, 0.1], [0.0, 1.0]], dtype=np.float32)
    rhythm = np.array([[0.9, 0.1], [1.0, 0.0], [0.1, 0.9]], dtype=np.float32)
    combined = np.array([[0.95, 0.05], [0.94, 0.06], [0.05, 0.95]], dtype=np.float32)
    np.save(root / "index" / "phrase_ids.npy", ids)
    np.save(root / "index" / "melody_embeddings.npy", melody)
    np.save(root / "index" / "rhythm_embeddings.npy", rhythm)
    np.save(root / "index" / "combined_embeddings.npy", combined)
    (root / "index" / "index_manifest.json").write_text(json.dumps({"kind": "synthetic"}), encoding="utf-8")
    return {"root": root, "cfg": make_vocabulary_config(), "phrases": phrases}

