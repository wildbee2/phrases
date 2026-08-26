from __future__ import annotations

from pathlib import Path

import pandas as pd

from phrase_lab.vocabulary.evaluate import summarize_blind_trials, summarize_cluster_reviews
from phrase_lab.vocabulary.manifest import vocabulary_root


def test_vocabulary_summaries_and_status(tmp_path: Path):
    root = tmp_path
    (root / "reviews").mkdir(parents=True)
    cluster_reviews = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "space": "melody", "k": 2, "token": "M_0000", "cluster_size": 2, "sampled_phrase_ids": '["p1"]', "sampling_seed": 42, "rating": "Strongly coherent", "note": "", "codebook_manifest_hash": "x"},
            {"timestamp": "2026-01-02T00:00:00Z", "space": "melody", "k": 2, "token": "M_0001", "cluster_size": 2, "sampled_phrase_ids": '["p2"]', "sampling_seed": 42, "rating": "Not coherent", "note": "", "codebook_manifest_hash": "x"},
        ]
    )
    cluster_reviews.to_csv(root / "reviews" / "vocabulary_cluster_reviews.csv", index=False)
    blind_trials = pd.DataFrame(
        [
            {"timestamp": "2026-01-01T00:00:00Z", "trial_id": "t1", "space": "melody", "k": 2, "query_phrase_id": "p1", "query_token": "M_0000", "candidate_a_phrase_id": "p2", "candidate_b_phrase_id": "p3", "candidate_a_token": "M_0000", "candidate_b_token": "M_0001", "same_cluster_side": "a", "visible_vote": "A", "same_cluster_won": True, "tie": False, "neither": False, "negative_sampling_method": "nearest_other_token", "match_starting_pitch": True, "fixed_tempo": True, "codebook_manifest_hash": "x"},
            {"timestamp": "2026-01-02T00:00:00Z", "trial_id": "t2", "space": "melody", "k": 2, "query_phrase_id": "p2", "query_token": "M_0000", "candidate_a_phrase_id": "p3", "candidate_b_phrase_id": "p4", "candidate_a_token": "M_0001", "candidate_b_token": "M_0000", "same_cluster_side": "b", "visible_vote": "B", "same_cluster_won": True, "tie": False, "neither": False, "negative_sampling_method": "nearest_other_token", "match_starting_pitch": True, "fixed_tempo": True, "codebook_manifest_hash": "x"},
            {"timestamp": "2026-01-03T00:00:00Z", "trial_id": "t3", "space": "melody", "k": 2, "query_phrase_id": "p3", "query_token": "M_0001", "candidate_a_phrase_id": "p5", "candidate_b_phrase_id": "p6", "candidate_a_token": "M_0001", "candidate_b_token": "M_0000", "same_cluster_side": "a", "visible_vote": "Tie", "same_cluster_won": False, "tie": True, "neither": False, "negative_sampling_method": "nearest_other_token", "match_starting_pitch": True, "fixed_tempo": True, "codebook_manifest_hash": "x"},
        ]
    )
    blind_trials.to_csv(root / "reviews" / "vocabulary_blind_trials.csv", index=False)
    cluster = summarize_cluster_reviews(root, "melody", 2)
    assert cluster["unique_clusters_reviewed"] == 2
    blind = summarize_blind_trials(root, "melody", 2)
    assert blind["decisive"] == 2
    assert blind["same_cluster_wins"] == 2
    assert (vocabulary_root(root) / "melody" / "k2" / "human_cluster_summary.json").exists()

