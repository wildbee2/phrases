from __future__ import annotations

import pandas as pd

from phrase_lab.vocabulary.sampling import sample_cluster_members, select_hard_negative


def test_sampling_modes_and_hard_negative():
    assignments = pd.DataFrame(
        [
            {"phrase_id": "p1", "token": "M_0000", "cosine_to_centroid": 0.9, "assignment_margin": 0.4, "score_id": "s1", "n_bars": 2.0, "n_notes": 3},
            {"phrase_id": "p2", "token": "M_0000", "cosine_to_centroid": 0.8, "assignment_margin": 0.2, "score_id": "s2", "n_bars": 2.2, "n_notes": 3},
            {"phrase_id": "p3", "token": "M_0000", "cosine_to_centroid": 0.6, "assignment_margin": 0.1, "score_id": "s3", "n_bars": 2.1, "n_notes": 3},
            {"phrase_id": "p4", "token": "M_0001", "cosine_to_centroid": 0.7, "assignment_margin": 0.3, "score_id": "s4", "n_bars": 2.0, "n_notes": 3},
        ]
    )
    random_a = sample_cluster_members(assignments, "M_0000", mode="random", n=2, seed=7)
    random_b = sample_cluster_members(assignments, "M_0000", mode="random", n=2, seed=7)
    assert list(random_a["phrase_id"]) == list(random_b["phrase_id"])
    nearest = sample_cluster_members(assignments, "M_0000", mode="centroid-nearest", n=1)
    assert nearest.iloc[0]["phrase_id"] == "p1"
    low = sample_cluster_members(assignments, "M_0000", mode="low-confidence", n=1)
    assert low.iloc[0]["phrase_id"] == "p3"
    neg = select_hard_negative(assignments, "p1", "M_0000", assignments[["phrase_id", "score_id", "n_bars", "n_notes"]])
    assert neg["token"] == "M_0001"
    assert neg["score_id"] != "s1"

