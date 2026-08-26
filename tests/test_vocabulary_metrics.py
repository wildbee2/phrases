from __future__ import annotations

import math

import numpy as np
import pandas as pd

from phrase_lab.vocabulary.metrics import compute_codebook_metrics, compute_joint_token_stats, compute_sequence_stats


def test_metrics_are_hand_computable():
    assignments = pd.DataFrame(
        [
            {"phrase_id": "p1", "token": "M_0000", "cluster_id": 0, "cosine_to_centroid": 0.9, "assignment_margin": 0.2},
            {"phrase_id": "p2", "token": "M_0000", "cluster_id": 0, "cosine_to_centroid": 0.8, "assignment_margin": 0.1},
            {"phrase_id": "p3", "token": "M_0001", "cluster_id": 1, "cosine_to_centroid": 0.7, "assignment_margin": 0.3},
        ]
    )
    metrics, cluster_stats = compute_codebook_metrics(assignments, np.eye(2, dtype=np.float32), np.eye(2, dtype=np.float32))
    assert metrics["occupancy"]["nonempty_clusters"] == 2
    assert math.isclose(metrics["quantization_error"]["mean"], 0.2, rel_tol=1e-6)
    assert cluster_stats.shape[0] == 2
    joint = compute_joint_token_stats(pd.DataFrame({"melody_token": ["M_0", "M_0", "M_1"], "rhythm_token": ["R_0", "R_1", "R_0"]}))
    assert joint["observed_pairs"] == 3
    assert joint["approx_mutual_information"] >= 0.0
    seq = compute_sequence_stats(pd.DataFrame({"sequence_length": [1, 2, 3]}))
    assert seq["sequence_count"] == 3
    assert seq["median_phrases_per_sequence"] == 2.0

