from __future__ import annotations

from pathlib import Path

import pandas as pd

from phrase_lab.app.similarity_review import build_visible_trial_payload, random_ab_assignment
from phrase_lab.learning.evaluate import _wilson_interval, summarize_blind_votes


def test_blind_assignment_and_summary(tmp_path: Path):
    rng = __import__("numpy").random.default_rng(0)
    a, b = random_ab_assignment(rng)
    assert sorted([a, b]) == ["handcrafted", "learned"]

    trial = {"trial_id": "t1", "query_phrase_id": "q1", "system_a_backend": "handcrafted", "system_b_backend": "learned", "comparison_tempo": 100.0, "match_starting_pitch": True}
    payload = build_visible_trial_payload(trial)
    assert "system_a_backend" not in payload
    assert payload["query_phrase_id"] == "q1"

    root = tmp_path
    (root / "reviews").mkdir(parents=True)
    pd.DataFrame(
        [
            {"timestamp": "2026-08-23T00:00:00Z", "trial_id": "t1", "query_phrase_id": "q1", "run_id": "r1", "winner_backend": "learned", "tie": False, "both_poor": False},
            {"timestamp": "2026-08-23T00:01:00Z", "trial_id": "t2", "query_phrase_id": "q1", "run_id": "r1", "winner_backend": "handcrafted", "tie": False, "both_poor": False},
            {"timestamp": "2026-08-23T00:02:00Z", "trial_id": "t3", "query_phrase_id": "q2", "run_id": "r1", "winner_backend": "", "tie": True, "both_poor": False},
        ]
    ).to_csv(root / "reviews" / "encoder_blind_votes.csv", index=False)
    summary = summarize_blind_votes(root, "r1", None)
    assert summary["unique_queries"] == 2
    assert summary["learned_wins"] == 1
    assert summary["baseline_wins"] == 0
    assert summary["ties"] == 1
    assert summary["decisive"] == 1
    lo, hi = _wilson_interval(1, 1)
    assert 0.0 <= lo <= hi <= 1.0
