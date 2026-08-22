from __future__ import annotations

import pandas as pd

from phrase_lab.pdmx.metadata import filter_scores


def test_unsafe_rows_are_removed():
    df = pd.DataFrame(
        [
            {"score_id": "a", "subset:no_license_conflict": True, "subset:deduplicated": True, "mxl": "a.mxl"},
            {"score_id": "b", "subset:no_license_conflict": False, "subset:deduplicated": True, "mxl": "b.mxl"},
            {"score_id": "c", "subset:no_license_conflict": True, "subset:deduplicated": False, "mxl": "c.mxl"},
        ]
    )
    out = filter_scores(df, subset="safe_deduplicated")
    assert list(out["score_id"]) == ["a"]

