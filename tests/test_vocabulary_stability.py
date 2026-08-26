from __future__ import annotations

import numpy as np

from phrase_lab.vocabulary.metrics import compute_stability_metrics


def test_stability_metrics_identical_and_permuted_labels():
    identical = compute_stability_metrics([np.array([0, 1, 0]), np.array([0, 1, 0])])
    assert identical["mean_ari"] == 1.0
    assert identical["mean_nmi"] == 1.0
    permuted = compute_stability_metrics([np.array([0, 1, 0]), np.array([1, 0, 1])])
    assert permuted["mean_ari"] == 1.0
    assert permuted["mean_nmi"] == 1.0
    different = compute_stability_metrics([np.array([0, 0, 1]), np.array([0, 1, 1])])
    assert different["mean_ari"] < 1.0

