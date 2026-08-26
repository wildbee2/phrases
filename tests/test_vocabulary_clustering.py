from __future__ import annotations

import numpy as np

from phrase_lab.vocabulary.clustering import assign_to_centroids, fit_spherical_codebook


def test_spherical_clustering_and_assignment():
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    fit = fit_spherical_codebook(vectors, k=2, seed=42, batch_size=2, max_iter=20, n_init=2)
    assert fit.centroids.shape == (2, 2)
    assert np.all(np.isfinite(fit.centroids))
    assert np.allclose(np.linalg.norm(fit.centroids, axis=1), 1.0, atol=1e-5)
    assigned = assign_to_centroids(vectors, fit.centroids)
    assert set(assigned["cluster_id"]) <= {0, 1}
    assert np.all(assigned["cosine_to_centroid"] <= 1.0 + 1e-6)
    assert np.all(assigned["assignment_margin"] >= -1e-6)

