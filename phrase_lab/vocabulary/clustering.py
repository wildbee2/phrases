from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return vectors / norms


@dataclass(frozen=True)
class FitResult:
    centroids: np.ndarray
    fitted_on: int
    sample_indices: np.ndarray


def fit_spherical_codebook(
    vectors: np.ndarray,
    k: int,
    seed: int = 42,
    batch_size: int = 8192,
    max_iter: int = 300,
    n_init: int = 3,
    reassignment_ratio: float = 0.01,
    fit_sample_size: int | None = None,
) -> FitResult:
    vectors = l2_normalize_rows(vectors)
    if len(vectors) == 0:
        raise ValueError("cannot fit a codebook on zero vectors")
    rng = np.random.default_rng(seed)
    if fit_sample_size is not None and fit_sample_size < len(vectors):
        sample_indices = rng.choice(len(vectors), size=int(fit_sample_size), replace=False)
        fit_vectors = vectors[sample_indices]
    else:
        sample_indices = np.arange(len(vectors))
        fit_vectors = vectors
    model = MiniBatchKMeans(
        n_clusters=int(k),
        batch_size=int(batch_size),
        max_iter=int(max_iter),
        n_init=int(n_init),
        reassignment_ratio=float(reassignment_ratio),
        random_state=int(seed),
    )
    model.fit(fit_vectors)
    centroids = l2_normalize_rows(model.cluster_centers_.astype(np.float32))
    return FitResult(centroids=centroids, fitted_on=int(len(fit_vectors)), sample_indices=np.asarray(sample_indices))


def assign_to_centroids(vectors: np.ndarray, centroids: np.ndarray) -> dict[str, np.ndarray]:
    vectors = l2_normalize_rows(vectors)
    centroids = l2_normalize_rows(centroids)
    if len(vectors) == 0:
        return {
            "cluster_id": np.zeros((0,), dtype=int),
            "cosine_to_centroid": np.zeros((0,), dtype=np.float32),
            "second_best_cosine": np.zeros((0,), dtype=np.float32),
            "assignment_margin": np.zeros((0,), dtype=np.float32),
        }
    scores = vectors @ centroids.T
    if centroids.shape[0] == 1:
        best = np.zeros(len(vectors), dtype=int)
        best_cos = scores[:, 0]
        second = np.full(len(vectors), -1.0, dtype=np.float32)
    else:
        top2 = np.argpartition(scores, kth=max(0, centroids.shape[0] - 2), axis=1)[:, -2:]
        top2_scores = np.take_along_axis(scores, top2, axis=1)
        order = np.argsort(top2_scores, axis=1)
        second = top2_scores[np.arange(len(vectors)), order[:, 0]]
        best = top2[np.arange(len(vectors)), order[:, 1]]
        best_cos = top2_scores[np.arange(len(vectors)), order[:, 1]]
    margin = best_cos - second
    return {
        "cluster_id": best.astype(int),
        "cosine_to_centroid": best_cos.astype(np.float32),
        "second_best_cosine": second.astype(np.float32),
        "assignment_margin": margin.astype(np.float32),
    }

