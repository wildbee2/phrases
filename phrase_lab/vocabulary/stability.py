from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .clustering import assign_to_centroids, fit_spherical_codebook, l2_normalize_rows
from .manifest import atomic_write_json, vocabulary_root
from .metrics import compute_stability_metrics


def build_stability_report(root: str | Path, cfg: dict[str, Any], space: str, k: int, sample_size: int | None = None) -> dict[str, Any]:
    root = Path(root)
    eligible = pd.read_parquet(vocabulary_root(root) / "eligible_phrases.parquet")
    vectors = np.load(root / "index" / f"{space}_embeddings.npy")
    ids = np.load(root / "index" / "phrase_ids.npy", allow_pickle=True).astype(str)
    id_to_index = {phrase_id: idx for idx, phrase_id in enumerate(ids.tolist())}
    indices = np.asarray([id_to_index[p] for p in eligible["phrase_id"].astype(str)], dtype=int)
    vectors = l2_normalize_rows(np.asarray(vectors[indices], dtype=np.float32))
    if sample_size is not None and sample_size < len(vectors):
        rng = np.random.default_rng(42)
        vectors = vectors[rng.choice(len(vectors), size=int(sample_size), replace=False)]
    partitions = []
    for seed in cfg["stability"]["repeated_seeds"]:
        fit = fit_spherical_codebook(
            vectors,
            int(k),
            seed=int(seed),
            batch_size=int(cfg["clustering"]["batch_size"]),
            max_iter=int(cfg["clustering"]["max_iter"]),
            n_init=int(cfg["clustering"]["n_init"]),
            reassignment_ratio=float(cfg["clustering"]["reassignment_ratio"]),
            fit_sample_size=min(len(vectors), int(sample_size) if sample_size is not None else len(vectors)),
        )
        partitions.append(assign_to_centroids(vectors, fit.centroids)["cluster_id"])
    stability = compute_stability_metrics(partitions)
    atomic_write_json(vocabulary_root(root) / space / f"k{k}" / "stability_metrics.json", stability)
    return stability

