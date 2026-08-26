from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .clustering import l2_normalize_rows


def _quantiles(values: np.ndarray, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> dict[str, float]:
    if len(values) == 0:
        return {f"q{int(q * 100)}": 0.0 for q in qs}
    return {f"q{int(q * 100)}": float(np.quantile(values, q)) for q in qs}


def compute_codebook_metrics(assignments: pd.DataFrame, centroids: np.ndarray, vectors: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    if assignments.empty:
        metrics = {
            "occupancy": {"nonempty_clusters": 0},
            "cohesion": {},
            "assignment_margin": {},
            "quantization_error": {},
            "effective_vocabulary": {"entropy": 0.0, "effective_vocab": 0.0, "effective_fraction": 0.0},
        }
        return metrics, pd.DataFrame(columns=["cluster_id", "token", "size"])
    vectors = l2_normalize_rows(vectors)
    centroids = l2_normalize_rows(centroids)
    cluster_sizes = assignments.groupby("cluster_id").size().sort_index()
    token_counts = assignments["token"].value_counts().sort_index()
    size_values = cluster_sizes.to_numpy(dtype=np.int64)
    occupancy = {
        "nonempty_clusters": int((cluster_sizes > 0).sum()),
        "min_cluster_size": int(size_values.min()) if len(size_values) else 0,
        "median_cluster_size": float(np.median(size_values)) if len(size_values) else 0.0,
        "mean_cluster_size": float(np.mean(size_values)) if len(size_values) else 0.0,
        "max_cluster_size": int(size_values.max()) if len(size_values) else 0,
        "size_quantiles": {f"q{q}": float(v) for q, v in zip([1, 5, 10, 25, 50, 75, 90, 95, 99], np.quantile(size_values, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))} if len(size_values) else {},
        "fraction_phrases_in_largest_1pct_clusters": 0.0,
        "fraction_phrases_in_largest_5pct_clusters": 0.0,
        "fraction_phrases_in_largest_10pct_clusters": 0.0,
        "fraction_clusters_lt_5": float((size_values < 5).mean()) if len(size_values) else 0.0,
        "fraction_clusters_lt_10": float((size_values < 10).mean()) if len(size_values) else 0.0,
        "fraction_clusters_lt_25": float((size_values < 25).mean()) if len(size_values) else 0.0,
    }
    if len(size_values):
        sorted_sizes = np.sort(size_values)[::-1]
        total = float(sorted_sizes.sum())
        for pct, key in [(0.01, "fraction_phrases_in_largest_1pct_clusters"), (0.05, "fraction_phrases_in_largest_5pct_clusters"), (0.10, "fraction_phrases_in_largest_10pct_clusters")]:
            n = max(1, int(math.ceil(len(sorted_sizes) * pct)))
            occupancy[key] = float(sorted_sizes[:n].sum() / total) if total else 0.0
    cluster_stats = assignments.groupby("cluster_id").agg(
        token=("token", "first"),
        size=("phrase_id", "size"),
        mean_cosine=("cosine_to_centroid", "mean"),
        median_cosine=("cosine_to_centroid", "median"),
        p10_cosine=("cosine_to_centroid", lambda s: float(np.quantile(s.to_numpy(), 0.10))),
        mean_margin=("assignment_margin", "mean"),
        median_margin=("assignment_margin", "median"),
    ).reset_index()
    cosine_values = assignments["cosine_to_centroid"].to_numpy(dtype=np.float32)
    margin_values = assignments["assignment_margin"].to_numpy(dtype=np.float32)
    quant_error = 1.0 - cosine_values
    probs = token_counts.to_numpy(dtype=np.float64) / max(1, int(token_counts.sum()))
    entropy = float(-(probs * np.log(np.maximum(probs, 1e-12))).sum())
    effective_vocab = float(np.exp(entropy))
    metrics = {
        "occupancy": occupancy,
        "cohesion": {
            "mean": float(np.mean(cosine_values)),
            "median": float(np.median(cosine_values)),
            "p10": float(np.quantile(cosine_values, 0.10)),
        },
        "assignment_margin": {
            "mean": float(np.mean(margin_values)),
            "median": float(np.median(margin_values)),
            "quantiles": _quantiles(margin_values),
        },
        "quantization_error": {
            "mean": float(np.mean(quant_error)),
            "median": float(np.median(quant_error)),
            "quantiles": _quantiles(quant_error),
        },
        "effective_vocabulary": {
            "entropy": entropy,
            "effective_vocab": effective_vocab,
            "effective_fraction": float(effective_vocab / max(1, len(token_counts))),
        },
    }
    return metrics, cluster_stats


def compute_stability_metrics(partitions: list[np.ndarray]) -> dict[str, Any]:
    from itertools import combinations
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    pairs = []
    for a, b in combinations(range(len(partitions)), 2):
        pairs.append(
            {
                "seed_a": a,
                "seed_b": b,
                "ari": float(adjusted_rand_score(partitions[a], partitions[b])),
                "nmi": float(normalized_mutual_info_score(partitions[a], partitions[b])),
            }
        )
    if not pairs:
        return {"pairwise": [], "mean_ari": 1.0, "mean_nmi": 1.0}
    return {
        "pairwise": pairs,
        "mean_ari": float(np.mean([p["ari"] for p in pairs])),
        "mean_nmi": float(np.mean([p["nmi"] for p in pairs])),
    }


def compute_joint_token_stats(df: pd.DataFrame, melody_col: str = "melody_token", rhythm_col: str = "rhythm_token") -> dict[str, Any]:
    if df.empty:
        return {
            "observed_pairs": 0,
            "joint_entropy": 0.0,
            "most_frequent_pairs": [],
            "highest_melody_diversity": [],
            "highest_rhythm_diversity": [],
            "approx_mutual_information": 0.0,
        }
    pair_counts = df.groupby([melody_col, rhythm_col]).size().sort_values(ascending=False)
    melody_counts = df[melody_col].value_counts()
    rhythm_counts = df[rhythm_col].value_counts()
    pair_probs = pair_counts.to_numpy(dtype=np.float64) / float(pair_counts.sum())
    joint_entropy = float(-(pair_probs * np.log(np.maximum(pair_probs, 1e-12))).sum())
    mel_probs = melody_counts.to_numpy(dtype=np.float64) / float(melody_counts.sum())
    rhy_probs = rhythm_counts.to_numpy(dtype=np.float64) / float(rhythm_counts.sum())
    Hm = float(-(mel_probs * np.log(np.maximum(mel_probs, 1e-12))).sum())
    Hr = float(-(rhy_probs * np.log(np.maximum(rhy_probs, 1e-12))).sum())
    mi = float(Hm + Hr - joint_entropy)
    melody_div = df.groupby(melody_col)[rhythm_col].nunique().sort_values(ascending=False)
    rhythm_div = df.groupby(rhythm_col)[melody_col].nunique().sort_values(ascending=False)
    return {
        "observed_pairs": int(len(pair_counts)),
        "joint_entropy": joint_entropy,
        "most_frequent_pairs": [{"melody_token": m, "rhythm_token": r, "count": int(c)} for (m, r), c in pair_counts.head(20).items()],
        "highest_melody_diversity": [{"melody_token": t, "rhythm_diversity": int(v)} for t, v in melody_div.head(20).items()],
        "highest_rhythm_diversity": [{"rhythm_token": t, "melody_diversity": int(v)} for t, v in rhythm_div.head(20).items()],
        "approx_mutual_information": mi,
    }


def compute_sequence_stats(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"sequence_count": 0}
    lens = df["sequence_length"].astype(int).to_numpy()
    return {
        "sequence_count": int(len(df)),
        "median_phrases_per_sequence": float(np.median(lens)),
        "length_quantiles": _quantiles(lens.astype(float), qs=(0.1, 0.5, 0.9)),
    }

