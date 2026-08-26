from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.index.search import load_embeddings
from phrase_lab.storage.manifest import save_json

from .assign import format_token
from .clustering import assign_to_centroids, fit_spherical_codebook, l2_normalize_rows
from .manifest import atomic_write_parquet, hash_dict, hash_file, codebook_root, space_root, vocabulary_root
from .metrics import compute_codebook_metrics


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def discover_codebooks(root: str | Path, space: str, version: str = "003") -> list[Path]:
    base = space_root(root, space, version)
    if not base.exists():
        return []
    out = []
    for k_dir in sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("k")]):
        for cb in sorted([p for p in k_dir.iterdir() if p.is_dir()]):
            if (cb / "codebook_manifest.json").exists():
                out.append(cb)
    return out


def load_codebook(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "centroids": np.load(path / "centroids.npy"),
        "assignments": pd.read_parquet(path / "assignments.parquet"),
        "cluster_stats": pd.read_parquet(path / "cluster_stats.parquet"),
        "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
        "manifest": json.loads((path / "codebook_manifest.json").read_text(encoding="utf-8")),
    }


def _align_vectors(root: Path, eligible: pd.DataFrame, space: str) -> np.ndarray:
    embeddings = load_embeddings(root / "index")
    phrase_ids = np.asarray(embeddings["phrase_ids"], dtype=str)
    id_to_index = {phrase_id: idx for idx, phrase_id in enumerate(phrase_ids.tolist())}
    vectors = np.asarray(embeddings[space], dtype=np.float32)
    indices = np.asarray([id_to_index[p] for p in eligible["phrase_id"].astype(str).tolist()], dtype=int)
    return vectors[indices]


def build_codebook(
    root: str | Path,
    cfg: dict[str, Any],
    space: str,
    k: int,
    max_phrases: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    root = Path(root)
    seed = int(seed if seed is not None else cfg["experiment"]["seed"])
    eligible = pd.read_parquet(vocabulary_root(root) / "eligible_phrases.parquet")
    if eligible.empty:
        raise ValueError("prepare-vocabulary-data must run first")
    if max_phrases is not None and len(eligible) > max_phrases:
        eligible = eligible.sort_values(["score_id", "start_q", "phrase_id"], kind="mergesort").head(max_phrases).copy()
    vectors = l2_normalize_rows(_align_vectors(root, eligible, space))
    fit = fit_spherical_codebook(
        vectors,
        k=int(k),
        seed=seed,
        batch_size=int(cfg["clustering"]["batch_size"]),
        max_iter=int(cfg["clustering"]["max_iter"]),
        n_init=int(cfg["clustering"]["n_init"]),
        reassignment_ratio=float(cfg["clustering"]["reassignment_ratio"]),
        fit_sample_size=cfg["clustering"].get("fit_sample_size"),
    )
    assignments = assign_to_centroids(vectors, fit.centroids)
    token_cfg = cfg["export"]
    width = int(token_cfg["token_width"])
    assignment_df = eligible.copy().reset_index(drop=True)
    assignment_df["cluster_id"] = assignments["cluster_id"]
    assignment_df["token"] = [format_token(space, cid, width=width, cfg=token_cfg) for cid in assignment_df["cluster_id"].tolist()]
    assignment_df["cosine_to_centroid"] = assignments["cosine_to_centroid"]
    assignment_df["second_best_cosine"] = assignments["second_best_cosine"]
    assignment_df["assignment_margin"] = assignments["assignment_margin"]
    assignment_df["rank_within_cluster_by_centroid_similarity"] = 0
    for _, grp in assignment_df.groupby("cluster_id", sort=False):
        order = grp["cosine_to_centroid"].rank(method="first", ascending=False).astype(int)
        assignment_df.loc[grp.index, "rank_within_cluster_by_centroid_similarity"] = order.to_numpy()
    metrics, cluster_stats = compute_codebook_metrics(assignment_df, fit.centroids, vectors)
    cluster_stats = cluster_stats.sort_values(["size", "mean_cosine"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    assignment_df = assignment_df.sort_values(["cluster_id", "rank_within_cluster_by_centroid_similarity", "phrase_id"], kind="mergesort").reset_index(drop=True)
    codebook_state = {
        "space": space,
        "k": int(k),
        "seed": seed,
        "max_phrases": max_phrases,
        "fit_sample_size": cfg["clustering"].get("fit_sample_size"),
        "algorithm": str(cfg["clustering"]["algorithm"]),
        "dataset_manifest_hash": hash_dict(json.loads((vocabulary_root(root) / "dataset_manifest.json").read_text(encoding="utf-8"))),
        "config_hash": hash_dict(cfg),
        "phrase_ids_hash": hashlib.sha1("\n".join(assignment_df["phrase_id"].astype(str).tolist()).encode("utf-8")).hexdigest(),
    }
    codebook_id = hashlib.sha1(json.dumps(codebook_state, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    out_dir = codebook_root(root, space, int(k), codebook_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "centroids.npy", fit.centroids.astype(np.float32))
    atomic_write_parquet(out_dir / "assignments.parquet", assignment_df)
    atomic_write_parquet(out_dir / "cluster_stats.parquet", cluster_stats)
    save_json(out_dir / "metrics.json", metrics)
    manifest = {
        **codebook_state,
        "codebook_id": codebook_id,
        "phrase_checksum": hash_file(vocabulary_root(root) / "eligible_phrases.parquet"),
        "centroid_checksum": hashlib.sha1(fit.centroids.tobytes()).hexdigest(),
        "assignment_count": int(len(assignment_df)),
        "centroid_count": int(len(fit.centroids)),
        "embedding_dimensions": int(vectors.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    save_json(out_dir / "codebook_manifest.json", manifest)
    from .stability import build_stability_report

    stability = build_stability_report(root, cfg, space, int(k), sample_size=cfg["stability"].get("sample_size"))
    save_json(out_dir / "stability_metrics.json", stability)
    return {"codebook_id": codebook_id, "path": str(out_dir), "metrics": metrics, "manifest": manifest, "cluster_stats": cluster_stats}


def build_vocabulary_sweep(root: str | Path, cfg: dict[str, Any], space: str) -> list[dict[str, Any]]:
    return [build_codebook(root, cfg, space, int(k)) for k in cfg["spaces"][space]["cluster_sizes"]]
