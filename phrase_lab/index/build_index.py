from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    import faiss
except Exception:  # pragma: no cover
    faiss = None


def _manifest_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_faiss_index(vectors: np.ndarray, exact_max: int = 500000, hnsw_m: int = 32):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.size == 0:
        return {"kind": "numpy", "vectors": vectors}
    if faiss is None:
        return {"kind": "numpy", "vectors": vectors}
    if len(vectors) <= exact_max:
        index = faiss.IndexFlatIP(vectors.shape[1])
    else:
        index = faiss.IndexHNSWFlat(vectors.shape[1], hnsw_m)
        index.hnsw.efSearch = 64
    index.add(vectors)
    return index


def save_index(index, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(index, dict):
        np.save(path.with_suffix(".npy"), index["vectors"])
        path.write_text("numpy-fallback-index", encoding="utf-8")
    else:
        faiss.write_index(index, str(path))


def build_all_indexes(root: str | Path, phrase_df, feature_cfg: dict[str, Any], index_cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    index_dir = root / "index"
    from phrase_lab.features.build_features import build_embeddings

    embeddings = build_embeddings(phrase_df, feature_cfg, index_dir)
    manifests = {}
    for name in ["melody", "rhythm", "combined"]:
        idx = build_faiss_index(embeddings[name], exact_max=int(index_cfg["exact_index_max_phrases"]), hnsw_m=int(index_cfg["hnsw_m"]))
        save_index(idx, index_dir / f"{name}.faiss")
        manifests[name] = {
            "n_phrases": int(len(phrase_df)),
            "embedding_dim": int(embeddings[name].shape[1]),
            "feature_config_hash": _manifest_hash(feature_cfg),
            "source_phrase_checksum": hashlib.sha1(Path(root / "extracted" / "phrases.parquet").read_bytes()).hexdigest(),
            "index_type": "flatip" if len(phrase_df) <= int(index_cfg["exact_index_max_phrases"]) else "hnsw",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    with open(index_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifests, f, indent=2)
    return manifests
