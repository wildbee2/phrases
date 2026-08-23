from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.index.build_index import build_faiss_index, save_index
from phrase_lab.learning.dataset import load_prepared_dataset
from phrase_lab.learning.embed import encode_batches
from phrase_lab.learning.model import EncoderConfig, LearnedPhraseEncoder
from phrase_lab.learning.checkpoint import load_checkpoint
from phrase_lab.storage.manifest import save_json


def _hash_dict(data: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def build_learned_index(root: str | Path, run_id: str, batch_size: int = 256) -> dict[str, Any]:
    root = Path(root)
    from phrase_lab.learning.runs import run_dir

    run = run_dir(root, run_id)
    prepared = load_prepared_dataset(root)
    ckpt = load_checkpoint(run / "checkpoints" / "best.pt", map_location="cpu")
    model = LearnedPhraseEncoder(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    tokens = prepared.tokens
    phrase_df = prepared.phrase_metadata.copy()
    embeddings = encode_batches(model, tokens, batch_size=batch_size)
    retrieval_dir = run / "retrieval"
    retrieval_dir.mkdir(parents=True, exist_ok=True)
    np.save(retrieval_dir / "embeddings.npy", embeddings.astype(np.float32))
    np.save(retrieval_dir / "phrase_ids.npy", phrase_df["phrase_id"].to_numpy())
    np.save(retrieval_dir / "split_labels.npy", phrase_df["split"].to_numpy())
    phrase_df.to_parquet(retrieval_dir / "phrase_metadata.parquet", index=False)
    index = build_faiss_index(embeddings, exact_max=500000, hnsw_m=32)
    save_index(index, retrieval_dir / "learned.faiss")
    manifest = {
        "run_id": run_id,
        "n_phrases": int(len(phrase_df)),
        "embedding_dim": int(embeddings.shape[1]) if len(embeddings) else 0,
        "dataset_manifest_hash": _hash_dict(prepared.dataset_manifest),
        "token_manifest_hash": _hash_dict(prepared.token_manifest),
        "model_config_hash": _hash_dict(ckpt["model_config"]),
        "checkpoint_path": str(run / "checkpoints" / "best.pt"),
    }
    save_json(retrieval_dir / "index_manifest.json", manifest)
    return manifest
