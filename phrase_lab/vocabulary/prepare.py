from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.index.search import load_embeddings
from phrase_lab.storage.manifest import save_json

from .manifest import atomic_write_parquet, hash_dict, hash_file, vocabulary_root


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _valid_notes(notes: Any) -> bool:
    if isinstance(notes, str):
        import json

        try:
            notes = json.loads(notes)
        except Exception:
            return False
    if not isinstance(notes, list) or not notes:
        return False
    for item in notes:
        if not isinstance(item, dict) or not all(k in item for k in ("p", "o", "d", "v")):
            return False
    return True


def prepare_vocabulary_data(root: str | Path, cfg: dict[str, Any], max_phrases: int | None = None) -> pd.DataFrame:
    root = Path(root)
    phrases_path = root / "extracted" / "phrases.parquet"
    if not phrases_path.exists():
        raise FileNotFoundError(phrases_path)
    phrases = pd.read_parquet(phrases_path)
    if "subset:no_license_conflict" not in phrases.columns:
        raise KeyError("subset:no_license_conflict column is required for Experiment 003")
    mask = phrases["subset:no_license_conflict"].astype(bool) & (phrases["extraction_mode"].astype(str) == "explicit_voice")
    mask &= phrases["notes_json"].apply(_valid_notes)
    mask &= phrases["n_notes"].astype(float).between(float(cfg["dataset"]["min_notes"]), float(cfg["dataset"]["max_notes"]))
    mask &= phrases["n_bars"].astype(float).between(float(cfg["dataset"]["min_bars"]), float(cfg["dataset"]["max_bars"]))
    filtered = phrases.loc[mask].copy()
    if max_phrases is not None and len(filtered) > max_phrases:
        filtered = filtered.sort_values(["score_id", "start_q", "phrase_id"], kind="mergesort").head(max_phrases).copy()
    filtered["phrase_id"] = filtered["phrase_id"].astype(str)
    filtered = filtered.drop_duplicates(subset=["phrase_id"], keep="first").reset_index(drop=True)
    embeddings = load_embeddings(root / "index")
    phrase_ids = np.asarray(embeddings["phrase_ids"], dtype=str)
    id_to_index = {phrase_id: idx for idx, phrase_id in enumerate(phrase_ids.tolist())}
    required_spaces = [space for space in ["melody", "rhythm", "combined"] if bool(cfg.get("spaces", {}).get(space, {}).get("enabled", True))]
    valid_mask = np.ones(len(phrase_ids), dtype=bool)
    for space in required_spaces:
        vecs = np.asarray(embeddings[space], dtype=np.float32)
        if len(vecs) != len(phrase_ids):
            raise ValueError(f"{space} embeddings and phrase_ids are misaligned")
        norms = np.linalg.norm(vecs, axis=1)
        valid_mask &= np.isfinite(vecs).all(axis=1) & np.isfinite(norms) & (norms > 0)
    eligible = filtered[filtered["phrase_id"].isin(set(phrase_ids[valid_mask]))].copy()
    eligible = eligible.drop_duplicates(subset=["phrase_id"], keep="first").reset_index(drop=True)
    if eligible.empty:
        raise ValueError("no eligible explicit_voice phrases remain after filtering")
    eligible["phrase_index"] = eligible["phrase_id"].map(id_to_index).astype(int)
    for space in required_spaces:
        eligible[f"{space}_index"] = eligible["phrase_index"]
        eligible[f"{space}_valid"] = True
    out_dir = vocabulary_root(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(out_dir / "eligible_phrases.parquet", eligible)
    alignment = {
        "source_phrase_checksum": hash_file(phrases_path),
        "source_index_manifest_hash": hash_file(root / "index" / "index_manifest.json"),
        "source_embedding_checksums": {space: hash_file(root / "index" / f"{space}_embeddings.npy") for space in required_spaces},
        "filters": {
            "require_no_license_conflict": bool(cfg["dataset"]["require_no_license_conflict"]),
            "extraction_modes": list(cfg["dataset"]["extraction_modes"]),
            "min_notes": int(cfg["dataset"]["min_notes"]),
            "max_notes": int(cfg["dataset"]["max_notes"]),
            "min_bars": float(cfg["dataset"]["min_bars"]),
            "max_bars": float(cfg["dataset"]["max_bars"]),
            "max_phrases": max_phrases,
        },
        "embedding_dimensions": {space: int(np.asarray(embeddings[space]).shape[1]) for space in required_spaces},
        "eligible_count": int(len(eligible)),
        "rejected_count": int(len(filtered) - len(eligible)),
        "config_hash": hash_dict(cfg),
        "git_commit": _git_commit(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(out_dir / "embedding_alignment.json", alignment)
    save_json(out_dir / "dataset_manifest.json", alignment)
    return eligible
