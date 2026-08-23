from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.features.normalize import l2_normalize
from phrase_lab.index.search import load_embeddings
from phrase_lab.music.piano_roll import phrase_piano_roll, figure_to_png_bytes
from phrase_lab.storage.manifest import save_json
from .dataset import load_prepared_dataset
from .runs import version_root


def _hash_dict(data: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def _notes(row: pd.Series) -> list[dict[str, Any]]:
    from .tokenize import notes_from_any

    return notes_from_any(row.get("notes_json"))


def _overlaps(a: pd.Series, b: pd.Series) -> bool:
    return not (float(a["end_q"]) <= float(b["start_q"]) or float(b["end_q"]) <= float(a["start_q"]))


def mine_positive_pairs(root: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(root)
    try:
        prepared = load_prepared_dataset(root)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "prepared encoder data is required before mining positives; run "
            "`python -m phrase_lab.cli prepare-encoder-data ...` first"
        ) from exc
    phrases = prepared.phrase_metadata.copy()
    if "split" not in phrases.columns:
        raise KeyError("prepared dataset split labels are required before mining positives")
    embeddings = load_embeddings(root / "index")
    ids = np.asarray(embeddings["phrase_ids"], dtype=str)
    vecs = embeddings[cfg["positive_mining"]["baseline_space"]]
    phrases = phrases.copy()
    phrases["phrase_id"] = phrases["phrase_id"].astype(str)
    aligned = phrases.set_index("phrase_id").reindex(ids)
    present = aligned["split"].notna()
    if not bool(present.any()):
        raise ValueError("no phrase ids overlap between the prepared dataset and the embedding index")
    if not bool(present.all()):
        ids = ids[present.to_numpy()]
        vecs = vecs[present.to_numpy()]
        aligned = aligned.loc[present]
    phrases = aligned.reset_index()
    mined: list[dict[str, Any]] = []
    top_k = int(cfg["positive_mining"]["reciprocal_top_k"])
    minimum_similarity = float(cfg["positive_mining"]["minimum_similarity"])
    same_part_only = bool(cfg["positive_mining"]["same_part_only"])
    max_pairs_per_phrase = int(cfg["positive_mining"]["max_pairs_per_phrase"])
    for i, anchor in phrases.iterrows():
        candidates = []
        for j, other in phrases.iterrows():
            if i == j or anchor["score_id"] != other["score_id"] or anchor["split"] != other["split"]:
                continue
            if same_part_only and str(anchor.get("part_id")) != str(other.get("part_id")):
                continue
            if _overlaps(anchor, other):
                continue
            length_ratio = float(other["n_bars"]) / max(1e-6, float(anchor["n_bars"]))
            lo, hi = cfg["positive_mining"]["min_length_ratio"], cfg["positive_mining"]["max_length_ratio"]
            if not (lo <= length_ratio <= hi):
                continue
            sim = float(np.dot(vecs[i], vecs[j]))
            if sim >= minimum_similarity:
                candidates.append((j, sim, length_ratio))
        candidates.sort(key=lambda t: t[1], reverse=True)
        kept = 0
        for j, sim, length_ratio in candidates[:top_k]:
            reciprocal = float(np.dot(vecs[j], vecs[i]))
            if reciprocal < minimum_similarity:
                continue
            if kept >= max_pairs_per_phrase:
                break
            other = phrases.iloc[j]
            mined.append(
                {
                    "phrase_id_a": anchor["phrase_id"],
                    "phrase_id_b": other["phrase_id"],
                    "score_id": anchor["score_id"],
                    "split": anchor["split"],
                    "baseline_similarity": sim,
                    "length_ratio": length_ratio,
                    "same_part": bool(str(anchor.get("part_id")) == str(other.get("part_id"))),
                    "start_q_a": float(anchor["start_q"]),
                    "end_q_a": float(anchor["end_q"]),
                    "start_q_b": float(other["start_q"]),
                    "end_q_b": float(other["end_q"]),
                    "mining_config_hash": _hash_dict(cfg["positive_mining"]),
                }
            )
            kept += 1
    out = pd.DataFrame(mined)
    out_dir = version_root(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "mined_positive_pairs.parquet", index=False)
    save_json(out_dir / "mining_report.json", {"count": int(len(out)), "config_hash": _hash_dict(cfg["positive_mining"])})
    if len(out):
        sample = out.sample(n=min(25, len(out)), random_state=42).reset_index(drop=True)
        md = ["# Mined positive pairs", ""]
        for _, row in sample.iterrows():
            md.append(f"## {row['phrase_id_a']} -> {row['phrase_id_b']}")
            md.append(f"- score_id: {row['score_id']}")
            md.append(f"- similarity: {row['baseline_similarity']:.3f}")
            md.append(f"- length_ratio: {row['length_ratio']:.3f}")
        (out_dir / "mined_positive_pairs_report.md").write_text("\n".join(md), encoding="utf-8")
    return out
