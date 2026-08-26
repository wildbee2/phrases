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


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _checkpoint_state_path(out_dir: Path) -> Path:
    return out_dir / "mining_checkpoint_state.json"


def _checkpoint_data_path(out_dir: Path) -> Path:
    return out_dir / "mined_positive_pairs.checkpoint.parquet"


def _load_checkpoint(out_dir: Path, expected_state: dict[str, Any]) -> tuple[int, int, list[dict[str, Any]]]:
    state_path = _checkpoint_state_path(out_dir)
    data_path = _checkpoint_data_path(out_dir)
    if not state_path.exists() or not data_path.exists():
        return 0, 0, []
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return 0, 0, []
    if any(state.get(key) != value for key, value in expected_state.items()):
        return 0, 0, []
    try:
        mined = pd.read_parquet(data_path)
    except Exception:
        return 0, 0, []
    return int(state.get("next_group_idx", 0)), int(state.get("next_anchor_idx", 0)), mined.to_dict("records")


def _save_checkpoint(
    out_dir: Path,
    expected_state: dict[str, Any],
    next_group_idx: int,
    next_anchor_idx: int,
    mined_rows: list[dict[str, Any]],
) -> None:
    _atomic_write_parquet(pd.DataFrame(mined_rows), _checkpoint_data_path(out_dir))
    state = dict(expected_state)
    state.update(
        {
            "next_group_idx": int(next_group_idx),
            "next_anchor_idx": int(next_anchor_idx),
            "rows": int(len(mined_rows)),
        }
    )
    _atomic_write_json(state, _checkpoint_state_path(out_dir))


def mine_positive_pairs(root: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    root = Path(root)
    prepared = None
    try:
        prepared = load_prepared_dataset(root)
        phrases = prepared.phrase_metadata.copy()
    except FileNotFoundError:
        phrases = pd.read_parquet(root / "extracted" / "phrases.parquet")
    if "split" not in phrases.columns:
        raise KeyError("prepared dataset split labels are required before mining positives")
    embeddings = load_embeddings(root / "index")
    ids = np.asarray(embeddings["phrase_ids"], dtype=str)
    vecs = embeddings[cfg["positive_mining"]["baseline_space"]]
    phrases = phrases.copy()
    phrases["phrase_id"] = phrases["phrase_id"].astype(str)
    phrases = phrases.drop_duplicates(subset=["phrase_id"], keep="first").copy()
    aligned = phrases.set_index("phrase_id").reindex(ids)
    present = aligned["split"].notna()
    if not bool(present.any()):
        raise ValueError("no phrase ids overlap between the prepared dataset and the embedding index")
    if not bool(present.all()):
        ids = ids[present.to_numpy()]
        vecs = vecs[present.to_numpy()]
        aligned = aligned.loc[present]
    phrases = aligned.reset_index()
    phrases["phrase_id"] = phrases["phrase_id"].astype(str)
    phrases["score_id"] = phrases["score_id"].astype(str)
    phrases["split"] = phrases["split"].astype(str)
    if "part_id" in phrases.columns:
        phrases["part_id"] = phrases["part_id"].astype(str)
    phrases["_source_row"] = np.arange(len(phrases), dtype=np.int64)
    phrases = phrases.sort_values(["score_id", "split", "phrase_id"], kind="mergesort").reset_index(drop=True)
    top_k = int(cfg["positive_mining"]["reciprocal_top_k"])
    minimum_similarity = float(cfg["positive_mining"]["minimum_similarity"])
    same_part_only = bool(cfg["positive_mining"]["same_part_only"])
    max_pairs_per_phrase = int(cfg["positive_mining"]["max_pairs_per_phrase"])
    out_dir = version_root(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_state = {
        "config_hash": _hash_dict(cfg["positive_mining"]),
        "dataset_manifest_hash": _hash_dict(prepared.dataset_manifest) if prepared is not None else None,
        "phrase_ids_hash": hashlib.sha1("\n".join(ids.tolist()).encode("utf-8")).hexdigest(),
        "baseline_space": str(cfg["positive_mining"]["baseline_space"]),
    }
    group_start, anchor_start, mined_rows = _load_checkpoint(out_dir, expected_state)
    group_items = list(phrases.groupby(["score_id", "split"], sort=True))
    if group_start >= len(group_items):
        group_start = 0
        anchor_start = 0
        mined_rows = []
    checkpoint_every = int(cfg["positive_mining"].get("checkpoint_every_anchors", 100))
    processed_since_checkpoint = 0
    length_lo = float(cfg["positive_mining"]["min_length_ratio"])
    length_hi = float(cfg["positive_mining"]["max_length_ratio"])
    try:
        for gidx in range(group_start, len(group_items)):
            _, group = group_items[gidx]
            group = group.reset_index(drop=True)
            group_vecs = vecs[group["_source_row"].to_numpy()]
            group_starts = group["start_q"].astype(float).to_numpy()
            group_ends = group["end_q"].astype(float).to_numpy()
            group_bars = group["n_bars"].astype(float).to_numpy()
            group_part_ids = group["part_id"].astype(str).to_numpy() if "part_id" in group.columns else None
            start_idx = anchor_start if gidx == group_start else 0
            for i in range(start_idx, len(group)):
                anchor = group.iloc[i]
                anchor_vec = group_vecs[i]
                sims = group_vecs @ anchor_vec
                mask = np.ones(len(group), dtype=bool)
                mask[i] = False
                if same_part_only and group_part_ids is not None:
                    mask &= group_part_ids == str(anchor.get("part_id"))
                non_overlap = (group_ends <= float(anchor["start_q"])) | (group_starts >= float(anchor["end_q"]))
                mask &= non_overlap
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratios = group_bars / max(1e-6, float(anchor["n_bars"]))
                mask &= (ratios >= length_lo) & (ratios <= length_hi)
                cand_idx = np.flatnonzero(mask & (sims >= minimum_similarity))
                if len(cand_idx):
                    cand_idx = cand_idx[np.argsort(-sims[cand_idx], kind="mergesort")]
                    kept = 0
                    for j in cand_idx[:top_k]:
                        if kept >= max_pairs_per_phrase:
                            break
                        other = group.iloc[j]
                        mined_rows.append(
                            {
                                "phrase_id_a": anchor["phrase_id"],
                                "phrase_id_b": other["phrase_id"],
                                "score_id": anchor["score_id"],
                                "split": anchor["split"],
                                "baseline_similarity": float(sims[j]),
                                "length_ratio": float(ratios[j]),
                                "same_part": bool(str(anchor.get("part_id")) == str(other.get("part_id"))),
                                "start_q_a": float(anchor["start_q"]),
                                "end_q_a": float(anchor["end_q"]),
                                "start_q_b": float(other["start_q"]),
                                "end_q_b": float(other["end_q"]),
                                "mining_config_hash": _hash_dict(cfg["positive_mining"]),
                            }
                        )
                        kept += 1
                processed_since_checkpoint += 1
                if checkpoint_every > 0 and processed_since_checkpoint >= checkpoint_every:
                    _save_checkpoint(out_dir, expected_state, gidx, i + 1, mined_rows)
                    processed_since_checkpoint = 0
            anchor_start = 0
            _save_checkpoint(out_dir, expected_state, gidx + 1, 0, mined_rows)
    except KeyboardInterrupt:
        _save_checkpoint(out_dir, expected_state, gidx if "gidx" in locals() else group_start, i + 1 if "i" in locals() else anchor_start, mined_rows)
        raise
    except Exception:
        _save_checkpoint(out_dir, expected_state, gidx if "gidx" in locals() else group_start, i + 1 if "i" in locals() else anchor_start, mined_rows)
        raise
    out = pd.DataFrame(mined_rows)
    out.to_parquet(out_dir / "mined_positive_pairs.parquet", index=False)
    save_json(out_dir / "mining_report.json", {"count": int(len(out)), "config_hash": _hash_dict(cfg["positive_mining"])})
    for checkpoint_path in [_checkpoint_data_path(out_dir), _checkpoint_state_path(out_dir)]:
        if checkpoint_path.exists():
            checkpoint_path.unlink()
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
