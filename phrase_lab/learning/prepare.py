from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.pdmx.metadata import load_metadata
from phrase_lab.storage.manifest import load_json, save_json
from .runs import version_root
from .tokenize import TokenizerConfig, notes_from_any, token_manifest_hash, tokenize_phrase


def _json_hash(data: Any) -> str:
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def _split_for_score(score_id: str, seed: int, fractions: tuple[float, float, float]) -> str:
    token = f"{seed}|{score_id}".encode("utf-8")
    digest = hashlib.sha1(token).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    train_f, val_f, _ = fractions
    if value < train_f:
        return "train"
    if value < train_f + val_f:
        return "validation"
    return "test"


def _valid_notes(notes: list[dict[str, Any]]) -> bool:
    return all(key in note for note in notes for key in ("p", "o", "d", "v"))


def prepare_encoder_dataset(root: str | Path, cfg: dict[str, Any], max_phrases: int | None = None) -> pd.DataFrame:
    root = Path(root)
    phrases = pd.read_parquet(root / "extracted" / "phrases.parquet")
    source_checksum = hashlib.sha1((root / "extracted" / "phrases.parquet").read_bytes()).hexdigest()
    exp_hash = _json_hash(cfg)
    tok_cfg = TokenizerConfig(**cfg["tokenizer"])
    base = version_root(root)
    base.mkdir(parents=True, exist_ok=True)
    if "subset:no_license_conflict" not in phrases.columns:
        raise KeyError("subset:no_license_conflict column is required for Experiment 002")
    mask = phrases["subset:no_license_conflict"].astype(bool) & (phrases["extraction_mode"].astype(str) == "explicit_voice")
    filtered = phrases.loc[mask].copy()
    if max_phrases is not None and len(filtered) > max_phrases:
        filtered = filtered.sort_values(["score_id", "start_q", "phrase_id"], kind="mergesort").head(max_phrases).copy()
    rejected: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    token_rows: list[np.ndarray] = []
    split_rows: list[dict[str, Any]] = []
    fractions = (
        float(cfg["split"]["train_fraction"]),
        float(cfg["split"]["val_fraction"]),
        float(cfg["split"]["test_fraction"]),
    )
    for _, row in filtered.iterrows():
        notes = notes_from_any(row.get("notes_json"))
        reason = None
        if not notes:
            reason = "missing_or_malformed_notes_json"
        elif not _valid_notes(notes):
            reason = "invalid_note_schema"
        elif len(notes) < int(cfg["dataset"]["min_notes"]):
            reason = "too_few_notes"
        elif len(notes) > int(cfg["dataset"]["max_notes"]):
            reason = "too_many_notes"
        elif float(row.get("n_bars", 0.0)) < float(cfg["dataset"]["min_bars"]):
            reason = "too_short_in_bars"
        elif float(row.get("n_bars", 0.0)) > float(cfg["dataset"]["max_bars"]):
            reason = "too_long_in_bars"
        if reason is not None:
            rejected.append({"phrase_id": row.get("phrase_id"), "score_id": row.get("score_id"), "reason": reason})
            continue
        try:
            tokens, n_notes, duration = tokenize_phrase(notes, tok_cfg)
        except Exception as exc:
            rejected.append({"phrase_id": row.get("phrase_id"), "score_id": row.get("score_id"), "reason": f"tokenize_error:{type(exc).__name__}"})
            continue
        split = _split_for_score(str(row.get("score_id")), int(cfg["split"]["seed"]), fractions)
        kept = row.to_dict()
        kept["split"] = split
        kept["token_length"] = int(n_notes)
        kept["phrase_duration"] = float(duration)
        kept_rows.append(kept)
        token_rows.append(tokens.astype(np.uint16))
        split_rows.append({"phrase_id": row.get("phrase_id"), "score_id": row.get("score_id"), "split": split})
    phrase_metadata = pd.DataFrame(kept_rows)
    split_assignments = pd.DataFrame(split_rows)
    tokens = np.stack(token_rows, axis=0) if token_rows else np.zeros((0, tok_cfg.max_notes, 5), dtype=np.uint16)
    phrase_metadata.to_parquet(base / "phrase_metadata.parquet", index=False)
    split_assignments.to_parquet(base / "split_assignments.parquet", index=False)
    np.save(base / "tokens.npy", tokens)
    token_manifest = {
        "tokenizer": tok_cfg.__dict__,
        "tokenizer_hash": _json_hash(tok_cfg.__dict__),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    dataset_manifest = {
        "source_phrases_checksum": source_checksum,
        "source_extraction_run_manifest_hash": hashlib.sha1((root / "extracted" / "run_manifest.json").read_bytes()).hexdigest() if (root / "extracted" / "run_manifest.json").exists() else None,
        "selection_rules": {
            "require_no_license_conflict": bool(cfg["dataset"]["require_no_license_conflict"]),
            "extraction_modes": list(cfg["dataset"]["extraction_modes"]),
            "min_notes": int(cfg["dataset"]["min_notes"]),
            "max_notes": int(cfg["dataset"]["max_notes"]),
            "min_bars": float(cfg["dataset"]["min_bars"]),
            "max_bars": float(cfg["dataset"]["max_bars"]),
            "max_phrases": max_phrases,
        },
        "rows_before_filtering": int(len(phrases)),
        "rows_after_filtering": int(len(phrase_metadata)),
        "explicit_voice_count": int((phrases["extraction_mode"].astype(str) == "explicit_voice").sum()),
        "rejected_count": int(len(rejected)),
        "train_phrase_count": int((phrase_metadata["split"] == "train").sum()) if len(phrase_metadata) else 0,
        "validation_phrase_count": int((phrase_metadata["split"] == "validation").sum()) if len(phrase_metadata) else 0,
        "test_phrase_count": int((phrase_metadata["split"] == "test").sum()) if len(phrase_metadata) else 0,
        "train_score_count": int(phrase_metadata.loc[phrase_metadata["split"] == "train", "score_id"].nunique()) if len(phrase_metadata) else 0,
        "validation_score_count": int(phrase_metadata.loc[phrase_metadata["split"] == "validation", "score_id"].nunique()) if len(phrase_metadata) else 0,
        "test_score_count": int(phrase_metadata.loc[phrase_metadata["split"] == "test", "score_id"].nunique()) if len(phrase_metadata) else 0,
        "tokenizer_config_hash": token_manifest["tokenizer_hash"],
        "experiment_config_hash": exp_hash,
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
    }
    save_json(base / "dataset_manifest.json", dataset_manifest)
    save_json(base / "token_manifest.json", token_manifest)
    if rejected:
        pd.DataFrame(rejected).to_csv(base / "rejected_phrases.csv", index=False)
    else:
        pd.DataFrame(columns=["phrase_id", "score_id", "reason"]).to_csv(base / "rejected_phrases.csv", index=False)
    return phrase_metadata

