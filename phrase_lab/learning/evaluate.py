from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.index.backends import HandcraftedRetrievalBackend, LearnedRetrievalBackend
from phrase_lab.index.search import search_neighbors
from phrase_lab.music.piano_roll import phrase_piano_roll
from phrase_lab.storage.manifest import save_json
from .dataset import load_prepared_dataset
from .embed import encode_batches
from .runs import run_dir
from .tokenize import notes_from_any


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = (z * math.sqrt((p * (1 - p) / total) + z * z / (4 * total * total))) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def select_fixed_queries(phrases: pd.DataFrame, count: int = 100, seed: int = 42) -> pd.DataFrame:
    df = phrases.copy()
    if "split" in df.columns:
        df = df[df["split"].astype(str) == "test"]
    df = df[df["extraction_mode"].astype(str) == "explicit_voice"]
    if df.empty:
        return df.head(0)
    df = df.sort_values(["score_id", "n_bars", "phrase_id"], kind="mergesort")
    if len(df) <= count:
        return df.reset_index(drop=True)
    indices = np.linspace(0, len(df) - 1, num=count, dtype=int)
    return df.iloc[indices].reset_index(drop=True)


def build_evaluation_artifacts(root: str | Path, run_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(root)
    run = run_dir(root, run_id)
    prepared = load_prepared_dataset(root)
    phrase_df = prepared.phrase_metadata.copy()
    query_df = select_fixed_queries(phrase_df, count=int(cfg["evaluation"]["fixed_query_count"]))
    eval_dir = run / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    query_df.to_parquet(eval_dir / "fixed_eval_queries.parquet", index=False)
    baseline_backend = HandcraftedRetrievalBackend(root)
    learned_backend = LearnedRetrievalBackend(root, run_id)
    baseline_rows = []
    learned_rows = []
    positive_rows = []
    mined_path = root / "learning" / "voice_v1" / "mined_positive_pairs.parquet"
    mined_pairs = pd.read_parquet(mined_path) if mined_path.exists() else pd.DataFrame()
    for _, row in query_df.iterrows():
        query_id = str(row["phrase_id"])
        if baseline_backend.contains(query_id):
            baseline_nn = baseline_backend.search(query_id, k=int(cfg["evaluation"]["neighbors"]), exclude_same_score=bool(cfg["evaluation"]["exclude_same_score"]), candidate_split="test")
            baseline_rows.append(baseline_nn.assign(query_phrase_id=query_id))
        if learned_backend.contains(query_id):
            learned_nn = learned_backend.search(query_id, k=int(cfg["evaluation"]["neighbors"]), exclude_same_score=bool(cfg["evaluation"]["exclude_same_score"]), candidate_split="test")
            learned_rows.append(learned_nn.assign(query_phrase_id=query_id))
    if baseline_rows:
        pd.concat(baseline_rows, ignore_index=True).to_parquet(eval_dir / "fixed_query_neighbors_baseline.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(eval_dir / "fixed_query_neighbors_baseline.parquet", index=False)
    if learned_rows:
        pd.concat(learned_rows, ignore_index=True).to_parquet(eval_dir / "fixed_query_neighbors_learned.parquet", index=False)
    else:
        pd.DataFrame().to_parquet(eval_dir / "fixed_query_neighbors_learned.parquet", index=False)
    if len(mined_pairs):
        test_pairs = mined_pairs[mined_pairs["split"].astype(str).isin({"validation", "test"})].copy()
        for _, pair in test_pairs.head(1000).iterrows():
            a = str(pair["phrase_id_a"])
            b = str(pair["phrase_id_b"])
            if learned_backend.contains(a) and learned_backend.contains(b):
                idx_a = int(np.flatnonzero(learned_backend.embeddings["phrase_ids"].astype(str) == a)[0])
                idx_b = int(np.flatnonzero(learned_backend.embeddings["phrase_ids"].astype(str) == b)[0])
                pos_sim = float(np.dot(learned_backend.embeddings["learned"][idx_a], learned_backend.embeddings["learned"][idx_b]))
                positive_rows.append({"phrase_id_a": a, "phrase_id_b": b, "similarity": pos_sim, "split": pair["split"]})
    if positive_rows:
        pd.DataFrame(positive_rows).to_parquet(eval_dir / "positive_pair_retrieval.parquet", index=False)
    else:
        pd.DataFrame(columns=["phrase_id_a", "phrase_id_b", "similarity", "split"]).to_parquet(eval_dir / "positive_pair_retrieval.parquet", index=False)
    metrics = {
        "fixed_query_count": int(len(query_df)),
        "held_out_positive_pairs": int(len(positive_rows)),
        "note": "This evaluation bundle is a machine-generated checkpoint; human judgments are recorded separately.",
    }
    save_json(eval_dir / "metrics.json", metrics)
    report = [
        "# Encoder Evaluation Report",
        "",
        "This bundle is generated from the held-out explicit_voice test split.",
        "",
        json.dumps(metrics, indent=2),
    ]
    (eval_dir / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    return metrics


def summarize_blind_votes(root: str | Path, run_id: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(root)
    path = root / "reviews" / "encoder_blind_votes.csv"
    if not path.exists():
        out = {"decisive": 0, "learned_wins": 0, "baseline_wins": 0, "ties": 0, "both_poor": 0, "status": "inconclusive"}
        save_json(run_dir(root, run_id) / "evaluation" / "human_evaluation_summary.json", out)
        return out
    df = pd.read_csv(path)
    if df.empty:
        out = {"decisive": 0, "learned_wins": 0, "baseline_wins": 0, "ties": 0, "both_poor": 0, "status": "inconclusive"}
        save_json(run_dir(root, run_id) / "evaluation" / "human_evaluation_summary.json", out)
        return out
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["run_id", "query_phrase_id"], keep="first")
    learned_wins = int((df["winner_backend"].astype(str) == "learned").sum())
    baseline_wins = int((df["winner_backend"].astype(str) == "handcrafted").sum())
    ties = int(df["tie"].astype(bool).sum()) if "tie" in df.columns else 0
    both_poor = int(df["both_poor"].astype(bool).sum()) if "both_poor" in df.columns else 0
    decisive = int(((~df.get("tie", pd.Series(False, index=df.index)).astype(bool)) & (~df.get("both_poor", pd.Series(False, index=df.index)).astype(bool))).sum())
    ci_low, ci_high = _wilson_interval(learned_wins, max(1, decisive))
    if decisive >= 50 and ci_low > 0.5:
        status = "pass"
    elif decisive >= 50 and ci_high < 0.5:
        status = "fail"
    else:
        status = "inconclusive"
    out = {
        "unique_queries": int(df["query_phrase_id"].nunique()),
        "learned_wins": learned_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "both_poor": both_poor,
        "decisive": decisive,
        "learned_win_fraction": float(learned_wins / decisive) if decisive else 0.0,
        "wilson_ci_lower": float(ci_low),
        "wilson_ci_upper": float(ci_high),
        "status": status,
    }
    eval_dir = run_dir(root, run_id) / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    save_json(eval_dir / "human_evaluation_summary.json", out)
    report = [
        "# Human Evaluation Summary",
        "",
        "This report reflects the user's judgments only and is not population-level evidence.",
        "",
        json.dumps(out, indent=2),
    ]
    (eval_dir / "human_evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    return out
