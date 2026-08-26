from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from phrase_lab.storage.manifest import save_json

from .manifest import vocabulary_root


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    radius = (z * math.sqrt((p * (1 - p) / total) + z * z / (4 * total * total))) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def summarize_cluster_reviews(root: str | Path, space: str, k: int) -> dict[str, Any]:
    root = Path(root)
    path = root / "reviews" / "vocabulary_cluster_reviews.csv"
    if not path.exists():
        out = {"unique_clusters_reviewed": 0, "coherent_clusters": 0, "coherent_fraction": 0.0, "wilson_ci_lower": 0.0, "wilson_ci_upper": 0.0, "status": "inconclusive"}
        save_json(vocabulary_root(root) / space / f"k{k}" / "human_cluster_summary.json", out)
        return out
    df = pd.read_csv(path)
    if df.empty:
        out = {"unique_clusters_reviewed": 0, "coherent_clusters": 0, "coherent_fraction": 0.0, "wilson_ci_lower": 0.0, "wilson_ci_upper": 0.0, "status": "inconclusive"}
        save_json(vocabulary_root(root) / space / f"k{k}" / "human_cluster_summary.json", out)
        return out
    df = df[df["space"].astype(str) == str(space)]
    df = df[df["k"].astype(int) == int(k)]
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["token"], keep="first")
    coherent = df["rating"].astype(str).isin({"Strongly coherent", "Mostly coherent"})
    unique_clusters = int(df["token"].nunique())
    coherent_clusters = int(coherent.sum())
    frac = float(coherent_clusters / unique_clusters) if unique_clusters else 0.0
    ci_low, ci_high = _wilson_interval(coherent_clusters, max(1, unique_clusters))
    if unique_clusters >= 40 and frac >= 0.65:
        status = "pass"
    elif unique_clusters >= 40 and frac < 0.40:
        status = "fail"
    else:
        status = "inconclusive"
    out = {
        "unique_clusters_reviewed": unique_clusters,
        "coherent_clusters": coherent_clusters,
        "coherent_fraction": frac,
        "wilson_ci_lower": float(ci_low),
        "wilson_ci_upper": float(ci_high),
        "status": status,
    }
    save_json(vocabulary_root(root) / space / f"k{k}" / "human_cluster_summary.json", out)
    return out


def summarize_blind_trials(root: str | Path, space: str, k: int) -> dict[str, Any]:
    root = Path(root)
    path = root / "reviews" / "vocabulary_blind_trials.csv"
    if not path.exists():
        out = {"decisive": 0, "same_cluster_wins": 0, "different_cluster_wins": 0, "ties": 0, "neither": 0, "status": "inconclusive"}
        save_json(vocabulary_root(root) / space / f"k{k}" / "human_blind_summary.json", out)
        return out
    df = pd.read_csv(path)
    if df.empty:
        out = {"decisive": 0, "same_cluster_wins": 0, "different_cluster_wins": 0, "ties": 0, "neither": 0, "status": "inconclusive"}
        save_json(vocabulary_root(root) / space / f"k{k}" / "human_blind_summary.json", out)
        return out
    df = df[df["space"].astype(str) == str(space)]
    df = df[df["k"].astype(int) == int(k)]
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="mergesort")
    df = df.drop_duplicates(subset=["trial_id"], keep="first")
    ties = int(df["tie"].astype(bool).sum()) if "tie" in df.columns else 0
    neither = int(df["neither"].astype(bool).sum()) if "neither" in df.columns else 0
    decisive = int(((~df.get("tie", pd.Series(False, index=df.index)).astype(bool)) & (~df.get("neither", pd.Series(False, index=df.index)).astype(bool))).sum())
    same_cluster_wins = int(df["same_cluster_won"].astype(bool).sum())
    different_cluster_wins = max(0, decisive - same_cluster_wins)
    ci_low, ci_high = _wilson_interval(same_cluster_wins, max(1, decisive))
    if decisive >= 50 and ci_low > 0.5:
        status = "pass"
    elif decisive >= 50 and ci_high < 0.5:
        status = "fail"
    else:
        status = "inconclusive"
    out = {
        "decisive": decisive,
        "same_cluster_wins": same_cluster_wins,
        "different_cluster_wins": different_cluster_wins,
        "ties": ties,
        "neither": neither,
        "same_cluster_win_fraction": float(same_cluster_wins / decisive) if decisive else 0.0,
        "wilson_ci_lower": float(ci_low),
        "wilson_ci_upper": float(ci_high),
        "status": status,
    }
    save_json(vocabulary_root(root) / space / f"k{k}" / "human_blind_summary.json", out)
    return out


def summarize_vocabulary_evaluation(root: str | Path, space: str, k: int) -> dict[str, Any]:
    cluster = summarize_cluster_reviews(root, space, k)
    blind = summarize_blind_trials(root, space, k)
    if cluster["status"] == "pass" and blind["status"] == "pass":
        status = "pass"
    elif cluster["status"] == "fail" or blind["status"] == "fail":
        status = "fail"
    else:
        status = "inconclusive"
    out = {"cluster": cluster, "blind": blind, "status": status}
    save_json(vocabulary_root(root) / space / f"k{k}" / "evaluation_summary.json", out)
    return out

