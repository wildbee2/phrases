from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phrase_lab.storage.manifest import save_json

from .manifest import atomic_write_parquet, vocabulary_root
from .metrics import compute_joint_token_stats, compute_sequence_stats
from phrase_lab.index.search import load_embeddings, search_neighbors
from phrase_lab.storage.phrase_store import PhraseStore
from .codebook import discover_codebooks, load_codebook


def _load_assignments(root: Path, space: str, k: int) -> pd.DataFrame:
    base = vocabulary_root(root) / space / f"k{k}"
    codebooks = [p for p in base.iterdir() if p.is_dir() and (p / "assignments.parquet").exists()]
    if not codebooks:
        raise FileNotFoundError(f"no codebook found for {space} k={k}")
    return pd.read_parquet(sorted(codebooks)[-1] / "assignments.parquet")


def export_phrase_tokens(root: str | Path, melody_k: int, rhythm_k: int, combined_k: int | None = None) -> pd.DataFrame:
    root = Path(root)
    melody = _load_assignments(root, "melody", int(melody_k))[["phrase_id", "token", "cosine_to_centroid", "assignment_margin"]].rename(columns={"token": "melody_token", "cosine_to_centroid": "melody_centroid_similarity", "assignment_margin": "melody_assignment_margin"})
    rhythm = _load_assignments(root, "rhythm", int(rhythm_k))[["phrase_id", "token", "cosine_to_centroid", "assignment_margin"]].rename(columns={"token": "rhythm_token", "cosine_to_centroid": "rhythm_centroid_similarity", "assignment_margin": "rhythm_assignment_margin"})
    phrase_tokens = melody.merge(rhythm, on="phrase_id", how="inner")
    phrase_tokens = phrase_tokens.merge(pd.read_parquet(vocabulary_root(root) / "eligible_phrases.parquet"), on="phrase_id", how="left")
    if combined_k is not None:
        try:
            combined = _load_assignments(root, "combined", int(combined_k))[["phrase_id", "token", "cosine_to_centroid"]].rename(columns={"token": "combined_token_optional", "cosine_to_centroid": "combined_centroid_similarity_optional"})
            phrase_tokens = phrase_tokens.merge(combined, on="phrase_id", how="left")
        except Exception:
            phrase_tokens["combined_token_optional"] = None
            phrase_tokens["combined_centroid_similarity_optional"] = np.nan
    out_dir = vocabulary_root(root)
    atomic_write_parquet(out_dir / "phrase_tokens.parquet", phrase_tokens)
    save_json(out_dir / "joint_token_stats.json", compute_joint_token_stats(phrase_tokens))
    sequences = export_phrase_sequences(root, phrase_tokens)
    save_json(out_dir / "sequence_stats.json", compute_sequence_stats(sequences))
    save_json(out_dir / "token_concentration.json", compute_token_concentration(phrase_tokens, out_dir / "token_concentration.parquet"))
    save_json(out_dir / "transition_probabilities.json", build_transition_probabilities(out_dir / "token_transition_counts.json"))
    return phrase_tokens


def export_phrase_sequences(root: str | Path, phrase_tokens: pd.DataFrame) -> pd.DataFrame:
    root = Path(root)
    if phrase_tokens.empty:
        out = pd.DataFrame(columns=["score_id", "part_id", "voice_id", "sequence_length", "phrase_ids", "melody_tokens", "rhythm_tokens", "start_qs", "end_qs"])
        atomic_write_parquet(vocabulary_root(root) / "phrase_sequences.parquet", out)
        return out
    rows = []
    for keys, grp in phrase_tokens.sort_values(["score_id", "part_id", "voice_id", "start_q", "phrase_id"], kind="mergesort").groupby(["score_id", "part_id", "voice_id"], dropna=False):
        rows.append(
            {
                "score_id": keys[0],
                "part_id": keys[1],
                "voice_id": keys[2],
                "sequence_length": int(len(grp)),
                "phrase_ids": grp["phrase_id"].astype(str).tolist(),
                "melody_tokens": grp["melody_token"].astype(str).tolist(),
                "rhythm_tokens": grp["rhythm_token"].astype(str).tolist(),
                "start_qs": grp["start_q"].astype(float).tolist(),
                "end_qs": grp["end_q"].astype(float).tolist(),
            }
        )
    out = pd.DataFrame(rows)
    atomic_write_parquet(vocabulary_root(root) / "phrase_sequences.parquet", out)
    melody_bigrams = Counter()
    melody_trigrams = Counter()
    rhythm_bigrams = Counter()
    rhythm_trigrams = Counter()
    melody_token_freq = Counter()
    rhythm_token_freq = Counter()
    for _, row in out.iterrows():
        melody_toks = list(row["melody_tokens"])
        rhythm_toks = list(row["rhythm_tokens"])
        melody_token_freq.update(melody_toks)
        rhythm_token_freq.update(rhythm_toks)
        melody_bigrams.update(zip(melody_toks[:-1], melody_toks[1:]))
        melody_trigrams.update(zip(melody_toks[:-2], melody_toks[1:-1], melody_toks[2:]))
        rhythm_bigrams.update(zip(rhythm_toks[:-1], rhythm_toks[1:]))
        rhythm_trigrams.update(zip(rhythm_toks[:-2], rhythm_toks[1:-1], rhythm_toks[2:]))
    save_json(
        vocabulary_root(root) / "token_transition_counts.json",
        {
            "melody_token_frequencies": {k: int(v) for k, v in melody_token_freq.items()},
            "rhythm_token_frequencies": {k: int(v) for k, v in rhythm_token_freq.items()},
            "melody_bigram_counts": {f"{a}->{b}": int(c) for (a, b), c in melody_bigrams.items()},
            "melody_trigram_counts": {f"{a}->{b}->{c}": int(n) for (a, b, c), n in melody_trigrams.items()},
            "rhythm_bigram_counts": {f"{a}->{b}": int(c) for (a, b), c in rhythm_bigrams.items()},
            "rhythm_trigram_counts": {f"{a}->{b}->{c}": int(n) for (a, b, c), n in rhythm_trigrams.items()},
        },
    )
    return out


def build_transition_probabilities(counts_path: str | Path) -> dict[str, Any]:
    path = Path(counts_path)
    counts = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    out: dict[str, Any] = {}
    for space in ["melody", "rhythm"]:
        bigrams = counts.get(f"{space}_bigram_counts", {})
        token_freq = counts.get(f"{space}_token_frequencies", {})
        totals: dict[str, int] = {}
        for key, val in bigrams.items():
            src, _ = key.split("->", 1)
            totals[src] = totals.get(src, 0) + int(val)
        conditional_probabilities = []
        for key, val in bigrams.items():
            src, dst = key.split("->", 1)
            conditional_probabilities.append(
                {
                    "space": space,
                    "current_token": src,
                    "next_token": dst,
                    "count": int(val),
                    "probability": float(val / max(1, totals.get(src, 0))),
                }
            )
        out[space] = {
            "token_frequencies": token_freq,
            "conditional_probabilities": conditional_probabilities,
        }
    return out


def compare_vocabulary_sizes(root: str | Path, space: str) -> pd.DataFrame:
    root = Path(root)
    base = vocabulary_root(root) / space
    rows = []
    if not base.exists():
        return pd.DataFrame()
    for k_dir in sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("k")]):
        for codebook_dir in sorted([p for p in k_dir.iterdir() if p.is_dir() and (p / "metrics.json").exists()]):
            metrics = json.loads((codebook_dir / "metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((codebook_dir / "codebook_manifest.json").read_text(encoding="utf-8"))
            stability = json.loads((codebook_dir / "stability_metrics.json").read_text(encoding="utf-8"))
            eval_summary = None
            eval_path = vocabulary_root(root) / space / f"k{manifest['k']}" / "evaluation_summary.json"
            if eval_path.exists():
                eval_summary = json.loads(eval_path.read_text(encoding="utf-8"))
            rows.append({
                "space": space,
                "k": int(manifest["k"]),
                "codebook_id": manifest["codebook_id"],
                "quantization_error_mean": metrics["quantization_error"]["mean"],
                "effective_fraction": metrics["effective_vocabulary"]["effective_fraction"],
                "tiny_cluster_fraction": metrics["occupancy"]["fraction_clusters_lt_10"],
                "cohesion_mean": metrics["cohesion"]["mean"],
                "assignment_margin_mean": metrics["assignment_margin"]["mean"],
                "stability_ari": stability.get("mean_ari", 1.0),
                "human_cluster_status": eval_summary["cluster"]["status"] if eval_summary else None,
                "human_cluster_fraction": eval_summary["cluster"]["coherent_fraction"] if eval_summary else None,
                "human_blind_status": eval_summary["blind"]["status"] if eval_summary else None,
                "human_blind_same_cluster_win_fraction": eval_summary["blind"]["same_cluster_win_fraction"] if eval_summary else None,
            })
    df = pd.DataFrame(rows).sort_values(["k", "codebook_id"], kind="mergesort").reset_index(drop=True)
    if not df.empty:
        atomic_write_parquet(vocabulary_root(root) / space / "vocabulary_size_comparison.parquet", df)
        df.to_csv(vocabulary_root(root) / space / "vocabulary_size_comparison.csv", index=False)
        summary = {"rows": int(len(df)), "best_k_by_quantization_error": int(df.sort_values("quantization_error_mean").iloc[0]["k"])}
        save_json(vocabulary_root(root) / space / "vocabulary_size_report.json", summary)
        _write_vocabulary_size_report(root, space, df, summary)
        build_multi_resolution_analysis(root, space)
    return df


def vocabulary_report(root: str | Path, space: str, k: int) -> Path:
    root = Path(root)
    codebook_dir = sorted([p for p in (vocabulary_root(root) / space / f"k{k}").iterdir() if p.is_dir() and (p / "assignments.parquet").exists()])[-1]
    df = pd.read_parquet(codebook_dir / "assignments.parquet")
    plot_paths = {
        "cluster_size_histogram.png": _save_histogram(df["cluster_id"].value_counts().to_numpy(), codebook_dir / "cluster_size_histogram.png", f"{space} k={k} cluster sizes"),
        "cohesion_histogram.png": _save_histogram(df["cosine_to_centroid"].to_numpy(), codebook_dir / "cohesion_histogram.png", f"{space} k={k} cohesion"),
        "assignment_margin_histogram.png": _save_histogram(df["assignment_margin"].to_numpy(), codebook_dir / "assignment_margin_histogram.png", f"{space} k={k} assignment margin"),
        "quantization_error_histogram.png": _save_histogram((1.0 - df["cosine_to_centroid"]).to_numpy(), codebook_dir / "quantization_error_histogram.png", f"{space} k={k} quantization error"),
        "token_frequency_rank.png": _save_rank_plot(df["token"].value_counts(), codebook_dir / "token_frequency_rank.png", f"{space} k={k} token frequency rank"),
    }
    manifest = json.loads((codebook_dir / "codebook_manifest.json").read_text(encoding="utf-8"))
    html = [
        "<html><body>",
        f"<h1>Vocabulary Report: {space} k={k}</h1>",
        f"<p>Assignments: {len(df)}</p>",
        f"<p>Mean centroid cosine: {float(df['cosine_to_centroid'].mean()):.4f}</p>",
        "".join(f"<p><img src='{name}' /></p>" for name in plot_paths),
        "<h2>Representative samples</h2>",
        df.sort_values("cosine_to_centroid", ascending=False).head(3)[["phrase_id", "token", "cosine_to_centroid"]].to_html(index=False),
        "<h2>Random samples</h2>",
        df.sample(n=min(5, len(df)), random_state=42)[["phrase_id", "token", "cosine_to_centroid"]].to_html(index=False),
        "<h2>Boundary samples</h2>",
        df.sort_values("assignment_margin", ascending=True).head(3)[["phrase_id", "token", "assignment_margin"]].to_html(index=False),
        "<pre>",
        json.dumps(manifest, indent=2),
        "</pre>",
        "</body></html>",
    ]
    out = codebook_dir / "vocabulary_report.html"
    out.write_text("\n".join(html), encoding="utf-8")
    return out


def build_multi_resolution_analysis(root: str | Path, space: str) -> pd.DataFrame:
    root = Path(root)
    codebooks = discover_codebooks(root, space)
    if not codebooks:
        return pd.DataFrame()
    by_k: dict[int, pd.DataFrame] = {}
    for cb in codebooks:
        loaded = load_codebook(cb)
        k = int(loaded["manifest"]["k"])
        by_k[k] = loaded["assignments"][["phrase_id", "token"]].rename(columns={"token": f"token_k{k}"})
    rows = []
    for parent_k, child_k in zip(sorted(by_k)[:-1], sorted(by_k)[1:]):
        parent = by_k[parent_k]
        child = by_k[child_k]
        merged = parent.merge(child, on="phrase_id", how="inner")
        for ptoken, grp in merged.groupby(f"token_k{parent_k}", dropna=False):
            parent_size = int(len(grp))
            child_sizes = grp[f"token_k{child_k}"].value_counts()
            for ctoken, count in child_sizes.items():
                child_total = int((merged[f"token_k{child_k}"] == ctoken).sum())
                rows.append(
                    {
                        "space": space,
                        "parent_k": int(parent_k),
                        "child_k": int(child_k),
                        "parent_token": ptoken,
                        "child_token": ctoken,
                        "overlap_count": int(count),
                        "fraction_of_parent": float(count / max(1, parent_size)),
                        "fraction_of_child": float(count / max(1, child_total)),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        atomic_write_parquet(vocabulary_root(root) / space / "multi_resolution_analysis.parquet", out)
        out.to_csv(vocabulary_root(root) / space / "multi_resolution_analysis.csv", index=False)
    return out


def compute_token_concentration(phrase_tokens: pd.DataFrame, out_path: str | Path | None = None) -> dict[str, Any]:
    if phrase_tokens.empty:
        return {"tokens": [], "flagged_tokens": []}
    rows = []
    for token, grp in phrase_tokens.groupby("melody_token", dropna=False):
        row = {
            "melody_token": token,
            "count": int(len(grp)),
            "distinct_scores": int(grp["score_id"].nunique()) if "score_id" in grp.columns else 0,
            "distinct_composers": int(grp["composer_name"].nunique()) if "composer_name" in grp.columns else 0,
            "largest_single_score_fraction": float(grp["score_id"].value_counts(normalize=True).iloc[0]) if "score_id" in grp.columns else 0.0,
            "largest_single_composer_fraction": float(grp["composer_name"].value_counts(normalize=True).iloc[0]) if "composer_name" in grp.columns else 0.0,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    flagged = out[(out["largest_single_score_fraction"] >= 0.5) | (out["largest_single_composer_fraction"] >= 0.5)].copy()
    if out_path is not None:
        atomic_write_parquet(out_path, out)
    return {
        "tokens": rows,
        "flagged_tokens": flagged.to_dict("records"),
    }


def compute_quantization_neighbor_preservation(
    root: str | Path,
    space: str,
    k: int,
    query_phrase_ids: list[str] | None = None,
    top_n: int = 10,
    top_k: int = 5,
) -> dict[str, Any]:
    root = Path(root)
    store = PhraseStore(root)
    phrase_df = store.get_dataframe()
    embeddings = load_embeddings(root / "index")
    codebook_dir = sorted([p for p in (vocabulary_root(root) / space / f"k{k}").iterdir() if p.is_dir() and (p / "assignments.parquet").exists()])[-1]
    assignments = pd.read_parquet(codebook_dir / "assignments.parquet")
    query_ids = query_phrase_ids or assignments["phrase_id"].astype(str).head(min(50, len(assignments))).tolist()
    stats = []
    for qid in query_ids:
        qid = str(qid)
        if qid not in set(assignments["phrase_id"].astype(str)):
            continue
        query_row = phrase_df[phrase_df["phrase_id"].astype(str) == qid]
        if query_row.empty:
            continue
        continuous = search_neighbors(qid, phrase_df, embeddings, mode=space, k=top_n, exclude_same_score=True)
        query_token = assignments.loc[assignments["phrase_id"].astype(str) == qid, "token"].iloc[0]
        same_token = assignments[(assignments["token"].astype(str) == str(query_token)) & (assignments["phrase_id"].astype(str) != qid)]
        same_token = same_token.sort_values("cosine_to_centroid", ascending=False).head(top_n)
        nearest_tokens = assignments.sort_values("cosine_to_centroid", ascending=False)["token"].drop_duplicates().head(max(1, top_k)).tolist()
        near_df = assignments[assignments["token"].isin(nearest_tokens) & (assignments["phrase_id"].astype(str) != qid)]
        near_df = near_df.sort_values("cosine_to_centroid", ascending=False).head(top_n)
        cont_ids = continuous["phrase_id"].astype(str).tolist()
        same_ids = same_token["phrase_id"].astype(str).tolist()
        near_ids = near_df["phrase_id"].astype(str).tolist()
        cont_set = set(cont_ids)
        same_set = set(same_ids)
        near_set = set(near_ids)
        stats.append(
            {
                "query_phrase_id": qid,
                "topk_overlap_same_token": int(len(cont_set & same_set)),
                "topk_overlap_nearest_tokens": int(len(cont_set & near_set)),
                "jaccard_same_token": float(len(cont_set & same_set) / max(1, len(cont_set | same_set))),
                "jaccard_nearest_tokens": float(len(cont_set & near_set) / max(1, len(cont_set | near_set))),
                "mean_original_cosine_same_token": float(same_token["cosine_to_centroid"].mean()) if len(same_token) else 0.0,
                "mean_original_cosine_nearest_tokens": float(near_df["cosine_to_centroid"].mean()) if len(near_df) else 0.0,
            }
        )
    out = {
        "query_count": int(len(stats)),
        "mean_topk_overlap_same_token": float(np.mean([r["topk_overlap_same_token"] for r in stats])) if stats else 0.0,
        "mean_topk_overlap_nearest_tokens": float(np.mean([r["topk_overlap_nearest_tokens"] for r in stats])) if stats else 0.0,
        "mean_jaccard_same_token": float(np.mean([r["jaccard_same_token"] for r in stats])) if stats else 0.0,
        "mean_jaccard_nearest_tokens": float(np.mean([r["jaccard_nearest_tokens"] for r in stats])) if stats else 0.0,
        "mean_original_cosine_same_token": float(np.mean([r["mean_original_cosine_same_token"] for r in stats])) if stats else 0.0,
        "mean_original_cosine_nearest_tokens": float(np.mean([r["mean_original_cosine_nearest_tokens"] for r in stats])) if stats else 0.0,
        "per_query": stats,
    }
    save_json(vocabulary_root(root) / space / f"k{k}" / "quantization_neighbor_preservation.json", out)
    atomic_write_parquet(vocabulary_root(root) / space / f"k{k}" / "quantization_neighbor_preservation.parquet", pd.DataFrame(stats))
    return out


def _write_vocabulary_size_report(root: Path, space: str, df: pd.DataFrame, summary: dict[str, Any]) -> None:
    _save_line_plot(df["k"], df["quantization_error_mean"], vocabulary_root(root) / space / "k_vs_quantization_error.png", "K vs quantization error")
    _save_line_plot(df["k"], df["effective_fraction"], vocabulary_root(root) / space / "k_vs_effective_fraction.png", "K vs effective vocabulary fraction")
    _save_line_plot(df["k"], df["stability_ari"], vocabulary_root(root) / space / "k_vs_stability.png", "K vs stability")
    _save_line_plot(df["k"], df["tiny_cluster_fraction"], vocabulary_root(root) / space / "k_vs_tiny_cluster_fraction.png", "K vs tiny-cluster fraction")
    table = df.to_csv(index=False)
    md = [
        f"# Vocabulary Size Report for {space}",
        "",
        f"- Rows: {summary['rows']}",
        f"- Best K by quantization error: {summary['best_k_by_quantization_error']}",
        "",
        "```csv",
        table.strip(),
        "```",
    ]
    (vocabulary_root(root) / space / "vocabulary_size_report.md").write_text("\n".join(md), encoding="utf-8")


def _save_histogram(values: np.ndarray, path: Path, title: str) -> Path:
    fig, ax = plt.subplots()
    ax.hist(np.asarray(values, dtype=np.float32), bins=20)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_line_plot(x, y, path: Path, title: str) -> Path:
    fig, ax = plt.subplots()
    ax.plot(list(x), list(y), marker="o")
    ax.set_xlabel("K")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_rank_plot(counter: pd.Series, path: Path, title: str) -> Path:
    fig, ax = plt.subplots()
    values = np.asarray(counter.sort_values(ascending=False).to_numpy(), dtype=np.float32)
    ax.plot(np.arange(1, len(values) + 1), values)
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
