from __future__ import annotations

import argparse
import json
import logging
import hashlib
import traceback
import sys
from datetime import datetime, timezone
from pathlib import Path

from phrase_lab.logging_utils import setup_logging
from phrase_lab.pdmx.acquire import download_pdmx


def _default_paths(root: str | Path):
    root = Path(root)
    return root / "PDMX.csv", root / "mxl"


def cmd_download(args):
    download_pdmx(args.root, record_id=args.zenodo_record_id, include_mid=args.include_mid)


def cmd_extract(args):
    import matplotlib.pyplot as plt
    import pandas as pd
    from phrase_lab.config import load_config, merge_cli_overrides
    from phrase_lab.music.melody import extract_melodic_lines
    from phrase_lab.music.parse import load_score, score_id_from_path
    from phrase_lab.music.segment import segment_line
    from phrase_lab.pdmx.metadata import filter_scores, load_metadata, selected_scores_path
    from phrase_lab.storage.manifest import save_json

    cfg = load_config(args.config)
    cfg = merge_cli_overrides(cfg, {k: v for k, v in {
        "pdmx.root": args.root,
        "pdmx.subset": args.subset,
        "pdmx.max_scores": args.max_scores,
        "pdmx.sampling": args.sampling,
        "pdmx.random_seed": args.random_seed,
        "extraction.workers": args.workers,
    }.items() if v is not None})
    root = Path(args.root)
    csv_path, mxl_root = _default_paths(root)
    meta = load_metadata(csv_path, root=root)
    selected = filter_scores(meta, subset=args.subset, max_scores=args.max_scores, sampling=args.sampling, random_seed=args.random_seed)
    selected_scores_path(root).parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(selected_scores_path(root), index=False)
    extracted = []
    failures = []
    melodic_lines_extracted = 0
    start_time = datetime.now(timezone.utc)
    cue_counts = {k: 0 for k in cfg["segmentation"]["boundary_weights"]}
    out_dir = root / "extracted"
    phrase_parts_dir = out_dir / "phrase_parts"
    phrase_parts_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest = out_dir / "completed_scores.csv"
    completed_ids: set[str] = set()
    if args.force and completed_manifest.exists():
        completed_manifest.unlink()
    if completed_manifest.exists() and not args.force:
        try:
            completed_df = pd.read_csv(completed_manifest)
            completed_ids = set(completed_df["score_id"].astype(str).tolist())
        except Exception:
            completed_ids = set()
    seg_cfg = cfg["segmentation"]
    completed_rows = []
    part_idx = 1
    for _, row in selected.iterrows():
        try:
            score_id_fallback = str(row.get("score_id") or row.get("id") or row.get("path") or "")
            if score_id_fallback in completed_ids and not args.force:
                continue
            path = row["mxl_path"] if "mxl_path" in row and pd.notna(row["mxl_path"]) else str((mxl_root / str(row["mxl"])).resolve())
            score = load_score(path)
            score_id = str(row.get("score_id") or score_id_from_path(path))
            if score_id in completed_ids and not args.force:
                continue
            lines = extract_melodic_lines(score, score_id, min_notes_per_line=cfg["extraction"]["min_notes_per_line"])
            melodic_lines_extracted += len(lines)
            score_rows = []
            for line in lines:
                for phrase in segment_line(line, seg_cfg):
                    phrase_row = phrase.to_row()
                    for cue, value in phrase.left_boundary.reasons.items():
                        if value:
                            cue_counts[cue] = cue_counts.get(cue, 0) + 1
                    for cue, value in phrase.right_boundary.reasons.items():
                        if value:
                            cue_counts[cue] = cue_counts.get(cue, 0) + 1
                    for col in ["path", "mxl", "metadata", "song_name", "title", "subtitle", "artist_name", "composer_name", "genres", "tags", "rating", "n_ratings", "n_favorites", "n_views", "complexity", "n_tracks", "tracks", "song_length.bars", "n_notes", "subset:no_license_conflict", "subset:deduplicated", "subset:rated", "subset:rated_deduplicated"]:
                        if col in row.index:
                            phrase_row[col if col != "path" else "source_path"] = row[col]
                    phrase_row["source_mxl"] = path
                    phrase_row["title"] = row.get("title")
                    phrase_row["song_name"] = row.get("song_name")
                    phrase_row["composer_name"] = row.get("composer_name")
                    phrase_row["artist_name"] = row.get("artist_name")
                    phrase_row["genres"] = row.get("genres")
                    score_rows.append(phrase_row)
                    extracted.append(phrase_row)
            if score_rows:
                pd.DataFrame(score_rows).to_parquet(
                    phrase_parts_dir / f"part-{part_idx:06d}.parquet",
                    index=False,
                    compression="zstd",
                )
                part_idx += 1
            completed_ids.add(score_id)
            completed_rows.append({"score_id": score_id, "mxl_path": path})
            pd.DataFrame(completed_rows).to_csv(completed_manifest, mode="a", index=False, header=not completed_manifest.exists())
            completed_rows.clear()
        except Exception as e:
            failures.append(
                {
                    "score_id": row.get("score_id"),
                    "mxl path": row.get("mxl_path"),
                    "exception type": type(e).__name__,
                    "message": str(e),
                    "traceback summary": traceback.format_exc(limit=3),
                }
            )
    if extracted:
        out = pd.DataFrame(extracted)
    else:
        out = pd.DataFrame(
            columns=[
                "phrase_id",
                "score_id",
                "source_mxl",
                "title",
                "song_name",
                "composer_name",
                "artist_name",
                "genres",
                "part_id",
                "part_name",
                "instrument_name",
                "voice_id",
                "extraction_mode",
                "start_q",
                "end_q",
                "start_measure",
                "end_measure",
                "n_bars",
                "n_notes",
                "left_boundary_score",
                "right_boundary_score",
                "left_boundary_reasons_json",
                "right_boundary_reasons_json",
                "notes_json",
            ]
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    if part_idx == 1:
        (phrase_parts_dir / "part-000001.parquet").parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(phrase_parts_dir / "part-000001.parquet", index=False, compression="zstd")
    out.to_parquet(out_dir / "phrases.parquet", index=False, compression="zstd")
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "extraction_failures.csv", index=False)
    summary = {
        "scores_selected": int(len(selected)),
        "scores_parsed": int(len(selected) - len(failures)),
        "scores_failed": int(len(failures)),
        "melodic_lines_extracted": int(melodic_lines_extracted),
        "phrases_extracted": int(len(out)),
        "median_phrases_per_score": float(out.groupby("score_id").size().median()) if len(out) else 0.0,
        "median_bars_per_phrase": float(out["n_bars"].median()) if len(out) else 0.0,
        "boundary_cue_frequencies": cue_counts,
        "fraction_using_explicit_voice": float((out["extraction_mode"] == "explicit_voice").mean()) if len(out) else 0.0,
        "fraction_using_skyline": float((out["extraction_mode"] == "skyline").mean()) if len(out) else 0.0,
    }
    save_json(
        out_dir / "run_manifest.json",
        {
            "PDMX_record_id": cfg["pdmx"]["zenodo_record_id"],
            "metadata_filter": {"subset": args.subset, "max_scores": args.max_scores, "sampling": args.sampling, "random_seed": args.random_seed},
            "selected_scores": int(len(selected)),
            "segmentation_config": seg_cfg,
            "python_version": sys.version,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "success_count": int(len(selected) - len(failures)),
            "failure_count": int(len(failures)),
            "phrases_emitted": int(len(out)),
            "segmentation_settings_hash": hashlib.sha1(json.dumps(seg_cfg, sort_keys=True).encode("utf-8")).hexdigest(),
        },
    )
    save_json(out_dir / "extraction_summary.json", summary)
    if len(out):
        fig, ax = plt.subplots()
        ax.hist(out["n_bars"], bins=20)
        ax.set_title("Phrase length histogram")
        fig.tight_layout()
        fig.savefig(out_dir / "phrase_length_histogram.png")
        plt.close(fig)
        fig, ax = plt.subplots()
        ax.bar(list(cue_counts.keys()), list(cue_counts.values()))
        ax.set_title("Boundary cue frequency")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / "boundary_cue_frequency.png")
        plt.close(fig)


def cmd_build_index(args):
    import pandas as pd
    from phrase_lab.config import load_config
    from phrase_lab.index.build_index import build_all_indexes

    root = Path(args.root)
    phrases_arg = getattr(args, "phrases", None)
    phrases_path = Path(phrases_arg) if phrases_arg else root / "extracted" / "phrases.parquet"
    phrases = pd.read_parquet(phrases_path)
    cfg = load_config(args.config)
    build_all_indexes(root, phrases, cfg["features"], cfg["index"])


def cmd_app(args):
    from phrase_lab.app.gradio_app import launch

    launch(root=args.root, share=args.share)


def cmd_pipeline(args):
    cmd_download(args)
    cmd_extract(args)
    cmd_build_index(args)


def cmd_inspect(args):
    import pandas as pd
    from phrase_lab.index.search import load_embeddings, search_neighbors
    from phrase_lab.storage.phrase_store import PhraseStore

    root = Path(args.root)
    phrases = pd.read_parquet(root / "extracted" / "phrases.parquet")
    phrases = phrases.sample(n=min(20, len(phrases)), random_state=42)
    html = ["<html><body><h1>PDMX Phrase Lab Inspection</h1>"]
    store = PhraseStore(root)
    embeddings = None
    try:
        embeddings = load_embeddings(root / "index")
    except Exception:
        embeddings = None
    for _, row in phrases.iterrows():
        html.append(f"<h2>{row.get('title')} - {row.get('composer_name')}</h2>")
        html.append(f"<p>{row['phrase_id']} {row['start_measure']} - {row['end_measure']}</p>")
        html.append(f"<pre>{json.dumps(row['left_boundary_reasons_json'], indent=2)}</pre>")
        if embeddings is not None and row["phrase_id"] in list(embeddings["phrase_ids"]):
            nn = search_neighbors(row["phrase_id"], store.get_dataframe(), embeddings, k=3)
            html.append(nn.to_html(index=False))
    html.append("</body></html>")
    out = root / "extracted" / "inspection_report.html"
    out.write_text("\n".join(html), encoding="utf-8")


def build_parser():
    p = argparse.ArgumentParser(prog="phrase_lab")
    p.add_argument("--config", default="configs/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["download", "extract", "build-index", "app", "pipeline", "inspect"]:
        sp = sub.add_parser(name)
        sp.add_argument("--root", default="data/raw/PDMX")
    sub.choices["download"].add_argument("--zenodo-record-id", type=int, default=15571083)
    sub.choices["download"].add_argument("--include-mid", action="store_true")
    sub.choices["pipeline"].add_argument("--zenodo-record-id", type=int, default=15571083)
    sub.choices["pipeline"].add_argument("--include-mid", action="store_true")
    for nm in ["extract", "pipeline"]:
        sp = sub.choices[nm]
        sp.add_argument("--subset", default="safe_deduplicated")
        sp.add_argument("--max-scores", type=int, default=None)
        sp.add_argument("--sampling", default="random")
        sp.add_argument("--random-seed", type=int, default=42)
        sp.add_argument("--workers", type=int, default=2)
        sp.add_argument("--force", action="store_true")
    sub.choices["build-index"].add_argument("--phrases", default=None)
    sub.choices["pipeline"].add_argument("--phrases", default=None)
    sub.choices["app"].add_argument("--share", action="store_true")
    return p


def main(argv: list[str] | None = None):
    setup_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd.replace("-", "_")
    globals()[f"cmd_{cmd}"](args)


if __name__ == "__main__":
    main()
