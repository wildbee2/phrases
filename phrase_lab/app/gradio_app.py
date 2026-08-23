from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from phrase_lab.app.review_store import append_review
from phrase_lab.storage.manifest import load_json
from phrase_lab.index.search import load_embeddings, search_neighbors
from phrase_lab.music.piano_roll import phrase_piano_roll
from phrase_lab.music.render import synthesize_phrase
from phrase_lab.storage.phrase_store import PhraseStore


def _load(root: str | Path):
    store = PhraseStore(root)
    embeddings = load_embeddings(Path(root) / "index")
    return store, embeddings


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _notes_json(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return _notes_json(loaded)
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    notes = []
    for item in value:
        if isinstance(item, dict):
            notes.append({str(k): _jsonable(v) for k, v in item.items()})
    return notes


def _phrase_row(row: dict[str, Any]) -> dict[str, Any]:
    out = {str(k): _jsonable(v) for k, v in row.items()}
    out["notes_json"] = _notes_json(row.get("notes_json"))
    return out


def _boundary_text(row: dict[str, Any], side: str) -> str:
    score = row.get(f"{side}_boundary_score", 0.0)
    reasons = row.get(f"{side}_boundary_reasons_json", {}) or {}
    parts = [f"{side.capitalize()} boundary score: {score:.3f}"]
    for key, value in reasons.items():
        if isinstance(value, (int, float)):
            parts.append(f"{key:16s} {value:.2f}")
        else:
            parts.append(f"{key:16s} {value}")
    return "\n".join(parts)


def _neighbor_audio(store: PhraseStore, row: dict[str, Any], playback_mode: str, tempo_mode: str, bpm: float):
    notes = _notes_json(row.get("notes_json"))
    target = None
    if playback_mode == "match" and notes:
        target = int(notes[0]["p"])
    neighbor_bpm = bpm if tempo_mode == "fixed" else bpm
    return synthesize_phrase(notes, bpm=neighbor_bpm, target_start_pitch=target)


def launch(root: str | Path = "data/raw/PDMX", share: bool = False):
    import gradio as gr

    store, embeddings = _load(root)
    phrase_df = store.get_dataframe()
    try:
        run_manifest = load_json(Path(root) / "extracted" / "run_manifest.json")
    except Exception:
        run_manifest = {}

    def search_scores(title, composer, instrument, genre):
        return store.search_metadata(title=title or "", composer=composer or "", instrument=instrument or "", genre=genre or "")

    def select_score(score_id):
        if not score_id:
            return pd.DataFrame()
        return store.get_score_phrases(score_id)[
            ["phrase_id", "start_measure", "end_measure", "n_bars", "n_notes", "left_boundary_score", "right_boundary_score"]
        ]

    def select_phrase(phrase_id):
        if not phrase_id:
            return {}, None, None, ""
        row = _phrase_row(store.get_phrase(phrase_id))
        notes = row["notes_json"]
        audio = synthesize_phrase(notes, bpm=100.0)
        title = f"{row.get('title')} - {row.get('composer_name')}"
        return row, (audio[1], audio[0]), phrase_piano_roll(notes, title=title), _boundary_text(row, "left") + "\n\n" + _boundary_text(row, "right")

    def find_neighbors(phrase_id, mode, exclude_same_score, same_instrument, length_only):
        if not phrase_id:
            return pd.DataFrame(), []
        ratio = (0.5, 2.0) if length_only else None
        nn = search_neighbors(
            phrase_id,
            phrase_df,
            embeddings,
            mode=mode,
            k=10,
            exclude_same_score=exclude_same_score,
            same_instrument=same_instrument,
            length_ratio=ratio,
        )
        return nn, nn.to_dict("records")

    def select_neighbor(rank, neighbor_rows, playback_mode, tempo_mode, bpm, query_row):
        if not neighbor_rows:
            return None, None, ""
        idx = max(0, min(int(rank) - 1, len(neighbor_rows) - 1))
        neighbor_row = _phrase_row(neighbor_rows[idx])
        phrase_row = _phrase_row(store.get_phrase(neighbor_row["phrase_id"]))
        phrase_row.update({k: v for k, v in neighbor_row.items() if k not in phrase_row or phrase_row[k] is None})
        audio = _neighbor_audio(store, phrase_row, playback_mode, tempo_mode, bpm)
        title = f"{phrase_row.get('title')} - {phrase_row.get('composer')}"
        return (audio[1], audio[0]), phrase_piano_roll(_notes_json(phrase_row.get("notes_json")), title=title), json.dumps(json_like(phrase_row), indent=2)

    def play_ab(query_row, neighbor_rows, rank, playback_mode, tempo_mode, bpm):
        if not query_row or not neighbor_rows:
            return None
        query_row = _phrase_row(query_row)
        q_notes = _notes_json(query_row.get("notes_json"))
        q_audio = synthesize_phrase(q_notes, bpm=bpm, target_start_pitch=(q_notes[0]["p"] if playback_mode == "match" and q_notes else None))
        idx = max(0, min(int(rank) - 1, len(neighbor_rows) - 1))
        nrow = _phrase_row(neighbor_rows[idx])
        n_audio = _neighbor_audio(store, nrow, playback_mode, tempo_mode, bpm)
        import numpy as np

        q_arr, sr = q_audio
        n_arr, _ = n_audio
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        return (sr, np.concatenate([q_arr, silence, n_arr]).astype(np.float32))

    def review_phrase(query_row, label, note):
        if not query_row:
            return "No phrase selected"
        append_review(root, query_row["phrase_id"], label, note or "", segmentation_config_hash=str(run_manifest.get("segmentation_settings_hash", "")))
        return f"Saved review for {query_row['phrase_id']}"

    def json_like(row):
        return {
            "phrase_id": row.get("phrase_id"),
            "score_id": row.get("score_id"),
            "similarity": row.get("similarity"),
            "title": row.get("title"),
            "composer": row.get("composer"),
            "instrument": row.get("instrument"),
            "measures": row.get("measures"),
            "bars": row.get("bars"),
            "notes": row.get("notes"),
        }

    with gr.Blocks() as demo:
        gr.Markdown("# PDMX Phrase Lab")
        query_state = gr.State({})
        neighbor_state = gr.State([])

        with gr.Row():
            title = gr.Textbox(label="Title contains")
            composer = gr.Textbox(label="Composer contains")
            instrument = gr.Textbox(label="Instrument contains")
            genre = gr.Textbox(label="Genre contains")
        search_btn = gr.Button("Search")
        search_tbl = gr.Dataframe(label="Matching phrases")
        search_btn.click(search_scores, [title, composer, instrument, genre], search_tbl)

        score_id = gr.Textbox(label="Score ID")
        score_btn = gr.Button("Show score phrases")
        score_tbl = gr.Dataframe(label="Phrases from score")
        score_btn.click(select_score, score_id, score_tbl)

        phrase_id = gr.Textbox(label="Phrase ID")
        phrase_btn = gr.Button("Load phrase")
        with gr.Row():
            query_audio = gr.Audio(label="Query audio", type="numpy")
            query_plot = gr.Plot(label="Query piano roll")
        query_info = gr.Textbox(label="Boundary evidence")
        phrase_btn.click(select_phrase, phrase_id, [query_state, query_audio, query_plot, query_info])

        nn_mode = gr.Radio(["combined", "melody", "rhythm"], value="combined", label="Similarity")
        exclude = gr.Checkbox(value=True, label="Exclude phrases from same score")
        same_inst = gr.Checkbox(value=False, label="Same instrument only")
        length_only = gr.Checkbox(value=False, label="Similar phrase length only")
        nn_btn = gr.Button("Find nearest phrases")
        nn_tbl = gr.Dataframe(label="Nearest neighbors")
        nn_btn.click(find_neighbors, [phrase_id, nn_mode, exclude, same_inst, length_only], [nn_tbl, neighbor_state])

        with gr.Row():
            rank = gr.Dropdown(choices=[str(i) for i in range(1, 11)], value="1", label="Neighbor rank")
            playback_mode = gr.Radio(["original", "match"], value="original", label="Playback")
            tempo_mode = gr.Radio(["original", "fixed"], value="fixed", label="Tempo")
            bpm = gr.Slider(40, 180, value=100, label="Comparison BPM")
        neighbor_audio = gr.Audio(label="Neighbor audio", type="numpy")
        neighbor_plot = gr.Plot(label="Neighbor piano roll")
        neighbor_info = gr.Textbox(label="Neighbor metadata")
        rank.change(select_neighbor, [rank, neighbor_state, playback_mode, tempo_mode, bpm, query_state], [neighbor_audio, neighbor_plot, neighbor_info])

        ab_btn = gr.Button("Play query then neighbor")
        ab_audio = gr.Audio(label="A/B audio", type="numpy")
        ab_btn.click(play_ab, [query_state, neighbor_state, rank, playback_mode, tempo_mode, bpm], ab_audio)

        review_note = gr.Textbox(label="Review note")
        review_label = gr.Radio(["Good phrase", "Bad phrase", "Boundary questionable"], value="Good phrase", label="Review label")
        review_btn = gr.Button("Save review")
        review_msg = gr.Textbox(label="Review status")
        review_btn.click(review_phrase, [query_state, review_label, review_note], review_msg)

    demo.launch(share=share)
