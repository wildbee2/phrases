from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.app.review_store import append_blind_vote, append_review, append_similarity_review
from phrase_lab.index.backends import HandcraftedRetrievalBackend, LearnedRetrievalBackend, available_learned_runs
from phrase_lab.index.search import load_embeddings, search_neighbors
from phrase_lab.learning.evaluate import select_fixed_queries
from phrase_lab.music.piano_roll import phrase_piano_roll
from phrase_lab.music.render import synthesize_phrase
from phrase_lab.storage.manifest import load_json
from phrase_lab.storage.phrase_store import PhraseStore


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
            return _notes_json(json.loads(value))
        except Exception:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    notes = []
    for item in value or []:
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
        parts.append(f"{key}: {value}")
    return "\n".join(parts)


def _display_phrase(row: dict[str, Any]) -> tuple[Any, Any, str, dict[str, Any]]:
    notes = _notes_json(row.get("notes_json"))
    audio = synthesize_phrase(notes, bpm=100.0)
    title = f"{row.get('title')} - {row.get('composer_name')}"
    return (audio[1], audio[0]), phrase_piano_roll(notes, title=title), _boundary_text(row, "left") + "\n\n" + _boundary_text(row, "right"), row


def launch(root: str | Path = "data/raw/PDMX", share: bool = False):
    import gradio as gr

    root = Path(root)
    store = PhraseStore(root)
    phrase_df = store.get_dataframe()
    try:
        run_manifest = load_json(root / "extracted" / "run_manifest.json")
    except Exception:
        run_manifest = {}
    learned_runs = available_learned_runs(root)
    from phrase_lab.app.vocabulary_browser import add_vocabulary_tab

    handcrafted_backend = HandcraftedRetrievalBackend(root)
    learned_backend_cache: dict[str, LearnedRetrievalBackend] = {}

    def backend_for(name: str, run_id: str | None):
        if name == "learned" and run_id:
            if run_id not in learned_backend_cache:
                learned_backend_cache[run_id] = LearnedRetrievalBackend(root, run_id)
            return learned_backend_cache[run_id]
        return handcrafted_backend

    def search_scores(title, composer, instrument, genre):
        return store.search_metadata(title=title or "", composer=composer or "", instrument=instrument or "", genre=genre or "")

    def select_score(score_id):
        if not score_id:
            return pd.DataFrame()
        cols = ["phrase_id", "start_measure", "end_measure", "n_bars", "n_notes", "left_boundary_score", "right_boundary_score"]
        return store.get_score_phrases(score_id)[cols]

    def select_phrase(phrase_id, backend_name, run_id):
        if not phrase_id:
            return {}, None, None, "", ""
        backend = backend_for(backend_name, run_id)
        if backend_name == "learned" and not backend.contains(phrase_id):
            row = _phrase_row(store.get_phrase(phrase_id))
            return row, None, None, "This Experiment 002 model was trained/indexed on explicit_voice phrases only.", ""
        row = _phrase_row(store.get_phrase(phrase_id))
        audio, plot, info, row = _display_phrase(row)
        backend_name_text = backend.name if hasattr(backend, "name") else backend_name
        return row, audio, plot, info, backend_name_text

    def find_neighbors(phrase_id, backend_name, run_id, mode, exclude_same_score, same_instrument, length_only, candidate_split):
        if not phrase_id:
            return pd.DataFrame(), []
        backend = backend_for(backend_name, run_id)
        if backend_name == "learned" and not backend.contains(phrase_id):
            return pd.DataFrame([{"message": "This Experiment 002 model was trained/indexed on explicit_voice phrases only."}]), []
        ratio = (0.5, 2.0) if length_only else None
        if backend_name == "learned":
            nn = backend.search(
                phrase_id,
                k=10,
                exclude_same_score=exclude_same_score,
                same_instrument=same_instrument,
                length_ratio=ratio,
                candidate_split=candidate_split or None,
            )
        else:
            nn = search_neighbors(
                phrase_id,
                backend.phrase_df,
                backend.embeddings,
                mode=mode,
                k=10,
                exclude_same_score=exclude_same_score,
                same_instrument=same_instrument,
                length_ratio=ratio,
                candidate_split=candidate_split or None,
            )
        return nn, nn.to_dict("records")

    def select_neighbor(rank, neighbor_rows, playback_mode, tempo_mode, bpm, query_row):
        if not neighbor_rows:
            return None, None, ""
        idx = max(0, min(int(rank) - 1, len(neighbor_rows) - 1))
        neighbor_row = _phrase_row(neighbor_rows[idx])
        phrase_row = _phrase_row(store.get_phrase(neighbor_row["phrase_id"]))
        phrase_row.update({k: v for k, v in neighbor_row.items() if k not in phrase_row or phrase_row[k] is None})
        notes = _notes_json(phrase_row.get("notes_json"))
        target = int(notes[0]["p"]) if playback_mode == "match" and notes else None
        audio = synthesize_phrase(notes, bpm=bpm, target_start_pitch=target)
        title = f"{phrase_row.get('title')} - {phrase_row.get('composer_name')}"
        return (audio[1], audio[0]), phrase_piano_roll(notes, title=title), json.dumps({"phrase_id": phrase_row.get("phrase_id"), "score_id": phrase_row.get("score_id"), "similarity": phrase_row.get("similarity")}, indent=2)

    def play_ab(query_row, neighbor_rows, rank, playback_mode, tempo_mode, bpm):
        if not query_row or not neighbor_rows:
            return None
        query_row = _phrase_row(query_row)
        q_notes = _notes_json(query_row.get("notes_json"))
        q_audio = synthesize_phrase(q_notes, bpm=bpm, target_start_pitch=(q_notes[0]["p"] if playback_mode == "match" and q_notes else None))
        idx = max(0, min(int(rank) - 1, len(neighbor_rows) - 1))
        nrow = _phrase_row(neighbor_rows[idx])
        n_notes = _notes_json(nrow.get("notes_json"))
        n_audio = synthesize_phrase(n_notes, bpm=bpm, target_start_pitch=(n_notes[0]["p"] if playback_mode == "match" and n_notes else None))
        silence = np.zeros(int(0.5 * q_audio[1]), dtype=np.float32)
        return (q_audio[1], np.concatenate([q_audio[0], silence, n_audio[0]]).astype(np.float32))

    def review_phrase(query_row, label, note):
        if not query_row:
            return "No phrase selected"
        append_review(root, query_row["phrase_id"], label, note or "", segmentation_config_hash=str(run_manifest.get("segmentation_settings_hash", "")))
        return f"Saved review for {query_row['phrase_id']}"

    def label_similarity(query_id, neighbor_id, label, backend_name, run_id):
        append_similarity_review(
            root,
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "query_phrase_id": query_id,
                "neighbor_phrase_id": neighbor_id,
                "backend": backend_name,
                "run_id": run_id or "",
                "similarity": "",
                "label": label,
            },
        )
        return "Saved similarity review"

    fixed_queries = pd.DataFrame()
    for run_name in reversed(learned_runs):
        candidate = root / "runs" / "002_contrastive_encoder" / run_name / "evaluation" / "fixed_eval_queries.parquet"
        if candidate.exists():
            fixed_queries = pd.read_parquet(candidate)
            break
    if fixed_queries.empty:
        fixed_queries = select_fixed_queries(phrase_df.assign(split=phrase_df.get("split", "test")), count=100)

    def next_blind_trial(run_id, seen_json):
        seen = set(json.loads(seen_json) if seen_json else [])
        candidates = fixed_queries if not fixed_queries.empty else phrase_df
        candidates = candidates[candidates["extraction_mode"].astype(str) == "explicit_voice"]
        candidates = candidates[candidates["phrase_id"].astype(str).isin(store.get_dataframe()["phrase_id"].astype(str))]
        candidates = candidates[~candidates["phrase_id"].astype(str).isin(seen)]
        if candidates.empty:
            return {}, None, None, "", seen_json, None, None, None, None, "", ""
        row = candidates.sample(n=1, random_state=None).iloc[0].to_dict()
        assignment = np.random.default_rng().permutation(["handcrafted", "learned"])
        q_row = _phrase_row(row)
        q_notes = _notes_json(q_row.get("notes_json"))
        q_audio = synthesize_phrase(q_notes, bpm=100.0, target_start_pitch=(q_notes[0]["p"] if q_notes else None))
        state = {
            "trial_id": pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%f"),
            "query_phrase_id": q_row["phrase_id"],
            "run_id": run_id or "",
            "system_a_backend": str(assignment[0]),
            "system_b_backend": str(assignment[1]),
            "system_a_phrase_ids": [],
            "system_b_phrase_ids": [],
            "comparison_tempo": 100.0,
            "match_starting_pitch": True,
        }
        return (
            state,
            (q_audio[1], q_audio[0]),
            phrase_piano_roll(q_notes, title="Blind query"),
            json.dumps({"query_phrase_id": q_row["phrase_id"]}, indent=2),
            seen_json,
            None,
            None,
            None,
            None,
            "",
            "",
        )

    def blind_select_neighbor(trial_state, system_key, rank):
        if not trial_state:
            return None, None, "", trial_state
        backend_name = trial_state[f"{system_key}_backend"]
        run_id = trial_state.get("run_id")
        backend = backend_for(backend_name, run_id) if backend_name == "learned" else handcrafted_backend
        query_id = trial_state["query_phrase_id"]
        if backend_name == "learned" and not backend.contains(query_id):
            return None, None, "This Experiment 002 model was trained/indexed on explicit_voice phrases only.", trial_state
        nn = backend.search(query_id, k=5, exclude_same_score=True, candidate_split="test" if "split" in backend.phrase_df.columns else None)
        if nn.empty:
            return None, None, "No neighbors available.", trial_state
        idx = max(0, min(int(rank) - 1, len(nn) - 1))
        neighbor_row = _phrase_row(store.get_phrase(nn.iloc[idx]["phrase_id"]))
        trial_state = dict(trial_state)
        trial_state[f"{system_key}_phrase_ids"] = nn["phrase_id"].astype(str).tolist()
        notes = _notes_json(neighbor_row.get("notes_json"))
        audio = synthesize_phrase(notes, bpm=100.0, target_start_pitch=(notes[0]["p"] if notes else None))
        return (audio[1], audio[0]), phrase_piano_roll(notes, title="Hidden neighbor"), "", trial_state

    def save_blind_vote(trial_state, query_row, winner, note, seen_json):
        if not trial_state:
            return "No active trial", seen_json
        vote = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "trial_id": trial_state["trial_id"],
            "query_phrase_id": trial_state["query_phrase_id"],
            "run_id": trial_state.get("run_id", ""),
            "baseline_manifest_hash": str(run_manifest.get("segmentation_settings_hash", "")),
            "dataset_manifest_hash": str(run_manifest.get("segmentation_settings_hash", "")),
            "system_a_backend": trial_state["system_a_backend"],
            "system_b_backend": trial_state["system_b_backend"],
            "system_a_phrase_ids": json.dumps(trial_state.get("system_a_phrase_ids", [])),
            "system_b_phrase_ids": json.dumps(trial_state.get("system_b_phrase_ids", [])),
            "vote_visible_choice": winner,
            "winner_backend": trial_state["system_a_backend"] if winner == "System A" else trial_state["system_b_backend"] if winner == "System B" else "",
            "tie": winner == "Tie",
            "both_poor": winner == "Both poor",
            "comparison_tempo": trial_state.get("comparison_tempo", 100.0),
            "match_starting_pitch": trial_state.get("match_starting_pitch", True),
            "app_version": "gradio_app_v2",
        }
        append_blind_vote(root, vote)
        seen = set(json.loads(seen_json) if seen_json else [])
        seen.add(trial_state["query_phrase_id"])
        visible = f"Saved vote. System A = {trial_state['system_a_backend']}, System B = {trial_state['system_b_backend']}"
        return visible, json.dumps(sorted(seen))

    with gr.Blocks() as demo:
        gr.Markdown("# PDMX Phrase Lab")
        query_state = gr.State({})
        neighbor_state = gr.State([])
        blind_state = gr.State({})
        blind_seen = gr.State("[]")

        with gr.Tab("Neighbor Browser"):
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

            with gr.Row():
                phrase_id = gr.Textbox(label="Phrase ID")
                retrieval_system = gr.Radio(["Handcrafted baseline", "Learned encoder"], value="Handcrafted baseline", label="Retrieval system")
                learned_run = gr.Dropdown(choices=learned_runs or [""], value=learned_runs[0] if learned_runs else "", label="Learned run")
            phrase_btn = gr.Button("Load phrase")
            with gr.Row():
                query_audio = gr.Audio(label="Query audio", type="numpy")
                query_plot = gr.Plot(label="Query piano roll")
            query_info = gr.Textbox(label="Boundary evidence")
            backend_info = gr.Textbox(label="Backend status")
            phrase_btn.click(select_phrase, [phrase_id, retrieval_system, learned_run], [query_state, query_audio, query_plot, query_info, backend_info])

            nn_mode = gr.Radio(["combined", "melody", "rhythm"], value="combined", label="Similarity")
            exclude = gr.Checkbox(value=True, label="Exclude phrases from same score")
            same_inst = gr.Checkbox(value=False, label="Same instrument only")
            length_only = gr.Checkbox(value=False, label="Similar phrase length only")
            candidate_split = gr.Radio(["", "train", "validation", "test"], value="", label="Candidate split")
            nn_btn = gr.Button("Find nearest phrases")
            nn_tbl = gr.Dataframe(label="Nearest neighbors")
            nn_btn.click(find_neighbors, [phrase_id, retrieval_system, learned_run, nn_mode, exclude, same_inst, length_only, candidate_split], [nn_tbl, neighbor_state])

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

            similarity_label = gr.Radio(["Very similar", "Somewhat similar", "Weakly related", "Unrelated"], value="Very similar", label="Neighbor similarity")
            similarity_btn = gr.Button("Save similarity label")
            similarity_msg = gr.Textbox(label="Similarity review status")
            similarity_btn.click(label_similarity, [query_state, phrase_id, similarity_label, retrieval_system, learned_run], similarity_msg)

        with gr.Tab("Blind Encoder Evaluation"):
            gr.Markdown("Query is fixed and the backend identity is hidden until after voting.")
            blind_run = gr.Dropdown(choices=learned_runs or [""], value=learned_runs[0] if learned_runs else "", label="Learned run")
            next_btn = gr.Button("Next blind query")
            with gr.Row():
                blind_query_audio = gr.Audio(label="Query audio", type="numpy")
                blind_query_plot = gr.Plot(label="Query piano roll")
            blind_query_info = gr.Textbox(label="Hidden query info")
            system_a_rank = gr.Dropdown(choices=[str(i) for i in range(1, 6)], value="1", label="System A neighbor rank")
            system_b_rank = gr.Dropdown(choices=[str(i) for i in range(1, 6)], value="1", label="System B neighbor rank")
            system_a_audio = gr.Audio(label="System A audio", type="numpy")
            system_a_plot = gr.Plot(label="System A piano roll")
            system_b_audio = gr.Audio(label="System B audio", type="numpy")
            system_b_plot = gr.Plot(label="System B piano roll")
            system_a_key = gr.State("system_a")
            system_b_key = gr.State("system_b")
            system_a_btn = gr.Button("Load System A neighbor")
            system_b_btn = gr.Button("Load System B neighbor")
            system_a_msg = gr.Textbox(label="System A status")
            system_b_msg = gr.Textbox(label="System B status")
            vote = gr.Radio(["System A", "System B", "Tie", "Both poor"], value="Tie", label="Which system returned the more musically similar set?")
            vote_note = gr.Textbox(label="Optional note")
            save_vote_btn = gr.Button("Save vote")
            vote_msg = gr.Textbox(label="Vote status")
            blind_next_state = gr.Textbox(visible=False, value="[]")

            next_btn.click(next_blind_trial, [blind_run, blind_seen], [blind_state, blind_query_audio, blind_query_plot, blind_query_info, blind_next_state, system_a_audio, system_a_plot, system_b_audio, system_b_plot, system_a_msg, system_b_msg])
            system_a_btn.click(blind_select_neighbor, [blind_state, system_a_key, system_a_rank], [system_a_audio, system_a_plot, system_a_msg, blind_state])
            system_b_btn.click(blind_select_neighbor, [blind_state, system_b_key, system_b_rank], [system_b_audio, system_b_plot, system_b_msg, blind_state])
            save_vote_btn.click(save_blind_vote, [blind_state, query_state, vote, vote_note, blind_next_state], [vote_msg, blind_seen])

        add_vocabulary_tab(gr, root, store)

    demo.launch(share=share)
