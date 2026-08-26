from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.app.review_store import append_vocabulary_blind_trial, append_vocabulary_cluster_review
from phrase_lab.music.piano_roll import phrase_piano_roll
from phrase_lab.music.render import synthesize_phrase
from phrase_lab.storage.phrase_store import PhraseStore
from phrase_lab.vocabulary.codebook import discover_codebooks, load_codebook
from phrase_lab.vocabulary.manifest import vocabulary_root
from phrase_lab.vocabulary.sampling import build_audio_montage, sample_cluster_members, select_hard_negative


def _notes_json(value: Any) -> list[dict[str, Any]]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            return _notes_json(json.loads(value))
        except Exception:
            return []
    notes = []
    for item in value or []:
        if isinstance(item, dict):
            notes.append(item)
    return notes


def _phrase_row(store: PhraseStore, phrase_id: str) -> dict[str, Any]:
    row = store.get_phrase(phrase_id)
    row["notes_json"] = _notes_json(row.get("notes_json"))
    return row


def add_vocabulary_tab(gr, root: str | Path, store: PhraseStore):
    root = Path(root)

    def list_spaces():
        base = vocabulary_root(root)
        if not base.exists():
            return []
        return [p.name for p in base.iterdir() if p.is_dir()]

    def list_ks(space: str):
        base = vocabulary_root(root) / space
        if not base.exists():
            return []
        return sorted({int(p.name[1:]) for p in base.iterdir() if p.is_dir() and p.name.startswith("k")})

    def list_tokens(space: str, k: int):
        codebooks = discover_codebooks(root, space)
        codebooks = [p for p in codebooks if int(load_codebook(p)["manifest"]["k"]) == int(k)]
        if not codebooks:
            return []
        assignments = load_codebook(sorted(codebooks)[-1])["assignments"]
        return sorted(assignments["token"].astype(str).unique().tolist())

    def summarize_token(space: str, k: int, token: str, sampling: str, n_phrases: int):
        codebooks = discover_codebooks(root, space)
        codebooks = [p for p in codebooks if int(load_codebook(p)["manifest"]["k"]) == int(k)]
        if not codebooks:
            return "No codebook available.", pd.DataFrame(), [], "[]", gr.update(choices=[], value=None)
        cb = load_codebook(sorted(codebooks)[-1])
        assignments = cb["assignments"]
        rows = sample_cluster_members(assignments, token, mode=sampling, n=int(n_phrases), seed=42)
        if rows.empty:
            return "No phrases found for token.", pd.DataFrame(), [], "[]", gr.update(choices=[], value=None)
        summary = rows.iloc[0][["token", "cluster_id", "cosine_to_centroid", "assignment_margin"]].to_dict()
        cluster = cb["cluster_stats"][cb["cluster_stats"]["token"].astype(str) == str(token)]
        if not cluster.empty:
            summary.update(cluster.iloc[0].to_dict())
        sample_df = rows[["phrase_id", "score_id", "part_id", "voice_id", "start_q", "end_q", "cosine_to_centroid", "assignment_margin", "rank_within_cluster_by_centroid_similarity"]].copy()
        selected = []
        for _, row in rows.iterrows():
            phrase = _phrase_row(store, str(row["phrase_id"]))
            selected.append(phrase)
        phrase_ids = list(rows["phrase_id"].astype(str))
        return json.dumps(summary, indent=2), sample_df, selected, json.dumps(selected), gr.update(choices=phrase_ids, value=(phrase_ids[0] if phrase_ids else None))

    def render_member(phrase_id: str, match_start_pitch: bool, fixed_tempo: bool, bpm: float):
        if not phrase_id:
            return None, None, ""
        row = _phrase_row(store, phrase_id)
        notes = row.get("notes_json", [])
        audio = synthesize_phrase(notes, bpm=bpm, target_start_pitch=(notes[0]["p"] if match_start_pitch and notes else None))
        plot = phrase_piano_roll(notes, title=f"{row.get('title')} - {row.get('composer_name')}")
        info = json.dumps(
            {
                "phrase_id": row.get("phrase_id"),
                "score_id": row.get("score_id"),
                "part_id": row.get("part_id"),
                "voice_id": row.get("voice_id"),
                "start_measure": row.get("start_measure"),
                "end_measure": row.get("end_measure"),
                "n_bars": row.get("n_bars"),
                "n_notes": row.get("n_notes"),
            },
            indent=2,
        )
        return (audio[1], audio[0]), plot, info

    def play_montage(rows: list[dict[str, Any]], match_start_pitch: bool, bpm: float):
        if not rows:
            return None
        df = pd.DataFrame(rows)
        audio, sr = build_audio_montage(df, bpm=bpm, match_start_pitch=match_start_pitch)
        return (sr, audio)

    def save_cluster_review(space: str, k: int, token: str, rating: str, note: str, sampled_rows_json: str):
        sampled_rows = json.loads(sampled_rows_json) if sampled_rows_json else []
        codebooks = discover_codebooks(root, space)
        codebooks = [p for p in codebooks if int(load_codebook(p)["manifest"]["k"]) == int(k)]
        manifest_hash = load_codebook(sorted(codebooks)[-1])["manifest"].get("config_hash", "") if codebooks else ""
        append_vocabulary_cluster_review(
            root,
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "space": space,
                "k": int(k),
                "token": token,
                "cluster_size": int(len(sampled_rows)),
                "sampled_phrase_ids": json.dumps([r.get("phrase_id") for r in sampled_rows]),
                "sampling_seed": 42,
                "rating": rating,
                "note": note,
                "codebook_manifest_hash": manifest_hash,
            },
        )
        return "Saved cluster review"

    def next_blind_trial(space: str, k: int, seen_json: str):
        codebooks = discover_codebooks(root, space)
        codebooks = [p for p in codebooks if int(load_codebook(p)["manifest"]["k"]) == int(k)]
        if not codebooks:
            return {}, None, None, None, None, None, None, None, seen_json
        cb = load_codebook(sorted(codebooks)[-1])
        assignments = cb["assignments"]
        phrase_df = store.get_dataframe()
        seen = set(json.loads(seen_json) if seen_json else [])
        candidates = phrase_df
        candidates = candidates[candidates["extraction_mode"].astype(str) == "explicit_voice"]
        candidates = candidates[candidates["phrase_id"].astype(str).isin(assignments["phrase_id"].astype(str))]
        candidates = candidates[~candidates["phrase_id"].astype(str).isin(seen)]
        if candidates.empty:
            return {}, None, None, None, None, None, None, None, seen_json
        query_row = candidates.sample(n=1, random_state=None).iloc[0].to_dict()
        query_row["notes_json"] = _notes_json(query_row.get("notes_json"))
        query_token = assignments[assignments["phrase_id"].astype(str) == str(query_row["phrase_id"])].iloc[0]["token"]
        same = assignments[(assignments["token"].astype(str) == str(query_token)) & (assignments["phrase_id"].astype(str) != str(query_row["phrase_id"]))]
        if same.empty:
            return {}, None, None, None, None, None, None, None, seen_json
        same_row = same.sample(n=1, random_state=None).iloc[0].to_dict()
        diff_row = select_hard_negative(assignments, str(query_row["phrase_id"]), str(query_token), phrase_df=phrase_df, seed=int(pd.Timestamp.utcnow().microsecond % (2**31 - 1)))
        assignment = np.random.default_rng().permutation(["a", "b"])
        a_row = same_row if assignment[0] == "a" else diff_row
        b_row = diff_row if assignment[0] == "a" else same_row
        a_phrase = _phrase_row(store, str(a_row["phrase_id"]))
        b_phrase = _phrase_row(store, str(b_row["phrase_id"]))
        q_notes = query_row["notes_json"]
        q_audio = synthesize_phrase(q_notes, bpm=100.0, target_start_pitch=(q_notes[0]["p"] if q_notes else None))
        a_audio = synthesize_phrase(a_phrase["notes_json"], bpm=100.0, target_start_pitch=(a_phrase["notes_json"][0]["p"] if a_phrase["notes_json"] else None))
        b_audio = synthesize_phrase(b_phrase["notes_json"], bpm=100.0, target_start_pitch=(b_phrase["notes_json"][0]["p"] if b_phrase["notes_json"] else None))
        state = {
            "trial_id": pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%f"),
            "space": space,
            "k": int(k),
            "query_phrase_id": str(query_row["phrase_id"]),
            "query_token": str(query_token),
            "candidate_a_phrase_id": str(a_row["phrase_id"]),
            "candidate_b_phrase_id": str(b_row["phrase_id"]),
            "candidate_a_token": str(a_row["token"]),
            "candidate_b_token": str(b_row["token"]),
            "same_cluster_side": "a" if a_row["phrase_id"] == same_row["phrase_id"] else "b",
            "negative_sampling_method": "nearest_other_token",
            "match_starting_pitch": True,
            "fixed_tempo": True,
            "codebook_manifest_hash": cb["manifest"].get("config_hash", ""),
        }
        return (
            state,
            (q_audio[1], q_audio[0]),
            phrase_piano_roll(q_notes, title="Blind query"),
            (a_audio[1], a_audio[0]),
            phrase_piano_roll(a_phrase["notes_json"], title="Candidate A"),
            (b_audio[1], b_audio[0]),
            phrase_piano_roll(b_phrase["notes_json"], title="Candidate B"),
            json.dumps({"query_phrase_id": query_row["phrase_id"]}, indent=2),
            seen_json,
        )

    def save_blind_vote(trial_state: dict[str, Any], visible_vote: str, seen_json: str):
        if not trial_state:
            return "No active trial", "", seen_json
        same_side = trial_state["same_cluster_side"]
        same_cluster_won = (visible_vote == "A" and same_side == "a") or (visible_vote == "B" and same_side == "b")
        append_vocabulary_blind_trial(
            root,
            {
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "trial_id": trial_state["trial_id"],
                "space": trial_state["space"],
                "k": int(trial_state["k"]),
                "query_phrase_id": trial_state["query_phrase_id"],
                "query_token": trial_state["query_token"],
                "candidate_a_phrase_id": trial_state["candidate_a_phrase_id"],
                "candidate_b_phrase_id": trial_state["candidate_b_phrase_id"],
                "candidate_a_token": trial_state["candidate_a_token"],
                "candidate_b_token": trial_state["candidate_b_token"],
                "same_cluster_side": trial_state["same_cluster_side"],
                "visible_vote": visible_vote,
                "same_cluster_won": bool(same_cluster_won),
                "tie": bool(visible_vote == "Tie"),
                "neither": bool(visible_vote == "Neither"),
                "negative_sampling_method": trial_state["negative_sampling_method"],
                "match_starting_pitch": bool(trial_state["match_starting_pitch"]),
                "fixed_tempo": bool(trial_state["fixed_tempo"]),
                "codebook_manifest_hash": trial_state["codebook_manifest_hash"],
            },
        )
        seen = set(json.loads(seen_json) if seen_json else [])
        seen.add(trial_state["query_phrase_id"])
        reveal = f"Saved blind vote. Same-cluster candidate was {same_side.upper()}."
        return "Saved blind vote", reveal, json.dumps(sorted(seen))

    with gr.Tab("Phrase Vocabulary Explorer"):
        space = gr.Dropdown(choices=list_spaces(), value=(list_spaces()[0] if list_spaces() else None), label="Space")
        k = gr.Dropdown(choices=list_ks(list_spaces()[0]) if list_spaces() else [], value=(list_ks(list_spaces()[0])[0] if list_spaces() and list_ks(list_spaces()[0]) else None), label="Vocabulary size")
        token = gr.Dropdown(choices=[], label="Token", allow_custom_value=True)
        sampling = gr.Radio(["Random members", "Centroid-nearest", "Low-confidence"], value="Random members", label="Sampling")
        n_phrases = gr.Slider(1, 12, value=5, step=1, label="Number of phrases")
        summary = gr.Textbox(label="Cluster summary")
        members_tbl = gr.Dataframe(label="Sampled members")
        member_state = gr.State([])
        sampled_rows_state = gr.State("[]")
        review_rating = gr.Radio(["Strongly coherent", "Mostly coherent", "Mixed", "Not coherent"], value="Mostly coherent", label="Cluster coherence")
        review_note = gr.Textbox(label="What seems shared?")
        review_btn = gr.Button("Save cluster review")
        review_msg = gr.Textbox(label="Review status")
        member_selector = gr.Dropdown(choices=[], label="Member phrase")
        member_audio = gr.Audio(label="Member audio", type="numpy")
        member_plot = gr.Plot(label="Member piano roll")
        member_info = gr.Textbox(label="Member info")
        montage_btn = gr.Button("Play montage")
        montage_audio = gr.Audio(label="Cluster montage", type="numpy")
        blind_next_btn = gr.Button("Next blind trial")
        blind_query_audio = gr.Audio(label="Query audio", type="numpy")
        blind_query_plot = gr.Plot(label="Query piano roll")
        blind_a_audio = gr.Audio(label="Candidate A audio", type="numpy")
        blind_a_plot = gr.Plot(label="Candidate A piano roll")
        blind_b_audio = gr.Audio(label="Candidate B audio", type="numpy")
        blind_b_plot = gr.Plot(label="Candidate B piano roll")
        blind_vote = gr.Radio(["A", "B", "Tie", "Neither"], value="Tie", label="Which candidate is more musically similar to the query?")
        blind_msg = gr.Textbox(label="Blind status")
        blind_reveal = gr.Textbox(label="Reveal after vote")
        blind_state = gr.State({})
        blind_seen = gr.State("[]")
        blind_query_info = gr.Textbox(label="Hidden query info")

        def _refresh(space_value: str, k_value: int, token_value: str | None, sampling_value: str, n_value: int):
            ks = list_ks(space_value) if space_value else []
            tokens = list_tokens(space_value, int(k_value)) if space_value and k_value else []
            token_value = token_value or (tokens[0] if tokens else None)
            if not token_value:
                return gr.update(choices=ks, value=(ks[0] if ks else None)), gr.update(choices=tokens, value=None), "", pd.DataFrame(), [], "[]", gr.update(choices=[]), None, None, ""
            summary_text, members, selected_rows, rows, phrase_ids = summarize_token(space_value, int(k_value), token_value, sampling_value.lower(), int(n_value))
            return gr.update(choices=ks, value=k_value), gr.update(choices=tokens, value=token_value), summary_text, members, selected_rows, json.dumps([r if isinstance(r, dict) else {} for r in selected_rows]), gr.update(choices=phrase_ids, value=(phrase_ids[0] if phrase_ids else None)), rows, phrase_ids, summary_text

        space.change(lambda s: gr.update(choices=list_ks(s), value=(list_ks(s)[0] if list_ks(s) else None)), space, k)
        k.change(lambda s, kv: gr.update(choices=list_tokens(s, int(kv)), value=(list_tokens(s, int(kv))[0] if list_tokens(s, int(kv)) else None)) if s and kv else gr.update(choices=[], value=None), [space, k], token)
        token.change(lambda s, kv, t, samp, n: summarize_token(s, int(kv), t, samp.lower(), int(n)), [space, k, token, sampling, n_phrases], [summary, members_tbl, member_state, sampled_rows_state, member_selector])
        sampling.change(lambda s, kv, t, samp, n: summarize_token(s, int(kv), t, samp.lower(), int(n)), [space, k, token, sampling, n_phrases], [summary, members_tbl, member_state, sampled_rows_state, member_selector])
        n_phrases.change(lambda s, kv, t, samp, n: summarize_token(s, int(kv), t, samp.lower(), int(n)), [space, k, token, sampling, n_phrases], [summary, members_tbl, member_state, sampled_rows_state, member_selector])
        member_selector.change(lambda pid: render_member(pid, True, True, 100.0), member_selector, [member_audio, member_plot, member_info])
        montage_btn.click(lambda rows_json, _: play_montage(json.loads(rows_json) if rows_json else [], True, 100.0), [sampled_rows_state, member_selector], montage_audio)
        review_btn.click(save_cluster_review, [space, k, token, review_rating, review_note, sampled_rows_state], review_msg)
        blind_next_btn.click(next_blind_trial, [space, k, blind_seen], [blind_state, blind_query_audio, blind_query_plot, blind_a_audio, blind_a_plot, blind_b_audio, blind_b_plot, blind_query_info, blind_seen])
        blind_vote.change(save_blind_vote, [blind_state, blind_vote, blind_seen], [blind_msg, blind_reveal, blind_seen])

        def _list_sequences():
            path = vocabulary_root(root) / "phrase_sequences.parquet"
            if not path.exists():
                return pd.DataFrame()
            return pd.read_parquet(path)

        def _source_lists():
            seq = _list_sequences()
            if seq.empty:
                return [], [], []
            return sorted(seq["score_id"].astype(str).dropna().unique().tolist()), sorted(seq["part_id"].astype(str).dropna().unique().tolist()), sorted(seq["voice_id"].astype(str).dropna().unique().tolist())

        def _load_source_sequence(score_id: str, part_id: str, voice_id: str):
            seq = _list_sequences()
            if seq.empty:
                return pd.DataFrame(), [], None, None, ""
            rows = seq.copy()
            rows = rows[rows["score_id"].astype(str) == str(score_id)]
            rows = rows[rows["part_id"].astype(str) == str(part_id)]
            rows = rows[rows["voice_id"].astype(str) == str(voice_id)]
            if rows.empty:
                return pd.DataFrame(), [], None, None, ""
            row = rows.iloc[0]
            phrase_ids = row["phrase_ids"]
            table = pd.DataFrame(
                {
                    "index": list(range(1, len(phrase_ids) + 1)),
                    "phrase_id": phrase_ids,
                    "melody_token": row["melody_tokens"],
                    "rhythm_token": row["rhythm_tokens"],
                    "start_q": row["start_qs"],
                    "end_q": row["end_qs"],
                }
            )
            return table, phrase_ids, gr.update(choices=phrase_ids, value=(phrase_ids[0] if phrase_ids else None)), phrase_ids, json.dumps({"sequence_length": int(row["sequence_length"])}, indent=2)

        def _render_source_phrase(phrase_id: str):
            if not phrase_id:
                return None, None, ""
            row = _phrase_row(store, phrase_id)
            notes = row.get("notes_json", [])
            audio = synthesize_phrase(notes, bpm=100.0, target_start_pitch=(notes[0]["p"] if notes else None))
            plot = phrase_piano_roll(notes, title=f"{row.get('title')} - {row.get('composer_name')}")
            info = json.dumps({"phrase_id": row.get("phrase_id"), "score_id": row.get("score_id"), "part_id": row.get("part_id"), "voice_id": row.get("voice_id"), "start_q": row.get("start_q"), "end_q": row.get("end_q")}, indent=2)
            return (audio[1], audio[0]), plot, info

        score_choices, part_choices, voice_choices = _source_lists()

    with gr.Tab("Source Work Token Browser"):
        source_member_state = gr.State([])
        source_rows_state = gr.State("[]")
        source_score3 = gr.Dropdown(choices=score_choices, label="Score ID")
        source_part3 = gr.Dropdown(choices=part_choices, label="Part ID")
        source_voice3 = gr.Dropdown(choices=voice_choices, label="Voice ID")
        source_btn3 = gr.Button("Load source work sequence")
        source_table3 = gr.Dataframe(label="Phrase sequence")
        source_member3 = gr.Dropdown(choices=[], label="Selected phrase")
        source_audio3 = gr.Audio(label="Selected phrase audio", type="numpy")
        source_plot3 = gr.Plot(label="Selected phrase piano roll")
        source_info3 = gr.Textbox(label="Selected phrase info")
        source_btn3.click(_load_source_sequence, [source_score3, source_part3, source_voice3], [source_table3, source_member_state, source_member3, source_rows_state, source_info3])
        source_member3.change(_render_source_phrase, source_member3, [source_audio3, source_plot3, source_info3])
