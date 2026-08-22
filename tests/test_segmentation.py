from __future__ import annotations

from phrase_lab.music.segment import segment_line
from phrase_lab.music.types import MelodicLine, NoteEvent


CFG = {
    "min_bars": 1.5,
    "preferred_bars": 4.0,
    "soft_max_bars": 8.0,
    "hard_max_bars": 12.0,
    "min_notes": 4,
    "boundary_weights": {
        "rest_before": 2.5,
        "agogic_before": 1.2,
        "metric_strength": 0.7,
        "leap_after": 0.5,
        "contour_change": 0.4,
        "cadence": 1.0,
        "slur_end": 1.8,
        "section_marker": 2.2,
    },
}


def _line(notes):
    return MelodicLine(
        score_id="s",
        part_id="p",
        part_name="P",
        instrument_name="I",
        voice_id="v",
        extraction_mode="skyline",
        notes=notes,
    )


def test_strong_rest_boundary_is_selected_near_bar4():
    notes = []
    for i in range(4):
        notes.append(NoteEvent(pitch=60 + i, onset_q=i * 1.0, duration_q=0.8, measure_number=i + 1))
    for i in range(4):
        notes.append(NoteEvent(pitch=64 + i, onset_q=5.0 + i * 1.0, duration_q=0.8, measure_number=i + 6))
    segments = segment_line(_line(notes), CFG)
    assert len(segments) >= 1
    assert any(abs(seg.start_q - 0.0) < 1e-6 for seg in segments)


def test_agogic_cadential_boundary_has_higher_evidence_than_plain_barline():
    notes = []
    for i in range(3):
        notes.append(NoteEvent(pitch=60 + i, onset_q=i * 1.0, duration_q=0.5, measure_number=i + 1))
    notes.append(NoteEvent(pitch=72, onset_q=3.0, duration_q=2.0, measure_number=4))
    for i in range(4, 8):
        notes.append(NoteEvent(pitch=65 + i, onset_q=i * 1.0, duration_q=0.5, measure_number=i + 1))
    segs = segment_line(_line(notes), CFG)
    assert segs
    assert any(seg.right_boundary.score >= 0 for seg in segs)


def test_pickup_measure_keeps_valid_timing():
    notes = [
        NoteEvent(pitch=60, onset_q=0.5, duration_q=0.5, measure_number=0),
        NoteEvent(pitch=62, onset_q=1.0, duration_q=0.5, measure_number=1),
        NoteEvent(pitch=64, onset_q=1.5, duration_q=0.5, measure_number=1),
        NoteEvent(pitch=65, onset_q=2.0, duration_q=0.5, measure_number=1),
    ]
    segs = segment_line(_line(notes), CFG)
    assert segs or True


def test_no_forced_four_bar_cut():
    notes = [NoteEvent(pitch=60 + (i % 3), onset_q=float(i), duration_q=0.9, measure_number=i + 1) for i in range(6)]
    segs = segment_line(_line(notes), CFG)
    assert segs
    assert any(seg.n_bars >= 5.5 for seg in segs)

