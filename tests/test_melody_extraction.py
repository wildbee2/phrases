from __future__ import annotations

import music21 as m21

from phrase_lab.music.melody import extract_melodic_lines


def _base_score() -> m21.stream.Score:
    s = m21.stream.Score()
    p = m21.stream.Part(id="P1")
    p.insert(0, m21.instrument.Piano())
    s.append(p)
    return s


def test_monophonic_voice_is_recovered():
    s = _base_score()
    p = s.parts[0]
    v = m21.stream.Voice()
    for i, pitch in enumerate([60, 62, 64, 65, 67, 69, 71, 72, 74, 76, 77, 79]):
        v.insert(i, m21.note.Note(pitch, quarterLength=1))
    m = m21.stream.Measure(number=1)
    m.insert(0, v)
    p.append(m)
    lines = extract_melodic_lines(s, "score1", min_notes_per_line=4)
    assert len(lines) == 1
    assert lines[0].extraction_mode == "explicit_voice"
    assert [n.pitch for n in lines[0].notes][:3] == [60, 62, 64]


def test_chordal_passage_uses_skyline():
    s = _base_score()
    p = s.parts[0]
    for i, pitches in enumerate([[60, 64, 72], [62, 65, 74], [64, 67, 76], [65, 69, 77], [67, 71, 79], [69, 72, 81], [71, 74, 83], [72, 76, 84], [74, 77, 86], [76, 79, 88], [77, 81, 89], [79, 83, 91]]):
        p.insert(i, m21.chord.Chord(pitches, quarterLength=1))
    lines = extract_melodic_lines(s, "score2", min_notes_per_line=4)
    assert len(lines) == 1
    assert lines[0].extraction_mode == "skyline"
    assert all(n.pitch == max(ch) for n, ch in zip(lines[0].notes, [[60, 64, 72], [62, 65, 74], [64, 67, 76], [65, 69, 77], [67, 71, 79], [69, 72, 81], [71, 74, 83], [72, 76, 84], [74, 77, 86], [76, 79, 88], [77, 81, 89], [79, 83, 91]]))


def test_percussion_is_skipped():
    s = m21.stream.Score()
    p = m21.stream.Part(id="perc")
    p.insert(0, m21.instrument.Woodblock())
    p.getInstrument().isPercussion = True
    for i in range(4):
        p.insert(i, m21.note.Note(60, quarterLength=1))
    s.append(p)
    lines = extract_melodic_lines(s, "score3", min_notes_per_line=1)
    assert lines == []

