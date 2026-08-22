from __future__ import annotations

import numpy as np

from phrase_lab.features.phrase_features import contour_vector, rhythm_vector


def _phrase(pitches, tempo_scale=1.0):
    notes = []
    t = 0.0
    for p in pitches:
        notes.append({"p": p, "o": t, "d": 1.0 * tempo_scale, "v": 80})
        t += 1.0 * tempo_scale
    return notes


def test_transposition_invariance():
    a = _phrase([60, 62, 64, 67])
    b = _phrase([67, 69, 71, 74])
    ca = contour_vector(a, steps=32)
    cb = contour_vector(b, steps=32)
    sim = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb)))
    assert sim > 0.98


def test_tempo_invariance():
    a = _phrase([60, 62, 64, 67], tempo_scale=1.0)
    b = _phrase([60, 62, 64, 67], tempo_scale=2.0)
    ra = rhythm_vector(a, steps=32)
    rb = rhythm_vector(b, steps=32)
    sim = float(np.dot(ra, rb) / (np.linalg.norm(ra) * np.linalg.norm(rb)))
    assert sim > 0.95


def test_contour_discrimination():
    up = _phrase([60, 62, 64, 65, 67])
    down = _phrase([67, 65, 64, 62, 60])
    cu = contour_vector(up, steps=32)
    cd = contour_vector(down, steps=32)
    sim = float(np.dot(cu, cd) / (np.linalg.norm(cu) * np.linalg.norm(cd)))
    assert sim < 0.5

