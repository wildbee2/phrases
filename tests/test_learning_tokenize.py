from __future__ import annotations

import numpy as np

from phrase_lab.learning.tokenize import TokenizerConfig, tokenize_phrase


def _phrase(pitches, tempo_scale=1.0):
    notes = []
    t = 0.0
    for p in pitches:
        notes.append({"p": p, "o": t, "d": 1.0 * tempo_scale, "v": 80})
        t += 1.0 * tempo_scale
    return notes


def test_tokenization_is_transposition_and_tempo_invariant():
    cfg = TokenizerConfig(max_notes=8, relative_pitch_clip=48, interval_clip=24, onset_bins=16, duration_bins=16, ioi_bins=16)
    a, n_a, _ = tokenize_phrase(_phrase([60, 62, 64, 67]), cfg)
    b, n_b, _ = tokenize_phrase(_phrase([67, 69, 71, 74]), cfg)
    c, n_c, _ = tokenize_phrase(_phrase([60, 62, 64, 67], tempo_scale=1.5), cfg)
    assert n_a == n_b == n_c == 4
    assert np.array_equal(a, b)
    assert np.array_equal(a, c)


def test_pad_behavior_and_max_length():
    cfg = TokenizerConfig(max_notes=4, relative_pitch_clip=48, interval_clip=24, onset_bins=16, duration_bins=16, ioi_bins=16)
    tokens, n_notes, _ = tokenize_phrase(_phrase([60, 62]), cfg)
    assert tokens.shape == (4, 5)
    assert n_notes == 2
    assert np.all(tokens[2:] == 0)
    too_long = _phrase([60, 61, 62, 63, 64])
    try:
        tokenize_phrase(too_long, cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("expected max length rejection")
