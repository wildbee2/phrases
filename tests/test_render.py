from __future__ import annotations

import numpy as np

from phrase_lab.music.render import synthesize_phrase


def test_render_returns_audio():
    audio, sr = synthesize_phrase([{"p": 60, "o": 0.0, "d": 1.0, "v": 80}, {"p": 64, "o": 1.0, "d": 1.0, "v": 80}], bpm=100)
    assert sr > 0
    assert len(audio) > 0
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) <= 1.0 + 1e-6

