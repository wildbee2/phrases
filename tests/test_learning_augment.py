from __future__ import annotations

import numpy as np

from phrase_lab.learning.augment import AugmentationConfig, augment_tokens
from phrase_lab.learning.tokenize import TokenizerConfig, tokenize_phrase


def _phrase():
    notes = []
    t = 0.0
    for p in [60, 62, 64, 65, 67]:
        notes.append({"p": p, "o": t, "d": 1.0, "v": 80})
        t += 1.0
    return notes


def test_augmentations_keep_boundary_notes_and_can_change_internal_tokens():
    tokens, _, _ = tokenize_phrase(_phrase(), TokenizerConfig(max_notes=8, relative_pitch_clip=48, interval_clip=24, onset_bins=16, duration_bins=16, ioi_bins=16))
    rng = np.random.default_rng(0)
    out = augment_tokens(
        tokens,
        rng,
        AugmentationConfig(
            note_mask_probability=1.0,
            feature_mask_probability=0.0,
            timing_jitter_probability=0.0,
            duration_jitter_probability=0.0,
            ornament_dropout_probability=0.0,
        ),
    )
    assert np.array_equal(out[0], tokens[0])
    assert np.array_equal(out[4], tokens[4])
    assert not np.array_equal(out[1], tokens[1])


def test_jitter_keeps_ids_valid():
    tokens, _, _ = tokenize_phrase(_phrase(), TokenizerConfig(max_notes=8, relative_pitch_clip=48, interval_clip=24, onset_bins=16, duration_bins=16, ioi_bins=16))
    rng = np.random.default_rng(1)
    out = augment_tokens(
        tokens,
        rng,
        AugmentationConfig(
            note_mask_probability=0.0,
            feature_mask_probability=0.0,
            timing_jitter_probability=1.0,
            duration_jitter_probability=1.0,
            ornament_dropout_probability=0.0,
        ),
    )
    assert out.shape == tokens.shape
    assert np.all(out >= 0)
