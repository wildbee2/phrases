from __future__ import annotations

import numpy as np
import pytest


torch = pytest.importorskip("torch")

from phrase_lab.learning.loss import nt_xent_loss
from phrase_lab.learning.model import EncoderConfig, LearnedPhraseEncoder
from phrase_lab.learning.tokenize import TokenizerConfig, tokenize_phrase


def _tokens():
    notes = []
    t = 0.0
    for p in [60, 62, 64, 67]:
        notes.append({"p": p, "o": t, "d": 1.0, "v": 80})
        t += 1.0
    return tokenize_phrase(notes, TokenizerConfig(max_notes=8, relative_pitch_clip=48, interval_clip=24, onset_bins=16, duration_bins=16, ioi_bins=16))[0]


def test_model_forward_shape_and_normalization():
    model = LearnedPhraseEncoder(EncoderConfig(d_model=32, n_layers=2, n_heads=4, ff_multiplier=2, dropout=0.0, embedding_dim=16, max_notes=8))
    batch = torch.as_tensor(np.stack([_tokens(), _tokens()]), dtype=torch.long)
    out = model(batch)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()
    norms = torch.linalg.norm(out, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_contrastive_loss_and_tiny_optimization():
    z1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    z2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    shuffled = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    good = float(nt_xent_loss(z1, z2).item())
    bad = float(nt_xent_loss(z1, shuffled).item())
    assert np.isfinite(good)
    assert good < bad

    param = torch.nn.Parameter(torch.randn(2, 2))
    optimizer = torch.optim.SGD([param], lr=0.5)
    before = float(nt_xent_loss(param, param).item())
    for _ in range(20):
        optimizer.zero_grad()
        loss = nt_xent_loss(param, param)
        loss.backward()
        optimizer.step()
    after = float(nt_xent_loss(param, param).item())
    assert after <= before
