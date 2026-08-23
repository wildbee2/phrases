from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def encode_batches(model, tokens: np.ndarray, batch_size: int = 256, device: Any | None = None) -> np.ndarray:
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(tokens), batch_size):
            batch = torch.as_tensor(tokens[start : start + batch_size], device=device)
            out.append(model(batch).detach().cpu().numpy())
    if not out:
        return np.zeros((0, getattr(model.cfg, "embedding_dim", 128)), dtype=np.float32)
    return np.vstack(out).astype(np.float32)

