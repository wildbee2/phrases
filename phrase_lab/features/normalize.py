from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True) if x.ndim == 2 else np.linalg.norm(x)
    norm = np.maximum(norm, 1e-8)
    return x / norm

