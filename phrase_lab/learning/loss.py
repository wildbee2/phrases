from __future__ import annotations

import torch
import torch.nn.functional as F


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")
    if z1.ndim != 2:
        raise ValueError("expected [batch, dim] embeddings")
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    z = torch.cat([z1, z2], dim=0)
    logits = z @ z.t() / max(temperature, 1e-6)
    logits = logits - torch.eye(logits.size(0), device=logits.device) * 1e9
    targets = torch.arange(z1.size(0), device=z1.device)
    loss_a = F.cross_entropy(logits[: z1.size(0), z1.size(0) :], targets)
    loss_b = F.cross_entropy(logits[z1.size(0) :, : z1.size(0)], targets)
    return 0.5 * (loss_a + loss_b)

