from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class EncoderConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.10
    embedding_dim: int = 128
    max_notes: int = 96
    relative_pitch_vocab: int = 101
    interval_vocab: int = 53
    onset_vocab: int = 130
    duration_vocab: int = 66
    ioi_vocab: int = 66


class LearnedPhraseEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig | dict[str, Any]):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = EncoderConfig(**{k: cfg.get(k, getattr(EncoderConfig, k)) for k in EncoderConfig.__annotations__})
        self.cfg = cfg
        self.pad_token = 0
        self.mask_token = 1
        self.channel_embeddings = nn.ModuleList(
            [
                nn.Embedding(cfg.relative_pitch_vocab, cfg.d_model),
                nn.Embedding(cfg.interval_vocab, cfg.d_model),
                nn.Embedding(cfg.onset_vocab, cfg.d_model),
                nn.Embedding(cfg.duration_vocab, cfg.d_model),
                nn.Embedding(cfg.ioi_vocab, cfg.d_model),
            ]
        )
        self.position_embedding = nn.Embedding(cfg.max_notes + 1, cfg.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * cfg.ff_multiplier,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.proj = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.embedding_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("expected [batch, notes, channels] input")
        if tokens.size(-1) != 5:
            raise ValueError("expected 5 token channels")
        tokens = tokens.long()
        mask = tokens[:, :, 0] == self.pad_token
        parts = []
        for i, emb in enumerate(self.channel_embeddings):
            parts.append(emb(tokens[:, :, i].clamp(min=0, max=emb.num_embeddings - 1)))
        x = torch.stack(parts, dim=0).sum(dim=0)
        pos = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0).expand(tokens.size(0), -1)
        x = x + self.position_embedding(pos.clamp(max=self.position_embedding.num_embeddings - 1))
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        pad = torch.cat([torch.zeros(tokens.size(0), 1, dtype=torch.bool, device=tokens.device), mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=pad)
        out = self.proj(x[:, 0])
        return F.normalize(out, dim=-1)

    def num_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

