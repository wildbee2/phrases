from __future__ import annotations

import hashlib
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phrase_lab.storage.manifest import save_json
from .augment import AugmentationConfig, augment_tokens
from .checkpoint import save_checkpoint, load_checkpoint
from .dataset import load_prepared_dataset
from .embed import encode_batches
from .loss import nt_xent_loss
from .model import EncoderConfig, LearnedPhraseEncoder
from .runs import make_run_id, run_dir, version_root


def _torch():
    import torch

    return torch


def _device():
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_mined_pairs(root: Path) -> dict[str, list[str]]:
    path = version_root(root) / "mined_positive_pairs.parquet"
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    mapping: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        mapping.setdefault(str(row["phrase_id_a"]), []).append(str(row["phrase_id_b"]))
        mapping.setdefault(str(row["phrase_id_b"]), []).append(str(row["phrase_id_a"]))
    return mapping


def _batch_indices(df: pd.DataFrame, batch_size: int, seed: int = 42) -> list[list[int]]:
    from .dataset import score_aware_batches

    return score_aware_batches(df, batch_size=batch_size, seed=seed)


def train_encoder(root: str | Path, cfg: dict[str, Any], run_id: str | None = None, resume: bool = False, max_train_batches: int | None = None) -> dict[str, Any]:
    root = Path(root)
    prepared = load_prepared_dataset(root)
    base = version_root(root)
    model_cfg = EncoderConfig(**cfg["model"])
    aug_cfg = AugmentationConfig(**cfg["augmentation"])
    config_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()
    run_id = run_id or make_run_id(config_hash)
    run = run_dir(root, run_id)
    if run.exists() and not resume:
        raise FileExistsError(f"run directory already exists: {run}")
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "tensorboard").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(parents=True, exist_ok=True)
    import yaml

    with open(run / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    save_json(run / "dataset_manifest.json", prepared.dataset_manifest)
    torch = _torch()
    device, device_name = _device()
    _set_seed(int(cfg["experiment"]["seed"]))
    model = LearnedPhraseEncoder(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["training"]["epochs"])), eta_min=float(cfg["training"]["min_learning_rate"]))
    start_epoch = 0
    best_val = float("inf")
    if resume and (run / "checkpoints" / "last.pt").exists():
        ckpt = load_checkpoint(run / "checkpoints" / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val_loss", float("inf")))
    train_mask = prepared.phrase_metadata["split"].astype(str) == "train"
    val_mask = prepared.phrase_metadata["split"].astype(str) == "validation"
    train_idx = np.flatnonzero(train_mask.to_numpy())
    val_idx = np.flatnonzero(val_mask.to_numpy())
    mined = _load_mined_pairs(root)
    effective_batch = int(cfg["training"]["batch_size"]) * int(cfg["training"]["gradient_accumulation_steps"])
    print(f"trainable parameter count: {model.num_parameters()}")
    print(f"estimated number of training examples: {len(train_idx)}")
    print(f"effective batch size: {effective_batch}")
    print(f"device: {device_name}")
    print(f"AMP status: {bool(cfg['training']['amp'] and device_name in {'cuda', 'mps'})}")
    metrics_path = run / "metrics.csv"
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        if metrics_path.stat().st_size == 0:
            writer.writeheader()
        patience = 0
        for epoch in range(start_epoch, int(cfg["training"]["epochs"])):
            model.train()
            rng = np.random.default_rng(int(cfg["experiment"]["seed"]) + epoch)
            order = np.array(train_idx)
            rng.shuffle(order)
            total_loss = 0.0
            seen = 0
            optimizer.zero_grad(set_to_none=True)
            amp_enabled = bool(cfg["training"]["amp"] and device_name in {"cuda", "mps"})
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and device_name == "cuda")
            for batch_no, start in enumerate(range(0, len(order), int(cfg["training"]["batch_size"]))):
                batch_idx = order[start : start + int(cfg["training"]["batch_size"])]
                view1 = []
                view2 = []
                for idx in batch_idx:
                    tokens = prepared.tokens[idx]
                    pos_ids = mined.get(str(prepared.phrase_ids[idx]), [])
                    pos_idx = idx
                    if pos_ids and rng.random() < float(cfg["training"]["mined_positive_probability"]):
                        choice = rng.choice(pos_ids)
                        matches = np.flatnonzero(prepared.phrase_ids == choice)
                        if len(matches):
                            pos_idx = int(matches[0])
                    pos_tokens = prepared.tokens[pos_idx]
                    view1.append(augment_tokens(tokens, rng, aug_cfg))
                    view2.append(augment_tokens(pos_tokens, rng, aug_cfg))
                v1 = torch.as_tensor(np.stack(view1), device=device)
                v2 = torch.as_tensor(np.stack(view2), device=device)
                with torch.autocast(device_type=device_name, enabled=amp_enabled):
                    z1 = model(v1)
                    z2 = model(v2)
                    loss = nt_xent_loss(z1, z2, temperature=float(cfg["training"]["temperature"]))
                scaler.scale(loss / int(cfg["training"]["gradient_accumulation_steps"])).backward()
                if (batch_no + 1) % int(cfg["training"]["gradient_accumulation_steps"]) == 0:
                    if float(cfg["training"]["gradient_clip_norm"]) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["training"]["gradient_clip_norm"]))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.item()) * len(batch_idx)
                seen += len(batch_idx)
                if max_train_batches is not None and batch_no + 1 >= max_train_batches:
                    break
            train_loss = total_loss / max(1, seen)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for start in range(0, len(val_idx), int(cfg["training"]["batch_size"])):
                    batch_idx = val_idx[start : start + int(cfg["training"]["batch_size"])]
                    if not len(batch_idx):
                        continue
                    batch_tokens = prepared.tokens[batch_idx]
                    rng_val = np.random.default_rng(int(cfg["experiment"]["seed"]) + epoch + 1000)
                    v1 = torch.as_tensor(np.stack([augment_tokens(t, rng_val, aug_cfg) for t in batch_tokens]), device=device)
                    v2 = torch.as_tensor(np.stack([augment_tokens(t, rng_val, aug_cfg) for t in batch_tokens]), device=device)
                    z1 = model(v1)
                    z2 = model(v2)
                    val_losses.append(float(nt_xent_loss(z1, z2, temperature=float(cfg["training"]["temperature"])).item()))
            val_loss = float(np.mean(val_losses)) if val_losses else train_loss
            scheduler.step()
            lr = float(optimizer.param_groups[0]["lr"])
            writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr})
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val,
                "model_config": asdict(model_cfg),
                "tokenizer_hash": prepared.token_manifest["tokenizer_hash"],
                "dataset_manifest_hash": hashlib.sha1(json.dumps(prepared.dataset_manifest, sort_keys=True).encode("utf-8")).hexdigest(),
                "experiment_config_hash": hashlib.sha1(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest(),
                "seed": int(cfg["experiment"]["seed"]),
            }
            save_checkpoint(run / "checkpoints" / f"epoch_{epoch + 1:03d}.pt", state)
            save_checkpoint(run / "checkpoints" / "last.pt", state)
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
                patience = 0
                state["best_val_loss"] = best_val
                save_checkpoint(run / "checkpoints" / "best.pt", state)
            else:
                patience += 1
            if patience >= int(cfg["training"]["early_stopping_patience"]):
                break
    save_json(run / "run_manifest.json", {"run_id": run_id, "device": device_name, "seed": int(cfg["experiment"]["seed"])})
    return {"run_id": run_id, "run_dir": str(run), "best_val_loss": best_val}
