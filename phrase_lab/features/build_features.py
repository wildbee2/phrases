from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .normalize import l2_normalize
from .phrase_features import contour_vector, interval_features, phrase_shape_descriptors, relative_pitch_class_profile, rhythm_vector


def _embed_row(row: pd.Series, contour_steps: int, rhythm_steps: int, interval_clip: int) -> dict[str, np.ndarray]:
    notes = list(row["notes_json"]) if row["notes_json"] is not None else []
    contour = contour_vector(notes, steps=contour_steps)
    intervals = interval_features(notes, clip=interval_clip)
    pc = relative_pitch_class_profile(notes)
    rhythm = rhythm_vector(notes, steps=rhythm_steps)
    shape = phrase_shape_descriptors(notes, float(row["n_bars"]))
    melody = np.concatenate([contour, intervals, pc]).astype(np.float32)
    combined = np.concatenate([melody, rhythm, shape]).astype(np.float32)
    return {"melody": melody, "rhythm": rhythm, "combined": combined}


def build_embeddings(
    phrases: pd.DataFrame,
    feature_cfg: dict[str, Any],
    index_dir: str | Path,
    fit_pca: bool = True,
) -> dict[str, np.ndarray]:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    contour_steps = int(feature_cfg["contour_steps"])
    rhythm_steps = int(feature_cfg["rhythm_steps"])
    interval_clip = int(feature_cfg["interval_clip"])
    melody_dim = contour_steps + (2 * interval_clip + 5) + 12
    rhythm_dim = rhythm_steps * 2 + 8
    combined_dim = melody_dim + rhythm_dim + 7
    if phrases.empty:
        melody_e = np.zeros((0, melody_dim), dtype=np.float32)
        rhythm_e = np.zeros((0, rhythm_dim), dtype=np.float32)
        combined_e = np.zeros((0, combined_dim), dtype=np.float32)
        phrase_ids = np.array([], dtype=object)
        np.save(index_dir / "melody_embeddings.npy", melody_e)
        np.save(index_dir / "rhythm_embeddings.npy", rhythm_e)
        np.save(index_dir / "combined_embeddings.npy", combined_e)
        np.save(index_dir / "phrase_ids.npy", phrase_ids)
        return {"melody": melody_e, "rhythm": rhythm_e, "combined": combined_e, "phrase_ids": phrase_ids}
    embedded = [_embed_row(row, contour_steps, rhythm_steps, interval_clip) for _, row in phrases.iterrows()]
    phrase_ids = phrases["phrase_id"].to_numpy()
    melody = np.vstack([e["melody"] for e in embedded]).astype(np.float32)
    rhythm = np.vstack([e["rhythm"] for e in embedded]).astype(np.float32)
    combined = np.vstack([e["combined"] for e in embedded]).astype(np.float32)
    scaler_m = StandardScaler().fit(melody)
    scaler_r = StandardScaler().fit(rhythm)
    scaler_c = StandardScaler().fit(combined)
    melody_s = scaler_m.transform(melody)
    rhythm_s = scaler_r.transform(rhythm)
    combined_s = scaler_c.transform(combined)
    joblib.dump({"melody": scaler_m, "rhythm": scaler_r, "combined": scaler_c}, index_dir / "feature_scaler.joblib")
    if fit_pca and len(phrases) > 1:
        sample_size = min(int(feature_cfg.get("pca_fit_sample", len(phrases))), len(phrases))
        sample_idx = np.random.default_rng(42).choice(len(phrases), size=sample_size, replace=False) if sample_size < len(phrases) else np.arange(len(phrases))
        if len(sample_idx) >= 2:
            pca_dims = min(int(feature_cfg["pca_dimensions"]), melody_s.shape[1], rhythm_s.shape[1], combined_s.shape[1], len(sample_idx))
            pca_m = PCA(n_components=pca_dims).fit(melody_s[sample_idx])
            pca_r = PCA(n_components=min(pca_dims, rhythm_s.shape[1])).fit(rhythm_s[sample_idx])
            pca_c = PCA(n_components=min(pca_dims, combined_s.shape[1])).fit(combined_s[sample_idx])
            melody_e = pca_m.transform(melody_s)
            rhythm_e = pca_r.transform(rhythm_s)
            combined_e = pca_c.transform(combined_s)
            joblib.dump(pca_m, index_dir / "pca_melody.joblib")
            joblib.dump(pca_r, index_dir / "pca_rhythm.joblib")
            joblib.dump(pca_c, index_dir / "pca_combined.joblib")
        else:
            melody_e, rhythm_e, combined_e = melody_s, rhythm_s, combined_s
    else:
        melody_e, rhythm_e, combined_e = melody_s, rhythm_s, combined_s
    melody_e = l2_normalize(melody_e.astype(np.float32))
    rhythm_e = l2_normalize(rhythm_e.astype(np.float32))
    combined_e = l2_normalize(combined_e.astype(np.float32))
    np.save(index_dir / "melody_embeddings.npy", melody_e)
    np.save(index_dir / "rhythm_embeddings.npy", rhythm_e)
    np.save(index_dir / "combined_embeddings.npy", combined_e)
    np.save(index_dir / "phrase_ids.npy", phrase_ids)
    return {"melody": melody_e, "rhythm": rhythm_e, "combined": combined_e, "phrase_ids": phrase_ids}
