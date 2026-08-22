from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


SAFE_SUBSET_MAP = {
    "safe_deduplicated": ("subset:no_license_conflict", "subset:deduplicated"),
    "safe_rated_deduplicated": ("subset:no_license_conflict", "subset:rated_deduplicated"),
    "safe_all": ("subset:no_license_conflict",),
}


def _boolize(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.astype(str).str.lower().isin({"1", "true", "t", "yes", "y"})


def load_metadata(csv_path: str | Path, root: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    for col in [c for c in df.columns if c.startswith("subset:")]:
        df[col] = _boolize(df[col])
    if "mxl" in df.columns:
        df["mxl"] = df["mxl"].astype("string")
    if root is not None and "mxl" in df.columns:
        root = Path(root)
        def _resolve(p):
            if pd.isna(p) or not str(p):
                return None
            candidate = (root / str(p)).resolve()
            if candidate.exists():
                return str(candidate)
            alt = (root / "mxl" / str(p)).resolve()
            return str(alt)

        df["mxl_path"] = df["mxl"].apply(_resolve)
    return df


def _quality_score(df: pd.DataFrame) -> pd.Series:
    rating = pd.to_numeric(df.get("rating"), errors="coerce").fillna(0.0)
    views = pd.to_numeric(df.get("n_views"), errors="coerce").fillna(0.0)
    favs = pd.to_numeric(df.get("n_favorites"), errors="coerce").fillna(0.0)
    nratings = pd.to_numeric(df.get("n_ratings"), errors="coerce").fillna(0.0)
    return rating + 0.001 * views + 0.01 * favs + 0.1 * nratings


def filter_scores(
    df: pd.DataFrame,
    subset: str = "safe_deduplicated",
    max_scores: int | None = None,
    sampling: str = "random",
    random_seed: int = 42,
) -> pd.DataFrame:
    if subset not in SAFE_SUBSET_MAP:
        raise ValueError(f"Unsupported subset: {subset}")
    mask = pd.Series(True, index=df.index)
    for col in SAFE_SUBSET_MAP[subset]:
        if col not in df.columns:
            raise KeyError(f"Missing required column {col}")
        mask &= _boolize(df[col])
    if "mxl" in df.columns:
        mask &= df["mxl"].notna() & (df["mxl"].astype(str).str.len() > 0)
    out = df.loc[mask].copy()
    if max_scores is not None and len(out) > max_scores:
        if sampling == "random":
            out = out.sample(n=max_scores, random_state=random_seed)
        elif sampling == "quality":
            out = out.assign(_quality=_quality_score(out)).sort_values(["_quality", "score_id"], ascending=[False, True]).head(max_scores).drop(columns=["_quality"])
        elif sampling == "first":
            out = out.head(max_scores)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling}")
    return out.reset_index(drop=True)


def selected_scores_path(root: str | Path) -> Path:
    return Path(root) / "extracted" / "selected_scores.parquet"
