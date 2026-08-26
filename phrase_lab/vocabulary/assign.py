from __future__ import annotations


def space_prefix(space: str, cfg: dict[str, str] | None = None) -> str:
    if cfg and f"token_prefix_{space}" in cfg:
        return str(cfg[f"token_prefix_{space}"])
    return {"melody": "M", "rhythm": "R", "combined": "C"}.get(space, space[:1].upper())


def format_token(space: str, cluster_id: int, width: int = 4, cfg: dict[str, str] | None = None) -> str:
    return f"{space_prefix(space, cfg)}_{int(cluster_id):0{int(width)}d}"

