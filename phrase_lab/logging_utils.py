from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers = [logging.StreamHandler()]
    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path / "phrase_lab.log", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

