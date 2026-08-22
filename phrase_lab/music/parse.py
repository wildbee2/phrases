from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import music21 as m21


def score_id_from_path(path: str | Path) -> str:
    p = Path(path)
    return hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:16]


def load_score(path: str | Path) -> m21.stream.Score:
    return m21.converter.parse(str(path))


def _is_percussion_part(part: m21.stream.Part) -> bool:
    try:
        inst = part.getInstrument(returnDefault=True)
        return bool(getattr(inst, "isPercussion", False))
    except Exception:
        return False


def iter_score_parts(score: m21.stream.Score) -> Iterable[m21.stream.Part]:
    for part in score.parts:
        if not _is_percussion_part(part):
            yield part


def part_identity(part: m21.stream.Part, index: int) -> tuple[str, str | None, str | None]:
    part_id = getattr(part, "id", None) or f"part_{index}"
    part_name = getattr(part, "partName", None)
    instrument_name = None
    try:
        instrument_name = part.getInstrument(returnDefault=True).instrumentName
    except Exception:
        instrument_name = None
    return str(part_id), part_name, instrument_name

